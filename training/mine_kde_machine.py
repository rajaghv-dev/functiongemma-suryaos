#!/usr/bin/env python3
"""
mine_kde_machine.py — real-data miner for function-calling training pairs.

Probes the LIVE KDE Plasma / Wayland machine and produces grounded
JSONL training pairs for the 12 tools in `tools/tool_schemas.json`.

The intent is to BALANCE the existing dispatch_pairs.jsonl which is 93%
dominated by `kde_krunner_launch`. Each "mine" function below pulls real
data from a different source on the machine; pairs include a
`provenance` field describing the exact command/file/line they came from.

Output: one JSONL file per mine under `dataset/real_sources/`.

Usage:
    python3 training/mine_kde_machine.py
    python3 training/mine_kde_machine.py --out dataset/real_sources/

Constraints honored:
  - Pure stdlib + subprocess (no extra deps).
  - Idempotent: per-mine output is sorted/deduped before write.
  - Every pair carries a `provenance` field.
  - Failures are SKIPped, never raised.
  - Per-tool cap: 200 pairs per mine to avoid flooding any single class.
"""

from __future__ import annotations

import argparse
import configparser
import io
import json
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# =============================================================================
# Paths & constants
# =============================================================================

REPO_ROOT       = Path(__file__).resolve().parent.parent
TOOL_SCHEMAS    = REPO_ROOT / "tools" / "tool_schemas.json"
DEFAULT_OUT_DIR = REPO_ROOT / "dataset" / "real_sources"

# Per-tool soft cap to prevent any one mine from dominating.
PER_TOOL_CAP = 200

# Map dotted tool names → mcp tool names (matches existing dispatch_pairs).
DOTTED_TO_MCP = {
    "kde.dialog.confirm":     "kde_dialog_confirm",
    "kde.krunner.launch":     "kde_krunner_launch",
    "kde.notifications.send": "kde_notifications_send",
    "kde.window.focus":       "kde_window_focus",
    "linux.battery.status":   "linux_battery_status",
    "linux.brightness.set":   "linux_brightness_set",
    "linux.disk.usage":       "linux_disk_usage",
    "linux.memory.usage":     "linux_memory_usage",
    "linux.metrics.summary":  "linux_metrics_summary",
    "linux.network.status":   "linux_network_status",
    "linux.service.status":   "linux_service_status",
    "linux.volume.set":       "linux_volume_set",
}


# =============================================================================
# Output helpers
# =============================================================================

def info(msg: str) -> None:    print(f"  [INFO]   {msg}")
def ok(msg: str) -> None:      print(f"  [OK]     {msg}")
def skip(msg: str) -> None:    print(f"  [SKIP]   {msg}")
def warn(msg: str) -> None:    print(f"  [WARN]   {msg}", file=sys.stderr)


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# =============================================================================
# Schemas — one mcp_schema per tool, used as the `tools` field of each pair
# =============================================================================

def load_schemas() -> dict[str, dict]:
    """Return mcp_name -> mcp_schema dict."""
    raw = json.loads(TOOL_SCHEMAS.read_text())
    out: dict[str, dict] = {}
    for dotted, entry in raw.items():
        mcp_name = entry.get("mcp_name") or DOTTED_TO_MCP.get(dotted)
        schema = entry.get("mcp_schema")
        if not mcp_name or not schema:
            warn(f"missing mcp_schema for {dotted}")
            continue
        out[mcp_name] = schema
    return out


# =============================================================================
# Pair construction
# =============================================================================

def make_pair(
    *,
    user_query: str,
    tool_name: str,
    arguments: dict,
    schema: dict,
    source: str,
    provenance: str,
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Call the right tool."},
            {"role": "user",   "content": user_query.strip()},
        ],
        "tools":   [schema],
        "target":  {"name": tool_name, "arguments": arguments},
        "source":  f"kde_machine_{source}",
        "provenance": provenance,
    }


def dedupe_and_cap(
    pairs: list[dict],
    cap_per_tool: int | dict[str, int] = PER_TOOL_CAP,
) -> list[dict]:
    """Stable dedupe by (user_query, target.name, json(arguments)); cap per tool.

    `cap_per_tool` may be a single int (applies to all tools) or a dict mapping
    tool name → cap (missing keys fall back to PER_TOOL_CAP).
    """
    seen: set[tuple] = set()
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        key = (
            p["messages"][1]["content"].strip().lower(),
            p["target"]["name"],
            json.dumps(p["target"].get("arguments", {}), sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        tool = p["target"]["name"]
        if isinstance(cap_per_tool, dict):
            cap = cap_per_tool.get(tool, PER_TOOL_CAP)
        else:
            cap = cap_per_tool
        if len(by_tool[tool]) >= cap:
            continue
        by_tool[tool].append(p)
    out: list[dict] = []
    for tool in sorted(by_tool):
        # Sort within each tool by query text for idempotency.
        out.extend(sorted(by_tool[tool], key=lambda x: x["messages"][1]["content"]))
    return out


def write_jsonl(path: Path, pairs: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            n += 1
    return n


def run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a command, return (rc, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            check=False,
        )
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout: {' '.join(shlex.quote(c) for c in cmd)}"
    except Exception as exc:  # pragma: no cover — defensive
        return 1, "", f"{type(exc).__name__}: {exc}"


# =============================================================================
# Mine 1: .desktop files → krunner_launch + window_focus
# =============================================================================

def _parse_desktop_file(path: Path) -> dict | None:
    """Return {name, generic, comment, exec, categories, hidden} or None on parse fail."""
    try:
        # Use configparser; .desktop is INI-like. interpolation off (% chars in Exec).
        cp = configparser.RawConfigParser(strict=False)
        cp.read(path, encoding="utf-8")
        if "Desktop Entry" not in cp:
            return None
        de = cp["Desktop Entry"]
        if de.get("NoDisplay", "false").lower() == "true":
            return None
        if de.get("Hidden", "false").lower() == "true":
            return None
        if de.get("Type", "").strip().lower() != "application":
            return None
        return {
            "name":       (de.get("Name") or "").strip(),
            "generic":    (de.get("GenericName") or "").strip(),
            "comment":    (de.get("Comment") or "").strip(),
            "exec":       (de.get("Exec") or "").strip(),
            "categories": (de.get("Categories") or "").strip(),
        }
    except (configparser.Error, OSError, UnicodeDecodeError):
        return None


def _exec_to_binary(exec_line: str) -> str | None:
    """Strip Exec= field codes (%U, %F, etc) and return the basename."""
    if not exec_line:
        return None
    # Remove field codes
    cleaned = re.sub(r"%[a-zA-Z]", "", exec_line).strip()
    parts = shlex.split(cleaned, posix=True) if cleaned else []
    if not parts:
        return None
    binary = Path(parts[0]).name
    # Reject env wrappers
    if binary in {"env", "sh", "bash", "flatpak", "snap"} and len(parts) > 1:
        # Try the next non-flag token
        for tok in parts[1:]:
            if not tok.startswith("-") and "=" not in tok:
                return Path(tok).name
    return binary


def mine_desktop_files(schemas: dict[str, dict]) -> list[dict]:
    info("scanning /usr/share/applications and ~/.local/share/applications")
    paths: list[Path] = []
    for d in [Path("/usr/share/applications"),
              Path.home() / ".local/share/applications",
              Path("/var/lib/flatpak/exports/share/applications"),
              Path("/var/lib/snapd/desktop/applications")]:
        if d.is_dir():
            paths.extend(sorted(d.glob("*.desktop")))
    info(f"found {len(paths)} .desktop files")

    pairs: list[dict] = []
    krunner = schemas.get("kde_krunner_launch")
    focus   = schemas.get("kde_window_focus")
    if not krunner or not focus:
        skip("krunner_launch or window_focus schema missing")
        return []

    apps_seen = 0
    for p in paths:
        meta = _parse_desktop_file(p)
        if not meta:
            continue
        name = meta["name"]
        generic = meta["generic"]
        comment = meta["comment"]
        binary = _exec_to_binary(meta["exec"])
        if not name or not binary:
            continue
        # Skip kcm_* settings panels — too many noisy pairs
        if p.name.startswith("kcm_"):
            continue
        apps_seen += 1
        prov = f".desktop:{p.name}"

        # Krunner pairs: a mix of phrasings, each grounded in real metadata.
        launch_queries: list[str] = [
            f"open {name}",
            f"launch {name}",
            f"start {name}",
        ]
        if generic and generic.lower() != name.lower():
            launch_queries.append(f"open the {generic.lower()}")
            launch_queries.append(f"launch {generic.lower()}")
        if comment:
            # Comment fields are typically noun phrases ("Image viewer",
            # "Browse all unicode characters"). Phrase as "open the X" /
            # "I need an X" so they remain grammatical.
            c = comment.rstrip(".").strip()
            if 8 <= len(c) <= 80 and "\n" not in c:
                cl = c.lower()
                # Pick prefix based on first word to keep grammar sane.
                first = cl.split(None, 1)[0]
                if first in {"a", "an", "the"}:
                    launch_queries.append(f"open {cl}")
                elif first.endswith("er") or first in {"file", "image",
                        "video", "audio", "music", "system", "settings",
                        "browser", "calendar"}:
                    launch_queries.append(f"open the {cl}")
                else:
                    # Verb-led comment like "Browse all unicode characters"
                    launch_queries.append(f"i want to {cl}")
        for q in launch_queries:
            pairs.append(make_pair(
                user_query=q,
                tool_name="kde_krunner_launch",
                arguments={"app": binary},
                schema=krunner,
                source="desktop_files",
                provenance=prov,
            ))

        # Window focus pairs: real apps that may have a window with this title.
        focus_queries: list[str] = [
            f"switch to {name}",
            f"focus {name}",
            f"bring {name} to the front",
        ]
        for q in focus_queries:
            pairs.append(make_pair(
                user_query=q,
                tool_name="kde_window_focus",
                arguments={"title": name},
                schema=focus,
                source="desktop_files",
                provenance=prov,
            ))

    ok(f"parsed {apps_seen} apps, generated {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Mine 2: systemctl → linux_service_status
# =============================================================================

def _systemctl_units(scope: str) -> list[tuple[str, str]]:
    """scope in {'system', 'user'}; return list of (unit_name, description)."""
    cmd = ["systemctl"]
    if scope == "user":
        cmd.append("--user")
    cmd += ["list-units", "--type=service", "--no-legend",
            "--no-pager", "--state=active", "--plain"]
    rc, out, err = run(cmd)
    if rc != 0:
        skip(f"systemctl {scope} list-units rc={rc}: {err.strip()[:120]}")
        return []
    units: list[tuple[str, str]] = []
    for line in out.splitlines():
        # active line columns: UNIT LOAD ACTIVE SUB DESCRIPTION
        # --plain drops leading bullet; split on whitespace, max 4 splits
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        unit, _load, active, _sub, desc = parts
        if active != "active":
            continue
        if not unit.endswith(".service"):
            continue
        units.append((unit, desc.strip()))
    return units


def mine_systemctl(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("linux_service_status")
    if not schema:
        skip("linux_service_status schema missing")
        return []

    pairs: list[dict] = []
    total_units = 0
    for scope in ("system", "user"):
        units = _systemctl_units(scope)
        info(f"systemctl {scope}: {len(units)} active services")
        total_units += len(units)
        for unit, desc in units:
            short = unit[: -len(".service")]  # strip .service
            prov = f"systemctl {scope}: {unit}"

            # 3-4 phrasings per unit, grounded in the unit name and description.
            queries = [
                f"is {short} running",
                f"check if {short} is active",
                f"status of {short} service",
            ]
            if desc and len(desc) <= 80 and desc.lower() not in {short.lower(), unit.lower()}:
                # Use real description as a more natural query.
                queries.append(f"is the {desc.lower().rstrip('.')} service up")
            for q in queries:
                pairs.append(make_pair(
                    user_query=q,
                    tool_name="linux_service_status",
                    arguments={"name": short},
                    schema=schema,
                    source=f"systemctl_{scope}",
                    provenance=prov,
                ))
    ok(f"systemctl: {total_units} units → {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Mine 3: qdbus6 org.kde.* services → kde_* tools (only describable services)
# =============================================================================

# Describable D-Bus services we know enough to ground a query in.
# Each entry: dbus name → (tool, args_factory(schemas), template_query)
KDE_SERVICE_DESCRIPTIONS: dict[str, tuple[str, dict, list[str]]] = {
    "org.kde.plasmashell": (
        "kde_notifications_send",
        {"title": "Reminder", "message": "from plasmashell"},
        [
            "send a desktop alert via plasma",
            "show me a plasma notification",
        ],
    ),
    "org.kde.KWin": (
        "kde_window_focus",
        {"title": "terminal"},
        [
            "switch focus using kwin",
            "use kwin to bring the terminal forward",
        ],
    ),
    "org.kde.krunner": (
        "kde_krunner_launch",
        {"app": "konsole"},
        [
            "use krunner to open konsole",
            "ask krunner to launch the terminal",
        ],
    ),
    "org.kde.kglobalaccel": (
        "kde_krunner_launch",
        {"app": "krunner"},
        [
            "trigger the global launcher",
        ],
    ),
    "org.kde.StatusNotifierWatcher": (
        "kde_notifications_send",
        {"title": "Status", "message": "tray update"},
        [
            "push a tray notification",
        ],
    ),
    "org.kde.kded6": (
        "linux_service_status",
        {"name": "plasma-kded"},
        [
            "is the kde daemon running",
            "check status of kded",
        ],
    ),
    "org.kde.Solid.PowerManagement": (
        "linux_battery_status",
        {},
        [
            "ask power management about battery",
            "what does solid say about my battery",
        ],
    ),
    "org.kde.KScreen": (
        "linux_brightness_set",
        {"direction": "down"},
        [
            "use kscreen to dim the display",
        ],
    ),
}


def mine_qdbus6(schemas: dict[str, dict]) -> list[dict]:
    rc, out, err = run(["qdbus6"])
    if rc != 0:
        skip(f"qdbus6 rc={rc}: {err.strip()[:120]}")
        return []
    services = sorted({s.strip() for s in out.splitlines() if s.strip()})
    kde_services = [s for s in services if s.startswith("org.kde.")]
    info(f"qdbus6: {len(services)} services, {len(kde_services)} org.kde.*")

    pairs: list[dict] = []
    for svc in kde_services:
        if svc not in KDE_SERVICE_DESCRIPTIONS:
            continue  # skip unknown — we can't honestly describe its purpose
        tool, args, queries = KDE_SERVICE_DESCRIPTIONS[svc]
        schema = schemas.get(tool)
        if not schema:
            continue
        prov = f"qdbus6 service: {svc}"
        for q in queries:
            pairs.append(make_pair(
                user_query=q,
                tool_name=tool,
                arguments=args,
                schema=schema,
                source="qdbus6",
                provenance=prov,
            ))
    ok(f"qdbus6: produced {len(pairs)} grounded pairs from {len(KDE_SERVICE_DESCRIPTIONS)} known services")
    return pairs


# =============================================================================
# Mine 4: /proc/meminfo + battery + backlight → grounded numeric pairs
# =============================================================================

def _read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into a {key: kB} dict."""
    info_dict: dict[str, int] = {}
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z()_]+):\s+(\d+)\s*kB", line)
        if m:
            info_dict[m.group(1)] = int(m.group(2))
    return info_dict


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_str(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def mine_meminfo_battery_brightness(schemas: dict[str, dict]) -> list[dict]:
    pairs: list[dict] = []

    # ---- /proc/meminfo ----
    mem = _read_meminfo()
    mem_schema = schemas.get("linux_memory_usage")
    if mem and mem_schema and "MemTotal" in mem and "MemAvailable" in mem:
        total = mem["MemTotal"]
        avail = mem["MemAvailable"]
        used_pct = round(100 * (total - avail) / total, 1)
        prov = f"/proc/meminfo: MemTotal={total}kB MemAvailable={avail}kB used={used_pct}%"
        # Phrasings vary by load level — but all map to linux_memory_usage.
        queries = [
            "how much ram is being used",
            "show memory usage",
            "is my system running low on memory",
            "what's my ram looking like",
            "free memory check",
        ]
        if used_pct >= 75:
            queries += [
                "is my memory full",
                "am i out of ram",
                "memory feels tight, check it",
            ]
        elif used_pct <= 30:
            queries += [
                "how much ram is free",
                "do i have memory headroom",
            ]
        for q in queries:
            pairs.append(make_pair(
                user_query=q,
                tool_name="linux_memory_usage",
                arguments={},
                schema=mem_schema,
                source="proc_meminfo",
                provenance=prov,
            ))
        ok(f"meminfo: used={used_pct}% → {len(queries)} pairs")
    else:
        skip("/proc/meminfo unavailable or missing keys")

    # ---- battery ----
    bat_schema = schemas.get("linux_battery_status")
    bat_dir = Path("/sys/class/power_supply")
    if bat_schema and bat_dir.is_dir():
        batteries = sorted([d for d in bat_dir.iterdir()
                            if d.is_dir() and d.name.upper().startswith("BAT")])
        if not batteries:
            skip("no BAT* under /sys/class/power_supply")
        for bat in batteries:
            cap = _read_int(bat / "capacity")
            status = _read_str(bat / "status")
            if cap is None or status is None:
                continue
            prov = f"{bat}: capacity={cap} status={status}"
            queries = [
                "how's the battery",
                "battery percentage",
                "is the laptop charging",
                "check battery status",
            ]
            if cap is not None and cap <= 25:
                queries += [
                    "battery dying?",
                    "is my battery low",
                    "how much battery left",
                ]
            elif cap is not None and cap >= 90:
                queries += [
                    "is the battery full",
                    "fully charged?",
                ]
            if status and status.lower() == "charging":
                queries.append("am i plugged in")
            for q in queries:
                pairs.append(make_pair(
                    user_query=q,
                    tool_name="linux_battery_status",
                    arguments={},
                    schema=bat_schema,
                    source="sys_battery",
                    provenance=prov,
                ))
            ok(f"battery {bat.name}: capacity={cap}% status={status}")
    elif not bat_schema:
        skip("linux_battery_status schema missing")
    else:
        skip("/sys/class/power_supply not a directory")

    # ---- backlight (brightness) ----
    bl_schema = schemas.get("linux_brightness_set")
    bl_dir = Path("/sys/class/backlight")
    if bl_schema and bl_dir.is_dir():
        bls = sorted([d for d in bl_dir.iterdir() if d.is_dir()])
        if not bls:
            skip("no backlight devices in /sys/class/backlight")
        for bl in bls:
            curr = _read_int(bl / "brightness")
            maxv = _read_int(bl / "max_brightness")
            if curr is None or not maxv:
                continue
            pct = round(100 * curr / maxv)
            prov = f"{bl}: brightness={curr} max={maxv} ({pct}%)"
            # Direction depends on current level.
            if pct >= 70:
                pairs.append(make_pair(
                    user_query="dim the screen",
                    tool_name="linux_brightness_set",
                    arguments={"direction": "down"},
                    schema=bl_schema,
                    source="sys_backlight",
                    provenance=prov,
                ))
                pairs.append(make_pair(
                    user_query="lower the brightness",
                    tool_name="linux_brightness_set",
                    arguments={"direction": "down"},
                    schema=bl_schema,
                    source="sys_backlight",
                    provenance=prov,
                ))
                pairs.append(make_pair(
                    user_query="too bright, turn it down",
                    tool_name="linux_brightness_set",
                    arguments={"direction": "down", "step": 20},
                    schema=bl_schema,
                    source="sys_backlight",
                    provenance=prov,
                ))
            else:
                pairs.append(make_pair(
                    user_query="brighten the display",
                    tool_name="linux_brightness_set",
                    arguments={"direction": "up"},
                    schema=bl_schema,
                    source="sys_backlight",
                    provenance=prov,
                ))
                pairs.append(make_pair(
                    user_query="make the screen brighter",
                    tool_name="linux_brightness_set",
                    arguments={"direction": "up"},
                    schema=bl_schema,
                    source="sys_backlight",
                    provenance=prov,
                ))
                pairs.append(make_pair(
                    user_query="bump brightness up by 20",
                    tool_name="linux_brightness_set",
                    arguments={"direction": "up", "step": 20},
                    schema=bl_schema,
                    source="sys_backlight",
                    provenance=prov,
                ))
            ok(f"backlight {bl.name}: {pct}%")
    elif not bl_schema:
        skip("linux_brightness_set schema missing")
    else:
        skip("/sys/class/backlight not a directory")

    return pairs


# =============================================================================
# Mine 5: nmcli → linux_network_status
# =============================================================================

def mine_nmcli(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("linux_network_status")
    if not schema:
        skip("linux_network_status schema missing")
        return []

    pairs: list[dict] = []

    # Devices: TYPE:DEVICE:STATE:CONNECTION
    rc, out, err = run(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device"])
    if rc != 0:
        skip(f"nmcli device rc={rc}: {err.strip()[:120]}")
    else:
        for line in out.splitlines():
            # Format: DEVICE:TYPE:STATE:CONNECTION (colons inside fields are escaped \: by nmcli -t)
            cols = re.split(r"(?<!\\):", line)
            cols = [c.replace(r"\:", ":") for c in cols]
            if len(cols) < 4:
                continue
            dev, dtype, state, conn = cols[:4]
            if not dev or dev == "lo":
                continue
            prov = f"nmcli device: {dev} type={dtype} state={state} conn={conn}"
            qbase = [
                "am i online",
                "what network am i on",
                "show network status",
                "is wifi connected",
            ]
            qextra: list[str] = []
            if dtype == "wifi":
                qextra += [
                    f"is the wifi adapter {dev} up",
                    "what wifi am i connected to",
                ]
            elif dtype == "ethernet":
                qextra += [
                    f"is ethernet {dev} connected",
                    "is wired networking up",
                ]
            for q in qbase + qextra:
                pairs.append(make_pair(
                    user_query=q,
                    tool_name="linux_network_status",
                    arguments={},
                    schema=schema,
                    source="nmcli_device",
                    provenance=prov,
                ))

    # Active connections — pull SSIDs / connection names
    rc, out, err = run(["nmcli", "-t", "-f", "NAME,TYPE,DEVICE", "con", "show", "--active"])
    if rc != 0:
        skip(f"nmcli con show rc={rc}: {err.strip()[:120]}")
    else:
        for line in out.splitlines():
            cols = re.split(r"(?<!\\):", line)
            cols = [c.replace(r"\:", ":") for c in cols]
            if len(cols) < 3:
                continue
            name, ctype, dev = cols[:3]
            if not name:
                continue
            prov = f"nmcli con active: {name} type={ctype} dev={dev}"
            queries = [
                f"am i still on {name}",
                f"check connection {name}",
                "is my network connection active",
            ]
            for q in queries:
                pairs.append(make_pair(
                    user_query=q,
                    tool_name="linux_network_status",
                    arguments={},
                    schema=schema,
                    source="nmcli_active",
                    provenance=prov,
                ))

    ok(f"nmcli: produced {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Mine 6: pactl → linux_volume_set
# =============================================================================

def mine_pactl(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("linux_volume_set")
    if not schema:
        skip("linux_volume_set schema missing")
        return []

    pairs: list[dict] = []

    # List sinks (audio outputs).
    rc, out, err = run(["pactl", "list", "short", "sinks"])
    if rc != 0:
        skip(f"pactl list short sinks rc={rc}: {err.strip()[:120]}")
        return []
    sinks: list[str] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            sinks.append(parts[1])
    info(f"pactl: {len(sinks)} sinks")

    # Default sink + its volume
    rc, default_sink, _ = run(["pactl", "get-default-sink"])
    default_sink = default_sink.strip() if rc == 0 else ""

    current_pct: int | None = None
    if default_sink:
        rc, vol_out, _ = run(["pactl", "get-sink-volume", default_sink])
        if rc == 0:
            # Volume line includes "  XX%"
            m = re.search(r"(\d+)%", vol_out)
            if m:
                current_pct = int(m.group(1))

    def add(query: str, args: dict, prov: str, src: str = "pactl") -> None:
        pairs.append(make_pair(
            user_query=query,
            tool_name="linux_volume_set",
            arguments=args,
            schema=schema,
            source=src,
            provenance=prov,
        ))

    base_prov = f"pactl default sink: {default_sink} ~{current_pct}%"

    # Generic volume queries grounded in real default-sink existence.
    if default_sink:
        add("turn the volume up", {"direction": "up"}, base_prov)
        add("turn the volume down", {"direction": "down"}, base_prov)
        add("make it louder", {"direction": "up"}, base_prov)
        add("make it quieter", {"direction": "down"}, base_prov)
        add("increase audio level", {"direction": "up"}, base_prov)
        add("lower the audio", {"direction": "down"}, base_prov)
        add("bump volume up by 10", {"direction": "up", "step": 10}, base_prov)
        add("bump volume down by 10", {"direction": "down", "step": 10}, base_prov)

        if current_pct is not None and current_pct >= 80:
            add("too loud, lower it", {"direction": "down", "step": 20}, base_prov)
            add("turn it down a lot", {"direction": "down", "step": 20}, base_prov)
        elif current_pct is not None and current_pct <= 20:
            add("can't hear, turn it up", {"direction": "up", "step": 20}, base_prov)
            add("crank up the volume", {"direction": "up", "step": 30}, base_prov)

    # Per-sink prov (provides extra grounding without changing schema args).
    for sink in sinks:
        prov = f"pactl sink: {sink}"
        add(f"raise volume on {sink}", {"direction": "up"}, prov, src="pactl_sink")
        add(f"lower volume on {sink}", {"direction": "down"}, prov, src="pactl_sink")

    ok(f"pactl: produced {len(pairs)} raw pairs (default sink {current_pct}%)")
    return pairs


# =============================================================================
# Mine 7: journalctl warnings → linux_service_status
# =============================================================================

def mine_journalctl(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("linux_service_status")
    if not schema:
        skip("linux_service_status schema missing")
        return []

    rc, out, err = run(
        ["journalctl", "-p", "warning", "-n", "200", "--no-pager",
         "--output=short-iso", "--quiet"],
        timeout=15,
    )
    if rc != 0:
        skip(f"journalctl rc={rc}: {err.strip()[:120]}")
        return []

    # Pattern: "<host> <unit>[<pid>]: ..."  — the unit is the second whitespace token
    # after the timestamp + host. We look for "name[pid]:" tokens.
    unit_re = re.compile(r"\s([a-zA-Z0-9_.@-]+)\[\d+\]:\s")
    seen_units: dict[str, str] = {}  # unit -> first warning text
    for line in out.splitlines():
        m = unit_re.search(line)
        if not m:
            continue
        unit = m.group(1).split("@")[0]
        if not unit or len(unit) > 64:
            continue
        # Skip pure numeric or kernel
        if unit in {"kernel", "systemd", "audit", "dbus-daemon"}:
            continue
        if unit not in seen_units:
            # Capture the line tail (post colon) for prov.
            tail = line.split(":", 3)[-1].strip() if ":" in line else line
            seen_units[unit] = tail[:160]

    info(f"journalctl: {len(seen_units)} distinct service-like units in warnings")
    pairs: list[dict] = []
    for unit, warn_text in sorted(seen_units.items()):
        prov = f"journalctl warning: {unit}: {warn_text}"
        queries = [
            f"is {unit} working",
            f"check {unit} status",
            f"why is {unit} complaining",
            f"is the {unit} service healthy",
        ]
        for q in queries:
            pairs.append(make_pair(
                user_query=q,
                tool_name="linux_service_status",
                arguments={"name": unit},
                schema=schema,
                source="journalctl",
                provenance=prov,
            ))
    ok(f"journalctl: produced {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Mine 8: KWin window list → kde_window_focus
# =============================================================================

def mine_window_list(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("kde_window_focus")
    if not schema:
        skip("kde_window_focus schema missing")
        return []

    pairs: list[dict] = []
    titles: list[str] = []

    # Try KWin scripting — works on Wayland but requires loading a script.
    # Easier path: use `qdbus6 org.kde.KWin /KWin` to enumerate methods, then
    # try the (older) activeWindow / activeClient API. We fall back to dbus
    # introspection.
    rc, _out, _err = run(["qdbus6", "org.kde.KWin"])
    if rc != 0:
        skip(f"qdbus6 KWin not available: {_err.strip()[:120]}")
    else:
        # KWin doesn't expose a window-list method directly; try a scripting
        # one-liner that prints captions. This may fail on locked-down envs.
        kwin_script = (
            "var clients = workspace.windowList ? workspace.windowList() : "
            "(workspace.clientList ? workspace.clientList() : []); "
            "for (var i = 0; i < clients.length; ++i) "
            "{ print(clients[i].caption); }"
        )
        rc, sid_out, err = run(
            ["qdbus6", "org.kde.KWin", "/Scripting", "loadDeclarativeScript",
             "/dev/stdin", "win-list-mine"],
            timeout=5,
        )
        # Most setups won't accept loadDeclarativeScript here. Treat as best-effort.
        if rc != 0:
            skip(f"kwin scripting not usable: {err.strip()[:120] or 'rc=' + str(rc)}")

    # Fallback / supplement: try wmctrl in case of XWayland.
    rc, wm_out, wm_err = run(["wmctrl", "-l"])
    if rc == 0:
        for line in wm_out.splitlines():
            # wmctrl -l format: <wid> <desktop> <host> <title>
            parts = line.split(None, 3)
            if len(parts) >= 4:
                titles.append(parts[3].strip())
        info(f"wmctrl: {len(titles)} windows (may be empty under pure Wayland)")
    else:
        skip(f"wmctrl unavailable: {wm_err.strip()[:120] or 'rc=' + str(rc)}")
        info("note: pure Wayland sessions don't support wmctrl; KWin scripting "
             "would be needed for a real window list. Limitation documented.")

    # Filter junk titles.
    titles = [t for t in titles if t and len(t) <= 120]
    seen: set[str] = set()
    for title in titles:
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        prov = f"wmctrl -l: {title!r}"
        for verb in ("switch to", "focus", "bring", "raise"):
            q = f"{verb} {title}"
            pairs.append(make_pair(
                user_query=q,
                tool_name="kde_window_focus",
                arguments={"title": title},
                schema=schema,
                source="window_list",
                provenance=prov,
            ))
    ok(f"window list: {len(seen)} unique titles → {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Mine 9: shell history → various tools (real CLI → NL)
# =============================================================================

# Map command prefix → (tool, arg_factory, NL queries).  arg_factory takes
# the matched shell line and returns the arguments dict.
def _hist_rules() -> list[tuple[re.Pattern, str, callable, list[str]]]:
    return [
        (re.compile(r"^\s*(?:sudo\s+)?free(?:\s|$)"),
         "linux_memory_usage",
         lambda line: {},
         ["how much memory is being used", "check ram usage"]),
        (re.compile(r"^\s*(?:sudo\s+)?(?:df|du)(?:\s|$)"),
         "linux_disk_usage",
         lambda line: {},
         ["check disk space", "how full is the disk"]),
        (re.compile(r"^\s*(?:sudo\s+)?systemctl\s+(?:status|is-active)\s+(\S+)"),
         "linux_service_status",
         lambda line: {"name": re.search(r"(?:status|is-active)\s+(\S+)", line).group(1).rstrip(".service")},
         ["is {name} running", "status of {name}"]),
        (re.compile(r"^\s*(?:sudo\s+)?nmcli(?:\s|$)"),
         "linux_network_status",
         lambda line: {},
         ["am i online", "check network"]),
        (re.compile(r"^\s*(?:sudo\s+)?(?:iwconfig|iw\s+dev)"),
         "linux_network_status",
         lambda line: {},
         ["is wifi up", "show wifi status"]),
        (re.compile(r"^\s*(?:sudo\s+)?(?:upower|acpi)(?:\s|$)"),
         "linux_battery_status",
         lambda line: {},
         ["how's the battery", "battery status"]),
        (re.compile(r"^\s*(?:sudo\s+)?pactl(?:\s|$)"),
         "linux_volume_set",
         lambda line: {"direction": "up"} if "+" in line else {"direction": "down"},
         ["adjust the volume"]),
        (re.compile(r"^\s*(?:sudo\s+)?brightnessctl(?:\s|$)"),
         "linux_brightness_set",
         lambda line: {"direction": "down"} if "-" in line else {"direction": "up"},
         ["change screen brightness"]),
        (re.compile(r"^\s*kstart\s+(\S+)"),
         "kde_krunner_launch",
         lambda line: {"app": re.search(r"kstart\s+(\S+)", line).group(1)},
         ["open {app}", "launch {app}"]),
    ]


def _read_history(path: Path) -> list[str]:
    """Best-effort read of bash/zsh history. Returns deduped command strings."""
    if not path.is_file():
        return []
    try:
        raw = path.read_text(errors="replace")
    except OSError:
        return []
    out: list[str] = []
    for line in raw.splitlines():
        # zsh extended history: ": 1700000000:0;cmd"
        if line.startswith(":"):
            idx = line.find(";")
            if idx != -1:
                line = line[idx + 1:]
        line = line.strip()
        if line:
            out.append(line)
    return out


def mine_history(schemas: dict[str, dict]) -> list[dict]:
    rules = _hist_rules()
    pairs: list[dict] = []
    total_lines = 0
    for hist_path in [Path.home() / ".bash_history", Path.home() / ".zsh_history"]:
        lines = _read_history(hist_path)
        total_lines += len(lines)
        if not lines:
            skip(f"empty/missing history: {hist_path}")
            continue
        info(f"{hist_path}: {len(lines)} commands")
        for line in lines:
            for pat, tool, args_fn, queries in rules:
                if not pat.search(line):
                    continue
                schema = schemas.get(tool)
                if not schema:
                    continue
                try:
                    args = args_fn(line)
                except Exception:
                    continue
                prov = f"{hist_path.name}: {line[:160]}"
                for q in queries:
                    try:
                        rendered = q.format(**args)
                    except (KeyError, IndexError):
                        rendered = q
                    pairs.append(make_pair(
                        user_query=rendered,
                        tool_name=tool,
                        arguments=args,
                        schema=schema,
                        source="history",
                        provenance=prov,
                    ))
                break  # one rule per line
    ok(f"history: {total_lines} total lines → {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Driver
# =============================================================================

# Per-mine cap overrides. The existing dataset is 93% kde_krunner_launch, so
# we deliberately throttle that tool when it appears as a side-product in a
# mine that's primarily about a different tool.
MINE_CAPS: dict[str, dict[str, int]] = {
    # desktop_files yields BOTH krunner_launch (already saturated upstream)
    # and window_focus (heavily under-represented). Cap krunner low to
    # honour the "do not flood" rule; keep focus near the global cap.
    "desktop_files": {"kde_krunner_launch": 80, "kde_window_focus": 200},
}

MINES = [
    ("desktop_files",            mine_desktop_files),
    ("systemctl",                mine_systemctl),
    ("qdbus6",                   mine_qdbus6),
    ("meminfo_battery_brightness", mine_meminfo_battery_brightness),
    ("nmcli",                    mine_nmcli),
    ("pactl",                    mine_pactl),
    ("journalctl",               mine_journalctl),
    ("window_list",              mine_window_list),
    ("history",                  mine_history),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help="Output directory (default: dataset/real_sources/)")
    args = parser.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not TOOL_SCHEMAS.is_file():
        warn(f"missing tool schemas at {TOOL_SCHEMAS}")
        return 2
    schemas = load_schemas()
    info(f"loaded schemas for: {sorted(schemas)}")

    summary: dict[str, dict[str, int]] = {}     # mine -> tool -> count
    written: dict[str, Path] = {}                # mine -> path

    for name, fn in MINES:
        banner(f"mine_{name}")
        try:
            raw = fn(schemas)
        except Exception as exc:
            warn(f"{name} crashed (caught): {type(exc).__name__}: {exc}")
            raw = []
        cap = MINE_CAPS.get(name, PER_TOOL_CAP)
        pairs = dedupe_and_cap(raw, cap)
        per_tool: dict[str, int] = defaultdict(int)
        for p in pairs:
            per_tool[p["target"]["name"]] += 1
        summary[name] = dict(per_tool)
        out_path = out_dir / f"mine_{name}.jsonl"
        n = write_jsonl(out_path, pairs)
        written[name] = out_path
        ok(f"wrote {n} pairs → {out_path.relative_to(REPO_ROOT)}")

    # ------- Final summary -------
    banner("SUMMARY")
    total = 0
    grand_per_tool: dict[str, int] = defaultdict(int)
    for name, _ in MINES:
        per = summary.get(name, {})
        n = sum(per.values())
        total += n
        if not per:
            print(f"  mine_{name:<28s} 0 pairs (skipped)")
            continue
        print(f"  mine_{name:<28s} {n:4d} pairs")
        for tool in sorted(per):
            cnt = per[tool]
            grand_per_tool[tool] += cnt
            print(f"      {tool:<26s} {cnt:4d}")
    print()
    print(f"  Total real pairs: {total}")
    print(f"  Per tool (across all mines):")
    for tool in sorted(grand_per_tool):
        print(f"      {tool:<26s} {grand_per_tool[tool]:4d}")
    print()
    print(f"  Output dir: {out_dir}")
    for name in sorted(written):
        print(f"    {written[name].relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
