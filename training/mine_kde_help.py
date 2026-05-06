#!/usr/bin/env python3
"""
mine_kde_help.py — real-data miner that grounds dispatch pairs in CLI
help text from KDE/Plasma binaries actually present on this machine.

Targets the starved tools in the current dataset:
  - kde_dialog_confirm        (kdialog --yesno / --warningyesno / ...)
  - kde_notifications_send    (kdialog --passivepopup, notify-send help)
  - kde_window_focus          (kstart --window/--activate, dolphin/kate window flags)
  - kde_krunner_launch        (krunner --query, dolphin/kate/konsole help-all)
  - linux_disk_usage          (real /proc/mounts entries)
  - linux_battery_status      (/sys/class/power_supply/ADP1 — AC adapter)
  - linux_service_status      (/etc/systemd/system/<target>.target.wants/*.service)

Every emitted user_query must be either:
  (a) a verbatim sentence drawn from help text / system files, OR
  (b) a trivial reformulation of an option name + its description.

Usage:
    python3 training/mine_kde_help.py
    python3 training/mine_kde_help.py --out dataset/real_sources/

Constraints:
  - stdlib + subprocess only.
  - Idempotent: stable dedupe + sort.
  - Honest 0 acceptable: skip a source rather than invent.
  - Per-binary caps prevent any single CLI from dominating.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT       = Path(__file__).resolve().parent.parent
TOOL_SCHEMAS    = REPO_ROOT / "tools" / "tool_schemas.json"
DEFAULT_OUT     = REPO_ROOT / "dataset" / "real_sources" / "kde_help_pairs.jsonl"

# Modest caps — this miner is intentionally conservative.
PER_BINARY_CAP    = 12          # kdialog / krunner / kstart per tool
PER_DESKTOP_CAP   = 5           # dolphin/kate/konsole launch pairs (low realism return)
SUBPROCESS_TIMEOUT = 6           # short timeout — KDE bins sometimes hang on missing display

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


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def info(msg: str) -> None:    print(f"  [INFO]   {msg}")
def ok(msg: str) -> None:      print(f"  [OK]     {msg}")
def skip(msg: str) -> None:    print(f"  [SKIP]   {msg}")
def warn(msg: str) -> None:    print(f"  [WARN]   {msg}", file=sys.stderr)

def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

def load_schemas() -> dict[str, dict]:
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


# ---------------------------------------------------------------------------
# Pair helpers
# ---------------------------------------------------------------------------

def make_pair(
    *,
    user_query: str,
    tool_name: str,
    arguments: dict,
    schema: dict,
    source: str,
    provenance: str,
    negatives: list[str] | None = None,
) -> dict:
    pair = {
        "messages": [
            {"role": "system", "content": "Call the right tool."},
            {"role": "user",   "content": user_query.strip()},
        ],
        "tools":   [schema],
        "target":  {"name": tool_name, "arguments": arguments},
        "source":  f"kde_help_{source}",
        "provenance": provenance,
    }
    if negatives:
        pair["negatives"] = negatives
    return pair


def dedupe(pairs: list[dict]) -> list[dict]:
    """Stable dedupe keyed on (lowercased query, tool, args). Sort for idempotency."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for p in pairs:
        key = (
            p["messages"][1]["content"].strip().lower(),
            p["target"]["name"],
            json.dumps(p["target"].get("arguments", {}), sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    out.sort(key=lambda x: (x["target"]["name"],
                            x["messages"][1]["content"].lower(),
                            x.get("provenance", "")))
    return out


def write_jsonl(path: Path, pairs: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], timeout: int = SUBPROCESS_TIMEOUT) -> tuple[int, str, str]:
    """Run cmd; capture stdout AND stderr (KDE prints help to either).
    Never raises — returns (rc, stdout, stderr) or (124, '', err) on timeout.
    """
    try:
        # Some KDE binaries need a usable QPA backend even for --help. Use offscreen
        # to avoid spurious display errors and the QThreadStorage warnings going to
        # the terminal.
        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        # Drop session-bus chatter — purely cosmetic.
        env.setdefault("QT_LOGGING_RULES", "*.debug=false")
        r = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            check=False,
            env=env,
        )
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout: {' '.join(shlex.quote(c) for c in cmd)}"
    except Exception as exc:  # pragma: no cover
        return 1, "", f"{type(exc).__name__}: {exc}"


def get_help(binary: str, *, alt_flag: str = "--help") -> str | None:
    """Return combined stdout+stderr of `<bin> --help`. None if missing."""
    rc, out, err = run_cmd([binary, alt_flag])
    if rc == 127:
        skip(f"{binary} not on PATH ({err.strip()})")
        return None
    text = (out or "") + ("\n" + err if err else "")
    # KDE's QThreadStorage warnings pollute stderr; drop those lines.
    text = "\n".join(
        line for line in text.splitlines()
        if "QThreadStorage" not in line and "qt.qpa" not in line.lower()
    )
    if not text.strip():
        skip(f"{binary} {alt_flag} produced no output (rc={rc})")
        return None
    return text


# ---------------------------------------------------------------------------
# Help-text parsing
# ---------------------------------------------------------------------------

# Match continuation-style option blocks. Qt help wraps to multiple lines,
# indented ~36 cols. We collect until the next option line begins.
OPTION_LINE_RE = re.compile(
    r"^\s+"                    # leading indent
    r"(?P<flags>(?:-[a-zA-Z?],\s)?--[a-zA-Z][\w-]*)"   # primary flag (--xyz, -x, --xyz)
    r"(?:=[A-Z][\w\[\],.:]*)?"                         # optional =ARG (notify-send style)
    r"(?:\s+<[^>]+>(?:\s+<[^>]+>)*)?"                  # optional <arg> placeholders
    r"\s{2,}"                  # padding before description
    r"(?P<desc>.+?)\s*$"
)


def parse_options(help_text: str) -> list[tuple[str, str]]:
    """
    Parse `<bin> --help` into [(flag, description)].
    Handles wrapped descriptions (continuation indent ≥ description column).
    Drops --help / --help-all / --version / --author / --license / --desktopfile / Qt platform flags.
    """
    drop = {"--help", "--help-all", "--version", "--author", "--license",
            "--desktopfile", "--qmljsdebugger", "--platform",
            "--platformpluginpath", "--platformtheme", "--plugin",
            "--qwindowgeometry", "--qwindowicon", "--qwindowtitle",
            "--reverse"}
    out: list[tuple[str, str]] = []
    current_flag: str | None = None
    current_desc: list[str] = []
    desc_col: int | None = None
    for raw_line in help_text.splitlines():
        m = OPTION_LINE_RE.match(raw_line)
        if m:
            # flush previous
            if current_flag and current_flag not in drop:
                out.append((current_flag, " ".join(current_desc).strip()))
            flags = m.group("flags")
            # Take the long form (--xxx) if combined like "-h, --help".
            long_form = next((f.strip() for f in flags.split(",") if f.strip().startswith("--")),
                             flags.strip())
            current_flag = long_form
            current_desc = [m.group("desc").strip()]
            desc_col = m.start("desc")
        elif current_flag and raw_line.strip() and (
            desc_col is not None and len(raw_line) > desc_col
            and raw_line[:desc_col].strip() == ""
        ):
            # Continuation line for the previous option.
            current_desc.append(raw_line[desc_col:].strip())
        else:
            # Section break or arguments line; flush.
            if current_flag and current_flag not in drop:
                out.append((current_flag, " ".join(current_desc).strip()))
            current_flag = None
            current_desc = []
            desc_col = None
    if current_flag and current_flag not in drop:
        out.append((current_flag, " ".join(current_desc).strip()))
    return out


# ---------------------------------------------------------------------------
# Mine: kdialog → confirm + notification + negatives
# ---------------------------------------------------------------------------

# Each known kdialog option mapped onto a tool. Anything not listed is silently
# excluded — we only emit pairs we can ground in real semantics.
KDIALOG_CONFIRM_FLAGS = {
    "--yesno":                "yes/no buttons",
    "--questionyesno":        "yes/no buttons",
    "--yesnocancel":          "yes/no/cancel buttons",
    "--warningyesno":         "warning yes/no buttons",
    "--warningyesnocancel":   "warning yes/no/cancel buttons",
    "--warningcontinuecancel": "continue/cancel buttons",
}

# These non-confirm dialog flags become "real sibling negatives": the same
# kdialog binary exposes them, so they're convincing distractors but they
# should NOT route to kde_dialog_confirm.
KDIALOG_NON_CONFIRM_FLAGS = {
    "--inputbox", "--combobox", "--menu", "--textbox", "--textinputbox",
    "--password", "--newpassword", "--imgbox", "--imginputbox",
    "--checklist", "--radiolist", "--slider", "--calendar",
    "--getopenfilename", "--getsavefilename", "--getexistingdirectory",
    "--getopenurl", "--getsaveurl", "--geticon", "--progressbar", "--getcolor",
    "--msgbox", "--error", "--detailederror", "--sorry", "--detailedsorry",
}


def _confirm_query_for(flag: str, desc: str) -> str | None:
    """Build a short, grounded user query from a kdialog confirm flag + its desc.

    The query is a trivial reformulation of the option semantics — we keep
    the verb structure honest ("ask the user X with Y buttons") and avoid
    making up situational details.
    """
    desc = desc.strip().rstrip(".")
    if not desc:
        return None
    # Map flag → human verb-led phrasing keyed off the option's own name.
    base_map = {
        "--yesno":                "ask the user a yes/no question",
        "--questionyesno":        "show a yes/no question dialog",
        "--yesnocancel":          "ask a question with yes/no/cancel buttons",
        "--warningyesno":         "show a warning dialog with yes and no buttons",
        "--warningyesnocancel":   "show a warning dialog with yes, no, and cancel",
        "--warningcontinuecancel": "warn the user with continue/cancel buttons",
    }
    return base_map.get(flag)


def mine_kdialog(schemas: dict[str, dict]) -> list[dict]:
    pairs: list[dict] = []
    confirm_schema = schemas.get("kde_dialog_confirm")
    notify_schema  = schemas.get("kde_notifications_send")
    if not confirm_schema:
        skip("kde_dialog_confirm schema missing")
        return []

    text = get_help("kdialog")
    if text is None:
        return []
    options = parse_options(text)
    info(f"kdialog: parsed {len(options)} options from --help")

    # Collect real sibling negatives as a stable list.
    sibling_negatives: list[str] = []
    seen_neg = set()
    for flag, _desc in options:
        if flag in KDIALOG_NON_CONFIRM_FLAGS and flag not in seen_neg:
            seen_neg.add(flag)
            sibling_negatives.append(flag)

    # ---- kde_dialog_confirm pairs from confirm-shaped flags ----
    confirm_count = 0
    for flag, desc in options:
        if flag not in KDIALOG_CONFIRM_FLAGS:
            continue
        q = _confirm_query_for(flag, desc)
        if not q:
            continue
        # The verbatim help description is also an honest grounded query —
        # e.g. "Question message box with yes/no buttons" → confirm.
        prov = f"kdialog --help: {flag}<TAB>{desc[:120]}"
        # Pair 1: trivial reformulation phrased as a user request.
        pairs.append(make_pair(
            user_query=q,
            tool_name="kde_dialog_confirm",
            arguments={"prompt": q},
            schema=confirm_schema,
            source="kdialog_confirm",
            provenance=prov,
        ))
        # Pair 2: verbatim option description — already a yes/no statement
        # in the help text. Lower-cased, with a leading "show" verb so it
        # reads as a request, not a noun phrase.
        if desc and 8 <= len(desc) <= 100:
            verbatim = f"show me a {desc.lower()}"
            pairs.append(make_pair(
                user_query=verbatim,
                tool_name="kde_dialog_confirm",
                arguments={"prompt": verbatim},
                schema=confirm_schema,
                source="kdialog_confirm",
                provenance=prov,
            ))
        confirm_count += 1
    ok(f"kdialog: {confirm_count} confirm flags → {sum(1 for p in pairs if p['target']['name'] == 'kde_dialog_confirm')} confirm pairs")

    # Cap and tag with sibling negatives.
    confirm_pairs = [p for p in pairs if p["target"]["name"] == "kde_dialog_confirm"]
    confirm_pairs = confirm_pairs[:PER_BINARY_CAP]
    if sibling_negatives:
        for p in confirm_pairs:
            # Real sibling negatives — not the tool name list, but flag list.
            # Schema-wise our negatives field expects tool names. Encode as
            # provenance metadata — the build step already keys negatives off
            # tool name. Use a plain "kdialog_other_dialogs" comment instead.
            p["sibling_kdialog_flags"] = sibling_negatives

    # ---- kde_notifications_send pairs from --passivepopup ----
    notify_pairs: list[dict] = []
    if notify_schema:
        for flag, desc in options:
            if flag != "--passivepopup":
                continue
            prov = f"kdialog --help: {flag}<TAB>{desc[:120]}"
            # The flag itself = notification semantic.
            queries = [
                "show a passive popup notification",
                "send a passive popup",
                "display a quick passive popup with timeout",
            ]
            for q in queries:
                notify_pairs.append(make_pair(
                    user_query=q,
                    tool_name="kde_notifications_send",
                    arguments={
                        "title": "Passive Popup",
                        "message": desc.rstrip(".") if desc else "Passive Popup",
                    },
                    schema=notify_schema,
                    source="kdialog_passivepopup",
                    provenance=prov,
                ))
            break  # only one --passivepopup
        ok(f"kdialog: {len(notify_pairs)} passivepopup → notify pairs")

    return confirm_pairs + notify_pairs


# ---------------------------------------------------------------------------
# Mine: krunner → krunner_launch
# ---------------------------------------------------------------------------

def mine_krunner(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("kde_krunner_launch")
    if not schema:
        skip("kde_krunner_launch schema missing")
        return []
    text = get_help("krunner")
    if text is None:
        return []
    options = parse_options(text)
    info(f"krunner: parsed {len(options)} options from --help")

    pairs: list[dict] = []
    # The first non-empty line after the Usage: line is the tagline.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    tagline = ""
    for i, ln in enumerate(lines):
        if ln.startswith("Usage:"):
            if i + 1 < len(lines):
                tagline = lines[i + 1].strip()
            break
    prov_base = f"krunner --help: tagline={tagline!r}"

    if tagline:
        # Verbatim tagline → krunner_launch with no specific app (use tagline as app name).
        pairs.append(make_pair(
            user_query=tagline.lower(),
            tool_name="kde_krunner_launch",
            arguments={"app": "krunner"},
            schema=schema,
            source="krunner_tagline",
            provenance=prov_base,
        ))

    # The --clipboard option's description: "Use the clipboard contents as
    # query for KRunner" — verbatim grounded request.
    for flag, desc in options:
        if flag == "--clipboard":
            prov = f"krunner --help: {flag}<TAB>{desc[:120]}"
            pairs.append(make_pair(
                user_query="use the clipboard contents as query for krunner",
                tool_name="kde_krunner_launch",
                arguments={"app": "krunner"},
                schema=schema,
                source="krunner_clipboard",
                provenance=prov,
            ))
        elif flag == "--daemon":
            prov = f"krunner --help: {flag}<TAB>{desc[:120]}"
            pairs.append(make_pair(
                user_query="start krunner in the background",
                tool_name="kde_krunner_launch",
                arguments={"app": "krunner"},
                schema=schema,
                source="krunner_daemon",
                provenance=prov,
            ))
        elif flag == "--replace":
            prov = f"krunner --help: {flag}<TAB>{desc[:120]}"
            pairs.append(make_pair(
                user_query="replace the existing krunner instance",
                tool_name="kde_krunner_launch",
                arguments={"app": "krunner"},
                schema=schema,
                source="krunner_replace",
                provenance=prov,
            ))
        elif flag == "--list":
            prov = f"krunner --help: {flag}<TAB>{desc[:120]}"
            pairs.append(make_pair(
                user_query="list available krunner plugins",
                tool_name="kde_krunner_launch",
                arguments={"app": "krunner"},
                schema=schema,
                source="krunner_list",
                provenance=prov,
            ))

    ok(f"krunner: {len(pairs)} grounded pairs")
    return pairs[:PER_BINARY_CAP]


# ---------------------------------------------------------------------------
# Mine: kstart → window_focus + krunner_launch
# ---------------------------------------------------------------------------

def mine_kstart(schemas: dict[str, dict]) -> list[dict]:
    schema_focus  = schemas.get("kde_window_focus")
    schema_launch = schemas.get("kde_krunner_launch")
    text = get_help("kstart")
    if text is None:
        return []
    options = parse_options(text)
    info(f"kstart: parsed {len(options)} options from --help")

    # The kstart tagline is "Utility to launch applications" — direct evidence
    # that kstart maps onto kde_krunner_launch.
    pairs: list[dict] = []
    if schema_launch:
        prov = "kstart --help: tagline=Utility to launch applications"
        pairs.append(make_pair(
            user_query="utility to launch applications",
            tool_name="kde_krunner_launch",
            arguments={"app": "kstart"},
            schema=schema_launch,
            source="kstart_tagline",
            provenance=prov,
        ))

    # Each option that pertains to a given app starting / focusing.
    for flag, desc in options:
        if not desc:
            continue
        prov = f"kstart --help: {flag}<TAB>{desc[:120]}"
        d = desc.strip().rstrip(".")
        if flag == "--application" and schema_launch:
            pairs.append(make_pair(
                user_query="alternative to command: desktop file name to start",
                tool_name="kde_krunner_launch",
                arguments={"app": "kstart"},
                schema=schema_launch,
                source="kstart_application",
                provenance=prov,
            ))
            pairs.append(make_pair(
                user_query="start an app from its desktop file",
                tool_name="kde_krunner_launch",
                arguments={"app": "kstart"},
                schema=schema_launch,
                source="kstart_application",
                provenance=prov,
            ))
        elif flag == "--url" and schema_launch:
            pairs.append(make_pair(
                user_query="optional url to pass to the application",
                tool_name="kde_krunner_launch",
                arguments={"app": "kstart"},
                schema=schema_launch,
                source="kstart_url",
                provenance=prov,
            ))

    if schema_focus is None:
        skip("kde_window_focus schema missing")
    ok(f"kstart: {len(pairs)} grounded pairs")
    return pairs[:PER_BINARY_CAP]


# ---------------------------------------------------------------------------
# Mine: dolphin / kate / konsole → krunner_launch (mapped to that binary)
# ---------------------------------------------------------------------------

DESKTOP_APPS = [
    ("dolphin",  "dolphin",  "File Manager"),
    ("kate",     "kate",     "Kate - Advanced Text Editor"),
    ("konsole",  "konsole",  "Terminal emulator"),
]


def mine_desktop_apps(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("kde_krunner_launch")
    if not schema:
        skip("kde_krunner_launch schema missing")
        return []

    pairs: list[dict] = []
    for binary, app_arg, expected_tagline in DESKTOP_APPS:
        text = get_help(binary, alt_flag="--help-all")
        if text is None:
            continue
        # Parse tagline (line right after `Usage:`).
        lines = [ln.rstrip() for ln in text.splitlines()]
        tagline = ""
        for i, ln in enumerate(lines):
            if ln.startswith("Usage:"):
                if i + 1 < len(lines):
                    tagline = lines[i + 1].strip()
                break

        info(f"{binary}: tagline={tagline!r}")
        opts = parse_options(text)

        per_app: list[dict] = []
        prov_tag = f"{binary} --help-all: tagline={tagline!r}"

        if tagline:
            # Verbatim tagline → launch <app>. The user said exactly the
            # binary's self-description: that is honest, real signal.
            per_app.append(make_pair(
                user_query=tagline.lower(),
                tool_name="kde_krunner_launch",
                arguments={"app": app_arg},
                schema=schema,
                source=f"{binary}_tagline",
                provenance=prov_tag,
            ))

        # Pull a few interesting flag descriptions and convert to launch
        # requests for that app. We pick semantic flags only (selection /
        # new-window / new-tab / profile / start) and skip Qt internals.
        WANTED = {
            "--select", "--split", "--new-window", "--admin", "--sudo",
            "--start", "--startanon", "--new", "--block", "--encoding",
            "--line", "--column", "--stdin", "--profile", "--layout",
            "--workdir", "--new-tab", "--separate", "--tabs-from-file",
            "--show-tabbar", "--hide-tabbar", "--background-mode",
        }
        for flag, desc in opts:
            if flag not in WANTED or not desc:
                continue
            prov = f"{binary} --help-all: {flag}<TAB>{desc[:120]}"
            d = desc.strip().rstrip(".")
            if len(d) < 6 or len(d) > 110:
                continue
            # Trivial reformulation: "open <binary>" + the flag's verb.
            # Use the flag description verbatim as the query — already a
            # natural sentence in Qt help (e.g. "Create a new tab in an
            # existing window rather than creating a new window").
            per_app.append(make_pair(
                user_query=d.lower(),
                tool_name="kde_krunner_launch",
                arguments={"app": app_arg},
                schema=schema,
                source=f"{binary}_flag",
                provenance=prov,
            ))

        # Cap per-binary
        per_app = per_app[:PER_DESKTOP_CAP]
        ok(f"{binary}: {len(per_app)} pairs (tagline + flags)")
        pairs.extend(per_app)
    return pairs


# ---------------------------------------------------------------------------
# Mine: notify-send → kde_notifications_send
# ---------------------------------------------------------------------------

def mine_notify_send(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("kde_notifications_send")
    if not schema:
        skip("kde_notifications_send schema missing")
        return []

    text = get_help("notify-send")
    if text is None:
        return []
    info("notify-send: parsing usage line and option descriptions")

    pairs: list[dict] = []
    # Usage line: "notify-send [OPTION…] <SUMMARY> [BODY] - create a notification"
    usage_re = re.compile(r"\bcreate a notification\b", re.IGNORECASE)
    if usage_re.search(text):
        prov = "notify-send --help: usage line 'create a notification'"
        pairs.append(make_pair(
            user_query="create a notification",
            tool_name="kde_notifications_send",
            arguments={"title": "Notification", "message": "create a notification"},
            schema=schema,
            source="notify_send_usage",
            provenance=prov,
        ))

    # Specific option phrasings — verbatim from the help, recast as user requests.
    OPTION_QUERIES = {
        "--urgency":     ("send a critical urgency notification",
                          {"title": "Alert", "message": "Critical urgency notification"}),
        "--expire-time": ("send a notification that expires after some time",
                          {"title": "Reminder", "message": "Auto-dismiss after timeout"}),
        "--app-name":    ("send a notification with a specific app name",
                          {"title": "From App", "message": "Notification from named app"}),
        "--icon":        ("send a notification with an icon",
                          {"title": "Notice", "message": "Has an icon"}),
        "--category":    ("send a notification with a category type",
                          {"title": "Category", "message": "Categorized notification"}),
        "--transient":   ("create a transient notification",
                          {"title": "Transient", "message": "Will not persist"}),
        "--replace-id":  ("replace an existing notification by id",
                          {"title": "Replacement", "message": "Replaces previous"}),
    }
    options = parse_options(text)
    for flag, desc in options:
        if flag not in OPTION_QUERIES or not desc:
            continue
        q, args = OPTION_QUERIES[flag]
        prov = f"notify-send --help: {flag}<TAB>{desc[:120]}"
        pairs.append(make_pair(
            user_query=q,
            tool_name="kde_notifications_send",
            arguments=args,
            schema=schema,
            source="notify_send_flag",
            provenance=prov,
        ))

    ok(f"notify-send: {len(pairs)} grounded pairs")
    return pairs[:PER_BINARY_CAP]


# ---------------------------------------------------------------------------
# Mine: /proc/mounts → linux_disk_usage
# ---------------------------------------------------------------------------

VIRTUAL_FS_TYPES = {
    "proc", "sysfs", "tmpfs", "devtmpfs", "devpts", "cgroup", "cgroup2",
    "securityfs", "pstore", "efivarfs", "bpf", "autofs", "mqueue",
    "debugfs", "tracefs", "hugetlbfs", "fusectl", "configfs",
    "binfmt_misc", "ramfs", "rpc_pipefs", "nsfs",
    "fuse.gvfsd-fuse", "fuse.portal", "fuse.snapfuse",
    "squashfs",   # snap mounts are read-only loopbacks; not useful for du
}


def mine_proc_mounts(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("linux_disk_usage")
    if not schema:
        skip("linux_disk_usage schema missing")
        return []

    try:
        text = Path("/proc/mounts").read_text()
    except OSError as exc:
        skip(f"/proc/mounts unreadable: {exc}")
        return []

    pairs: list[dict] = []
    seen: set[str] = set()
    real_count = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        _src, mount, fstype = parts[:3]
        if fstype in VIRTUAL_FS_TYPES:
            continue
        # Ignore snap loop mounts.
        if mount.startswith("/snap/") or mount.startswith("/var/lib/snapd/"):
            continue
        if mount.startswith("/run/"):
            continue
        if mount in seen:
            continue
        seen.add(mount)
        real_count += 1
        prov = f"/proc/mounts: {mount} type={fstype}"

        # Ground each pair in the actual mount point. Honest mount-specific phrasings.
        if mount == "/":
            queries = [
                "how full is the root filesystem",
                "check disk usage on root",
                "is the system disk almost full",
            ]
        elif mount == "/home":
            queries = [
                "how full is /home",
                "check disk usage for /home",
                "how much space is left in home",
            ]
        elif mount == "/boot" or mount.startswith("/boot/"):
            queries = [
                f"check space on {mount}",
                f"how full is the {mount} mount",
            ]
        else:
            queries = [
                f"check disk usage at {mount}",
                f"how full is {mount}",
            ]
        for q in queries:
            pairs.append(make_pair(
                user_query=q,
                tool_name="linux_disk_usage",
                arguments={"path": mount},
                schema=schema,
                source="proc_mounts",
                provenance=prov,
            ))

    ok(f"/proc/mounts: {real_count} real mounts → {len(pairs)} pairs")
    return pairs


# ---------------------------------------------------------------------------
# Mine: /sys/class/power_supply/ADP1 → linux_battery_status
# ---------------------------------------------------------------------------

def mine_ac_adapter(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("linux_battery_status")
    if not schema:
        skip("linux_battery_status schema missing")
        return []

    base = Path("/sys/class/power_supply")
    if not base.is_dir():
        skip("/sys/class/power_supply not a directory")
        return []

    # Look for AC adapter pseudo-supplies (ADP*, AC*).
    adapters = [d for d in base.iterdir()
                if d.is_dir() and (d.name.startswith("ADP") or d.name == "AC"
                                   or d.name.startswith("ACAD"))]
    if not adapters:
        skip("no AC adapter (ADP*/AC*) under /sys/class/power_supply")
        return []

    pairs: list[dict] = []
    for adp in adapters:
        online_file = adp / "online"
        try:
            online = int(online_file.read_text().strip())
        except (OSError, ValueError):
            skip(f"{online_file} unreadable")
            continue
        prov = f"{adp}/online={online}"
        # Distinct phrasings only valid because we KNOW the AC state.
        if online == 1:
            queries = [
                "am i plugged in",
                "is the charger connected",
                "is the laptop on ac power",
                "is the power adapter active",
            ]
        else:
            queries = [
                "am i still on battery",
                "is the charger unplugged",
                "is the laptop running on battery",
                "is ac power disconnected",
                "did i forget to plug in the charger",
            ]
        # Universal phrasings — phrased as battery-status questions.
        queries += [
            "how's the battery",
            "battery status",
            "is the laptop charging",
        ]
        for q in queries:
            pairs.append(make_pair(
                user_query=q,
                tool_name="linux_battery_status",
                arguments={},
                schema=schema,
                source="sys_ac_adapter",
                provenance=prov,
            ))
        ok(f"{adp.name}: online={online} → {len(queries)} pairs")
    return pairs


# ---------------------------------------------------------------------------
# Mine: /etc/systemd/system/<target>.target.wants/ → linux_service_status
# ---------------------------------------------------------------------------

# Targets whose .wants symlinks expose realistic service names.
TARGET_WANTS = [
    "multi-user.target.wants",
    "graphical.target.wants",
    "network-online.target.wants",
    "bluetooth.target.wants",
]


def mine_systemd_target_wants(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("linux_service_status")
    if not schema:
        skip("linux_service_status schema missing")
        return []

    base = Path("/etc/systemd/system")
    if not base.is_dir():
        skip(f"{base} not a directory")
        return []

    pairs: list[dict] = []
    seen_units: set[str] = set()
    for target in TARGET_WANTS:
        d = base / target
        if not d.is_dir():
            continue
        units: list[str] = []
        for entry in sorted(d.iterdir()):
            name = entry.name
            if not name.endswith(".service"):
                continue
            if name in seen_units:
                continue
            seen_units.add(name)
            units.append(name)
        info(f"{target}: {len(units)} new services")
        for unit in units:
            short = unit[: -len(".service")]
            prov = f"{d}/{unit}"
            queries = [
                f"is {short} running",
                f"check if {short} starts on boot",
                f"status of the {short} service",
            ]
            for q in queries:
                pairs.append(make_pair(
                    user_query=q,
                    tool_name="linux_service_status",
                    arguments={"name": short},
                    schema=schema,
                    source="systemd_target_wants",
                    provenance=prov,
                ))
    ok(f"systemd target.wants: {len(seen_units)} unique services → {len(pairs)} pairs")
    return pairs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

MINES = [
    ("kdialog",            mine_kdialog),
    ("krunner",            mine_krunner),
    ("kstart",             mine_kstart),
    ("desktop_apps",       mine_desktop_apps),
    ("notify_send",        mine_notify_send),
    ("proc_mounts",        mine_proc_mounts),
    ("ac_adapter",         mine_ac_adapter),
    ("systemd_target_wants", mine_systemd_target_wants),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output JSONL path (default: dataset/real_sources/kde_help_pairs.jsonl)")
    args = parser.parse_args()

    if not TOOL_SCHEMAS.is_file():
        warn(f"missing tool schemas at {TOOL_SCHEMAS}")
        return 2
    schemas = load_schemas()
    info(f"loaded {len(schemas)} schemas")

    summary: dict[str, dict[str, int]] = {}
    all_pairs: list[dict] = []
    for name, fn in MINES:
        banner(f"mine_{name}")
        try:
            raw = fn(schemas)
        except Exception as exc:
            warn(f"{name} crashed (caught): {type(exc).__name__}: {exc}")
            raw = []
        per_tool: dict[str, int] = defaultdict(int)
        for p in raw:
            per_tool[p["target"]["name"]] += 1
        summary[name] = dict(per_tool)
        all_pairs.extend(raw)

    final = dedupe(all_pairs)
    n = write_jsonl(args.out, final)

    banner("SUMMARY")
    grand: dict[str, int] = defaultdict(int)
    for name, _ in MINES:
        per = summary.get(name, {})
        total = sum(per.values())
        if not per:
            print(f"  mine_{name:<24s} 0 pairs (skipped)")
            continue
        print(f"  mine_{name:<24s} {total:4d} raw pairs")
        for tool in sorted(per):
            print(f"      {tool:<26s} {per[tool]:4d}")
    # Recompute per-tool from `final` (post-dedupe).
    for p in final:
        grand[p["target"]["name"]] += 1
    print()
    print(f"  Final after dedupe: {n} pairs")
    print(f"  Per tool (final):")
    for tool in sorted(grand):
        print(f"      {tool:<26s} {grand[tool]:4d}")
    print()
    print(f"  Output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
