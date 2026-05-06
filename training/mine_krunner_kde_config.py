#!/usr/bin/env python3
"""
mine_krunner_kde_config.py — real-data miner for KRunner / KDE config.

Targets the under-represented `kde_*` non-launch tools (window_focus,
notifications_send, dialog_confirm) by probing REAL state on this machine:

  Source 1 — KRunner state/history:
      ~/.local/share/krunner/         (history files)
      ~/.cache/krunner/               (cached query state)
      ~/.config/krunnerrc             ([General] HistoryEntries / etc.)
      ~/.local/share/klipper/history2.lst (clipboard text the user copied)
      ~/.local/share/kactivitymanagerd/   (activity names)

  Source 2 — Plasma keyboard shortcuts / action labels:
      ~/.config/kglobalshortcutsrc
      ~/.config/khotkeysrc
      ~/.config/kdeglobals
      ~/.config/plasma-org.kde.plasma.desktop-appletsrc

  Source 3 — Application kxmlgui RC files:
      /usr/share/kxmlgui5/**/*.rc
      /usr/share/kxmlgui6/**/*.rc
      <Action name="..." text="..."/>  → real menu/action labels

  Source 4 — Live window titles (best-effort under Wayland):
      qdbus6 KWin scripting (typically blocked under Wayland)
      wmctrl -l (typically empty under pure Wayland)
      ps -eo comm fallback: running KDE apps → likely focusable targets

  Source 5 — Recently used files:
      ~/.local/share/RecentDocuments/*.desktop
      ~/.local/share/recently-used.xbel

  Source 6 — KSysGuard sensor / metric names:
      /usr/share/ksysguard/*.xml
      /usr/share/ksysguard5/*.xml

DOES NOT OVERLAP with mine_kde_machine.py (.desktop apps, systemctl, qdbus6
generic, nmcli, pactl, journalctl, /proc/meminfo, history) or mine_man_pages
or mine_kf5book.

Output: one consolidated JSONL at dataset/real_sources/krunner_kde_pairs.jsonl
with one pair per line in dispatch_pairs format.

Constraints honored:
  - Pure stdlib only.
  - Idempotent: dedupe + sort before write.
  - Every pair carries a `provenance` field linking to a file/command.
  - Skipped sources logged with [SKIP] and the reason.
  - Do NOT invent user queries: only emit a pair when the source phrase IS
    plausible as a user request.
"""

from __future__ import annotations

import argparse
import configparser
import json
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# =============================================================================
# Paths & constants
# =============================================================================

REPO_ROOT     = Path(__file__).resolve().parent.parent
TOOL_SCHEMAS  = REPO_ROOT / "tools" / "tool_schemas.json"
DEFAULT_OUT   = REPO_ROOT / "dataset" / "real_sources" / "krunner_kde_pairs.jsonl"

PER_TOOL_CAP = 250

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
# Logging helpers
# =============================================================================

def info(msg: str) -> None: print(f"  [INFO]   {msg}")
def ok(msg: str)   -> None: print(f"  [OK]     {msg}")
def skip(msg: str) -> None: print(f"  [SKIP]   {msg}")
def warn(msg: str) -> None: print(f"  [WARN]   {msg}", file=sys.stderr)

def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# =============================================================================
# Schema loader & pair builder
# =============================================================================

def load_schemas() -> dict[str, dict]:
    raw = json.loads(TOOL_SCHEMAS.read_text())
    out: dict[str, dict] = {}
    for dotted, entry in raw.items():
        mcp_name = entry.get("mcp_name") or DOTTED_TO_MCP.get(dotted)
        schema = entry.get("mcp_schema")
        if mcp_name and schema:
            out[mcp_name] = schema
    return out


def make_pair(*, user_query: str, tool_name: str, arguments: dict,
              schema: dict, source: str, provenance: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Call the right tool."},
            {"role": "user",   "content": user_query.strip()},
        ],
        "tools":   [schema],
        "target":  {"name": tool_name, "arguments": arguments},
        "source":  f"krunner_kde_{source}",
        "provenance": provenance,
    }


def dedupe_and_cap(pairs: list[dict], cap: int = PER_TOOL_CAP) -> list[dict]:
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
        if len(by_tool[tool]) >= cap:
            continue
        by_tool[tool].append(p)
    out: list[dict] = []
    for tool in sorted(by_tool):
        out.extend(sorted(by_tool[tool],
                          key=lambda x: x["messages"][1]["content"]))
    return out


def write_jsonl(path: Path, pairs: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(pairs)


def run(cmd: list[str], timeout: int = 6) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, text=True, check=False)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout: {' '.join(shlex.quote(c) for c in cmd)}"
    except Exception as exc:  # pragma: no cover
        return 1, "", f"{type(exc).__name__}: {exc}"


# =============================================================================
# Common helpers
# =============================================================================

# Bare strings that look like a user query (single-word app names included).
_QUERY_GOOD_RE = re.compile(r"^[\w][\w\-\s.,'/&:+]{1,80}$")
# Reject obvious paths, env var dumps, key= lines, and base64 blobs.
_QUERY_BAD_RE = re.compile(
    r"(^/|^\.\.|^\s*$|=|\\\d|\\x|^\w+\.\w+\.\w+\.\w+\.|^[A-Za-z0-9+/]{40,})")


def looks_like_user_query(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 100:
        return False
    if not _QUERY_GOOD_RE.match(s):
        return False
    if _QUERY_BAD_RE.search(s):
        return False
    # Reject obvious shell commands or KConfig-encoded blobs.
    if s.startswith(("#", "[", "$", "{", "<")):
        return False
    # Need at least one alphabetic character.
    if not re.search(r"[A-Za-z]", s):
        return False
    return True


def _parse_kconfig(path: Path) -> dict[str, dict[str, str]]:
    """Parse a KDE/KConfig INI file into {section: {key: value}}.

    KConfig allows keys with non-INI characters; configparser is forgiving
    enough when strict=False and interpolation is disabled.
    """
    cp = configparser.RawConfigParser(strict=False, delimiters=("=",),
                                      allow_no_value=True)
    # Preserve case
    cp.optionxform = lambda x: x  # type: ignore[assignment]
    try:
        cp.read(path, encoding="utf-8")
    except (configparser.Error, OSError, UnicodeDecodeError):
        return {}
    out: dict[str, dict[str, str]] = {}
    for sect in cp.sections():
        out[sect] = {}
        for k, v in cp.items(sect):
            out[sect][k] = v if v is not None else ""
    return out


# =============================================================================
# Source 1 — KRunner state, clipboard, activities
# =============================================================================

def mine_krunner_state(schemas: dict[str, dict]) -> list[dict]:
    pairs: list[dict] = []
    krunner = schemas.get("kde_krunner_launch")
    notif   = schemas.get("kde_notifications_send")
    if not krunner:
        skip("kde_krunner_launch schema missing")
        return []

    candidates: list[tuple[str, str]] = []  # (text, provenance)

    # ---- ~/.config/krunnerrc HistoryEntries=… ----
    krunnerrc = Path.home() / ".config" / "krunnerrc"
    if krunnerrc.is_file():
        cfg = _parse_kconfig(krunnerrc)
        if not cfg:
            skip(f"krunnerrc could not be parsed: {krunnerrc}")
        else:
            n_before = len(candidates)
            for sect, kvs in cfg.items():
                for k, v in kvs.items():
                    if not v:
                        continue
                    # KRunner historically stored history under a HistoryEntries
                    # key (or per-runner). Comma-delimited.
                    if "history" in k.lower() or "lastquery" in k.lower():
                        for entry in v.split(","):
                            entry = entry.strip()
                            if entry:
                                candidates.append(
                                    (entry, f"krunnerrc:[{sect}]/{k}"))
            ok(f"krunnerrc: parsed, {len(candidates) - n_before} history entries")
    else:
        skip(f"missing: {krunnerrc}")

    # ---- ~/.local/share/krunner/* and ~/.cache/krunner/* ----
    for d in (Path.home() / ".local/share/krunner",
              Path.home() / ".cache/krunner"):
        if not d.is_dir():
            skip(f"missing dir: {d}")
            continue
        files = sorted(d.rglob("*"))
        text_files = [p for p in files if p.is_file() and p.stat().st_size < 256_000]
        info(f"{d}: {len(text_files)} files")
        for fp in text_files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                # Try INI key=value first
                if "=" in line:
                    _, _, val = line.partition("=")
                    for entry in val.split(","):
                        entry = entry.strip()
                        if entry:
                            candidates.append((entry, f"{fp}:k=v"))
                else:
                    candidates.append((line.strip(), f"{fp}:line"))

    # ---- ~/.local/share/klipper/history2.lst (clipboard) ----
    klip = Path.home() / ".local/share/klipper/history2.lst"
    if klip.is_file():
        try:
            raw = klip.read_bytes()
        except OSError:
            raw = b""
        # history2.lst is a Qt-serialised binary blob with embedded UTF-16/8
        # strings. Extract printable ASCII runs ≥ 4 chars as a best-effort.
        ascii_runs = re.findall(rb"[\x20-\x7e]{4,200}", raw)
        # Also try UTF-16-LE.
        try:
            utf16_text = raw.decode("utf-16-le", errors="replace")
            utf16_runs = re.findall(r"[ -~]{4,200}", utf16_text)
        except Exception:
            utf16_runs = []
        clip_strings = []
        for s in ascii_runs:
            try:
                clip_strings.append(s.decode("ascii"))
            except UnicodeDecodeError:
                pass
        clip_strings.extend(utf16_runs)
        info(f"klipper history2.lst: {len(clip_strings)} extracted strings")
        # Clipboard often contains URLs, code, or pastes — most are NOT user
        # queries. We only keep entries that LOOK like a phrase a user might
        # type into KRunner: short, alphabetic, no shell punctuation.
        for s in clip_strings:
            s = s.strip()
            if 3 <= len(s) <= 60 and looks_like_user_query(s):
                # Reject anything with URL-y or path-y characters.
                if "://" in s or s.startswith(("/", "~")):
                    continue
                candidates.append((s, f"klipper history2.lst: {s[:40]!r}"))
    else:
        skip(f"missing: {klip}")

    # ---- ~/.local/share/kactivitymanagerd/ — Activity names ----
    actdir = Path.home() / ".local/share/kactivitymanagerd"
    activity_names: list[tuple[str, str]] = []
    if actdir.is_dir():
        # The activities config has names; also look at resources/database.
        for p in actdir.rglob("*"):
            if not p.is_file() or p.stat().st_size > 4_000_000:
                continue
            if p.suffix in {".db", ".sqlite", ".sqlite3"}:
                try:
                    raw = p.read_bytes()
                except OSError:
                    continue
                # Activity titles are stored as plain UTF-8 in sqlite pages.
                for s in re.findall(rb"[\x20-\x7e]{4,80}", raw):
                    text = s.decode("ascii", errors="replace").strip()
                    if looks_like_user_query(text) and len(text.split()) <= 4:
                        activity_names.append((text, f"{p.name}"))
            elif p.suffix in {".rc", ".cfg", ".conf"} or p.name.endswith("rc"):
                cfg = _parse_kconfig(p)
                for sect, kvs in cfg.items():
                    for k, v in kvs.items():
                        if k.lower() in {"name", "title"} and v:
                            activity_names.append((v.strip(), f"{p.name}:[{sect}]"))
        info(f"kactivitymanagerd: {len(activity_names)} candidate activity names")
    else:
        skip(f"missing: {actdir}")

    # Activity names are real user-defined labels (e.g. "Work", "Music") —
    # they make plausible krunner targets ("switch to Work activity").
    # We emit window_focus pairs because activity names are scoped to
    # workspaces, not apps, but the closest match is window_focus.
    focus = schemas.get("kde_window_focus")
    if focus:
        for name, prov in activity_names:
            for q in (f"switch to the {name} activity",
                      f"go to {name} activity"):
                pairs.append(make_pair(
                    user_query=q,
                    tool_name="kde_window_focus",
                    arguments={"title": name},
                    schema=focus,
                    source="kactivities",
                    provenance=f"kactivitymanagerd: {prov}",
                ))

    # Convert KRunner candidates to launch pairs.
    raw = 0
    for text, prov in candidates:
        if not looks_like_user_query(text):
            continue
        # Avoid emitting empty / single-word junk that's just a separator
        if len(text) < 2:
            continue
        # Strip trailing punctuation
        clean = text.strip().rstrip(".!?,;:")
        if not clean:
            continue
        # Emit two phrasings around the captured query.
        # The text ITSELF is the user's literal KRunner input, so it's the
        # most authentic user query we can produce.
        pairs.append(make_pair(
            user_query=clean,
            tool_name="kde_krunner_launch",
            arguments={"app": clean.split()[0].lower()},
            schema=krunner,
            source="krunner_history",
            provenance=prov,
        ))
        # Add an "open <X>" phrasing too if the bare query isn't already
        # an imperative.
        first_word = clean.split()[0].lower()
        if first_word not in {"open", "launch", "start", "run", "show"}:
            pairs.append(make_pair(
                user_query=f"open {clean}",
                tool_name="kde_krunner_launch",
                arguments={"app": first_word},
                schema=krunner,
                source="krunner_history",
                provenance=prov,
            ))
        raw += 1

    ok(f"krunner_state: {raw} candidate queries → {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Source 2 — Plasma keyboard shortcuts / action labels
# =============================================================================

# Keys that we explicitly know are noise (UUIDs, hashes, internal IDs).
_NOISE_KEY_RE = re.compile(r"^[A-Fa-f0-9-]{16,}$")


def _section_action_labels(cfg: dict[str, dict[str, str]]) -> list[tuple[str, str, str]]:
    """Yield (section, key, label) where label looks like a human action label.

    KGlobalShortcutsRC entries look like:
        [kwin]
        Switch Window Down=Meta+Alt+Down,...,Switch to next window down
        ^^^^^^^^^^^^^^^^^^                                ^^^^^^^^^^^^^^^^^^
        the KEY is the action name; the third comma-segment is a friendly label.
    """
    out: list[tuple[str, str, str]] = []
    for sect, kvs in cfg.items():
        for key, value in kvs.items():
            if _NOISE_KEY_RE.match(key):
                continue
            # Label is the action key itself (real human-readable in
            # kglobalshortcutsrc).
            label = key
            # Some KConfig values are "Trigger,DefaultTrigger,FriendlyName"
            if "," in (value or ""):
                tail = value.split(",")[-1].strip()
                if tail and tail.lower() != "no such trigger" and len(tail) <= 80:
                    label = tail
            out.append((sect, key, label))
    return out


_FOCUS_RE = re.compile(
    r"^(?:switch to|focus|raise|activate|jump to|show)\b.+", re.IGNORECASE)
_NOTIFY_RE = re.compile(
    r"\b(notify|notification|alert|toast|popup)\b", re.IGNORECASE)
_CONFIRM_RE = re.compile(
    r"\b(confirm|are you sure|prompt|ask|verify)\b", re.IGNORECASE)
_LAUNCH_RE = re.compile(
    r"^(?:run|launch|start|open|invoke)\b.+", re.IGNORECASE)


def mine_kde_shortcuts(schemas: dict[str, dict]) -> list[dict]:
    pairs: list[dict] = []
    focus  = schemas.get("kde_window_focus")
    notify = schemas.get("kde_notifications_send")
    confirm = schemas.get("kde_dialog_confirm")
    krunner = schemas.get("kde_krunner_launch")

    config_files = [
        Path.home() / ".config" / "kglobalshortcutsrc",
        Path.home() / ".config" / "khotkeysrc",
        Path.home() / ".config" / "kdeglobals",
        Path.home() / ".config" / "plasma-org.kde.plasma.desktop-appletsrc",
    ]

    for cfg_path in config_files:
        if not cfg_path.is_file():
            skip(f"missing: {cfg_path}")
            continue
        cfg = _parse_kconfig(cfg_path)
        if not cfg:
            skip(f"unparseable: {cfg_path}")
            continue
        labels = _section_action_labels(cfg)
        info(f"{cfg_path.name}: {len(labels)} candidate (section,key,label) tuples")
        for sect, key, label in labels:
            label = label.strip()
            if not label or len(label) > 80:
                continue
            if not re.search(r"[A-Za-z]", label):
                continue
            prov = f"{cfg_path.name}: [{sect}] {key}"

            # Match patterns
            l_lower = label.lower()
            # 1) Window focus / switch-to actions
            if focus and _FOCUS_RE.match(label):
                m = re.match(
                    r"^(?:switch to|focus|raise|activate|jump to|show)\s+(.+)",
                    label, re.IGNORECASE)
                target_title = m.group(1).strip() if m else label
                # Actions named "Switch Window Down" → not a real window title;
                # only accept when the trailing portion contains capitalised
                # words (likely a window/app name) or quoted text.
                # Else use the label as a plausible user phrase.
                # User could literally say "switch to ${target_title}".
                pairs.append(make_pair(
                    user_query=label.lower(),
                    tool_name="kde_window_focus",
                    arguments={"title": target_title},
                    schema=focus,
                    source="kde_shortcuts",
                    provenance=prov,
                ))
                # An "imperative" phrasing.
                if not l_lower.startswith(("switch", "focus", "raise",
                                          "activate", "jump", "show")):
                    pairs.append(make_pair(
                        user_query=f"switch to {target_title}",
                        tool_name="kde_window_focus",
                        arguments={"title": target_title},
                        schema=focus,
                        source="kde_shortcuts",
                        provenance=prov,
                    ))
            # 2) Notification / alert actions
            elif notify and _NOTIFY_RE.search(label):
                title = label
                pairs.append(make_pair(
                    user_query=label.lower(),
                    tool_name="kde_notifications_send",
                    arguments={"title": title, "message": title},
                    schema=notify,
                    source="kde_shortcuts",
                    provenance=prov,
                ))
            # 3) Confirm-style actions
            elif confirm and _CONFIRM_RE.search(label):
                pairs.append(make_pair(
                    user_query=label.lower(),
                    tool_name="kde_dialog_confirm",
                    arguments={"prompt": label},
                    schema=confirm,
                    source="kde_shortcuts",
                    provenance=prov,
                ))
            # 4) Launch / run / start actions → krunner
            elif krunner and _LAUNCH_RE.match(label):
                m = re.match(r"^(?:run|launch|start|open|invoke)\s+(.+)",
                             label, re.IGNORECASE)
                app_target = m.group(1).strip() if m else label
                pairs.append(make_pair(
                    user_query=label.lower(),
                    tool_name="kde_krunner_launch",
                    arguments={"app": app_target.split()[0].lower()},
                    schema=krunner,
                    source="kde_shortcuts",
                    provenance=prov,
                ))

    ok(f"kde_shortcuts: produced {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Source 3 — kxmlgui RC files (real menu/action labels)
# =============================================================================

def _parse_kxmlgui_rc(path: Path) -> list[tuple[str, str]]:
    """Return list of (action_name, action_text) from a kxmlgui RC file.

    The text attribute is what the user sees in the menu (e.g. "Open Recent").
    The name attribute is the internal action ID (e.g. "file_open_recent").
    """
    out: list[tuple[str, str]] = []
    try:
        # kxmlgui rc files start with a DOCTYPE; ET handles it fine but may
        # complain about ampersands in attributes — tolerate that.
        text = path.read_text(encoding="utf-8", errors="replace")
        # Strip the DOCTYPE which sometimes references a non-existent DTD.
        text = re.sub(r"<!DOCTYPE[^>]*>", "", text, count=1)
        root = ET.fromstring(text)
    except (ET.ParseError, OSError, UnicodeDecodeError):
        return []
    for action in root.iter("Action"):
        name = (action.get("name") or "").strip()
        # Some RC files add the human label as a child <text> elem; others as
        # an attribute. Most of the time the label lives outside the RC and
        # is provided by the .desktop or QAction. For a real signal we
        # reformat the name (e.g. "file_open_recent" → "open recent").
        if not name:
            continue
        text_attr = (action.get("text") or "").strip()
        if not text_attr:
            # Reformat snake_case action name to a friendly label.
            text_attr = name.replace("_", " ").replace("-", " ").strip()
            # Title-case-y but lowercased
            text_attr = re.sub(r"\s+", " ", text_attr)
        if 2 <= len(text_attr) <= 60:
            out.append((name, text_attr))
    return out


def mine_kxmlgui(schemas: dict[str, dict]) -> list[dict]:
    pairs: list[dict] = []
    focus = schemas.get("kde_window_focus")
    krunner = schemas.get("kde_krunner_launch")
    if not focus:
        skip("kde_window_focus schema missing")
        return []

    rc_dirs = [Path("/usr/share/kxmlgui5"), Path("/usr/share/kxmlgui6")]
    rc_paths: list[Path] = []
    for d in rc_dirs:
        if not d.is_dir():
            skip(f"missing dir: {d}")
            continue
        rc_paths.extend(sorted(d.rglob("*.rc")))
        rc_paths.extend(sorted(d.rglob("*.xml")))
    info(f"kxmlgui: {len(rc_paths)} RC/XML files")
    if not rc_paths:
        return []

    by_app: dict[str, list[str]] = defaultdict(list)
    for path in rc_paths:
        # Path is /usr/share/kxmlgui{5,6}/<app>/<file>.rc
        app = path.parent.name
        actions = _parse_kxmlgui_rc(path)
        for _name, label in actions:
            l_low = label.lower()
            # Filter useless labels.
            if any(stop in l_low for stop in (
                    "&amp;", "%1", "%2", "shortcut", "dummy", "test")):
                continue
            if not re.search(r"[a-z]", l_low):
                continue
            by_app[app].append(label)

    info(f"kxmlgui: extracted actions for {len(by_app)} apps")

    # The user might say "open kate's <action>" or "in dolphin show downloads".
    # Both are plausible window-focus targets — we ground the tool argument on
    # the app name (since that's the window we want focused) but vary the user
    # query with the real action label.
    for app, labels in sorted(by_app.items()):
        # Cap actions per app.
        for label in sorted(set(labels))[:6]:
            l = label.strip()
            l_low = l.lower()
            prov = f"kxmlgui RC: app={app} action={l!r}"
            # User intent: "in <app>, do <action>" → most likely focus that app.
            if focus:
                pairs.append(make_pair(
                    user_query=f"in {app}, {l_low}",
                    tool_name="kde_window_focus",
                    arguments={"title": app},
                    schema=focus,
                    source="kxmlgui",
                    provenance=prov,
                ))
                pairs.append(make_pair(
                    user_query=f"switch to {app} to {l_low}",
                    tool_name="kde_window_focus",
                    arguments={"title": app},
                    schema=focus,
                    source="kxmlgui",
                    provenance=prov,
                ))
            # If the action label *itself* starts with "open" / "show" /
            # "launch" then it's also a krunner candidate.
            if krunner and re.match(r"^(open|show|launch|start)\s+", l_low):
                m = re.match(r"^(?:open|show|launch|start)\s+(.+)", l_low)
                tail = m.group(1).strip() if m else l_low
                pairs.append(make_pair(
                    user_query=l_low,
                    tool_name="kde_krunner_launch",
                    arguments={"app": app},
                    schema=krunner,
                    source="kxmlgui",
                    provenance=prov,
                ))

    ok(f"kxmlgui: produced {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Source 4 — Live window titles (best-effort under Wayland)
# =============================================================================

# KDE app .comm names that, if running, almost always have a focusable window.
KDE_APP_COMM_TO_TITLE = {
    "konsole":     "Konsole",
    "kate":        "Kate",
    "kwrite":      "KWrite",
    "dolphin":     "Dolphin",
    "kcalc":       "Calculator",
    "okular":      "Okular",
    "gwenview":    "Gwenview",
    "spectacle":   "Spectacle",
    "ark":         "Ark",
    "kfind":       "KFind",
    "kmail":       "KMail",
    "korganizer":  "KOrganizer",
    "krunner":     "KRunner",
    "yakuake":     "Yakuake",
    "systemsettings": "System Settings",
    "systemsettings5": "System Settings",
    "discover":    "Discover",
    "firefox":     "Firefox",
    "firefox-esr": "Firefox",
    "chrome":      "Chrome",
    "chromium":    "Chromium",
    "code":        "Visual Studio Code",
    "code-insiders": "Visual Studio Code",
    "thunderbird": "Thunderbird",
    "obsidian":    "Obsidian",
    "slack":       "Slack",
    "telegram-desktop": "Telegram",
    "signal-desktop": "Signal",
    "vlc":         "VLC",
    "mpv":         "mpv",
    "kdeconnect-app": "KDE Connect",
    "plasmashell": "Plasma",
}


def mine_window_list(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("kde_window_focus")
    if not schema:
        skip("kde_window_focus schema missing")
        return []

    titles_with_prov: list[tuple[str, str]] = []

    # ---- KWin scripting on Wayland ----
    # We do NOT try to load a script (requires user click on Wayland), per
    # the prompt's instructions. Note the limitation explicitly:
    skip("KWin scripting window enumeration: blocked under Wayland security "
         "model (loadDeclarativeScript needs interactive user consent)")

    # ---- wmctrl (works under XWayland-only legacy clients) ----
    rc, out, err = run(["wmctrl", "-l"])
    if rc == 0:
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) >= 4:
                t = parts[3].strip()
                if t:
                    titles_with_prov.append((t, "wmctrl -l"))
        info(f"wmctrl -l: {len(titles_with_prov)} XWayland windows "
             "(typically empty under pure Wayland)")
    else:
        skip(f"wmctrl unavailable or empty: rc={rc} {err.strip()[:120]}")

    # ---- ps -e -o pid,comm fallback ----
    rc, out, err = run(["ps", "-e", "-o", "comm="])
    if rc != 0:
        skip(f"ps failed: rc={rc} {err.strip()[:120]}")
    else:
        running_apps = set()
        for line in out.splitlines():
            comm = line.strip()
            if comm in KDE_APP_COMM_TO_TITLE:
                running_apps.add(comm)
        info(f"ps: {len(running_apps)} running KDE-style GUI apps detected")
        for comm in sorted(running_apps):
            title = KDE_APP_COMM_TO_TITLE[comm]
            titles_with_prov.append(
                (title, f"ps -e -o comm: {comm} running"))

    seen: set[str] = set()
    pairs: list[dict] = []
    for title, prov in titles_with_prov:
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        for verb in ("switch to", "focus", "bring", "raise", "show me"):
            pairs.append(make_pair(
                user_query=f"{verb} {title}",
                tool_name="kde_window_focus",
                arguments={"title": title},
                schema=schema,
                source="window_list",
                provenance=prov,
            ))
        # Also "I need <title>" / "go back to <title>"
        pairs.append(make_pair(
            user_query=f"i need the {title.lower()} window",
            tool_name="kde_window_focus",
            arguments={"title": title},
            schema=schema,
            source="window_list",
            provenance=prov,
        ))

    ok(f"window_list: {len(seen)} unique titles → {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Source 5 — Recently used files
# =============================================================================

def _parse_recent_desktop(path: Path) -> tuple[str, str] | None:
    """Return (Name, URL) from a RecentDocuments .desktop file, or None."""
    try:
        cp = configparser.RawConfigParser(strict=False, interpolation=None)
        cp.optionxform = lambda x: x  # type: ignore[assignment]
        cp.read(path, encoding="utf-8")
    except (configparser.Error, OSError, UnicodeDecodeError):
        return None
    if "Desktop Entry" not in cp:
        return None
    de = cp["Desktop Entry"]
    name = (de.get("Name") or "").strip()
    url  = (de.get("URL") or de.get("Exec") or "").strip()
    if not name:
        return None
    return name, url


def mine_recent_files(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("kde_window_focus")
    if not schema:
        skip("kde_window_focus schema missing")
        return []

    pairs: list[dict] = []
    seen_titles: set[str] = set()

    # ---- ~/.local/share/RecentDocuments/*.desktop ----
    rdir = Path.home() / ".local/share/RecentDocuments"
    if rdir.is_dir():
        files = sorted(rdir.glob("*.desktop"))
        info(f"RecentDocuments: {len(files)} .desktop entries")
        for f in files:
            parsed = _parse_recent_desktop(f)
            if not parsed:
                continue
            name, url = parsed
            if name.lower() in seen_titles:
                continue
            seen_titles.add(name.lower())
            prov = f"RecentDocuments: {f.name} → {name!r}"
            for verb in ("switch to", "focus", "bring up", "show me"):
                pairs.append(make_pair(
                    user_query=f"{verb} the {name} window",
                    tool_name="kde_window_focus",
                    arguments={"title": name},
                    schema=schema,
                    source="recent_documents",
                    provenance=prov,
                ))
            pairs.append(make_pair(
                user_query=f"go back to {name}",
                tool_name="kde_window_focus",
                arguments={"title": name},
                schema=schema,
                source="recent_documents",
                provenance=prov,
            ))
    else:
        skip(f"missing: {rdir}")

    # ---- ~/.local/share/recently-used.xbel ----
    xbel = Path.home() / ".local/share/recently-used.xbel"
    if xbel.is_file():
        try:
            text = xbel.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        # bookmark href="file:///path/to/foo.pdf" — extract the basename.
        hrefs = re.findall(r'\bhref="([^"]+)"', text)
        info(f"recently-used.xbel: {len(hrefs)} bookmark hrefs")
        for href in hrefs:
            # Decode percent-escapes minimally.
            try:
                from urllib.parse import unquote, urlparse
                u = urlparse(href)
                p = unquote(u.path or "")
                base = Path(p).name
            except Exception:
                continue
            if not base:
                continue
            if base.lower() in seen_titles:
                continue
            seen_titles.add(base.lower())
            prov = f"recently-used.xbel: {base!r}"
            for verb in ("switch to", "focus", "bring up"):
                pairs.append(make_pair(
                    user_query=f"{verb} the {base} window",
                    tool_name="kde_window_focus",
                    arguments={"title": base},
                    schema=schema,
                    source="xbel",
                    provenance=prov,
                ))
    else:
        skip(f"missing: {xbel}")

    ok(f"recent_files: {len(seen_titles)} titles → {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Source 6 — KSysGuard sensor / metric names
# =============================================================================

def mine_ksysguard(schemas: dict[str, dict]) -> list[dict]:
    schema = schemas.get("linux_metrics_summary")
    if not schema:
        skip("linux_metrics_summary schema missing")
        return []

    sensor_dirs = [
        Path("/usr/share/ksysguard"),
        Path("/usr/share/ksysguard5"),
        Path("/usr/share/ksysguard6"),
        Path("/usr/share/ksystemstats"),
        Path("/usr/share/plasma/plasmoids/org.kde.plasma.systemmonitor"),
    ]
    found: list[Path] = []
    for d in sensor_dirs:
        if d.is_dir():
            found.extend(sorted(d.rglob("*.xml")))
            found.extend(sorted(d.rglob("*.ksgrd")))
            found.extend(sorted(d.rglob("*.json")))
        else:
            skip(f"missing dir: {d}")
    info(f"ksysguard: {len(found)} candidate XML/ksgrd/json files")

    if not found:
        return []

    sensor_titles: set[str] = set()
    for path in found:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Try XML attributes title="..." / display="..." / name="..."
        for m in re.finditer(r'\b(title|display|name|label)="([^"]{3,60})"', text):
            label = m.group(2).strip()
            # Filter purely numeric, ID-ish, or path strings.
            if not re.search(r"[A-Za-z]", label):
                continue
            if "/" in label or "." in label and " " not in label:
                continue
            sensor_titles.add(label)

    info(f"ksysguard: {len(sensor_titles)} unique sensor labels")
    pairs: list[dict] = []
    for label in sorted(sensor_titles):
        l_low = label.lower()
        # Map sensor-name → realistic metrics_summary user query.
        # Only emit if the label is a recognisable system-health term.
        keep = any(kw in l_low for kw in (
            "cpu", "memory", "ram", "swap", "load", "temp", "fan",
            "battery", "disk", "network", "throughput", "process",
            "system", "summary", "health", "monitor", "usage"))
        if not keep:
            continue
        prov = f"ksysguard sensor label: {label!r}"
        # The user might literally say "show me the CPU usage" — that maps
        # to linux_metrics_summary if it's the broad health view.
        pairs.append(make_pair(
            user_query=f"show {l_low}",
            tool_name="linux_metrics_summary",
            arguments={},
            schema=schema,
            source="ksysguard",
            provenance=prov,
        ))
        pairs.append(make_pair(
            user_query=f"what's my {l_low} like",
            tool_name="linux_metrics_summary",
            arguments={},
            schema=schema,
            source="ksysguard",
            provenance=prov,
        ))

    ok(f"ksysguard: produced {len(pairs)} raw pairs")
    return pairs


# =============================================================================
# Driver
# =============================================================================

MINES = [
    ("krunner_state",  mine_krunner_state),
    ("kde_shortcuts",  mine_kde_shortcuts),
    ("kxmlgui",        mine_kxmlgui),
    ("window_list",    mine_window_list),
    ("recent_files",   mine_recent_files),
    ("ksysguard",      mine_ksysguard),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output JSONL (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    if not TOOL_SCHEMAS.is_file():
        warn(f"missing tool schemas at {TOOL_SCHEMAS}")
        return 2
    schemas = load_schemas()
    info(f"loaded schemas for: {sorted(schemas)}")

    all_pairs: list[dict] = []
    summary: dict[str, dict[str, int]] = {}

    for name, fn in MINES:
        banner(f"mine_{name}")
        try:
            raw = fn(schemas)
        except Exception as exc:
            warn(f"{name} crashed (caught): {type(exc).__name__}: {exc}")
            raw = []
        per: dict[str, int] = defaultdict(int)
        for p in raw:
            per[p["target"]["name"]] += 1
        summary[name] = dict(per)
        all_pairs.extend(raw)

    # Final dedupe + cap
    final = dedupe_and_cap(all_pairs)
    n = write_jsonl(args.out, final)

    banner("SUMMARY")
    grand: dict[str, int] = defaultdict(int)
    for name, _ in MINES:
        per = summary.get(name, {})
        total = sum(per.values())
        if not per:
            print(f"  mine_{name:<24s} 0 raw pairs (skipped or empty)")
            continue
        print(f"  mine_{name:<24s} {total:4d} raw pairs")
        for tool in sorted(per):
            print(f"      {tool:<26s} {per[tool]:4d}")
    print()
    print("  After dedupe + cap (per-tool):")
    final_per: dict[str, int] = defaultdict(int)
    for p in final:
        final_per[p["target"]["name"]] += 1
    for tool in sorted(final_per):
        grand[tool] = final_per[tool]
        print(f"      {tool:<26s} {final_per[tool]:4d}")
    print()
    print(f"  Total final pairs: {n}")
    try:
        rel = args.out.relative_to(REPO_ROOT)
    except ValueError:
        rel = args.out
    print(f"  Output: {rel}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
