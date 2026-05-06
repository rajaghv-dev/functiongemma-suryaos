#!/usr/bin/env python3
"""
mine_kde_issues.py — real-data miner that grounds dispatch pairs in REAL
issue titles from KDE's GitLab (invent.kde.org).

Why issue titles?
  Issue titles are concise natural-language descriptions of user-observed
  intents and bugs ("brightness slider not working", "notifications don't
  show on lock screen", "KRunner doesn't find apps starting with K").
  These are real user phrasings — exactly the distribution the model needs
  to learn. We refuse to invent a query that isn't grounded in a title.

Strategy:
  1. KDE's web UI is JS-rendered; the issue list HTML contains no titles.
     We use the public GitLab REST API instead:
       GET /api/v4/projects/<urlencoded path>/issues?state=all&per_page=100
     Returns JSON with `title` for every issue. No auth required.
  2. We cache each project's JSON at
       dataset/real_sources/issues_cache/<safe-project>.json
     so re-runs are idempotent and don't hammer KDE's infra. --no-cache
     forces refetch.
  3. We normalize each title (strip [Bug]/[Feature]/version-suffixes/etc.)
     and FILTER OUT internal-state bug reports (crash/segfault/build-fail/
     test-flake/etc.) that aren't natural user requests. We KEEP titles that
     describe a user-shaped action ("brightness reverts on resume", "let me
     mute notifications", "open krunner with Meta+S").
  4. Each emitted pair carries the verbatim issue title in its provenance,
     plus the project URL — fully auditable.

Per-source cap is 30 issues, per the spec.

Usage:
    python3 training/mine_kde_issues.py
    python3 training/mine_kde_issues.py --no-cache       # force refetch
    python3 training/mine_kde_issues.py --out /tmp/x.jsonl

Constraints:
  - stdlib only (urllib, json, re, html). No bs4, no requests.
  - Each pair: source="kde_issues", provenance={url, issue_title, project}.
  - Tool schema comes from tools/tool_schemas.json (canonical mcp_schema).
  - 10 s timeout per request.
  - On fetch failure we log [SKIP] and continue.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT       = Path(__file__).resolve().parent.parent
TOOL_SCHEMAS    = REPO_ROOT / "tools" / "tool_schemas.json"
CACHE_DIR       = REPO_ROOT / "dataset" / "real_sources" / "issues_cache"
DEFAULT_OUT     = REPO_ROOT / "dataset" / "real_sources" / "kde_issues_pairs.jsonl"

GITLAB_HOST     = "https://invent.kde.org"
API_TIMEOUT_S   = 10
PER_SOURCE_CAP  = 30
USER_AGENT      = "functiongemma-suryaos-miner/1.0 (+real-data-grounding)"

# Each tuple is (project_path, target_tool_mcp_name, optional_label_filter).
# project_path is the GitLab namespace path (e.g. "plasma/plasma-pa") —
# used both as URL component AND as cache filename basis.
SOURCES: list[tuple[str, str]] = [
    ("frameworks/kwidgetsaddons", "kde_dialog_confirm"),
    ("frameworks/knotifications", "kde_notifications_send"),
    ("frameworks/krunner",        "kde_krunner_launch"),
    ("plasma/kwin",               "kde_window_focus"),
    ("plasma/powerdevil",         "linux_brightness_set"),
    ("plasma/plasma-pa",          "linux_volume_set"),
]


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
# Schema loading (canonical mcp_schema)
# ---------------------------------------------------------------------------

def load_schemas() -> dict[str, dict]:
    raw = json.loads(TOOL_SCHEMAS.read_text())
    out: dict[str, dict] = {}
    for dotted, entry in raw.items():
        mcp_name = entry.get("mcp_name")
        schema = entry.get("mcp_schema")
        if not mcp_name or not schema:
            warn(f"missing mcp_schema for {dotted}")
            continue
        out[mcp_name] = schema
    return out


# ---------------------------------------------------------------------------
# Fetch (urllib, with cache)
# ---------------------------------------------------------------------------

def _safe_filename(project: str) -> str:
    """Map "plasma/plasma-pa" → "plasma__plasma-pa.json"."""
    return project.replace("/", "__") + ".json"


def fetch_issues(project: str, *, use_cache: bool) -> list[dict] | None:
    """
    Return the parsed JSON list of issues for one project, or None on failure.

    Caches under dataset/real_sources/issues_cache/<project>.json.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / _safe_filename(project)

    if use_cache and cache_path.is_file() and cache_path.stat().st_size > 2:
        try:
            data = json.loads(cache_path.read_text())
            if isinstance(data, list):
                info(f"cache hit  {project} ({len(data)} issues, {cache_path.stat().st_size} bytes)")
                return data
        except json.JSONDecodeError as exc:
            warn(f"cache for {project} unreadable, refetching: {exc}")

    pid = urllib.parse.quote(project, safe="")
    url = f"{GITLAB_HOST}/api/v4/projects/{pid}/issues?state=all&per_page=100"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_S) as r:
            body = r.read().decode("utf-8", errors="replace")
            x_total = r.headers.get("X-Total", "?")
    except urllib.error.HTTPError as exc:
        skip(f"{project}: HTTP {exc.code} {exc.reason}")
        return None
    except urllib.error.URLError as exc:
        skip(f"{project}: URL error {exc.reason}")
        return None
    except (TimeoutError, OSError) as exc:
        skip(f"{project}: network error {type(exc).__name__}: {exc}")
        return None

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        skip(f"{project}: bad JSON ({exc})")
        return None
    if not isinstance(data, list):
        skip(f"{project}: API did not return a list (got {type(data).__name__})")
        return None

    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    ok(f"fetched   {project} ({len(data)} issues, X-Total={x_total}) → {cache_path.name}")
    return data


# ---------------------------------------------------------------------------
# Title cleaning + filtering
# ---------------------------------------------------------------------------

# Strip leading bracketed/parenthesized tags + the common "RFC:"/"WIP:" prefixes.
PREFIX_RE = re.compile(
    r"^\s*"
    r"(?:"
        r"[\[\(][^\]\)]{0,40}[\]\)]\s*"    # [Bug] / (FEATURE) / [kf6] / etc
        r"|(?:rfc|wip|fyi|q|feature|bug|enhancement|todo|fix)[:\-]\s*"
    r")+",
    re.IGNORECASE,
)

# Strip trailing version-numbers / "(in 5.84.0)" / dates.
VERSION_TAIL_RE = re.compile(
    r"\s*[\(\[][^\)\]]*\b\d+\.\d+(?:\.\d+)?\b[^\)\]]*[\)\]]\s*$"
)
TRAILING_VERSION_RE = re.compile(r"\s+\b\d+\.\d+(?:\.\d+)?\b\s*$")

# Bug-report / dev-internal phrases — drop these, they are not user requests.
INTERNAL_PATTERNS = re.compile(
    r"("
    r"\bcrash(?:es|ed|ing)?\b|\bsegfault\b|\babort(?:s|ed|ing)?\b|"
    r"\bbuild(?:s|ing)? (?:fail|error)\b|\bcmake\b|"
    r"\bcompile (?:error|fail)\b|\bfpermissive\b|build_shared_libs|"
    r"\bunstable test\b|\btest(?:s)? (?:fail|flak|unstable)\b|"
    r"\bmtest\b|\btest suite is failing\b|\bbuild_shared\b|"
    r"\bmemory leak\b|\bleak in\b|\buse[- ]after[- ]free\b|\bdouble[- ]free\b|\basan\b|\bubsan\b|"
    r"\bassertion failed\b|\bassert\s*\(|\bdereference\b|\bnullptr\b|"
    r"\brefactor(?:ing)?\b|\brework\b|\bcleanup\b|\bport to\b|\babi break\b|"
    r"\bdeprecated api\b|\bdeprecation\b|"
    r"\breproducible build\b|\bnot reproducible\b|"
    r"^follow[- ]up from|^follow up from|^discussion(?: help needed)?:|"
    r"\(inadvertently|\bbrainstorm \d+\b|"
    r"^spike\b|^scoping\b|"
    r"\bi18n string\b|\bkconfigxt\b|\bqmlui\b|"
    r"\bmerge request\b|^todo:?\s*$|^wip:?\s*$|"
    r"\brfc:|\bspike|\bscoping|"
    r"\bxdg_toplevel\b|\bzwp_relative_pointer\b|\beffecthandler\b|"
    r"\bfollow up issues\b|\bremove backend plugins\b|"
    r"\bsuggestion\b|^idea\b|^q:?\s|\bquestion about\b|"
    r"\bapparmor\b|\bshould check /sys|\bin file\b|\bat line \d+\b"
    r")",
    re.IGNORECASE,
)

# Specific to test-suite / tooling titles in frameworks/* repos.
INTERNAL_NAME_RE = re.compile(r"^(?:[a-z][a-z0-9_]+test|test[_-][a-z0-9_]+)\b", re.IGNORECASE)

# Per-tool positive-keyword gate. The cleaned title must contain at least
# one of these words/substrings to be considered "shaped for" the tool.
# This makes filtering grounded in the tool's domain rather than guessing.
POSITIVE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "kde_dialog_confirm": (
        "dialog", "confirm", "confirmation", "yes/no", "yesno", "messagebox",
        "message box", "prompt", "kmessagebox", "kdialog", "warning dialog",
        "ask the user", "are you sure",
    ),
    "kde_notifications_send": (
        "notification", "notifications", "notify", "popup", "knotification",
        "transient", "default action", "passivepopup",
    ),
    "kde_krunner_launch": (
        "krunner", "launcher", "launch ", "open ", "shortcut to open",
        "calculator", "search ", "run ",
    ),
    "kde_window_focus": (
        "window", "focus", "raise", "minimize", "maximize", "activate",
        "bring to front", "switch to",
    ),
    "linux_brightness_set": (
        "brightness", "screen brightness", "backlight", "dim ", "dimming",
        "ambient light",
    ),
    "linux_volume_set": (
        "volume", "audio output", "speaker", "mute", "unmute",
        "absolute volume", "hardware volume", "application volume",
    ),
}


def clean_title(title: str) -> str:
    """Normalize an issue title into a user-shaped query.

    Steps:
      1. HTML-unescape (the API returns plain text already, but be safe).
      2. Strip leading [Bug]/[Feature]/RFC: prefixes.
      3. Strip trailing "(in 5.84.0)" or " 5.84.0" version suffix.
      4. Collapse whitespace, lowercase.
    """
    s = html_lib.unescape(title or "").strip()
    if not s:
        return ""
    # Repeatedly strip leading bracket-prefixes (handles "[Bug] [kf6] foo").
    prev = None
    while prev != s:
        prev = s
        s = PREFIX_RE.sub("", s).strip()
    s = VERSION_TAIL_RE.sub("", s).strip()
    s = TRAILING_VERSION_RE.sub("", s).strip()
    # Strip leading/trailing punctuation.
    s = s.strip(" .:;-")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def is_user_shaped(title_clean: str, tool: str) -> bool:
    """
    Heuristic gate: is this a query a real user would plausibly type for `tool`?
      - reject empty / too-short (<10 chars) / too-long (>120 chars)
      - reject internal / build / test / refactor reports
      - reject test-name leading titles (e.g. "ktwofingerswipetest is unstable")
      - REQUIRE at least one positive keyword for the target tool — this is
        the load-bearing filter: a brightness query must actually mention
        brightness/backlight/dimming, not "remove backend plugins".
    """
    if not title_clean:
        return False
    if len(title_clean) < 10 or len(title_clean) > 120:
        return False
    if INTERNAL_PATTERNS.search(title_clean):
        return False
    if INTERNAL_NAME_RE.match(title_clean):
        return False
    # Positive-keyword gate: must mention something on-domain.
    keywords = POSITIVE_KEYWORDS.get(tool, ())
    if keywords:
        if not any(kw in title_clean for kw in keywords):
            return False
    # Drop pure parenthetical bug-id-only titles.
    if re.search(r"\b(bug \d{4,}|kde-\d+|in file|at line \d+)\b", title_clean):
        return False
    return True


# ---------------------------------------------------------------------------
# Argument scaffolding (placeholder values for `arguments`)
# ---------------------------------------------------------------------------

# We don't try to extract argument values from the title — the title
# describes the intent, not parameters. Downstream `populate_arguments.py`
# fills these in. We supply the minimal-required-keys schema-shaped object,
# matching exactly the canonical mcp_schema in tools/tool_schemas.json.
def placeholder_args(tool: str) -> dict:
    # kde_dialog_confirm requires `prompt`
    if tool == "kde_dialog_confirm":
        return {"prompt": ""}
    # kde_notifications_send requires `title` + `message`
    if tool == "kde_notifications_send":
        return {"title": "", "message": ""}
    # kde_krunner_launch requires `app`
    if tool == "kde_krunner_launch":
        return {"app": ""}
    # kde_window_focus requires `title` (substring of window title)
    if tool == "kde_window_focus":
        return {"title": ""}
    # linux_brightness_set requires `direction`  (enum: up|down)
    if tool == "linux_brightness_set":
        return {"direction": "up"}
    # linux_volume_set requires `direction`  (enum: up|down)
    if tool == "linux_volume_set":
        return {"direction": "up"}
    return {}


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def make_pair(
    *,
    user_query: str,
    tool_name: str,
    schema: dict,
    project: str,
    issue_title: str,
    issue_url: str,
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Call the right tool."},
            {"role": "user",   "content": user_query},
        ],
        "tools": [schema],
        "target": {
            "name": tool_name,
            "arguments": placeholder_args(tool_name),
        },
        "source": "kde_issues",
        "provenance": {
            "url": issue_url,
            "issue_title": issue_title,
            "project": project,
        },
    }


def dedupe(pairs: list[dict]) -> list[dict]:
    """Keyed on (lowercased query, tool). Stable sort for idempotent output."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for p in pairs:
        key = (p["messages"][1]["content"].strip().lower(),
               p["target"]["name"])
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    out.sort(key=lambda p: (p["target"]["name"],
                            p["messages"][1]["content"]))
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
# Per-source mining
# ---------------------------------------------------------------------------

def mine_one(
    project: str,
    tool_name: str,
    schemas: dict[str, dict],
    *,
    use_cache: bool,
) -> tuple[list[dict], dict]:
    """Return (pairs, stats) for one (project, tool) source."""
    schema = schemas.get(tool_name)
    stats = {"project": project, "tool": tool_name,
             "fetched": 0, "kept": 0, "dropped": 0, "cache_bytes": 0}
    if not schema:
        skip(f"schema missing for {tool_name}")
        return [], stats

    issues = fetch_issues(project, use_cache=use_cache)
    if issues is None:
        return [], stats
    stats["fetched"] = len(issues)

    cache_path = CACHE_DIR / _safe_filename(project)
    if cache_path.is_file():
        stats["cache_bytes"] = cache_path.stat().st_size

    pairs: list[dict] = []
    dropped = 0
    for it in issues:
        if not isinstance(it, dict):
            continue
        title_raw = it.get("title") or ""
        web_url = it.get("web_url") or f"{GITLAB_HOST}/{project}/-/issues/{it.get('iid','')}"
        cleaned = clean_title(title_raw)
        if not is_user_shaped(cleaned, tool_name):
            dropped += 1
            continue
        pairs.append(make_pair(
            user_query=cleaned,
            tool_name=tool_name,
            schema=schema,
            project=project,
            issue_title=title_raw,
            issue_url=web_url,
        ))
    stats["dropped"] = dropped

    # Per-source cap.
    if len(pairs) > PER_SOURCE_CAP:
        info(f"{project}: capping {len(pairs)} → {PER_SOURCE_CAP}")
        pairs = pairs[:PER_SOURCE_CAP]
    stats["kept"] = len(pairs)
    ok(f"{project} → {tool_name}: kept {stats['kept']} / dropped {dropped} / fetched {stats['fetched']}")
    return pairs, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output JSONL path "
                             "(default: dataset/real_sources/kde_issues_pairs.jsonl)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force refetch from GitLab (ignore cache).")
    args = parser.parse_args()

    if not TOOL_SCHEMAS.is_file():
        warn(f"missing tool schemas at {TOOL_SCHEMAS}")
        return 2
    schemas = load_schemas()
    info(f"loaded {len(schemas)} schemas")
    info(f"sources: {len(SOURCES)}; per-source cap = {PER_SOURCE_CAP}")
    info(f"cache dir: {CACHE_DIR}")
    info(f"output:    {args.out}")

    all_pairs: list[dict] = []
    all_stats: list[dict] = []
    use_cache = not args.no_cache
    for project, tool in SOURCES:
        banner(f"mine_kde_issues  {project}  →  {tool}")
        pairs, stats = mine_one(project, tool, schemas, use_cache=use_cache)
        all_pairs.extend(pairs)
        all_stats.append(stats)

    final = dedupe(all_pairs)
    n = write_jsonl(args.out, final)

    banner("SUMMARY — per source")
    for s in all_stats:
        size_kb = s["cache_bytes"] / 1024.0
        print(f"  {s['project']:<32s} → {s['tool']:<24s} "
              f"fetched={s['fetched']:>3d} kept={s['kept']:>3d} "
              f"dropped={s['dropped']:>3d} cache={size_kb:6.1f} KB")

    banner("SUMMARY — per target tool")
    per_tool: dict[str, int] = defaultdict(int)
    for p in final:
        per_tool[p["target"]["name"]] += 1
    for tool in sorted(per_tool):
        print(f"  {tool:<28s} {per_tool[tool]:>4d}")
    print()
    print(f"  Total pairs (post-dedupe): {n}")
    print(f"  Output: {args.out}")

    if final:
        banner("EXAMPLES (up to 5)")
        for p in final[:5]:
            print(f"  • [{p['target']['name']}] {p['messages'][1]['content']}")
            print(f"      url={p['provenance']['url']}")
            print(f"      title={p['provenance']['issue_title']!r}")
    else:
        info("no pairs emitted — all fetches blocked or all titles filtered")

    return 0


if __name__ == "__main__":
    sys.exit(main())
