#!/usr/bin/env python3
"""
populate_arguments.py — extract real argument values from user queries.

Why this exists
---------------
The function-calling SFT dataset trains the model to emit `target.arguments`
along with the right tool name. Inspection of `dataset/dispatch_pairs.jsonl`
showed that 1564/1564 existing pairs had wrong or empty arguments:

    "Dim the screen."             arguments={"direction":"up"}     <-- WRONG
    "Lower the volume."           arguments={"direction":"up"}     <-- WRONG
    "Is Ollama running?"          arguments={"name":""}            <-- empty
    "Open Kate."                  arguments={"app":""}             <-- empty
    "Switch to Firefox."          arguments={"title":""}           <-- empty
    "Notify me build is done."    arguments={"title":"","message":""}<- empty
    "Lower brightness by 20%."    no `step`                        <-- empty

A model trained on those will emit `direction:"up"` for "lower volume" and
empty-string arguments for every required slot. That's exactly what
production audit traces show.

This script is a per-tool extractor library plus a CLI patcher.

Usage
-----
As CLI (rewrites a JSONL file in place, with a `.bak` backup):
    python3 training/populate_arguments.py --in dataset/dispatch_pairs.jsonl

As library (orchestrators import `populate(pair)`):
    from populate_arguments import populate
    fixed = populate(pair)

Honesty
-------
Each extractor only fills a slot when it has CONFIDENT evidence in the user
query. If the query is ambiguous, the slot stays empty and the pair is
tagged `provenance.arg_extraction = "ambiguous"` so a downstream filter can
drop it rather than train on a wrong value.

The extractors are HAND-WRITTEN per tool — there are 12 tools with very
different argument shapes. Generic heuristics produce more wrong labels
than empty ones did.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Common systemd service names users actually mention. Lowercase keys; the
# value is the canonical systemd unit name (without `.service` suffix —
# systemctl is forgiving about it).
KNOWN_SERVICES = {
    "ollama":           "ollama",
    "bluetooth":        "bluetooth",
    "bluetoothd":       "bluetooth",
    "bt":               "bluetooth",
    "networkmanager":   "NetworkManager",
    "network-manager":  "NetworkManager",
    "wifi":             "NetworkManager",
    "sshd":             "sshd",
    "ssh":              "sshd",
    "cups":             "cups",
    "printing":         "cups",
    "docker":           "docker",
    "pipewire":         "pipewire",
    "pulseaudio":       "pulseaudio",
    "audio":            "pipewire",
    "firewalld":        "firewalld",
    "ufw":              "ufw",
    "cron":             "cron",
    "crond":            "cron",
    "ntp":              "systemd-timesyncd",
    "chrony":           "chronyd",
    "tailscale":        "tailscaled",
    "wireguard":        "wg-quick",
    "sddm":             "sddm",
    "lightdm":          "lightdm",
    "gdm":              "gdm",
}

# Articles + filler words to strip before treating a token as an app name.
APP_NOISE_WORDS = {
    "the", "a", "an", "my", "please", "up", "open", "launch", "start", "run",
    "fire", "boot", "load", "kindly", "could", "you", "can", "for", "me",
    "to", "and", "that", "this",
}

# Direction-up cue lexemes (regex pattern).
UP_CUES = re.compile(
    r"\b(up|brighter|raise|increase|louder|higher|more|max(imum)?|"
    r"turn\s*up|crank|boost|amp\s*it\s*up|all\s*the\s*way\s*up|"
    r"bright(er|en)?|loud(er)?)\b",
    re.IGNORECASE,
)
DOWN_CUES = re.compile(
    r"\b(down|dim(mer)?|lower|decrease|softer|quiet(er)?|less|min(imum)?|"
    r"turn\s*down|reduce|drop|tone\s*down|all\s*the\s*way\s*down|"
    r"darker|hush|muffle)\b",
    re.IGNORECASE,
)

# "by 20 percent" / "by 20%" / "to 50" / "50 percent"
STEP_PCT = re.compile(
    r"(?:by|to|at)?\s*(\d{1,3})\s*(?:%|percent\b|pct\b)?",
    re.IGNORECASE,
)
# Numeric-only fallback (less aggressive).
INT_IN_QUERY = re.compile(r"\b(\d{1,3})\b")

# Filesystem path mentions for linux_disk_usage.
PATH_MENTIONED = re.compile(r"\B(/[A-Za-z0-9._/\-]+)\b")


# ---------------------------------------------------------------------------
# Per-tool extractors
# ---------------------------------------------------------------------------

def _strip_app_name(name: str) -> str:
    """Lowercase, drop trailing punctuation, drop noise words."""
    name = re.sub(r"[^\w\s.\-]", "", name).strip().lower()
    parts = [w for w in name.split() if w and w not in APP_NOISE_WORDS]
    return " ".join(parts).strip()


def extract_brightness(q: str) -> tuple[dict, str]:
    """Return (args_dict, status) for linux_brightness_set."""
    args: dict = {}
    direction = None
    if DOWN_CUES.search(q):
        direction = "down"
    if UP_CUES.search(q):
        # "more" + "down" is rare but DOWN wins (negation cue).
        if direction is None:
            direction = "up"
    if direction is None:
        return {}, "ambiguous"
    args["direction"] = direction

    # Step: prefer "by N percent" / "to N percent". Plain numbers are risky.
    pct = re.search(r"by\s+(\d{1,3})\s*(?:%|percent|pct)?", q, re.IGNORECASE)
    if not pct:
        pct = re.search(r"to\s+(\d{1,3})\s*(?:%|percent|pct)?", q, re.IGNORECASE)
    if not pct:
        pct = re.search(r"(\d{1,3})\s*(?:%|percent|pct)\b", q, re.IGNORECASE)
    if pct:
        v = int(pct.group(1))
        if 1 <= v <= 100:
            args["step"] = v
    return args, "ok"


def extract_volume(q: str) -> tuple[dict, str]:
    """linux_volume_set — same direction logic as brightness but distinct verbs."""
    args, status = extract_brightness(q)
    return args, status  # same shape; cue lexicon already covers volume


def extract_service(q: str) -> tuple[dict, str]:
    """linux_service_status — extract a service name from common phrasings."""
    # First try the SERVICE keyword list.
    q_low = q.lower()
    for kw, canonical in KNOWN_SERVICES.items():
        if re.search(rf"\b{re.escape(kw)}\b", q_low):
            return {"name": canonical}, "ok"

    # Fall back to "<X> service" / "is X running" / "status of X" patterns.
    patterns = [
        r"is\s+(?:the\s+)?([A-Za-z][A-Za-z0-9._-]+)\s+(?:running|active|up|started|down|dead|failed)",
        r"check\s+(?:if\s+)?([A-Za-z][A-Za-z0-9._-]+)\s+(?:is|service|running)?",
        r"status\s+of\s+(?:the\s+)?([A-Za-z][A-Za-z0-9._-]+)\s+(?:service|daemon)?",
        r"(?:service|daemon)\s+([A-Za-z][A-Za-z0-9._-]+)",
        r"^([A-Za-z][A-Za-z0-9._-]+)\s+service\b",
    ]
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name and name.lower() not in {"the", "is", "a", "an", "my"}:
                return {"name": name}, "ok"

    return {"name": ""}, "ambiguous"


def extract_disk(q: str) -> tuple[dict, str]:
    """linux_disk_usage — optional `path` arg."""
    paths = PATH_MENTIONED.findall(q)
    if paths:
        # Pick the longest mentioned path (most specific).
        return {"path": max(paths, key=len)}, "ok"
    return {}, "ok"  # default /home — schema allows omission


def extract_krunner_launch(q: str) -> tuple[dict, str]:
    """kde_krunner_launch — extract app/binary from real launch verbs."""
    # Pattern 1: explicit launch verb + target.
    pat = re.compile(
        r"(?:open|launch|start|run|fire\s*up|boot|load)\s+(?:up\s+|the\s+|my\s+|a\s+)?"
        r"([A-Za-z][\w\s.\-]*?)(?:\s+(?:please|now|app|application|window|browser))?[.\?\!]?\s*$",
        re.IGNORECASE,
    )
    m = pat.search(q)
    if m:
        candidate = _strip_app_name(m.group(1))
        if candidate:
            # If user said "open the file manager" / "open the browser", keep
            # those — generic but still real (downstream alias map handles).
            return {"app": candidate}, "ok"

    # Pattern 2: bare app name (often from KRunner history).
    if re.match(r"^\s*[A-Za-z][\w.\-]*\s*$", q):
        return {"app": q.strip().lower()}, "ok"

    return {"app": ""}, "ambiguous"


def extract_window_focus(q: str) -> tuple[dict, str]:
    """kde_window_focus — extract title from real focus verbs."""
    pat = re.compile(
        r"(?:switch\s+to|focus(?:\s+on)?|bring|show|raise|go\s+(?:back\s+)?to)\s+"
        r"(?:the\s+)?([A-Za-z][\w\s.\-]*?)(?:\s+(?:window|to\s+(?:the\s+)?(?:front|top)))?[.\?\!]?\s*$",
        re.IGNORECASE,
    )
    m = pat.search(q)
    if m:
        title = _strip_app_name(m.group(1))
        if title:
            return {"title": title}, "ok"
    return {"title": ""}, "ambiguous"


def extract_notification(q: str) -> tuple[dict, str]:
    """kde_notifications_send — derive title + message.
    Heuristic: "notify me X is done" → title="X", message=q stripped.
    Real source pairs (mined from journalctl etc.) should already have args
    populated; this extractor is for legacy/templated pairs.
    """
    q_strip = q.rstrip(".!? ")
    # Strip the leading "notify me / send a notification / show alert" preamble.
    preamble = re.compile(
        r"^\s*(notify\s+me\s+(?:that\s+|when\s+)?|send\s+(?:a\s+)?(?:desktop\s+)?(?:notification|alert)\s+(?:saying\s+|that\s+|about\s+)?|"
        r"show\s+(?:a\s+|an\s+)?(?:popup|alert|notification)\s+(?:saying\s+|that\s+)?|"
        r"remind\s+me\s+(?:when\s+|that\s+)?|"
        r"tell\s+me\s+(?:when\s+|that\s+)?)",
        re.IGNORECASE,
    )
    body = preamble.sub("", q_strip).strip()
    if not body:
        return {"title": "", "message": ""}, "ambiguous"

    # Title = first 1-3 nouns / first comma-separated chunk.
    if "," in body:
        first, rest = body.split(",", 1)
        title = first.strip().title()
        message = rest.strip().capitalize() + "."
    else:
        words = body.split()
        title = " ".join(words[:3]).rstrip(":,.").title()
        message = body.capitalize().rstrip(".") + "."
    return {"title": title, "message": message}, "ok"


def extract_dialog_confirm(q: str) -> tuple[dict, str]:
    """kde_dialog_confirm — derive a prompt sentence from the user's intent."""
    q_strip = q.rstrip(".!? ").strip()
    preamble = re.compile(
        r"^\s*(ask\s+(?:the\s+user\s+)?(?:before\s+|to\s+|if\s+)?|"
        r"confirm\s+(?:with\s+the\s+user\s+)?(?:that\s+(?:they\s+want\s+to\s+)?)?|"
        r"get\s+(?:user\s+)?(?:permission|approval|confirmation|consent)\s+(?:to\s+|for\s+|before\s+)?|"
        r"are\s+you\s+sure\s+(?:about\s+)?|"
        r"verify\s+(?:that\s+the\s+user\s+wants\s+)?(?:to\s+)?)",
        re.IGNORECASE,
    )
    body = preamble.sub("", q_strip).strip()
    if not body:
        return {"prompt": ""}, "ambiguous"
    # Make it a yes/no question.
    body_q = body[0].upper() + body[1:]
    if not body_q.endswith("?"):
        body_q = body_q.rstrip(".") + "?"
    return {"prompt": body_q}, "ok"


# Mapping tool name → extractor. Tools without args use the trivial extractor.
EXTRACTORS = {
    "linux_brightness_set":    extract_brightness,
    "linux_volume_set":        extract_volume,
    "linux_volume_change":     extract_volume,  # legacy alias
    "linux_service_status":    extract_service,
    "linux_disk_usage":        extract_disk,
    "kde_krunner_launch":      extract_krunner_launch,
    "kde_window_focus":        extract_window_focus,
    "kde_notifications_send":  extract_notification,
    "kde_dialog_confirm":      extract_dialog_confirm,
    # Tools that take no required args:
    "linux_memory_usage":      lambda q: ({}, "ok"),
    "linux_metrics_summary":   lambda q: ({}, "ok"),
    "linux_battery_status":    lambda q: ({}, "ok"),
    "linux_network_status":    lambda q: ({}, "ok"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def populate(pair: dict, *, force: bool = False) -> dict:
    """Return a copy of `pair` with target.arguments populated from the user query.

    If `force` is False (default), only OVERWRITE empty/missing args. If True,
    overwrite even non-empty args (use to repair labels like the
    `direction:"up"` mislabels).
    """
    out = json.loads(json.dumps(pair))  # deep copy
    target = out.get("target") or {}
    tool = target.get("name")
    if tool not in EXTRACTORS:
        return out

    user = ""
    for m in out.get("messages", []):
        if m.get("role") == "user":
            user = m.get("content", "")
            break
    if not user:
        return out

    new_args, status = EXTRACTORS[tool](user)
    cur_args = target.get("arguments") or {}

    if force:
        merged = {**cur_args, **new_args}
    else:
        # Only fill missing/empty slots.
        merged = dict(cur_args)
        for k, v in new_args.items():
            if k not in merged or merged[k] in ("", None):
                merged[k] = v

    target["arguments"] = merged
    out["target"] = target
    # Provenance may already be a string (e.g. "systemctl: NetworkManager") from
    # a miner. Promote to a dict so we can attach `arg_extraction` without
    # losing the original note.
    prov = out.get("provenance")
    if prov is None:
        prov = {}
    elif isinstance(prov, str):
        prov = {"note": prov}
    elif not isinstance(prov, dict):
        prov = {"note": str(prov)}
    prov["arg_extraction"] = status
    out["provenance"] = prov
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="JSONL file in dispatch_pairs schema to patch.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: rewrite in-place with .bak).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing non-empty args (repairs known mislabels).")
    ap.add_argument("--drop-ambiguous", action="store_true",
                    help="Drop pairs whose args remain unresolved.")
    args = ap.parse_args()

    inp: Path = args.inp
    if not inp.exists():
        print(f"input not found: {inp}", file=sys.stderr)
        return 1

    out_path: Path = args.out or inp
    if out_path == inp:
        bak = inp.with_suffix(inp.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(inp, bak)
            print(f"backup: {bak}")

    n_total = 0
    n_changed = 0
    by_tool: Counter = Counter()
    by_status: Counter = Counter()
    dropped = 0

    fixed_lines: list[str] = []
    for line in inp.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            fixed_lines.append(line)
            continue

        n_total += 1
        before = json.dumps(d.get("target", {}).get("arguments") or {}, sort_keys=True)
        d2 = populate(d, force=args.force)
        after = json.dumps(d2.get("target", {}).get("arguments") or {}, sort_keys=True)
        status = (d2.get("provenance") or {}).get("arg_extraction", "ok")
        by_status[status] += 1
        if before != after:
            n_changed += 1
            by_tool[d2["target"]["name"]] += 1

        if args.drop_ambiguous and status == "ambiguous":
            dropped += 1
            continue
        fixed_lines.append(json.dumps(d2, ensure_ascii=False))

    out_path.write_text("\n".join(fixed_lines) + "\n")

    print(f"\n  populate_arguments — {inp}")
    print(f"    total pairs : {n_total}")
    print(f"    args changed: {n_changed}")
    if dropped:
        print(f"    dropped     : {dropped} (--drop-ambiguous)")
    print(f"    by status   : {dict(by_status)}")
    print(f"    by tool     :")
    for t, n in by_tool.most_common():
        print(f"      {t:28s} {n}")
    print(f"    output      : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
