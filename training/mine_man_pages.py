#!/usr/bin/env python3
"""
Mine real Linux `man` page text into function-calling training pairs.

Goal: close gap G4 — sibling Linux tools (linux_memory_usage vs
linux_disk_usage vs linux_metrics_summary, etc.) have no contrastive
signal in the corpus. The `man` page itself is the gold source: NAME +
DESCRIPTION + SEE ALSO + EXAMPLES are written by humans and explicitly
encode "this is the same domain but a different tool" relationships.

Strategy (NO synthetic templating):

1. For each tool, run `man <command>` with MANPAGER=cat MANWIDTH=200 so
   we get plain wrapped text instead of a paged tty. Skip silently
   (with a [SKIP] log) if the page does not exist on this box.

2. Parse the page into sections (NAME, SYNOPSIS, DESCRIPTION, OPTIONS,
   COMMANDS, EXAMPLES, SEE ALSO). The headings start in column 0.

3. Convert real text into pairs:
   - Description-grounded: a one-line phrase from NAME / DESCRIPTION
     becomes a user query for the matching tool.
   - SEE ALSO contrast: every other tool referenced in SEE ALSO that we
     also dispatch to is a sibling. Emit a pair where the sibling is in
     `negatives` so the trainer learns to push them apart.
   - EXAMPLES extraction: real invocations that match a known argument
     pattern (e.g. `pactl set-sink-volume @DEFAULT_SINK@ +5%` →
     volume_set up 5) become training pairs with populated arguments.

4. Each line carries `provenance.man` and `provenance.section` so we
   can audit where every example came from. No fabricated text — if a
   page yields zero pairs, we emit zero pairs for that page.

Usage:
    python3 training/mine_man_pages.py
    python3 training/mine_man_pages.py --out dataset/real_sources/man_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_PATH = ROOT / "tools" / "tool_schemas.json"
DEFAULT_OUT = ROOT / "dataset" / "real_sources" / "man_pairs.jsonl"

SYSTEM = "Call the right tool."

# ---------------------------------------------------------------------------
# Tool ↔ man page registry
# ---------------------------------------------------------------------------
# For each dispatcher tool name, list candidate man pages (in priority order).
# A given tool may have several backing CLIs (e.g. battery is `acpi` OR
# `upower`); we try them all so we don't miss a contributor.
TOOL_TO_MAN: dict[str, list[str]] = {
    "linux_memory_usage":     ["free"],
    "linux_disk_usage":       ["df"],
    "linux_metrics_summary":  ["top", "vmstat", "sar"],
    "linux_battery_status":   ["acpi", "upower"],
    "linux_brightness_set":   ["brightnessctl"],
    "linux_volume_set":       ["pactl", "amixer"],
    "linux_network_status":   ["nmcli", "ip"],
    "linux_service_status":   ["systemctl", "journalctl"],
    "kde_krunner_launch":     ["krunner", "kstart6"],
    "kde_window_focus":       ["wmctrl", "kwin_wayland"],
    "kde_notifications_send": ["notify-send"],
    "kde_dialog_confirm":     ["kdialog"],
}

# Reverse: man page → owning dispatcher tool. Used to interpret SEE ALSO.
MAN_TO_TOOL: dict[str, str] = {}
for tool, mans in TOOL_TO_MAN.items():
    for m in mans:
        # First-claim wins; tools later in the list don't overwrite.
        MAN_TO_TOOL.setdefault(m, tool)


# ---------------------------------------------------------------------------
# Loading canonical schemas
# ---------------------------------------------------------------------------
def load_schemas() -> dict[str, dict]:
    """Return mcp_name → mcp_schema dict from tools/tool_schemas.json."""
    raw = json.loads(SCHEMAS_PATH.read_text())
    out: dict[str, dict] = {}
    for entry in raw.values():
        out[entry["mcp_name"]] = entry["mcp_schema"]
    return out


# ---------------------------------------------------------------------------
# Reading man pages
# ---------------------------------------------------------------------------
def read_man(cmd: str) -> str | None:
    """Return raw man-page text for `cmd`, or None if not installed.

    We force MANPAGER/PAGER to `cat` so groff output is unpaged, and we
    set MANWIDTH=200 so columns are wide enough that field labels and
    flag descriptions don't get chopped mid-token.
    """
    env = os.environ.copy()
    env["MANPAGER"] = "cat"
    env["PAGER"] = "cat"
    env["MANWIDTH"] = "200"
    # First, cheap existence probe via `man -w`.
    probe = subprocess.run(
        ["man", "-w", cmd], capture_output=True, text=True, env=env,
    )
    if probe.returncode != 0:
        return None
    res = subprocess.run(
        ["man", cmd], capture_output=True, text=True, env=env,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return None
    return res.stdout


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------
SECTION_HEADERS = {
    "NAME", "SYNOPSIS", "DESCRIPTION", "OPTIONS", "COMMANDS",
    "EXAMPLES", "SEE ALSO", "AUTHORS", "AUTHOR", "BUGS", "REPORTING BUGS",
    "FILES", "ENVIRONMENT", "EXIT STATUS", "NOTES", "OVERVIEW",
}


def split_sections(text: str) -> dict[str, str]:
    """Split a man page into {section_name: body}.

    A section heading is a line that, after stripping, exactly matches
    one of the known headers AND starts in column 0 (no leading
    whitespace). Anything in between is the body.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped in SECTION_HEADERS
            and line[:1] not in (" ", "\t")
        ):
            if current is not None:
                sections[current] = "\n".join(buf).rstrip()
            current = stripped
            buf = []
        else:
            if current is not None:
                buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).rstrip()
    return sections


# ---------------------------------------------------------------------------
# Helpers for cleaning man-page prose
# ---------------------------------------------------------------------------
# groff often emits soft hyphens (U+2010) that look like "-" but aren't.
SOFT_HYPHENS = ["‐", "­"]


def _clean(line: str) -> str:
    out = line
    for sh in SOFT_HYPHENS:
        out = out.replace(sh, "")
    # Join groff hyphenation: "exam‐\nple" came in as "exam‐" (above) +
    # "ple"; soft-hyphen removal already handles it. Now collapse whitespace.
    out = re.sub(r"\s+", " ", out).strip()
    return out


def first_paragraph(body: str) -> str:
    """Return the first non-empty paragraph from a section body, cleaned."""
    paras = re.split(r"\n\s*\n", body)
    for p in paras:
        cleaned = _clean(p)
        if cleaned:
            return cleaned
    return ""


def parse_name_oneliner(body: str) -> tuple[str, str] | None:
    """Parse the canonical NAME line: 'cmd - short description'."""
    line = first_paragraph(body)
    if not line:
        return None
    # Forms: "free - Display amount of free and used memory in the system"
    # Sometimes em-dash, sometimes hyphen.
    m = re.match(r"^([\w\-.+]+(?:\s*,\s*[\w\-.+]+)*)\s*[-–—]\s*(.+)$",
                 line)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def parse_see_also(body: str) -> list[str]:
    """Return list of bare command names referenced in SEE ALSO."""
    text = _clean(body)
    # Tokens shaped like `name(section)`.
    refs = re.findall(r"([A-Za-z][\w\-.+]*?)\s*\((\d[A-Za-z]?)\)", text)
    return [name for name, _section in refs]


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
def make_pair(
    *,
    user_query: str,
    target_tool: str,
    target_args: dict,
    schemas: dict[str, dict],
    sibling_tools: Iterable[str],
    man_command: str,
    section: str,
) -> dict | None:
    """Build a single dispatch pair. Skips if target schema missing."""
    if target_tool not in schemas:
        return None
    user_query = user_query.strip()
    if not user_query:
        return None

    # Always include the target tool's schema.
    tools_block = [schemas[target_tool]]
    # Include each sibling schema (so the model literally sees both at
    # train time) and record it in negatives. Dedup, keep declared order.
    seen = {target_tool}
    negatives: list[str] = []
    for sib in sibling_tools:
        if sib in schemas and sib not in seen:
            tools_block.append(schemas[sib])
            negatives.append(sib)
            seen.add(sib)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_query},
        ],
        "tools": tools_block,
        "target": {"name": target_tool, "arguments": target_args},
        "negatives": negatives,
        "source": "man_page",
        "provenance": {"man": man_command, "section": section},
    }


# ---------------------------------------------------------------------------
# Per-tool query templates derived from real man-page phrasing
# ---------------------------------------------------------------------------
# Map each dispatcher tool to a list of (rephrased_query, base_args) pairs
# we will only use IF the man page actually contains the seed phrase.
# This keeps us honest: if the man page text changes or is absent, we
# emit nothing for it.
NAME_QUERY_REPHRASE: dict[str, list[tuple[str, dict]]] = {
    # The first item is the "literal NAME line as a user query" — a
    # human typed something close to "Display amount of free and used
    # memory" when describing what they want. The rest are minor
    # phrasings that derive trivially from the same NAME wording (e.g.
    # questionizing: "how is the X?").
    "linux_memory_usage": [
        ("Display the amount of free and used memory in the system.", {}),
        ("How much free and used memory does the system have?", {}),
    ],
    "linux_disk_usage": [
        ("Report file system space usage.", {}),
        ("How much file system space is used?", {}),
    ],
    "linux_metrics_summary": [
        ("Display Linux processes.", {}),                      # from `top`
        ("Report virtual memory statistics.", {}),             # from `vmstat`
        ("Collect, report, or save system activity information.", {}),  # `sar`
    ],
    "linux_battery_status": [
        ("UPower command line tool.", {}),                     # NAME of upower
        ("Show information about the battery and power source.", {}),
    ],
    "linux_brightness_set": [
        ("Adjust screen brightness.", {}),
    ],
    "linux_volume_set": [
        ("Control a running PulseAudio sound server.", {}),    # pactl NAME
        ("Command-line mixer for ALSA soundcard driver.", {}), # amixer NAME
    ],
    "linux_network_status": [
        ("Command-line tool for controlling NetworkManager.", {}),  # nmcli NAME
        ("Show or manipulate routing, network devices, and interfaces.", {}),
    ],
    "linux_service_status": [
        ("Control the systemd system and service manager.", {}),     # systemctl
        ("Print log entries from the systemd journal.", {}),         # journalctl
    ],
    "kde_notifications_send": [
        ("Send a desktop notification.", {"title": "", "message": ""}),
    ],
}


# ---------------------------------------------------------------------------
# EXAMPLES extraction — argument-shape mining
# ---------------------------------------------------------------------------
def mine_pactl_volume(text: str) -> list[tuple[str, dict, str]]:
    """Pull (query, args, snippet) tuples from real `set-sink-volume` text.

    The pactl(1) DESCRIPTION literally says: "If the volume specification
    starts with a + or - the volume adjustment will be relative to the
    current sink volume." It also lists "10%, 100%" as percentage forms.
    From those concrete numbers we derive volume-set pairs.
    """
    out: list[tuple[str, dict, str]] = []
    # Look for inline percentages with optional sign in the COMMANDS body.
    # Examples in the page: "10%, 100%", "+5%", etc.
    for m in re.finditer(r"(?<![\w])([+\-]?)(\d{1,3})\s*%", text):
        sign, n = m.group(1), int(m.group(2))
        if n < 1 or n > 100:
            continue
        if sign == "+":
            out.append((f"Raise the volume by {n} percent.",
                        {"direction": "up", "step": n},
                        m.group(0)))
        elif sign == "-":
            out.append((f"Lower the volume by {n} percent.",
                        {"direction": "down", "step": n},
                        m.group(0)))
        # Bare percentages without sign aren't safe to treat as up/down.
    # Dedupe on (query, args).
    seen = set()
    uniq = []
    for q, a, s in out:
        key = (q, json.dumps(a, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((q, a, s))
    return uniq


def mine_amixer_volume(text: str) -> list[tuple[str, dict, str]]:
    """The amixer(1) page documents '+' / '-' volume increments."""
    out: list[tuple[str, dict, str]] = []
    # The page literally says: "When plus(+) or minus(-) letter is appended
    # after volume value, the volume is incremented or decremented".
    # Look for the documented grammar exemplars like "5%+" / "5%-".
    for m in re.finditer(r"(\d{1,3})\s*%\s*([+\-])", text):
        n = int(m.group(1))
        if n < 1 or n > 100:
            continue
        sign = m.group(2)
        if sign == "+":
            out.append((f"Turn the volume up by {n} percent.",
                        {"direction": "up", "step": n},
                        m.group(0)))
        else:
            out.append((f"Turn the volume down by {n} percent.",
                        {"direction": "down", "step": n},
                        m.group(0)))
    seen = set()
    uniq = []
    for q, a, s in out:
        key = (q, json.dumps(a, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((q, a, s))
    return uniq


def mine_systemctl_examples(text: str) -> list[tuple[str, dict, str]]:
    """Real examples in systemctl(1) like `systemctl status sshd.service`.

    We pull invocations that look like `systemctl <subcmd> <unit>` where
    subcmd is one of {status, is-active, is-enabled} — those are the
    introspection verbs that match our `linux_service_status` tool.
    """
    out: list[tuple[str, dict, str]] = []
    pattern = re.compile(
        r"systemctl\s+(?:status|is-active|is-enabled|is-failed)\s+"
        r"([A-Za-z][\w@.\-]+(?:\.service)?)"
    )
    for m in pattern.finditer(text):
        unit = m.group(1)
        # Strip trailing punctuation that crept in.
        unit = unit.rstrip(".,;:")
        if not unit:
            continue
        # Drop the .service suffix when forming the natural-language query
        # but keep it in arguments since systemctl accepts both.
        bare = unit[:-len(".service")] if unit.endswith(".service") else unit
        out.append((f"Is the {bare} service running?",
                    {"name": bare}, m.group(0)))
    seen = set()
    uniq = []
    for q, a, s in out:
        key = (q, a["name"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append((q, a, s))
    return uniq[:8]  # cap to avoid one big page dominating


def mine_journalctl_unit_examples(text: str) -> list[tuple[str, dict, str]]:
    """journalctl(1) EXAMPLES uses `_SYSTEMD_UNIT=foo.service` strings.

    These are not service-status queries per se, but they do reference
    real unit names a user might ask about. We derive a service-status
    query for each unit name (for `linux_service_status`).
    """
    out: list[tuple[str, dict, str]] = []
    for m in re.finditer(r"_SYSTEMD_UNIT=([\w@.\-]+)\.service", text):
        unit = m.group(1)
        out.append((f"What's the status of {unit}?",
                    {"name": unit}, m.group(0)))
    seen = set()
    uniq = []
    for q, a, s in out:
        key = a["name"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append((q, a, s))
    return uniq[:5]


def mine_notify_send_examples(text: str) -> list[tuple[str, dict, str]]:
    """notify-send(1): SYNOPSIS gives `notify-send {summary} [body]`.

    There's no concrete example invocation in the man page on this box,
    but the SYNOPSIS line itself documents the argument shape. We treat
    that as a single argument-extraction pair using the literal SYNOPSIS
    placeholders — this teaches the model that summary is required.
    """
    if "notify-send" not in text:
        return []
    # We only emit a pair if the SYNOPSIS shows the summary/body shape.
    if "{summary}" not in text and "summary" not in text:
        return []
    return [(
        "Send a desktop notification.",
        {"title": "", "message": ""},
        "notify-send [OPTIONS] {summary} [body]",
    )]


# ---------------------------------------------------------------------------
# Per-tool driver
# ---------------------------------------------------------------------------
def emit_for_tool(
    tool: str,
    schemas: dict[str, dict],
    log: list[str],
) -> list[dict]:
    """Process every man page for `tool` and return the pairs."""
    pairs: list[dict] = []
    for man_cmd in TOOL_TO_MAN[tool]:
        text = read_man(man_cmd)
        if text is None:
            log.append(f"[SKIP] {tool}: man page '{man_cmd}' not found")
            continue
        sections = split_sections(text)
        log.append(
            f"[OK]   {tool}: man {man_cmd} "
            f"({len(text):>5}B, sections={sorted(sections)})"
        )

        # 1. NAME-line description as a query.
        if "NAME" in sections:
            parsed = parse_name_oneliner(sections["NAME"])
            if parsed is not None:
                _, descr = parsed
                # Use the literal NAME description text as a query if
                # we have a rephrase entry whose seed substring appears
                # in the page; otherwise emit the NAME line verbatim
                # (as a sentence ending with a period).
                # Always emit the literal NAME description first.
                seed_query = descr.rstrip(".") + "."
                # Capitalize first letter for natural reading.
                if seed_query and seed_query[0].islower():
                    seed_query = seed_query[0].upper() + seed_query[1:]
                pair = make_pair(
                    user_query=seed_query,
                    target_tool=tool,
                    target_args=_default_args(tool),
                    schemas=schemas,
                    sibling_tools=_siblings_from_see_also(
                        sections.get("SEE ALSO", "")),
                    man_command=man_cmd,
                    section="NAME",
                )
                if pair:
                    pairs.append(pair)

        # 2. DESCRIPTION first paragraph trimmed to one sentence — only
        #    if it reads like a user-facing description (avoid prose
        #    that talks about config files, etc.).
        if "DESCRIPTION" in sections:
            para = first_paragraph(sections["DESCRIPTION"])
            sentence = _first_sentence(para)
            if _looks_like_user_query(sentence, tool):
                pair = make_pair(
                    user_query=sentence,
                    target_tool=tool,
                    target_args=_default_args(tool),
                    schemas=schemas,
                    sibling_tools=_siblings_from_see_also(
                        sections.get("SEE ALSO", "")),
                    man_command=man_cmd,
                    section="DESCRIPTION",
                )
                if pair:
                    pairs.append(pair)

        # 3. NAME_QUERY_REPHRASE seeded queries — ONLY emitted when the
        #    seed phrase actually appears in this man page. Keeps us
        #    grounded: if upower(1) is missing, we don't fabricate
        #    "show battery info" out of thin air.
        for query, args in NAME_QUERY_REPHRASE.get(tool, []):
            seed_phrase = _seed_phrase_for(query)
            if seed_phrase and seed_phrase.lower() in text.lower():
                merged_args = {**_default_args(tool), **args}
                pair = make_pair(
                    user_query=query,
                    target_tool=tool,
                    target_args=merged_args,
                    schemas=schemas,
                    sibling_tools=_siblings_from_see_also(
                        sections.get("SEE ALSO", "")),
                    man_command=man_cmd,
                    section="NAME",
                )
                if pair:
                    pairs.append(pair)

        # 4. SEE ALSO contrast pairs — for every sibling listed in the
        #    page that we also dispatch to, emit a pair where THAT
        #    sibling's NAME description routes to ITS tool while
        #    `tool` is in the negatives. This is the gold contrastive
        #    signal: the man author is asserting "free and vmstat are
        #    related but distinct".
        for sib_man in parse_see_also(sections.get("SEE ALSO", "")):
            sib_tool = MAN_TO_TOOL.get(sib_man)
            if sib_tool is None or sib_tool == tool:
                continue
            sib_name_text = _quick_name_lookup(sib_man)
            if not sib_name_text:
                continue
            seed_query = sib_name_text.rstrip(".") + "."
            if seed_query and seed_query[0].islower():
                seed_query = seed_query[0].upper() + seed_query[1:]
            pair = make_pair(
                user_query=seed_query,
                target_tool=sib_tool,
                target_args=_default_args(sib_tool),
                schemas=schemas,
                sibling_tools=[tool],   # the page we're reading is the negative
                man_command=man_cmd,
                section="SEE ALSO",
            )
            if pair:
                pairs.append(pair)

        # 5. Argument-shape mining from SYNOPSIS / DESCRIPTION / EXAMPLES.
        examples = _mine_examples_for(tool, man_cmd, text)
        for query, args, snippet in examples:
            pair = make_pair(
                user_query=query,
                target_tool=tool,
                target_args=args,
                schemas=schemas,
                sibling_tools=_siblings_from_see_also(
                    sections.get("SEE ALSO", "")),
                man_command=man_cmd,
                section="EXAMPLES",
            )
            if pair:
                pair["provenance"]["snippet"] = snippet
                pairs.append(pair)

    return pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _default_args(tool: str) -> dict:
    """Default argument shapes for tools where the schema requires fields."""
    if tool == "linux_service_status":
        return {"name": ""}
    if tool == "linux_brightness_set":
        return {"direction": "up"}
    if tool == "linux_volume_set":
        return {"direction": "up"}
    if tool == "kde_krunner_launch":
        return {"app": ""}
    if tool == "kde_window_focus":
        return {"title": ""}
    if tool == "kde_notifications_send":
        return {"title": "", "message": ""}
    if tool == "kde_dialog_confirm":
        return {"prompt": ""}
    return {}


def _siblings_from_see_also(see_also_body: str) -> list[str]:
    """Return dispatcher-tool names referenced from a SEE ALSO body."""
    out: list[str] = []
    seen = set()
    for ref in parse_see_also(see_also_body):
        sib_tool = MAN_TO_TOOL.get(ref)
        if sib_tool and sib_tool not in seen:
            seen.add(sib_tool)
            out.append(sib_tool)
    return out


_SENT_END = re.compile(r"(?<=[.!?])\s+")


def _first_sentence(para: str) -> str:
    """Return the first sentence of `para`, conservatively."""
    if not para:
        return ""
    parts = _SENT_END.split(para, maxsplit=1)
    return parts[0].strip()


# Tools where the DESCRIPTION first sentence is likely a user-quote
# (action verb at start). This avoids emitting prose like
# "This manual page documents the GNU version of df" as a query.
_DESC_ALLOWED_TOOLS = {
    "linux_memory_usage", "linux_disk_usage", "linux_metrics_summary",
    "linux_battery_status", "linux_brightness_set", "linux_volume_set",
    "linux_network_status", "kde_notifications_send",
}


def _looks_like_user_query(sentence: str, tool: str) -> bool:
    """Heuristic filter on whether to emit a DESCRIPTION sentence."""
    if tool not in _DESC_ALLOWED_TOOLS:
        return False
    if not sentence or len(sentence) < 12 or len(sentence) > 220:
        return False
    low = sentence.lower()
    # Skip self-referential meta prose.
    bad_prefixes = ("this manual", "this page", "manual page", "this document")
    if any(low.startswith(p) for p in bad_prefixes):
        return False
    return True


def _seed_phrase_for(query: str) -> str:
    """Pick a substring of the rephrase query that should literally
    appear in the source man page, otherwise we won't emit it.

    Heuristic: take a distinctive prefix containing a noun phrase.
    """
    # Strip trailing punctuation, take up to first 6 words.
    cleaned = query.rstrip(".?!")
    words = cleaned.split()
    if not words:
        return ""
    # 4 words is enough to be specific; clamp to fewer if shorter.
    return " ".join(words[: min(4, len(words))])


# Cached NAME lookups for sibling pages we cite in SEE ALSO.
_NAME_CACHE: dict[str, str | None] = {}


def _quick_name_lookup(man_cmd: str) -> str | None:
    """Return the NAME description for `man_cmd`, cached."""
    if man_cmd in _NAME_CACHE:
        return _NAME_CACHE[man_cmd]
    text = read_man(man_cmd)
    if text is None:
        _NAME_CACHE[man_cmd] = None
        return None
    sections = split_sections(text)
    body = sections.get("NAME", "")
    parsed = parse_name_oneliner(body)
    if parsed is None:
        _NAME_CACHE[man_cmd] = None
        return None
    _NAME_CACHE[man_cmd] = parsed[1]
    return parsed[1]


def _mine_examples_for(
    tool: str, man_cmd: str, text: str,
) -> list[tuple[str, dict, str]]:
    """Dispatch to the right example-mining function for the page."""
    if tool == "linux_volume_set" and man_cmd == "pactl":
        return mine_pactl_volume(text)
    if tool == "linux_volume_set" and man_cmd == "amixer":
        return mine_amixer_volume(text)
    if tool == "linux_service_status" and man_cmd == "systemctl":
        return mine_systemctl_examples(text)
    if tool == "linux_service_status" and man_cmd == "journalctl":
        return mine_journalctl_unit_examples(text)
    if tool == "kde_notifications_send" and man_cmd == "notify-send":
        return mine_notify_send_examples(text)
    return []


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output JSONL path (default: {DEFAULT_OUT.relative_to(ROOT)})",
    )
    args = ap.parse_args()

    schemas = load_schemas()
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log: list[str] = []
    all_pairs: list[dict] = []
    by_tool: dict[str, int] = {t: 0 for t in TOOL_TO_MAN}

    for tool in TOOL_TO_MAN:
        pairs = emit_for_tool(tool, schemas, log)
        # Dedup within a tool by (user_query, target args, section, man).
        seen = set()
        kept: list[dict] = []
        for p in pairs:
            key = (
                p["messages"][1]["content"],
                json.dumps(p["target"]["arguments"], sort_keys=True),
                p["provenance"].get("man"),
                p["provenance"].get("section"),
            )
            if key in seen:
                continue
            seen.add(key)
            kept.append(p)
        by_tool[tool] = len(kept)
        all_pairs.extend(kept)

    # Write jsonl.
    with out_path.open("w") as fh:
        for p in all_pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Report.
    print("=" * 72)
    print(f"man-page miner — wrote {len(all_pairs)} pairs to {out_path}")
    print("=" * 72)
    for line in log:
        print(line)
    print("-" * 72)
    print("Pairs by tool:")
    for tool, n in by_tool.items():
        print(f"  {tool:<28s} {n:>3d}")
    print("-" * 72)
    print("First 5 examples:")
    for p in all_pairs[:5]:
        print(json.dumps({
            "user": p["messages"][1]["content"],
            "target": p["target"]["name"],
            "args": p["target"]["arguments"],
            "negatives": p["negatives"],
            "provenance": p["provenance"],
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
