#!/usr/bin/env python3
"""
test_corpus_balance.py — quantify whether a corpus is good enough to train on.

Run as a script for a diagnostic report:
    python3 tests/test_corpus_balance.py
    python3 tests/test_corpus_balance.py --corpus path/to/corpus.txt

Run as pytest for hard pass/fail on iter #4 thresholds:
    pytest tests/test_corpus_balance.py -v

The thresholds encode the iter #4 design:
  - No tool > 25% of total lines (was 66% in iter #3)
  - Each of 12 tools >= 80 lines
  - >= 250 contrastive lines (was ~32)
  - >= 30% of tool-mentioning lines have the tool name OUTSIDE the
    `When the user says "X", call Y.` slot (varied position)
  - >= 80 multi-tool co-occurrence lines
  - Every new token from new_tokens.json appears >= 5 times
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "dataset" / "tokenizer" / "corpus.txt"
NEW_TOKENS     = REPO_ROOT / "dataset" / "tokenizer" / "new_tokens.json"

# Canonical tool names + every naming variant. Mirrors CORE_TOOLS in
# training/build_tokenizer_dataset.py — kept in sync by hand so the test
# stays runnable without importing the generator.
TOOL_VARIANTS: dict[str, list[str]] = {
    "linux_memory_usage":   ["linux_memory_usage", "memory_usage", "system_memory_usage", "linux.memory.usage"],
    "linux_disk_usage":     ["linux_disk_usage", "disk_usage", "system_disk_usage", "linux.disk.usage"],
    "linux_metrics_summary":["linux_metrics_summary", "metrics_summary", "system_metrics_summary", "linux.metrics.summary"],
    "linux_battery_status": ["linux_battery_status", "battery_status", "system_battery_status", "linux.battery.status"],
    "linux_brightness_set": ["linux_brightness_set", "brightness_set", "system_brightness_set", "linux.brightness.set"],
    "linux_volume_change":  ["linux_volume_change", "volume_change", "system_volume_change", "linux.volume.change"],
    "linux_network_status": ["linux_network_status", "network_status", "system_network_status", "linux.network.status"],
    "linux_service_status": ["linux_service_status", "service_status", "system_service_status", "linux.service.status"],
    "kde_krunner_launch":   ["kde_krunner_launch", "krunner_launch", "kde.krunner.launch"],
    "kde_window_focus":     ["kde_window_focus", "window_focus", "kde.window.focus"],
    "kde_notifications_send":["kde_notifications_send", "notifications_send", "kde.notifications.send"],
    "kde_dialog_confirm":   ["kde_dialog_confirm", "dialog_confirm", "kde.dialog.confirm"],
}

# Build a regex per tool that matches any of its naming variants as a
# whole token (longest-first so `linux_memory_usage` wins over `memory_usage`).
def _tool_regex(variants: list[str]) -> re.Pattern:
    sorted_vars = sorted(variants, key=len, reverse=True)
    pat = "|".join(re.escape(v) for v in sorted_vars)
    return re.compile(rf"(?<![A-Za-z0-9_.])({pat})(?![A-Za-z0-9_])")

TOOL_REGEX = {name: _tool_regex(vs) for name, vs in TOOL_VARIANTS.items()}

# A line is in the "dispatch slot" when the tool name appears at the very
# end of a `When the user says "X", call Y.` style sentence. This is the
# grammatical mold we are trying to break.
DISPATCH_SLOT = re.compile(
    r'^\s*[Ww]hen the user says\s+["\'].*?["\'],?\s+call\s+\S+\.?\s*$'
)

# Heuristic markers of a contrastive sentence (one that pushes embeddings
# apart). A line counts as contrastive if it contains at least one of these
# markers AND mentions two or more tool names.
CONTRAST_MARKERS = re.compile(
    r"\b(not|NOT|don't|do not|instead of|rather than|≠|vs\.?|versus|"
    r"different (from|domain|category)|unrelated|distinct|do not confuse|"
    r"is for .* not|is about .* not)\b",
    re.IGNORECASE,
)

# Thresholds — tunable single source of truth.
THRESHOLDS = {
    "max_tool_share_pct":     25.0,   # iter #3 had 66% in one tool
    "min_lines_per_tool":     80,     # iter #3 had 12 for some
    "min_contrastive":        250,    # iter #3 had ~32
    "min_varied_position_pct": 30.0,  # iter #3 was nearly 100% in dispatch slot
    "min_cooccurrence":       80,     # multi-tool sentences
    "min_token_coverage":     5,      # every new token mentioned >= 5 times
    "max_dispatch_slot_pct":  60.0,   # cap the templated mold
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    if path.suffix == ".jsonl":
        out = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line)["text"])
                except Exception:
                    continue
        return out
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def load_new_tokens() -> list[str]:
    if not NEW_TOKENS.exists():
        return []
    data = json.loads(NEW_TOKENS.read_text())
    out: list[str] = []
    for cat, items in data.items():
        for item in items:
            out.append(item["token"])
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_tool_counts(lines: list[str]) -> Counter:
    """Count the lines that mention each tool (any variant)."""
    counts: Counter = Counter()
    for ln in lines:
        for tool, rx in TOOL_REGEX.items():
            if rx.search(ln):
                counts[tool] += 1
    return counts


def primary_tool(line: str) -> str | None:
    """Return the canonical tool name whose variant appears FIRST in the line.
    Used to compute "dispatch share" — which tool dominates the corpus."""
    earliest_pos = None
    earliest_tool = None
    for tool, rx in TOOL_REGEX.items():
        m = rx.search(line)
        if m and (earliest_pos is None or m.start() < earliest_pos):
            earliest_pos = m.start()
            earliest_tool = tool
    return earliest_tool


def primary_tool_distribution(lines: list[str]) -> Counter:
    counts: Counter = Counter()
    for ln in lines:
        t = primary_tool(ln)
        if t:
            counts[t] += 1
    return counts


def count_contrastive(lines: list[str]) -> int:
    n = 0
    for ln in lines:
        tools_in_line = sum(1 for rx in TOOL_REGEX.values() if rx.search(ln))
        if tools_in_line >= 2 and CONTRAST_MARKERS.search(ln):
            n += 1
    return n


def count_cooccurrence(lines: list[str]) -> int:
    """Lines mentioning >= 2 distinct tool names."""
    n = 0
    for ln in lines:
        if sum(1 for rx in TOOL_REGEX.values() if rx.search(ln)) >= 2:
            n += 1
    return n


def position_breakdown(lines: list[str]) -> dict:
    """How many tool-mentioning lines are in the dispatch slot vs varied?"""
    dispatch = 0
    varied = 0
    for ln in lines:
        if not any(rx.search(ln) for rx in TOOL_REGEX.values()):
            continue
        if DISPATCH_SLOT.match(ln):
            dispatch += 1
        else:
            varied += 1
    total = dispatch + varied
    return {
        "dispatch_slot": dispatch,
        "varied":        varied,
        "total":         total,
        "varied_pct":    100.0 * varied / total if total else 0.0,
        "dispatch_pct":  100.0 * dispatch / total if total else 0.0,
    }


def token_coverage(lines: list[str], tokens: list[str]) -> dict[str, int]:
    text = "\n".join(lines)
    return {t: text.count(t) for t in tokens}


# ---------------------------------------------------------------------------
# Diagnostic report (script mode)
# ---------------------------------------------------------------------------

def report(corpus_path: Path) -> dict:
    lines = load_lines(corpus_path)
    new_tokens = load_new_tokens()

    per_tool = per_tool_counts(lines)
    primary = primary_tool_distribution(lines)
    n_contrast = count_contrastive(lines)
    n_cooc = count_cooccurrence(lines)
    pos = position_breakdown(lines)
    cov = token_coverage(lines, new_tokens)

    n_total = len(lines)
    top_tool, top_n = max(primary.items(), key=lambda x: x[1]) if primary else (None, 0)
    top_share = 100.0 * top_n / n_total if n_total else 0.0
    weak = [(t, n) for t, n in cov.items() if n < THRESHOLDS["min_token_coverage"]]

    bar = "=" * 72
    print(f"\n{bar}\n  Corpus balance report — {corpus_path}\n{bar}")
    print(f"  Total lines               : {n_total}")
    print(f"  Top tool (by primary pos) : {top_tool} = {top_n} lines "
          f"({top_share:.1f}% of corpus)")
    print(f"  Tools mentioned at all    : {len(per_tool)} / {len(TOOL_VARIANTS)}")
    print()

    print("  Per-tool primary-position distribution:")
    print(f"    {'tool':28s} {'lines':>7s} {'%':>6s}")
    for t in sorted(TOOL_VARIANTS, key=lambda x: -primary.get(x, 0)):
        n = primary.get(t, 0)
        pct = 100.0 * n / n_total if n_total else 0.0
        flag = ""
        if pct > THRESHOLDS["max_tool_share_pct"]:
            flag = " <-- DOMINATES"
        elif n < THRESHOLDS["min_lines_per_tool"]:
            flag = " <-- starved"
        print(f"    {t:28s} {n:7d} {pct:5.1f}%{flag}")
    print()

    print("  Position breakdown (tool-mentioning lines):")
    print(f"    in dispatch slot ('When the user says X, call Y'): {pos['dispatch_slot']:5d} ({pos['dispatch_pct']:5.1f}%)")
    print(f"    varied position                                  : {pos['varied']:5d} ({pos['varied_pct']:5.1f}%)")
    print()

    print(f"  Contrastive lines (>=2 tools + 'not'/'vs'/etc): {n_contrast}")
    print(f"  Co-occurrence lines (>=2 tools, any context)  : {n_cooc}")
    print()

    print(f"  New tokens ({len(new_tokens)}):")
    if weak:
        print(f"    {len(weak)} tokens with < {THRESHOLDS['min_token_coverage']} occurrences:")
        for t, n in sorted(weak, key=lambda x: x[1])[:10]:
            print(f"      {t!r:30s} {n}")
    else:
        print(f"    all {len(new_tokens)} tokens covered >= {THRESHOLDS['min_token_coverage']} times")
    print()

    # Pass/fail summary against thresholds
    checks = {
        "no tool dominates"     : top_share <= THRESHOLDS["max_tool_share_pct"],
        "all tools >= min lines": all(primary.get(t, 0) >= THRESHOLDS["min_lines_per_tool"]
                                      for t in TOOL_VARIANTS),
        "contrastive >= target" : n_contrast >= THRESHOLDS["min_contrastive"],
        "varied position >= 30%": pos["varied_pct"] >= THRESHOLDS["min_varied_position_pct"],
        "co-occurrence >= 80"   : n_cooc >= THRESHOLDS["min_cooccurrence"],
        "every token covered"   : len(weak) == 0,
        "dispatch slot <= 60%"  : pos["dispatch_pct"] <= THRESHOLDS["max_dispatch_slot_pct"],
    }
    print("  Threshold checks:")
    for name, ok in checks.items():
        mark = "PASS" if ok else "FAIL"
        print(f"    [{mark}] {name}")
    print()
    n_pass = sum(checks.values())
    print(f"  Result: {n_pass}/{len(checks)} checks passed.")
    print(bar)

    return {
        "n_total": n_total,
        "top_share_pct": top_share,
        "top_tool": top_tool,
        "primary": dict(primary),
        "n_contrastive": n_contrast,
        "n_cooccurrence": n_cooc,
        "position": pos,
        "weak_tokens": weak,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Pytest assertions
# ---------------------------------------------------------------------------

def _stats():
    return report(DEFAULT_CORPUS)


def test_no_tool_dominates():
    s = _stats()
    assert s["top_share_pct"] <= THRESHOLDS["max_tool_share_pct"], (
        f"{s['top_tool']} owns {s['top_share_pct']:.1f}% of the corpus "
        f"(threshold {THRESHOLDS['max_tool_share_pct']}%) — rebalance per_tool counts"
    )


def test_all_tools_have_min_lines():
    s = _stats()
    starved = [t for t in TOOL_VARIANTS
               if s["primary"].get(t, 0) < THRESHOLDS["min_lines_per_tool"]]
    assert not starved, (
        f"Starved tools (< {THRESHOLDS['min_lines_per_tool']} lines): "
        + ", ".join(f"{t}={s['primary'].get(t, 0)}" for t in starved)
    )


def test_contrastive_count():
    s = _stats()
    assert s["n_contrastive"] >= THRESHOLDS["min_contrastive"], (
        f"only {s['n_contrastive']} contrastive lines "
        f"(need {THRESHOLDS['min_contrastive']})"
    )


def test_position_diversity():
    s = _stats()
    assert s["position"]["varied_pct"] >= THRESHOLDS["min_varied_position_pct"], (
        f"only {s['position']['varied_pct']:.1f}% varied-position "
        f"(need {THRESHOLDS['min_varied_position_pct']}%)"
    )


def test_cooccurrence():
    s = _stats()
    assert s["n_cooccurrence"] >= THRESHOLDS["min_cooccurrence"]


def test_token_coverage():
    s = _stats()
    assert not s["weak_tokens"], (
        f"under-covered tokens: " + ", ".join(f"{t}={n}" for t, n in s["weak_tokens"][:5])
    )


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                    help=f"corpus file (default: {DEFAULT_CORPUS})")
    args = ap.parse_args()
    s = report(args.corpus)
    return 0 if all(s["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
