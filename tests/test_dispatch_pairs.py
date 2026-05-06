#!/usr/bin/env python3
"""
test_dispatch_pairs.py — validate the function-calling SFT dataset.

This is the dataset that fine-tunes the model on (query → tool call). Each
line of dataset/dispatch_pairs.jsonl is one supervised example:

    {"messages": [<system>, <user>], "tools": [<schema>, ...],
     "target": {"name": "<tool>", "arguments": {...}},
     "source": "<provenance>"}

Run as a script for a diagnostic report:
    python3 tests/test_dispatch_pairs.py
    python3 tests/test_dispatch_pairs.py --pairs path/to/file.jsonl

Run as pytest for hard pass/fail on iter #4 thresholds:
    pytest tests/test_dispatch_pairs.py -v

What it checks (function-calling specific):
  1. Schema integrity: every line parseable, has messages/tools/target.
  2. Per-tool count balance: no tool > MAX_TOOL_SHARE_PCT, all >= MIN.
  3. Query phrasing diversity per tool:
       - distinct lead words (not all "open X" / "launch X")
       - distinct query lengths (variance > floor)
       - bigram diversity ratio
  4. Tool schema consistency: tools[] in each line match tools/tool_schemas.json.
  5. Argument coverage per tool: how many distinct values does each
     required parameter take across the dataset?
  6. Source provenance: which source produced which tool, and how diverse
     is the source mix per tool.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS = REPO_ROOT / "dataset" / "dispatch_pairs.jsonl"
SCHEMAS_PATH  = REPO_ROOT / "tools" / "tool_schemas.json"

# Canonical mcp_name list — the values that may appear in target.name.
KNOWN_TOOLS = [
    "linux_memory_usage", "linux_disk_usage", "linux_metrics_summary",
    "linux_battery_status", "linux_brightness_set", "linux_volume_set",
    "linux_volume_change",  # legacy alias still seen in old data
    "linux_network_status", "linux_service_status",
    "kde_krunner_launch", "kde_window_focus",
    "kde_notifications_send", "kde_dialog_confirm",
]

THRESHOLDS = {
    "max_tool_share_pct"     : 25.0,
    "min_pairs_per_tool"     : 80,
    "min_lead_word_unique"   : 8,    # at least this many distinct first words per tool
    "min_bigram_diversity"   : 0.35, # distinct bigrams / total bigrams in tool's queries
    "min_length_stdev"       : 1.5,  # in words — penalises every-query-the-same-length
    "min_arg_uniqueness"     : 0.50, # distinct arg values / total pairs (per required arg)
    "min_sources_per_tool"   : 2,    # at least 2 provenance sources per tool
}


# ---------------------------------------------------------------------------

def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    out = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise AssertionError(f"line {i}: invalid JSON: {e}")
    return out


def load_schemas() -> dict:
    if not SCHEMAS_PATH.exists():
        return {}
    raw = json.loads(SCHEMAS_PATH.read_text())
    return {entry["mcp_name"]: entry for entry in raw.values()}


def user_query(pair: dict) -> str:
    for m in pair.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return ""


def lead_word(query: str) -> str:
    m = re.match(r"\s*(\S+)", query)
    return m.group(1).rstrip(".,!?;:").lower() if m else ""


def bigrams(query: str) -> list[tuple[str, str]]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9'_-]*", query.lower())
    return list(zip(toks, toks[1:]))


# ---------------------------------------------------------------------------
# Per-tool metrics
# ---------------------------------------------------------------------------

def tool_metrics(pairs: list[dict]) -> dict:
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        t = p.get("target", {}).get("name")
        if t:
            by_tool[t].append(p)

    metrics: dict[str, dict] = {}
    for tool, ps in by_tool.items():
        queries = [user_query(p) for p in ps]
        lengths = [len(q.split()) for q in queries if q]
        leads = Counter(lead_word(q) for q in queries if q)
        all_bigrams: list[tuple[str, str]] = []
        for q in queries:
            all_bigrams.extend(bigrams(q))
        unique_bg = len(set(all_bigrams))
        bg_ratio = unique_bg / max(1, len(all_bigrams))

        # Argument value diversity per required parameter
        arg_values: dict[str, set] = defaultdict(set)
        for p in ps:
            for k, v in (p.get("target", {}).get("arguments") or {}).items():
                if isinstance(v, (str, int, float, bool)):
                    arg_values[k].add(v)

        sources = Counter(p.get("source", "unknown") for p in ps)

        metrics[tool] = {
            "n":                  len(ps),
            "n_unique_queries":   len(set(queries)),
            "lead_words_unique":  len(leads),
            "lead_word_top5":     leads.most_common(5),
            "length_mean":        statistics.mean(lengths) if lengths else 0,
            "length_stdev":       statistics.stdev(lengths) if len(lengths) > 1 else 0,
            "bigram_diversity":   bg_ratio,
            "arg_values":         {k: len(v) for k, v in arg_values.items()},
            "arg_uniqueness":     {
                k: len(v) / max(1, len(ps)) for k, v in arg_values.items()
            },
            "sources":            dict(sources),
            "n_sources":          len(sources),
        }
    return metrics


# ---------------------------------------------------------------------------

def schema_consistency(pairs: list[dict], schemas: dict) -> list[str]:
    """Return list of (line_idx, error) violations."""
    errs = []
    for i, p in enumerate(pairs, start=1):
        tools = p.get("tools") or []
        target_name = p.get("target", {}).get("name")
        if target_name and target_name not in {t.get("name") for t in tools}:
            errs.append(f"line {i}: target {target_name!r} not in tools[]")
        for t in tools:
            name = t.get("name")
            if name in schemas:
                canonical = schemas[name]["mcp_schema"]["inputSchema"]
                actual = t.get("inputSchema")
                if actual and actual != canonical:
                    errs.append(f"line {i}: tool {name} schema drift")
    return errs


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------

def report(pairs_path: Path) -> dict:
    pairs = load_pairs(pairs_path)
    schemas = load_schemas()
    metrics = tool_metrics(pairs)
    drift = schema_consistency(pairs, schemas)

    n_total = len(pairs)
    bar = "=" * 78
    print(f"\n{bar}\n  Function-calling dataset report — {pairs_path}\n{bar}")
    print(f"  Total pairs              : {n_total}")
    print(f"  Tools represented        : {len(metrics)} / {len(KNOWN_TOOLS)}")
    missing = [t for t in KNOWN_TOOLS if t not in metrics]
    if missing:
        print(f"  Tools with ZERO pairs    : {', '.join(missing)}")
    print(f"  Schema drift errors      : {len(drift)}")
    if drift[:3]:
        for e in drift[:3]:
            print(f"      {e}")
        if len(drift) > 3:
            print(f"      (and {len(drift)-3} more)")
    print()

    # Per-tool table
    print(f"  {'tool':28s} {'n':>5s} {'%':>6s} {'uniq_q':>7s} "
          f"{'leads':>6s} {'len_sd':>7s} {'bg_div':>7s} {'srcs':>5s}")
    sorted_tools = sorted(metrics.items(), key=lambda x: -x[1]["n"])
    for tool, m in sorted_tools:
        pct = 100.0 * m["n"] / n_total
        flag = ""
        if pct > THRESHOLDS["max_tool_share_pct"]:
            flag = " <-- DOMINATES"
        elif m["n"] < THRESHOLDS["min_pairs_per_tool"]:
            flag = " <-- starved"
        print(f"  {tool:28s} {m['n']:5d} {pct:5.1f}% "
              f"{m['n_unique_queries']:7d} {m['lead_words_unique']:6d} "
              f"{m['length_stdev']:7.2f} {m['bigram_diversity']:7.2f} "
              f"{m['n_sources']:5d}{flag}")
    print()

    # Argument coverage
    print("  Argument value coverage (distinct values / total pairs per tool):")
    for tool, m in sorted_tools:
        if not m["arg_uniqueness"]:
            continue
        parts = ", ".join(f"{k}={v}/{m['n']}" for k, v in m["arg_values"].items())
        weak = any(u < THRESHOLDS["min_arg_uniqueness"] and m["n"] >= 10
                   for u in m["arg_uniqueness"].values())
        flag = " <-- low arg variety" if weak else ""
        print(f"    {tool:28s} {parts}{flag}")
    print()

    # Per-tool lead-word example
    print("  Top lead words per tool (a window into phrasing diversity):")
    for tool, m in sorted_tools:
        if m["n"] < 10:
            continue
        leads_str = ", ".join(f"{w}({n})" for w, n in m["lead_word_top5"])
        print(f"    {tool:28s} {leads_str}")
    print()

    # Source mix
    print("  Source mix per tool:")
    for tool, m in sorted_tools:
        srcs = ", ".join(f"{k}={v}" for k, v in sorted(m["sources"].items(), key=lambda x: -x[1]))
        flag = " <-- single source" if m["n_sources"] < THRESHOLDS["min_sources_per_tool"] else ""
        print(f"    {tool:28s} {srcs}{flag}")
    print()

    # Threshold checks
    top_tool, top_n = sorted_tools[0] if sorted_tools else (None, 0)
    top_pct = 100.0 * (top_n["n"] if top_tool else 0) / n_total

    starved = [t for t in KNOWN_TOOLS
               if metrics.get(t, {}).get("n", 0) < THRESHOLDS["min_pairs_per_tool"]]
    weak_leads = [t for t, m in metrics.items() if m["n"] >= 20
                  and m["lead_words_unique"] < THRESHOLDS["min_lead_word_unique"]]
    weak_lengths = [t for t, m in metrics.items() if m["n"] >= 20
                    and m["length_stdev"] < THRESHOLDS["min_length_stdev"]]
    weak_bg = [t for t, m in metrics.items() if m["n"] >= 20
               and m["bigram_diversity"] < THRESHOLDS["min_bigram_diversity"]]
    weak_args = [
        (t, k, m["arg_uniqueness"][k])
        for t, m in metrics.items() if m["n"] >= 20
        for k, u in m["arg_uniqueness"].items()
        if u < THRESHOLDS["min_arg_uniqueness"]
    ]
    weak_sources = [t for t, m in metrics.items() if m["n_sources"] < THRESHOLDS["min_sources_per_tool"]]

    checks = {
        "no tool dominates"     : top_pct <= THRESHOLDS["max_tool_share_pct"],
        "all tools >= min pairs": not starved,
        "lead-word diversity"   : not weak_leads,
        "length variation"      : not weak_lengths,
        "bigram diversity"      : not weak_bg,
        "argument variety"      : not weak_args,
        "multi-source per tool" : not weak_sources,
        "schema consistency"    : not drift,
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
        "n_total":          n_total,
        "metrics":          metrics,
        "drift":            drift,
        "starved":          starved,
        "weak_leads":       weak_leads,
        "weak_lengths":     weak_lengths,
        "weak_bg":          weak_bg,
        "weak_args":        weak_args,
        "weak_sources":     weak_sources,
        "top_share_pct":    top_pct,
        "top_tool":         top_tool,
        "checks":           checks,
    }


# ---------------------------------------------------------------------------
# Pytest assertions
# ---------------------------------------------------------------------------

def _stats():
    return report(DEFAULT_PAIRS)


def test_no_tool_dominates():
    s = _stats()
    assert s["top_share_pct"] <= THRESHOLDS["max_tool_share_pct"], (
        f"{s['top_tool']} owns {s['top_share_pct']:.1f}% of pairs"
    )


def test_all_tools_have_min_pairs():
    s = _stats()
    assert not s["starved"], f"starved tools: {s['starved']}"


def test_lead_word_diversity():
    s = _stats()
    assert not s["weak_leads"], (
        f"tools with too few distinct lead words: {s['weak_leads']}"
    )


def test_length_variation():
    s = _stats()
    assert not s["weak_lengths"], (
        f"tools with low query-length variance: {s['weak_lengths']}"
    )


def test_bigram_diversity():
    s = _stats()
    assert not s["weak_bg"], (
        f"tools with low bigram diversity: {s['weak_bg']}"
    )


def test_argument_variety():
    s = _stats()
    assert not s["weak_args"], (
        f"tools with low arg variety: {s['weak_args']}"
    )


def test_multi_source_per_tool():
    s = _stats()
    assert not s["weak_sources"], (
        f"tools sourced from only one place: {s['weak_sources']}"
    )


def test_schema_consistency():
    s = _stats()
    assert not s["drift"], "; ".join(s["drift"][:3])


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    args = ap.parse_args()
    s = report(args.pairs)
    return 0 if all(s["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
