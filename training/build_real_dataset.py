#!/usr/bin/env python3
"""
build_real_dataset.py — orchestrate real-source miners into one balanced
function-calling dataset.

Inputs (all in `dispatch_pairs.jsonl` schema):
  - dataset/dispatch_pairs.jsonl                (existing baseline)
  - dataset/real_sources/*.jsonl                 (output of every miner)

Pipeline:
  1. Load all sources, tag each line with its source filename.
  2. Apply `populate_arguments.populate(force=True)` to repair mislabels.
  3. Drop rows whose arg extraction stayed `ambiguous` AND have an empty
     required arg (the model would learn nothing from them).
  4. Deduplicate by (tool_name, user_query.lower()).
  5. Per-tool quota: cap any tool to MAX_PER_TOOL by even sub-sampling
     across sources. This is what fixes the 93% kde_krunner_launch flood.
  6. Hard floor: if a tool has < MIN_PER_TOOL after capping, log a warning
     — a downstream miner needs to produce more pairs for it.
  7. Write the rebalanced output to dataset/dispatch_pairs_v4.jsonl.
  8. Print a final balance report.

Why this exists
---------------
Iter #4 abandons synthetic templating. Instead, multiple miner scripts each
produce real-source JSONL fragments (man pages, KDE config, real systemd
state, real .desktop metadata, etc.). This script is the deterministic glue
that combines them into one training-ready file with verifiable balance.

Re-run after any miner produces new pairs:
    python3 training/build_real_dataset.py
    python3 tests/test_dispatch_pairs.py --pairs dataset/dispatch_pairs_v4.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow importing populate_arguments from this directory.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from populate_arguments import populate  # noqa: E402

REPO_ROOT       = HERE.parent
EXISTING_PAIRS  = REPO_ROOT / "dataset" / "dispatch_pairs.jsonl"
REAL_SOURCES_DIR= REPO_ROOT / "dataset" / "real_sources"
DEFAULT_OUT     = REPO_ROOT / "dataset" / "dispatch_pairs_v4.jsonl"

MAX_PER_TOOL = 200   # cap any tool's contribution
MIN_PER_TOOL = 80    # warn if a tool ends up below this
SEED = 42            # deterministic sub-sampling

KNOWN_TOOLS = [
    "linux_memory_usage", "linux_disk_usage", "linux_metrics_summary",
    "linux_battery_status", "linux_brightness_set", "linux_volume_set",
    "linux_network_status", "linux_service_status",
    "kde_krunner_launch", "kde_window_focus",
    "kde_notifications_send", "kde_dialog_confirm",
]

# Required-arg map per tool (from tools/tool_schemas.json). Used to detect
# pairs where extraction failed and the slot is still empty.
REQUIRED_ARGS: dict[str, list[str]] = {
    "linux_brightness_set":   ["direction"],
    "linux_volume_set":       ["direction"],
    "linux_volume_change":    ["direction"],
    "linux_service_status":   ["name"],
    "kde_krunner_launch":     ["app"],
    "kde_window_focus":       ["title"],
    "kde_notifications_send": ["title", "message"],
    "kde_dialog_confirm":     ["prompt"],
    # The other tools have no required args.
}


# ---------------------------------------------------------------------------

def _info(s: str)  -> None: print(f"  [INFO]    {s}")
def _ok(s: str)    -> None: print(f"  [OK]      {s}")
def _warn(s: str)  -> None: print(f"  [WARN]    {s}", file=sys.stderr)
def _drop(s: str)  -> None: print(f"  [DROP]    {s}")
def _banner(s: str)-> None: print("\n" + "=" * 78 + f"\n  {s}\n" + "=" * 78)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def is_arg_complete(pair: dict) -> bool:
    """True if all required args for this tool have non-empty values."""
    name = pair.get("target", {}).get("name")
    args = pair.get("target", {}).get("arguments") or {}
    for req in REQUIRED_ARGS.get(name, []):
        v = args.get(req)
        if v in (None, "", []):
            return False
    return True


def user_query(pair: dict) -> str:
    for m in pair.get("messages", []):
        if m.get("role") == "user":
            return (m.get("content") or "").strip().lower()
    return ""


# ---------------------------------------------------------------------------
# Per-tool quota with even sub-sampling across sources
# ---------------------------------------------------------------------------

def cap_per_tool(pairs: list[dict], cap: int, *, seed: int) -> list[dict]:
    """Keep up to `cap` pairs per tool, sampled evenly across distinct
    `source` values so no single source dominates a tool."""
    rnd = random.Random(seed)
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_tool[p.get("target", {}).get("name", "_unknown")].append(p)

    kept: list[dict] = []
    for tool, ps in by_tool.items():
        if len(ps) <= cap:
            kept.extend(ps)
            continue

        # Group by source, then round-robin across source buckets.
        by_src: dict[str, list[dict]] = defaultdict(list)
        for p in ps:
            by_src[p.get("source", "unknown")].append(p)
        # Shuffle each source bucket independently for fairness.
        for bucket in by_src.values():
            rnd.shuffle(bucket)

        sources = list(by_src.keys())
        rnd.shuffle(sources)
        picked: list[dict] = []
        while len(picked) < cap:
            progress = False
            for src in sources:
                if by_src[src]:
                    picked.append(by_src[src].pop())
                    progress = True
                    if len(picked) >= cap:
                        break
            if not progress:
                break
        kept.extend(picked)
        _info(f"capped {tool}: {len(ps)} → {len(picked)} (sources: {len(sources)})")
    return kept


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--existing", type=Path, default=EXISTING_PAIRS,
                    help=f"existing dispatch_pairs.jsonl (default: {EXISTING_PAIRS})")
    ap.add_argument("--real-sources-dir", type=Path, default=REAL_SOURCES_DIR,
                    help=f"directory of mined *.jsonl files (default: {REAL_SOURCES_DIR})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output path (default: {DEFAULT_OUT})")
    ap.add_argument("--max-per-tool", type=int, default=MAX_PER_TOOL)
    ap.add_argument("--min-per-tool", type=int, default=MIN_PER_TOOL)
    ap.add_argument("--balanced", action="store_true",
                    help=("down-sample EVERY tool to the minimum across the 12 "
                          "(produces a small but tightly-balanced set; useful "
                          "for first-pass training while real-source mining "
                          "is still catching up on starved tools)"))
    ap.add_argument("--balanced-target", type=int, default=None,
                    help="when --balanced, override floor with this exact count per tool")
    ap.add_argument("--keep-ambiguous", action="store_true",
                    help="don't drop rows whose required args came back empty")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    _banner("PHASE 1: load all sources")
    sources: list[tuple[Path, list[dict]]] = []
    if args.existing.exists():
        ps = load_jsonl(args.existing)
        sources.append((args.existing, ps))
        _ok(f"loaded {len(ps):5d} pairs from {args.existing.name}")
    else:
        _warn(f"no existing pairs file at {args.existing}")

    if args.real_sources_dir.exists():
        for f in sorted(args.real_sources_dir.glob("*.jsonl")):
            ps = load_jsonl(f)
            sources.append((f, ps))
            _ok(f"loaded {len(ps):5d} pairs from real_sources/{f.name}")
    else:
        _warn(f"no real-sources dir at {args.real_sources_dir} — only existing data will be used")

    flat: list[dict] = []
    for f, ps in sources:
        for p in ps:
            # Tag with source filename if not already set.
            p.setdefault("source", f.stem)
            flat.append(p)
    _ok(f"total raw pairs: {len(flat)}")

    _banner("PHASE 2: populate arguments (force-fix mislabels)")
    repaired = []
    statuses: Counter = Counter()
    for p in flat:
        p2 = populate(p, force=True)
        repaired.append(p2)
        statuses[(p2.get("provenance") or {}).get("arg_extraction", "ok")] += 1
    _ok(f"repaired arg statuses: {dict(statuses)}")

    _banner("PHASE 3: drop incomplete-arg rows" if not args.keep_ambiguous else "PHASE 3: keep all rows")
    if args.keep_ambiguous:
        kept_args = repaired
        _info("keeping all rows including ambiguous (per --keep-ambiguous)")
    else:
        kept_args = []
        dropped = 0
        for p in repaired:
            if is_arg_complete(p):
                kept_args.append(p)
            else:
                dropped += 1
        _ok(f"kept {len(kept_args)}; dropped {dropped} with unresolved required args")

    _banner("PHASE 4: dedupe by (tool, lower(user_query))")
    seen: set = set()
    deduped: list[dict] = []
    for p in kept_args:
        key = (p.get("target", {}).get("name", ""), user_query(p))
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    _ok(f"deduped: {len(kept_args)} → {len(deduped)} unique (query, tool) pairs")

    if args.balanced:
        # Determine the per-tool target: the minimum count across present tools,
        # or an explicit --balanced-target if given. Tools with zero pairs are
        # excluded from the min calculation.
        present_counts = Counter()
        for p in deduped:
            tn = p.get("target", {}).get("name")
            if tn:
                present_counts[tn] += 1
        present_min = min(c for c in present_counts.values() if c > 0) if present_counts else 0
        target = args.balanced_target if args.balanced_target is not None else present_min
        _banner(f"PHASE 5: --balanced — every tool down-sampled to {target}")
        capped = cap_per_tool(deduped, target, seed=args.seed)
        _ok(f"after balanced cap: {len(capped)} pairs ({target} × ~{len(present_counts)} tools)")
    else:
        _banner(f"PHASE 5: per-tool cap = {args.max_per_tool}")
        capped = cap_per_tool(deduped, args.max_per_tool, seed=args.seed)
        _ok(f"after cap: {len(capped)} pairs")

    _banner("PHASE 6: write output + balance report")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in capped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    _ok(f"wrote {args.out}")

    by_tool: Counter = Counter(p.get("target", {}).get("name") for p in capped)
    by_source: Counter = Counter(p.get("source") for p in capped)
    print()
    print(f"  Per-tool counts:")
    for t in KNOWN_TOOLS:
        n = by_tool.get(t, 0)
        flag = ""
        if n == 0:
            flag = " <-- ZERO"
        elif n < args.min_per_tool:
            flag = f" <-- below floor {args.min_per_tool}"
        print(f"    {t:28s} {n:5d}{flag}")

    print(f"\n  Source mix (top 10):")
    for s, n in by_source.most_common(10):
        print(f"    {s:40s} {n:5d}")

    print(f"\n  Total: {len(capped)} pairs across {len(by_tool)} tools and {len(by_source)} sources.")

    starved = [t for t in KNOWN_TOOLS if by_tool.get(t, 0) < args.min_per_tool]
    if starved:
        print()
        _warn(f"{len(starved)} tools below floor — need more real-source pairs:")
        for t in starved:
            print(f"             {t}: {by_tool.get(t, 0)}")
        return 1

    print()
    _ok("all tools meet minimum floor.")
    print()
    _info("next: pytest tests/test_dispatch_pairs.py --pairs dataset/dispatch_pairs_v4.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
