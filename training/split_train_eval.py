#!/usr/bin/env python3
"""
split_train_eval.py — stratified train/eval split for the function-calling SFT dataset.

Reads a JSONL file (default: dataset/dispatch_pairs_v4.jsonl) and writes two new
JSONL files (train + eval). The split is stratified per-tool and (within each
tool) per-source, so that:

  * every tool with >= 20 pairs has at least 5 eval examples (configurable),
  * eval per-tool is capped (default 30) so we don't waste eval budget on
    over-represented tools,
  * tools with < 20 pairs go entirely to train (eval coverage is impossible
    to measure for them with only a handful of pairs — emit a [WARN]).

Pair fields are preserved bit-for-bit — including any `negatives`, `provenance`,
or other future fields. The script never mutates the input file.

Determinism: with the same input file + same flags, the script produces the
same outputs bit-for-bit.

Run:
    python3 training/split_train_eval.py
    python3 training/split_train_eval.py --in dataset/dispatch_pairs_v4_with_negatives.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN  = REPO_ROOT / "dataset" / "dispatch_pairs_v4.jsonl"
DEFAULT_TRAIN = REPO_ROOT / "dataset" / "dispatch_pairs_v4_train.jsonl"
DEFAULT_EVAL  = REPO_ROOT / "dataset" / "dispatch_pairs_v4_eval.jsonl"


def _user_query(pair: dict) -> str:
    for m in pair.get("messages", []):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _key(pair: dict) -> tuple[str, str]:
    """Dedup / leakage key: (tool name, lowercased user query)."""
    tool = (pair.get("target") or {}).get("name", "")
    return (tool, _user_query(pair).lower())


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    pairs: list[dict] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            pairs.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"line {i}: invalid JSON: {e}")
    return pairs


def dedupe(pairs: list[dict]) -> list[dict]:
    """Keep first occurrence per (tool, lower(query)) key. Order preserved."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for p in pairs:
        k = _key(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def split_one_tool(
    tool: str,
    tool_pairs: list[dict],
    *,
    ratio: float,
    min_eval: int,
    max_eval: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Stratify ONE tool's pairs across sources, then pick eval size.
    Returns (train_pairs, eval_pairs, warnings).
    """
    warnings: list[str] = []
    n = len(tool_pairs)

    # Tools with too few pairs: all go to train, no eval coverage.
    if n < 20:
        warnings.append(
            f"[WARN] tool {tool!r} has only {n} pairs (<20) — no eval coverage; "
            f"all pairs go to train."
        )
        return list(tool_pairs), [], warnings

    # Bucket by source so we can sample evenly.
    by_source: dict[str, list[dict]] = defaultdict(list)
    for p in tool_pairs:
        by_source[p.get("source", "unknown")].append(p)

    # Deterministic source iteration order.
    source_order = sorted(by_source.keys())
    for s in source_order:
        rng.shuffle(by_source[s])

    # Target eval size: clamp ratio result into [min_eval, max_eval].
    target_eval = int(round(n * (1.0 - ratio)))
    target_eval = max(min_eval, target_eval)
    target_eval = min(max_eval, target_eval, n - 1)  # keep at least 1 in train

    # Round-robin pick across sources until target_eval reached.
    eval_set: list[dict] = []
    cursors = {s: 0 for s in source_order}
    keep_going = True
    while keep_going and len(eval_set) < target_eval:
        keep_going = False
        for s in source_order:
            if len(eval_set) >= target_eval:
                break
            idx = cursors[s]
            if idx < len(by_source[s]):
                eval_set.append(by_source[s][idx])
                cursors[s] = idx + 1
                keep_going = True

    eval_keys = {id(p) for p in eval_set}
    train_set = [p for p in tool_pairs if id(p) not in eval_keys]
    return train_set, eval_set, warnings


def write_jsonl(path: Path, pairs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False, sort_keys=False))
            f.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in",  dest="in_path",   type=Path, default=DEFAULT_IN,
                    help="input JSONL (default: dataset/dispatch_pairs_v4.jsonl)")
    ap.add_argument("--out-train", dest="out_train", type=Path, default=DEFAULT_TRAIN,
                    help="train output JSONL")
    ap.add_argument("--out-eval",  dest="out_eval",  type=Path, default=DEFAULT_EVAL,
                    help="eval output JSONL")
    ap.add_argument("--ratio",     type=float, default=0.9,
                    help="train fraction (default: 0.9 → 10%% eval)")
    ap.add_argument("--min-eval",  type=int,   default=5,
                    help="floor on eval size per tool (>=20 pairs); default 5")
    ap.add_argument("--max-eval",  type=int,   default=30,
                    help="ceiling on eval size per tool; default 30")
    ap.add_argument("--seed",      type=int,   default=42,
                    help="RNG seed (default: 42)")
    args = ap.parse_args(argv)

    if not (0.0 < args.ratio < 1.0):
        raise SystemExit(f"--ratio must be in (0, 1), got {args.ratio}")
    if args.min_eval < 0 or args.max_eval < args.min_eval:
        raise SystemExit("require 0 <= --min-eval <= --max-eval")

    print(f"[split] reading {args.in_path}")
    raw_pairs = load_pairs(args.in_path)
    n_raw = len(raw_pairs)
    pairs = dedupe(raw_pairs)
    n_unique = len(pairs)
    if n_unique != n_raw:
        print(f"[split] deduped {n_raw} -> {n_unique} pairs by (tool, lower(query))")

    # Group by tool, deterministically.
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_tool[(p.get("target") or {}).get("name", "")].append(p)

    tools_sorted = sorted(by_tool.keys())  # deterministic order
    rng_master = random.Random(args.seed)

    train_all: list[dict] = []
    eval_all:  list[dict] = []
    table_rows: list[tuple[str, int, int, int, int]] = []
    tools_ge_20: list[str] = []
    warnings_all: list[str] = []

    for tool in tools_sorted:
        # Per-tool RNG derived from master for reproducibility independent of order.
        tool_seed = rng_master.randrange(2**31)
        rng = random.Random(tool_seed)
        # Stable order before shuffling: rely on input order, then shuffle copy.
        tool_pairs = list(by_tool[tool])
        rng.shuffle(tool_pairs)

        train_part, eval_part, warns = split_one_tool(
            tool, tool_pairs,
            ratio=args.ratio,
            min_eval=args.min_eval,
            max_eval=args.max_eval,
            rng=rng,
        )
        warnings_all.extend(warns)
        train_all.extend(train_part)
        eval_all.extend(eval_part)

        n_total = len(tool_pairs)
        eval_sources = len({p.get("source", "unknown") for p in eval_part})
        table_rows.append((tool, n_total, len(train_part), len(eval_part), eval_sources))
        if n_total >= 20:
            tools_ge_20.append(tool)

    # ---- Sanity-check assertions -----------------------------------------
    train_keys = {_key(p) for p in train_all}
    eval_keys  = {_key(p) for p in eval_all}
    overlap = train_keys & eval_keys
    assert not overlap, (
        f"[FAIL] {len(overlap)} pair(s) leaked into both splits, "
        f"e.g. {next(iter(overlap))!r}"
    )

    eval_tools = {(p.get("target") or {}).get("name", "") for p in eval_all}
    missing = [t for t in tools_ge_20 if t not in eval_tools]
    assert not missing, (
        f"[FAIL] tools with >=20 pairs but zero eval examples: {missing}"
    )

    assert len(train_all) + len(eval_all) == n_unique, (
        f"[FAIL] train({len(train_all)}) + eval({len(eval_all)}) != "
        f"unique({n_unique})"
    )

    # ---- Write outputs ---------------------------------------------------
    write_jsonl(args.out_train, train_all)
    write_jsonl(args.out_eval,  eval_all)

    # ---- Emit warnings & table ------------------------------------------
    for w in warnings_all:
        print(w)

    print()
    print(f"  {'tool':30s} {'all':>5s} {'train':>7s} {'eval':>6s} {'eval_sources':>13s}")
    for tool, n_total, n_tr, n_ev, n_src in sorted(table_rows, key=lambda r: -r[1]):
        print(f"  {tool:30s} {n_total:5d} {n_tr:7d} {n_ev:6d} {n_src:13d}")
    print()
    print(f"[split] total unique pairs : {n_unique}")
    print(f"[split] train pairs        : {len(train_all)}  -> {args.out_train}")
    print(f"[split] eval pairs         : {len(eval_all)}  -> {args.out_eval}")
    print(f"[split] sanity checks      : 3/3 passed")
    print(f"        - no key in both train and eval")
    print(f"        - every tool with >=20 pairs has >=1 eval pair")
    print(f"        - train + eval = total unique pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
