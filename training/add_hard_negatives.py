#!/usr/bin/env python3
"""
add_hard_negatives.py — augment dispatch_pairs_v4.jsonl with `negatives` per
pair, and emit explicit cross-domain / sibling contrast pairs.

Why this exists
---------------
Iter #4 regressed cross-domain cosine 0.62 -> 0.70 (got worse). The model has
no signal that pushes different tool domains apart, because every training
pair currently lists only the correct tool in `tools` and most pairs have no
`negatives` field.

This script:
  1. Reads dataset/dispatch_pairs_v4.jsonl and adds a `negatives: [mcp_name,...]`
     list to every pair, derived from THREE sources of evidence:
       (a) cross-domain  — every linux_* pair gets one kde_* negative; every
           kde_* pair gets one linux_* negative. Deterministic via hash(query).
           This is the headline fix for cosine collapse.
       (b) within-domain siblings — hand-coded SIBLING_GROUPS pair semantically
           adjacent tools (status-readers, setters, kde fire-and-forget vs
           blocking, etc.).
       (c) preserved man-page negatives — mine_man_pages.py already emitted
           SEE-ALSO-derived negatives on 35 pairs. We never delete those; if a
           pair already has a `negatives` list, we union it with (a)+(b).

  2. Emits NEW explicit hard-contrast pairs to
     dataset/real_sources/hard_negatives_pairs.jsonl. The query for each new
     pair is verbatim text from tools/tool_schemas.json `description` (no
     invented sentences). Cross-domain pairs are capped at ~30, KDE-sibling
     pairs at ~10.

Outputs
-------
  dataset/dispatch_pairs_v4_with_negatives.jsonl  — every existing pair, now
                                                    with a negatives field.
  dataset/real_sources/hard_negatives_pairs.jsonl — new contrast pairs that
                                                    build_real_dataset.py will
                                                    fold into the next v4.

Idempotent: same input + same seed -> bit-for-bit identical output.

Run
---
    python3 training/add_hard_negatives.py
    python3 training/build_real_dataset.py        # fold new pairs into v4
    python3 tests/test_dispatch_pairs.py --pairs dataset/dispatch_pairs_v4.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parent.parent
SCHEMAS_PATH   = REPO_ROOT / "tools" / "tool_schemas.json"
DEFAULT_IN     = REPO_ROOT / "dataset" / "dispatch_pairs_v4.jsonl"
DEFAULT_OUT_PAIRS     = REPO_ROOT / "dataset" / "real_sources" / "hard_negatives_pairs.jsonl"
DEFAULT_OUT_AUGMENTED = REPO_ROOT / "dataset" / "dispatch_pairs_v4_with_negatives.jsonl"

LINUX_TOOLS = [
    "linux_memory_usage", "linux_disk_usage", "linux_metrics_summary",
    "linux_battery_status", "linux_brightness_set", "linux_volume_set",
    "linux_network_status", "linux_service_status",
]
KDE_TOOLS = [
    "kde_krunner_launch", "kde_window_focus",
    "kde_notifications_send", "kde_dialog_confirm",
]
ALL_TOOLS = LINUX_TOOLS + KDE_TOOLS

# Hand-coded sibling groups: each tool maps to a list of within-domain peers
# whose semantics overlap enough that the model would mis-rank them without
# explicit negative signal.
SIBLING_GROUPS: dict[str, list[str]] = {
    # linux status-readers — all answer "what's the system doing?"-shaped queries
    "linux_memory_usage":    ["linux_metrics_summary", "linux_disk_usage", "linux_battery_status"],
    "linux_disk_usage":      ["linux_metrics_summary", "linux_memory_usage"],
    "linux_metrics_summary": ["linux_memory_usage",   "linux_disk_usage", "linux_battery_status", "linux_network_status"],
    "linux_battery_status":  ["linux_metrics_summary", "linux_memory_usage"],
    "linux_network_status":  ["linux_metrics_summary", "linux_service_status"],
    "linux_service_status":  ["linux_network_status",  "linux_metrics_summary"],
    # linux setters — both "adjust X up/down by N percent"
    "linux_brightness_set":  ["linux_volume_set"],
    "linux_volume_set":      ["linux_brightness_set"],
    # kde — krunner.launch (start a new app) vs window.focus (raise existing)
    "kde_krunner_launch":    ["kde_window_focus"],
    "kde_window_focus":      ["kde_krunner_launch"],
    # kde — notifications.send (fire-and-forget) vs dialog.confirm (blocking yes/no)
    "kde_notifications_send":["kde_dialog_confirm"],
    "kde_dialog_confirm":    ["kde_notifications_send"],
}


# ---------------------------------------------------------------------------

def _info(s: str)  -> None: print(f"  [INFO]    {s}")
def _ok(s: str)    -> None: print(f"  [OK]      {s}")
def _warn(s: str)  -> None: print(f"  [WARN]    {s}", file=sys.stderr)
def _banner(s: str)-> None: print("\n" + "=" * 78 + f"\n  {s}\n" + "=" * 78)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(json.loads(ln))
    return out


def load_schemas() -> dict[str, dict]:
    """Return a dict keyed by mcp_name with each tool's metadata.

    Each entry has at minimum: name, mcp_name, description, mcp_schema."""
    raw = json.loads(SCHEMAS_PATH.read_text())
    return {entry["mcp_name"]: entry for entry in raw.values()}


def domain_of(mcp_name: str) -> str:
    if mcp_name.startswith("linux_"): return "linux"
    if mcp_name.startswith("kde_"):   return "kde"
    return "unknown"


def stable_hash(s: str) -> int:
    """Deterministic, stable hash that does not depend on PYTHONHASHSEED."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def user_query(pair: dict) -> str:
    for m in pair.get("messages", []):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Negative selection
# ---------------------------------------------------------------------------

def cross_domain_negative(target: str, query: str) -> str:
    """Pick exactly one cross-domain tool deterministically from the query."""
    pool = KDE_TOOLS if domain_of(target) == "linux" else LINUX_TOOLS
    if not pool:
        return ""
    return pool[stable_hash(query) % len(pool)]


def within_domain_negative(target: str, query: str) -> str:
    """Pick one sibling for `target` from SIBLING_GROUPS, deterministically."""
    sibs = SIBLING_GROUPS.get(target, [])
    if not sibs:
        return ""
    # Use a different mixer than cross_domain so we don't always co-pick the
    # same combination across domains.
    return sibs[stable_hash("sib::" + query) % len(sibs)]


def build_negatives(pair: dict) -> list[str]:
    """Return the union of (existing) + cross-domain + within-domain negs,
    minus the target tool, deduped, in stable order."""
    target = pair.get("target", {}).get("name", "")
    query  = user_query(pair)
    if not target:
        return []

    existing = list(pair.get("negatives") or [])
    cd  = cross_domain_negative(target, query)
    wd  = within_domain_negative(target, query)

    merged: list[str] = []
    seen: set = set()
    for n in (existing + [cd, wd]):
        if not n or n == target or n in seen:
            continue
        if n not in ALL_TOOLS:
            continue
        seen.add(n)
        merged.append(n)
    return merged


# ---------------------------------------------------------------------------
# Emit explicit contrast pairs (queries are real schema text)
# ---------------------------------------------------------------------------

def make_contrast_pair(target: str, distractors: list[str], schemas: dict,
                       *, origin_field: str, tool_a: str, tool_b: str) -> dict:
    """Build a dispatch-pairs-schema dict.

    The query for this pair is the canonical `description` of the target tool
    (from tool_schemas.json), so it is REAL evidence — not invented text. The
    `tools` array contains only the target schema (matching the rest of v4),
    and `negatives` lists the distractors that should NOT win.
    """
    s = schemas[target]
    query = s["description"]
    tool_block = {
        "name":        s["mcp_schema"]["name"],
        "description": s["mcp_schema"]["description"],
        "inputSchema": s["mcp_schema"]["inputSchema"],
    }
    return {
        "messages": [
            {"role": "system", "content": "Call the right tool."},
            {"role": "user",   "content": query},
        ],
        "tools": [tool_block],
        "target": {"name": target, "arguments": _default_args(target)},
        "negatives": [d for d in distractors if d != target],
        "source": "hard_negative_cross_domain" if domain_of(tool_a) != domain_of(tool_b) \
                  else "hard_negative_sibling",
        "provenance": {
            "origin": origin_field,
            "tool_a": tool_a,
            "tool_b": tool_b,
        },
    }


def _default_args(target: str) -> dict:
    """Minimal valid arguments to satisfy is_arg_complete() in build_real_dataset.

    We use the most generic allowable values from the schema so populate_arguments
    won't drop these rows. The query is the schema description itself, so any
    placeholder is fine semantically — the row exists only to teach ranking."""
    # Match REQUIRED_ARGS in build_real_dataset.py.
    if target in ("linux_brightness_set", "linux_volume_set"):
        return {"direction": "up"}
    if target == "linux_service_status":
        return {"name": "ollama"}
    if target == "kde_krunner_launch":
        return {"app": "kate"}
    if target == "kde_window_focus":
        return {"title": "Firefox"}
    if target == "kde_notifications_send":
        return {"title": "Status", "message": "ok"}
    if target == "kde_dialog_confirm":
        return {"prompt": "Proceed?"}
    return {}


def emit_cross_domain_pairs(schemas: dict, *, cap: int, seed: int) -> list[dict]:
    """For every (linux, kde) pair, in deterministic order, emit one contrast
    pair per direction until we hit `cap`. Each pair's query is the real
    description text of the target tool."""
    rnd = random.Random(seed)

    # All directional pairs (linux -> kde) and (kde -> linux).
    candidates: list[tuple[str, str]] = []
    for a in LINUX_TOOLS:
        for b in KDE_TOOLS:
            candidates.append((a, b))   # query=a description, negatives include b
            candidates.append((b, a))   # query=b description, negatives include a

    # Stable shuffle to avoid always picking the same first 30.
    rnd.shuffle(candidates)
    candidates = candidates[:cap]
    candidates.sort()  # final pass: bit-for-bit deterministic output ordering

    out: list[dict] = []
    for tool_a, tool_b in candidates:
        # Add 1-2 extra cross-domain distractors from the OTHER domain so the
        # negatives list is informative (not just one tool).
        other_domain = KDE_TOOLS if domain_of(tool_a) == "linux" else LINUX_TOOLS
        extras = [t for t in other_domain if t != tool_b][:2]
        distractors = [tool_b] + extras
        out.append(make_contrast_pair(
            target=tool_a, distractors=distractors, schemas=schemas,
            origin_field="tool_schemas.description",
            tool_a=tool_a, tool_b=tool_b,
        ))
    return out


def emit_kde_sibling_pairs(schemas: dict, *, cap: int) -> list[dict]:
    """Emit contrast pairs for the two KDE sibling axes:
       - kde_krunner_launch  vs kde_window_focus  (start vs raise)
       - kde_notifications_send vs kde_dialog_confirm (fire-and-forget vs blocking)
    Each direction gets one pair; we go around the four directions until cap."""
    pairs: list[tuple[str, str]] = [
        ("kde_krunner_launch",     "kde_window_focus"),
        ("kde_window_focus",       "kde_krunner_launch"),
        ("kde_notifications_send", "kde_dialog_confirm"),
        ("kde_dialog_confirm",     "kde_notifications_send"),
        # also linux setter siblings — same axis-shape (adjust X by N%)
        ("linux_brightness_set",   "linux_volume_set"),
        ("linux_volume_set",       "linux_brightness_set"),
        # linux status-reader axis (memory vs metrics is the most-confused pair)
        ("linux_memory_usage",     "linux_metrics_summary"),
        ("linux_metrics_summary",  "linux_memory_usage"),
        ("linux_disk_usage",       "linux_metrics_summary"),
        ("linux_network_status",   "linux_metrics_summary"),
    ]
    pairs = pairs[:cap]

    out: list[dict] = []
    for tool_a, tool_b in pairs:
        out.append(make_contrast_pair(
            target=tool_a, distractors=[tool_b], schemas=schemas,
            origin_field="tool_schemas.description",
            tool_a=tool_a, tool_b=tool_b,
        ))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(augmented: list[dict], emitted: list[dict]) -> None:
    _banner("Augmentation summary")

    by_tool = defaultdict(list)
    for p in augmented:
        by_tool[p.get("target", {}).get("name", "?")].append(len(p.get("negatives") or []))

    print(f"  {'tool':28s} {'pairs':>6s} {'avg_neg':>8s} {'min':>4s} {'max':>4s}")
    total_neg = 0
    for t in ALL_TOOLS:
        ns = by_tool.get(t, [])
        if not ns:
            print(f"  {t:28s} {0:6d} {'-':>8s} {'-':>4s} {'-':>4s}")
            continue
        avg = sum(ns) / len(ns)
        total_neg += sum(ns)
        print(f"  {t:28s} {len(ns):6d} {avg:8.2f} {min(ns):4d} {max(ns):4d}")
    print()

    # Histogram of negatives-list-length across all augmented pairs.
    hist = Counter(len(p.get("negatives") or []) for p in augmented)
    print(f"  Negatives-list-length histogram (over {len(augmented)} pairs):")
    for k in sorted(hist):
        bar = "#" * min(60, hist[k] // 10)
        print(f"    len={k:2d}  {hist[k]:5d}  {bar}")
    print()

    # Emitted pairs summary.
    by_src = Counter(p.get("source") for p in emitted)
    print(f"  Emitted contrast pairs: {len(emitted)}")
    for s, n in by_src.most_common():
        print(f"    {s:35s} {n}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in",            dest="in_path", type=Path, default=DEFAULT_IN,
                    help=f"input dispatch_pairs file (default: {DEFAULT_IN.name})")
    ap.add_argument("--out-pairs",     type=Path, default=DEFAULT_OUT_PAIRS,
                    help=f"emit new contrast pairs here (default: {DEFAULT_OUT_PAIRS.relative_to(REPO_ROOT)})")
    ap.add_argument("--out-augmented", type=Path, default=DEFAULT_OUT_AUGMENTED,
                    help=f"emit input + negatives field here (default: {DEFAULT_OUT_AUGMENTED.name})")
    ap.add_argument("--seed",          type=int,  default=42)
    ap.add_argument("--cross-domain-cap", type=int, default=30,
                    help="max number of cross-domain contrast pairs to emit (default 30)")
    ap.add_argument("--sibling-cap",   type=int, default=10,
                    help="max number of within-domain sibling contrast pairs (default 10)")
    args = ap.parse_args()

    _banner("PHASE 1: load inputs")
    schemas = load_schemas()
    pairs   = load_jsonl(args.in_path)
    _ok(f"loaded {len(pairs)} pairs from {args.in_path.name}")
    _ok(f"loaded {len(schemas)} tool schemas from {SCHEMAS_PATH.relative_to(REPO_ROOT)}")

    n_already_neg = sum(1 for p in pairs if p.get("negatives"))
    _info(f"pairs that already had a non-empty negatives list: {n_already_neg}")

    _banner("PHASE 2: augment every pair with negatives")
    augmented: list[dict] = []
    for p in pairs:
        # Don't mutate the input dict: copy first so re-runs are stable.
        p2 = dict(p)
        p2["negatives"] = build_negatives(p)
        augmented.append(p2)
    n_with_neg = sum(1 for p in augmented if p.get("negatives"))
    _ok(f"augmented {n_with_neg}/{len(augmented)} pairs with at least one negative")

    _banner("PHASE 3: emit explicit contrast pairs from tool_schemas.description")
    cd_pairs   = emit_cross_domain_pairs(schemas, cap=args.cross_domain_cap, seed=args.seed)
    sib_pairs  = emit_kde_sibling_pairs(schemas, cap=args.sibling_cap)
    emitted    = cd_pairs + sib_pairs
    _ok(f"emitted {len(cd_pairs)} cross-domain + {len(sib_pairs)} sibling = {len(emitted)} contrast pairs")

    _banner("PHASE 4: write outputs")
    args.out_augmented.parent.mkdir(parents=True, exist_ok=True)
    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with args.out_augmented.open("w") as f:
        for p in augmented:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    _ok(f"wrote {args.out_augmented}")
    with args.out_pairs.open("w") as f:
        for p in emitted:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    _ok(f"wrote {args.out_pairs}")

    print_summary(augmented, emitted)

    # Print 3 example before/after pairs for the report.
    _banner("Sample diffs (first 3 augmented pairs)")
    for orig, aug in zip(pairs[:3], augmented[:3]):
        q = user_query(orig)[:60]
        before = orig.get("negatives", "<missing>")
        after  = aug.get("negatives")
        print(f"  query : {q!r}")
        print(f"  target: {orig.get('target', {}).get('name')}")
        print(f"  before: {before}")
        print(f"  after : {after}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
