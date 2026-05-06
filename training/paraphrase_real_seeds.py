#!/usr/bin/env python3
"""
paraphrase_real_seeds.py — conservative, traceable paraphrasing of real-source
seed pairs to lift starved-tool counts in the dispatch dataset.

Why this exists
---------------
We mandate "minimize synthetic generation". But after iter #4, ~9 of 12 tools
are still well below the 80-pair floor (some at 12-31). Producing pairs from
thin air is forbidden; instead, we expand the REAL seeds we already mined
(`dataset/real_sources/*.jsonl`) by light, *programmatic* rephrasings — each
one a single-rule transformation traceable back to a verbatim seed.

Each emitted pair:
  - keeps the same target tool,
  - keeps the same arguments (re-extracted via `populate_arguments.populate`
    so e.g. step / direction stay consistent with the rephrased query),
  - carries `source = "paraphrase_real_seed"` plus `provenance.seed_query`
    and `provenance.transformation` (list of rule names applied).

Conservative rules ONLY:
  - synonym swap from a hand-curated, per-tool dict
  - tense / pronoun shift
  - politeness wrapper (please / could you / can you)
  - question/statement flip
No new entities, no new arg values, no multi-hop drift. If a seed admits
fewer than --max-per-seed paraphrases, that is fine — we do not pad.

Usage
-----
    python3 training/paraphrase_real_seeds.py
    # emits dataset/real_sources/paraphrase_pairs.jsonl

CLI flags:
    --out PATH               output path
    --max-per-seed N         cap per real seed (default 8)
    --starved-from PATH      compute starved tool list from this dispatch
                             dataset (default: dispatch_pairs_v4.jsonl). If
                             missing, fall back to a hardcoded list.

The script is idempotent: it overwrites --out and recomputes from scratch.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from populate_arguments import populate  # noqa: E402

REPO_ROOT       = HERE.parent
REAL_SOURCES_DIR= REPO_ROOT / "dataset" / "real_sources"
DEFAULT_OUT     = REAL_SOURCES_DIR / "paraphrase_pairs.jsonl"
DEFAULT_PAIRS   = REPO_ROOT / "dataset" / "dispatch_pairs_v4.jsonl"

DEFAULT_FLOOR = 80

STARVED_FALLBACK = [
    "linux_memory_usage", "linux_disk_usage", "linux_metrics_summary",
    "linux_battery_status", "linux_brightness_set", "linux_volume_set",
    "linux_network_status", "kde_notifications_send", "kde_dialog_confirm",
]

# ---------------------------------------------------------------------------
# Hand-curated, per-tool synonym dict.
#
# Each entry maps a *whole-word* token (or short phrase) found in real seeds
# to a list of equivalent rephrasings. Keys must match as standalone tokens
# (case-insensitive), to avoid mid-word substitutions.
#
# Curated by reading the real seed corpora for each starved tool — every
# entry below appears literally in at least one real seed query.
# ---------------------------------------------------------------------------
TOOL_SYNONYMS: dict[str, dict[str, list[str]]] = {
    "linux_memory_usage": {
        "memory":      ["ram"],
        "ram":         ["memory"],
        "free":        ["available"],
        "used":        ["consumed"],
        "show":        ["display", "report"],
        "display":     ["show"],
        "report":      ["show"],
        "how much":    ["what amount of"],
        "check":       ["inspect"],
    },
    "linux_disk_usage": {
        "disk":        ["storage", "drive"],
        "storage":     ["disk", "drive"],
        "drive":       ["disk", "storage"],
        "space":       ["capacity"],
        "free":        ["available"],
        "used":        ["consumed"],
        "show":        ["display", "report"],
        "check":       ["inspect"],
        "report":      ["show"],
        "filesystem":  ["fs"],
    },
    "linux_metrics_summary": {
        "show":        ["display", "report"],
        "display":     ["show"],
        "report":      ["show"],
        "system":      ["machine"],
        "stats":       ["metrics", "statistics"],
        "metrics":     ["stats", "statistics"],
        "snapshot":    ["overview", "summary"],
        "summary":     ["overview", "snapshot"],
        "health":      ["status"],
    },
    "linux_battery_status": {
        "battery":     ["batt"],
        "batt":        ["battery"],
        "charge":      ["charging"],
        "percentage":  ["percent", "level"],
        "percent":     ["percentage"],
        "level":       ["percentage"],
        "check":       ["inspect"],
        "show":        ["display", "report"],
        "laptop":      ["computer", "machine"],
        # The real seeds include "battry" (typo) — preserve typo style.
        "battry":      ["battery"],
    },
    "linux_brightness_set": {
        "screen":      ["display", "monitor"],
        "display":     ["screen", "monitor"],
        "monitor":     ["screen", "display"],
        "brightness":  ["backlight"],
        "backlight":   ["brightness"],
        "brighten":    ["raise", "increase"],
        "dim":         ["lower", "reduce"],
        "lower":       ["reduce", "decrease"],
        "raise":       ["increase", "boost"],
        "increase":    ["raise"],
        "decrease":    ["lower", "reduce"],
        "make":        ["turn"],
    },
    "linux_volume_set": {
        "volume":      ["audio", "sound"],
        "audio":       ["volume", "sound"],
        "sound":       ["volume", "audio"],
        "louder":      ["higher"],
        "quieter":     ["softer", "lower"],
        "raise":       ["increase", "boost"],
        "lower":       ["reduce", "decrease"],
        "increase":    ["raise"],
        "decrease":    ["lower", "reduce"],
        "turn":        ["bump"],
        "bump":        ["turn"],
        "make":        ["turn"],
    },
    "linux_network_status": {
        "network":     ["internet", "connection"],
        "internet":    ["network", "connection"],
        "connection":  ["network"],
        "wifi":        ["wireless", "wi-fi"],
        "wireless":    ["wifi"],
        "wi-fi":       ["wifi"],
        "online":      ["connected"],
        "connected":   ["online"],
        "active":      ["up", "running"],
        "up":          ["active"],
        "show":        ["display", "report"],
        "check":       ["inspect"],
    },
    "kde_notifications_send": {
        "notification":["alert", "popup"],
        "alert":       ["notification", "popup"],
        "popup":       ["notification"],
        "send":        ["push", "show"],
        "push":        ["send"],
        "show":        ["display"],
        "display":     ["show"],
        "remind":      ["notify"],
        "notify":      ["alert"],
        "desktop":     ["plasma"],
        "tray":        ["notification area"],
    },
    "kde_dialog_confirm": {
        "ask":         ["prompt"],
        "prompt":      ["ask"],
        "confirm":     ["verify"],
        "verify":      ["confirm"],
        "show":        ["display"],
        "display":     ["show"],
        "warn":        ["alert"],
        "dialog":      ["popup", "box"],
        "popup":       ["dialog"],
        "question":    ["query"],
        "query":       ["question"],
        "warning":     ["caution"],
    },
}

# ---------------------------------------------------------------------------
# Generic rule helpers
# ---------------------------------------------------------------------------

POLITENESS_PREFIX = ["please ", "could you ", "can you "]
POLITENESS_SUFFIX = [", please", " please"]

# Pronoun / tense shifts that don't change semantics for status queries.
PRONOUN_SHIFTS = [
    (re.compile(r"\bshow me\b", re.IGNORECASE), "show"),
    (re.compile(r"\btell me\b", re.IGNORECASE), "report"),
    (re.compile(r"\bwhat's\b",  re.IGNORECASE), "what is"),
    (re.compile(r"\bhow's\b",   re.IGNORECASE), "how is"),
    (re.compile(r"\bhow is\b",  re.IGNORECASE), "how's"),
    (re.compile(r"\bwhat is\b", re.IGNORECASE), "what's"),
    (re.compile(r"\bare\b",     re.IGNORECASE), "is"),
    (re.compile(r"\bi am\b",    re.IGNORECASE), "i'm"),
    (re.compile(r"\bi'm\b",     re.IGNORECASE), "i am"),
]

# Patterns that map a question to a statement and vice-versa, preserving
# meaning and never inventing new content.
Q_TO_STATEMENT = [
    # "what's the battery percentage" -> "battery percentage"
    (re.compile(r"^what'?s\s+the\s+(.+?)\??\s*$", re.IGNORECASE), r"\1"),
    (re.compile(r"^what'?s\s+(my|the)\s+(.+?)\??\s*$", re.IGNORECASE), r"\2"),
    (re.compile(r"^how'?s\s+(my|the)\s+(.+?)\??\s*$", re.IGNORECASE), r"\2 status"),
]
S_TO_QUESTION = [
    # "battery percentage" -> "what's the battery percentage"
    # Only fire on short bare noun-phrases (<= 4 tokens). Skip imperative
    # sentences that happen to end with a noun like "...space usage."
    (re.compile(r"^([a-z][a-z\s\-]{0,40}?(?:percent|percentage|level|status))\s*$",
                re.IGNORECASE),
     r"what's the \1"),
]


# ---------------------------------------------------------------------------
# Single-rule paraphrase generators. Each yields (new_query, rule_name).
# The caller chains exactly ONE rule per emitted paraphrase.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b([A-Za-z][A-Za-z\-']*)\b")


def synonym_swaps(q: str, table: dict[str, list[str]]):
    """Yield (new_q, "synonym_swap[X->Y]") for each whole-word synonym
    rewrite that actually changes the string."""
    if not table:
        return
    # Match longest keys first (handle multi-word like "how much").
    keys_sorted = sorted(table.keys(), key=lambda k: -len(k))
    seen: set[str] = set()
    for key in keys_sorted:
        # Word-boundary regex; case-insensitive; only first occurrence to keep
        # the rewrite a single, traceable swap.
        pat = re.compile(rf"(?<![A-Za-z]){re.escape(key)}(?![A-Za-z])",
                         re.IGNORECASE)
        if not pat.search(q):
            continue
        for repl in table[key]:
            new_q = pat.sub(repl, q, count=1)
            if new_q == q or new_q in seen:
                continue
            seen.add(new_q)
            yield new_q, f"synonym[{key}->{repl}]"


def pronoun_tense_shift(q: str):
    """Yield (new_q, rule_name) for each tense / pronoun shift that fires."""
    seen: set[str] = set()
    for pat, repl in PRONOUN_SHIFTS:
        if not pat.search(q):
            continue
        new_q = pat.sub(repl, q, count=1)
        if new_q == q or new_q in seen:
            continue
        seen.add(new_q)
        yield new_q, f"shift[{pat.pattern}->{repl}]"


def politeness_wrapper(q: str):
    """Yield (new_q, rule_name) for each polite prefix/suffix variant."""
    base = q.rstrip(".!? ")
    seen: set[str] = set()
    # Don't double-wrap if the seed already says "please" or "could you".
    low = base.lower()
    for pre in POLITENESS_PREFIX:
        if low.startswith(pre.strip()):
            continue
        new_q = pre + base[0].lower() + base[1:]
        if new_q in seen or new_q.lower() == q.lower():
            continue
        seen.add(new_q)
        yield new_q, f"politeness[prefix={pre.strip()}]"
    for suf in POLITENESS_SUFFIX:
        if "please" in low:
            continue
        new_q = base + suf
        if new_q in seen or new_q.lower() == q.lower():
            continue
        seen.add(new_q)
        yield new_q, f"politeness[suffix={suf.strip()}]"


def question_statement_flip(q: str):
    """Yield (new_q, rule_name) for question<->statement flips."""
    seen: set[str] = set()
    base = q.rstrip(" .")
    for pat, repl in Q_TO_STATEMENT:
        m = pat.match(base)
        if not m:
            continue
        new_q = pat.sub(repl, base).strip()
        if new_q and new_q not in seen and new_q.lower() != q.lower():
            seen.add(new_q)
            yield new_q, "flip[q->statement]"
    for pat, repl in S_TO_QUESTION:
        m = pat.match(base)
        if not m:
            continue
        new_q = pat.sub(repl, base).strip() + "?"
        if new_q and new_q not in seen and new_q.lower() != q.lower():
            seen.add(new_q)
            yield new_q, "flip[statement->q]"


def all_paraphrases(q: str, syn_table: dict[str, list[str]]):
    """Yield ALL single-rule paraphrases for query `q`, deduped on lowercase
    string. Order: synonym -> tense -> politeness -> flip. The caller picks
    the first N."""
    seen: set[str] = {q.strip().lower()}
    for gen in (
        synonym_swaps(q, syn_table),
        pronoun_tense_shift(q),
        politeness_wrapper(q),
        question_statement_flip(q),
    ):
        for new_q, rule in gen:
            key = new_q.strip().lower()
            if key in seen or not key:
                continue
            seen.add(key)
            yield new_q, rule


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
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


def starved_tools_from(pairs_path: Path, floor: int) -> list[str]:
    """Compute the list of starved tools from a dispatch_pairs JSONL file.

    Pairs whose source is `paraphrase_real_seed` are EXCLUDED from the
    starvation count: otherwise this script becomes non-idempotent —
    re-running after a successful build would shrink the starved set
    because we'd be counting our own previous output as if it were real
    data. We always paraphrase against the real-seed-only baseline.
    """
    if not pairs_path.exists():
        print(f"  [WARN]  no pairs file at {pairs_path} — using hardcoded fallback",
              file=sys.stderr)
        return list(STARVED_FALLBACK)
    counts: Counter = Counter()
    for ln in pairs_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if d.get("source") == "paraphrase_real_seed":
            continue
        name = (d.get("target") or {}).get("name")
        if name:
            counts[name] += 1
    starved = [t for t, n in counts.items() if n < floor]
    # Also include tools that exist in fallback but never appeared in pairs.
    for t in STARVED_FALLBACK:
        if t not in counts and t not in starved:
            starved.append(t)
    return starved


def get_user_query(pair: dict) -> str:
    for m in pair.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


def set_user_query(pair: dict, new_q: str) -> dict:
    """Return a deep copy of pair with the user message replaced."""
    p = json.loads(json.dumps(pair))
    for m in p.get("messages", []):
        if m.get("role") == "user":
            m["content"] = new_q
            break
    return p


def build_paraphrase(seed: dict, new_q: str, rule: str) -> dict:
    """Construct an emitted pair from a real-source seed plus one rule."""
    out = set_user_query(seed, new_q)
    # Re-extract args from the new query so direction/step/etc. stay correct.
    out = populate(out, force=True)

    seed_q = get_user_query(seed)
    seed_source = seed.get("source", "unknown")

    out["source"] = "paraphrase_real_seed"
    prov = out.get("provenance")
    if isinstance(prov, dict):
        prov_out = dict(prov)
    elif prov is None:
        prov_out = {}
    else:
        prov_out = {"note": str(prov)}
    prov_out["seed_source"]    = seed_source
    prov_out["seed_query"]     = seed_q
    prov_out["transformation"] = [rule]
    out["provenance"] = prov_out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-per-seed", type=int, default=8)
    ap.add_argument("--starved-from", type=Path, default=DEFAULT_PAIRS,
                    help="dispatch_pairs JSONL used to detect starved tools "
                         "(below the floor of 80).")
    ap.add_argument("--floor", type=int, default=DEFAULT_FLOOR,
                    help="tools with fewer pairs than this in --starved-from "
                         "are paraphrased.")
    args = ap.parse_args()

    starved = starved_tools_from(args.starved_from, args.floor)
    print(f"  starved tools (< {args.floor}): {starved}")

    # Load all real_source files.
    all_seeds: list[dict] = []
    by_src_count: Counter = Counter()
    for f in sorted(REAL_SOURCES_DIR.glob("*.jsonl")):
        # NEVER paraphrase our own output; that would be a feedback loop.
        if f.resolve() == args.out.resolve():
            continue
        ps = load_jsonl(f)
        for p in ps:
            p.setdefault("source", f.stem)
        all_seeds.extend(ps)
        by_src_count[f.stem] += len(ps)
    print(f"  loaded {len(all_seeds)} real seeds from "
          f"{sum(1 for _ in REAL_SOURCES_DIR.glob('*.jsonl'))} files")

    # Filter to seeds whose target tool is starved AND whose user query is
    # non-empty (some mined seeds have empty content; skip those).
    starved_set = set(starved)
    seeds: list[dict] = []
    for p in all_seeds:
        tool = (p.get("target") or {}).get("name")
        if tool not in starved_set:
            continue
        if not get_user_query(p).strip():
            continue
        seeds.append(p)
    seeds_per_tool: Counter = Counter(
        (p.get("target") or {}).get("name") for p in seeds
    )
    print(f"  seeds matching starved tools: {len(seeds)}")
    for t in starved:
        print(f"    {t:28s} seeds={seeds_per_tool.get(t, 0)}")

    # Generate paraphrases.
    emitted: list[dict] = []
    seed_emit_count: dict[int, int] = {}
    seen_keys: set[tuple[str, str]] = set()  # (tool, lower-stripped query)

    # Pre-seed seen_keys with the EXACT seed (tool, query) to avoid emitting
    # duplicates of the seed itself.
    for s in seeds:
        tool = s["target"]["name"]
        q    = get_user_query(s).strip().lower()
        seen_keys.add((tool, q))

    examples: list[tuple[str, str, str]] = []  # (seed_q, new_q, rule)

    for s in seeds:
        tool = s["target"]["name"]
        seed_q = get_user_query(s)
        syn = TOOL_SYNONYMS.get(tool, {})
        produced = 0
        for new_q, rule in all_paraphrases(seed_q, syn):
            if produced >= args.max_per_seed:
                break
            key = (tool, new_q.strip().lower())
            if key in seen_keys:
                continue
            built = build_paraphrase(s, new_q, rule)
            # Drop a paraphrase whose populate() couldn't recover required
            # args from the new wording (e.g. "lower the volume by ten" lost
            # the digit) — we'd just feed the trainer garbage.
            tgt = built.get("target", {})
            args_ok = True
            for req in ("direction",) if tool in ("linux_volume_set",
                                                  "linux_brightness_set") else ():
                if not (tgt.get("arguments") or {}).get(req):
                    args_ok = False
                    break
            if tool == "linux_service_status" and not (tgt.get("arguments") or {}).get("name"):
                args_ok = False
            if tool == "kde_notifications_send" and not (tgt.get("arguments") or {}).get("title"):
                args_ok = False
            if tool == "kde_dialog_confirm" and not (tgt.get("arguments") or {}).get("prompt"):
                args_ok = False
            if not args_ok:
                continue
            seen_keys.add(key)
            emitted.append(built)
            produced += 1
            seed_emit_count[id(s)] = produced
            if len(examples) < 5:
                examples.append((seed_q, new_q, rule))

    # Write output.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in emitted:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Per-tool summary.
    by_tool: Counter = Counter(p["target"]["name"] for p in emitted)
    by_seed_src: Counter = Counter(
        (p.get("provenance") or {}).get("seed_source", "unknown")
        for p in emitted
    )

    print()
    print(f"  per-tool emitted (cap {args.max_per_seed}/seed):")
    for t in starved:
        n_seed = seeds_per_tool.get(t, 0)
        n_emit = by_tool.get(t, 0)
        print(f"    {t:28s} seeds={n_seed:3d}  emitted={n_emit:4d}  total={n_seed + n_emit:4d}")

    print()
    print(f"  emitted by seed_source (top 10):")
    for s, n in by_seed_src.most_common(10):
        print(f"    {s:40s} {n}")

    print()
    print(f"  total emitted: {len(emitted)}")
    print(f"  output: {args.out}")

    print()
    print("  example (seed -> paraphrase) lines:")
    for sq, nq, rule in examples:
        print(f"    [{rule}]")
        print(f"      seed:       {sq}")
        print(f"      paraphrase: {nq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
