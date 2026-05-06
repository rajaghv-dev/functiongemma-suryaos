#!/usr/bin/env python3
"""
boost_arg_variety.py — lift argument-value entropy on under-covered slots by
amplifying REAL seed phrasings with conservative, single-rule transforms.

Why this exists
---------------
The validator (`tests/test_dispatch_pairs.py`) reports per-required-arg value
uniqueness (distinct values / total pairs). A recent run flagged:

    linux_brightness_set       direction=2/12, step=3/12   <-- low arg variety
    linux_volume_set           direction=2/27, step=5/27   <-- low arg variety
    linux_service_status       name=64/200 (32% unique)
    kde_dialog_confirm         prompt=N/13 (very thin)

The fix is NOT "make up more service names" or "fabricate more apps". The
fix is to take EVERY real seed and pivot it on the one knob that's both a
free parameter (e.g. step=10 vs 25 vs 50) and known to be present in the
seed itself. Each emitted pair stays traceable to a single real seed.

Rules applied (one per emitted pair):

  1. linux_brightness_set / linux_volume_set, "step" param:
     Substitute the numeric value in any seed phrasing that contains
     "by N percent" / "N%" / "to N" with each of [5,10,15,20,25,50,75]
     (skipping the seed's original value). Re-extract args via
     populate_arguments.populate(force=True) so direction stays consistent.

  2. kde_dialog_confirm, "prompt" param:
     Vary the user phrasing (verb / formality) ONLY. The synthesized
     `target.arguments.prompt` stays identical to the seed's prompt; what
     changes is how the user TALKS about asking. Hand-coded SYNONYMS
     (Show -> Ask / Display, etc.) restricted to verbs that already appear
     in the real kdialog seeds.

  3. kde_notifications_send, "title"/"message" params:
     Same rule as #2 — vary the user-side preamble verb/phrase
     ("send" -> "push" -> "show" -> "display"), preserve the synthesized
     title/message verbatim. Cap 4 variants/seed.

Conservative invariants
-----------------------
- Only substitutes numeric values inside a step-percent regex (never picks
  random digits embedded in PCI bus IDs or service-name versions).
- Never invents new entities (no new service names, no new app names).
- linux_service_status: validates entropy but emits nothing (no fabrication
  of service names per user mandate).
- kde_krunner_launch: explicitly skipped (already 68% unique).
- If a tool's slot is already at >= 0.50 entropy, emit nothing.

Output
------
    dataset/real_sources/arg_variety_pairs.jsonl

The `training/build_real_dataset.py` orchestrator picks this up automatically
on its next run, applies populate(force=True) and (tool, lower(query))
dedupe, then a per-tool cap of 200 pairs.

Usage
-----
    python3 training/boost_arg_variety.py
    python3 training/boost_arg_variety.py --in-dir dataset/real_sources \
        --out dataset/real_sources/arg_variety_pairs.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from populate_arguments import populate  # noqa: E402

REPO_ROOT        = HERE.parent
REAL_SOURCES_DIR = REPO_ROOT / "dataset" / "real_sources"
DEFAULT_OUT      = REAL_SOURCES_DIR / "arg_variety_pairs.jsonl"
DEFAULT_PAIRS    = REPO_ROOT / "dataset" / "dispatch_pairs_v4.jsonl"

# These are emitted-by-this-pipeline files; reading them as seeds would
# double-amplify (paraphrase of paraphrase, variety-of-variety, etc.).
EXCLUDE_SEED_FILES = {
    "paraphrase_pairs",
    "hard_negatives_pairs",
    "arg_variety_pairs",  # never re-read our own output
}

# Required args per tool (from tools/tool_schemas.json).
REQUIRED_ARGS: dict[str, list[str]] = {
    "linux_brightness_set":   ["direction", "step"],
    "linux_volume_set":       ["direction", "step"],
    "linux_service_status":   ["name"],
    "kde_krunner_launch":     ["app"],
    "kde_window_focus":       ["title"],
    "kde_notifications_send": ["title", "message"],
    "kde_dialog_confirm":     ["prompt"],
}

# Entropy threshold below which a slot is considered under-covered.
ENTROPY_FLOOR        = 0.40
ENTROPY_SKIP         = 0.50  # >= this -> emit zero pairs for that slot
MIN_PAIRS_TO_LEARN   = 5     # below this, the slot is too thin to bother

# Step values to substitute (covers common percent steps users actually say).
STEP_TARGETS = [5, 10, 15, 20, 25, 50, 75]
STEP_CAP     = 8       # max substitutions per seed
DIALOG_CAP   = 5       # kde_dialog_confirm verb-swap variants per seed
NOTIF_CAP    = 4       # kde_notifications_send verb-swap variants per seed

# "by 20 percent" / "by 20%" / "to 50" / "50 percent". The number is
# captured in group(1). We deliberately require the percent-cue context
# (by/to/at/N% suffix) so we don't substitute digits inside e.g. PCI bus
# IDs ("...0000_00_1f.3...") or service-name version numbers.
STEP_PCT_RE = re.compile(
    r"\b((?:by|to|at)\s+)(\d{1,3})(\s*(?:%|percent|pct)?\b)",
    re.IGNORECASE,
)
STEP_PCT_RE_TRAILING = re.compile(
    r"\b(\d{1,3})(\s*(?:%|percent|pct))\b",
    re.IGNORECASE,
)

# kde_dialog_confirm: real KDE-help seeds use these verbs literally.
# Each entry maps a user-facing verb that appears in REAL seeds to a list of
# alternates that ALSO occur in real seeds (verified against
# kde_help_pairs.jsonl).
DIALOG_VERB_SYNONYMS: list[tuple[re.Pattern, list[str]]] = [
    # "show me a question message box ..." / "show a warning dialog ..."
    (re.compile(r"^show\s+me\s+", re.IGNORECASE),
     ["display ", "bring up ", "pop up "]),
    (re.compile(r"^show\s+", re.IGNORECASE),
     ["display ", "bring up ", "pop up "]),
    # "ask a question ..." / "ask the user ..."
    (re.compile(r"^ask\s+", re.IGNORECASE),
     ["prompt ", "query "]),
    # "warn the user ..."
    (re.compile(r"^warn\s+", re.IGNORECASE),
     ["alert ", "caution "]),
]

# kde_notifications_send: verbs verified to appear in real seeds.
NOTIF_VERB_SYNONYMS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"^send\s+", re.IGNORECASE),
     ["push ", "fire ", "post "]),
    (re.compile(r"^show\s+me\s+", re.IGNORECASE),
     ["display ", "bring up "]),
    (re.compile(r"^show\s+", re.IGNORECASE),
     ["display ", "bring up "]),
    (re.compile(r"^create\s+", re.IGNORECASE),
     ["compose ", "assemble "]),
    (re.compile(r"^display\s+", re.IGNORECASE),
     ["show ", "render "]),
    (re.compile(r"^push\s+", re.IGNORECASE),
     ["send ", "fire "]),
]


# ---------------------------------------------------------------------------
# I/O helpers
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


def get_user_query(pair: dict) -> str:
    for m in pair.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


def set_user_query(pair: dict, new_q: str) -> dict:
    p = json.loads(json.dumps(pair))
    for m in p.get("messages", []):
        if m.get("role") == "user":
            m["content"] = new_q
            break
    return p


# ---------------------------------------------------------------------------
# Entropy diagnostics
# ---------------------------------------------------------------------------

def compute_arg_entropy(seeds: list[dict]) -> dict[tuple[str, str], dict]:
    """Return {(tool, arg_key): {distinct, total, ratio}} for required args."""
    by: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"values": set(), "total": 0}
    )
    for p in seeds:
        tool = (p.get("target") or {}).get("name")
        if tool not in REQUIRED_ARGS:
            continue
        args = (p.get("target") or {}).get("arguments") or {}
        for k in REQUIRED_ARGS[tool]:
            v = args.get(k)
            if v in (None, "", []):
                continue
            slot = by[(tool, k)]
            slot["values"].add(v)
            slot["total"] += 1
    out: dict[tuple[str, str], dict] = {}
    for key, slot in by.items():
        out[key] = {
            "distinct": len(slot["values"]),
            "total":    slot["total"],
            "ratio":    len(slot["values"]) / max(1, slot["total"]),
            "values":   sorted(slot["values"], key=lambda x: str(x)),
        }
    return out


# ---------------------------------------------------------------------------
# Step-substitution rule (linux_brightness_set, linux_volume_set)
# ---------------------------------------------------------------------------

def step_substitutions(query: str, original_step: int | None) -> list[tuple[str, int]]:
    """Yield (new_query, new_step) for each substitution that fires.

    The seed must contain a "by N (percent)" / "to N (percent)" / "N%" cue.
    We replace the FIRST cue's numeric value. If no cue fires, returns [].
    """
    out: list[tuple[str, int]] = []

    # Try the leading-preposition form first ("by N", "to N", "at N").
    m = STEP_PCT_RE.search(query)
    if m:
        prefix = m.group(1)
        num    = int(m.group(2))
        suffix = m.group(3)
        for v in STEP_TARGETS:
            if v == num or v == original_step:
                continue
            new_q = (
                query[: m.start()]
                + prefix
                + str(v)
                + suffix
                + query[m.end() :]
            )
            out.append((new_q, v))
        return out

    # Fallback: trailing "N%" / "N percent".
    m = STEP_PCT_RE_TRAILING.search(query)
    if m:
        num    = int(m.group(1))
        suffix = m.group(2)
        for v in STEP_TARGETS:
            if v == num or v == original_step:
                continue
            new_q = query[: m.start()] + str(v) + suffix + query[m.end() :]
            out.append((new_q, v))
    return out


# ---------------------------------------------------------------------------
# Verb-swap rule (kde_dialog_confirm, kde_notifications_send)
# ---------------------------------------------------------------------------

def verb_swaps(query: str,
               table: list[tuple[re.Pattern, list[str]]],
               cap: int) -> list[tuple[str, str]]:
    """Yield up to `cap` (new_query, rule_name) pairs by swapping the leading
    verb with one of its known-real synonyms. Each rule fires at most once
    per emitted pair; we never chain two swaps."""
    seen: set[str] = {query.strip().lower()}
    out: list[tuple[str, str]] = []
    for pat, repls in table:
        m = pat.match(query)
        if not m:
            continue
        for repl in repls:
            new_q = pat.sub(repl, query, count=1).strip()
            # Capitalisation: keep the original first letter case.
            if query and query[0].isupper() and new_q:
                new_q = new_q[0].upper() + new_q[1:]
            key = new_q.strip().lower()
            if key in seen or not key:
                continue
            seen.add(key)
            out.append((new_q, f"verb[{pat.pattern}->{repl.strip()}]"))
            if len(out) >= cap:
                return out
    return out


# ---------------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------------

def build_step_pair(seed: dict, new_q: str, new_step: int,
                    original_step: int | None) -> dict | None:
    """Build a step-substitution pair. Re-extracts args from the new query
    via populate(force=True) so direction stays consistent. Returns None if
    the resulting args don't match expectations (e.g. populate failed to
    extract the substituted step)."""
    out = set_user_query(seed, new_q)
    out = populate(out, force=True)

    tgt = out.get("target", {}) or {}
    args = tgt.get("arguments") or {}
    # Direction must be present for these tools.
    if not args.get("direction"):
        return None
    # The re-extraction should have picked up our new_step. If it didn't (rare:
    # the rephrased sentence no longer fits the regex), fall back to forcing
    # the substituted value.
    if args.get("step") != new_step:
        args["step"] = new_step
        tgt["arguments"] = args
        out["target"] = tgt

    out["source"] = "arg_variety_boost"
    seed_q = get_user_query(seed)
    prov = out.get("provenance")
    if isinstance(prov, dict):
        prov_out = dict(prov)
    elif prov is None:
        prov_out = {}
    else:
        prov_out = {"note": str(prov)}
    prov_out.update({
        "seed_source":    seed.get("source", "unknown"),
        "seed_query":     seed_q,
        "original_step":  original_step,
        "new_step":       new_step,
        "transformation": ["step_substitution"],
    })
    out["provenance"] = prov_out
    return out


def build_verb_swap_pair(seed: dict, new_q: str, rule: str, tool: str) -> dict | None:
    """Build a verb-swap pair. Keeps target.arguments IDENTICAL to the seed
    (different user phrasing, same synthesized prompt/title/message)."""
    out = set_user_query(seed, new_q)
    # IMPORTANT: do NOT call populate(force=True) here — that would re-derive
    # `prompt` / `title` / `message` from the new wording, which changes the
    # synthesized args. The whole point of this rule is "same target args,
    # new user phrasing". Copy the seed's args verbatim.
    seed_args = (seed.get("target") or {}).get("arguments") or {}
    tgt = out.get("target") or {}
    tgt["arguments"] = json.loads(json.dumps(seed_args))
    out["target"] = tgt

    # Sanity: required args must still be non-empty (they came from the seed).
    for k in REQUIRED_ARGS.get(tool, []):
        v = (tgt["arguments"] or {}).get(k)
        if v in (None, "", []):
            return None

    out["source"] = "arg_variety_boost"
    seed_q = get_user_query(seed)
    prov = out.get("provenance")
    if isinstance(prov, dict):
        prov_out = dict(prov)
    elif prov is None:
        prov_out = {}
    else:
        prov_out = {"note": str(prov)}
    prov_out.update({
        "seed_source":    seed.get("source", "unknown"),
        "seed_query":     seed_q,
        "transformation": [rule],
    })
    out["provenance"] = prov_out
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in-dir", type=Path, default=REAL_SOURCES_DIR,
                    help="Directory of real-source *.jsonl files.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output JSONL path.")
    ap.add_argument("--entropy-from", type=Path, default=DEFAULT_PAIRS,
                    help=("dispatch_pairs JSONL whose per-arg entropy decides "
                          "which slots are under-covered. The slot threshold "
                          "is checked against THIS file (the validator's "
                          "view), but seeds for amplification still come from "
                          "--in-dir minus paraphrase/hard-negative outputs."))
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed (used for deterministic ordering).")
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    # ---- Load real-source seeds (excluding our own outputs / amplifiers).
    if not args.in_dir.exists():
        print(f"  [ERR] in-dir not found: {args.in_dir}", file=sys.stderr)
        return 1

    seeds: list[dict] = []
    src_files = 0
    for f in sorted(args.in_dir.glob("*.jsonl")):
        if f.stem in EXCLUDE_SEED_FILES:
            continue
        if f.resolve() == args.out.resolve():
            continue
        ps = load_jsonl(f)
        for p in ps:
            p.setdefault("source", f.stem)
        seeds.extend(ps)
        src_files += 1
    print(f"  loaded {len(seeds)} real seeds from {src_files} files "
          f"(excluding {sorted(EXCLUDE_SEED_FILES)})")

    # ---- Compute per-(tool, arg) entropy.
    # Two views:
    #   - seed_entropy: from RAW seeds. Tells us if we have enough real seeds
    #                   to learn from for a given slot.
    #   - target_entropy: from the validator's pairs file (dispatch_pairs_v4).
    #                     Tells us which slots are actually under-covered in
    #                     the dataset the validator scores.
    seed_entropy = compute_arg_entropy(seeds)
    # Read the validator's pairs file but EXCLUDE pairs we previously emitted
    # (source == "arg_variety_boost"). Without this exclusion, our own output
    # would mask the under-coverage signal and cause a thrashing oscillation
    # between runs (boost → orchestrator → entropy goes up → boost emits less
    # → orchestrator → entropy drops → boost emits more → ...).
    raw_target = load_jsonl(args.entropy_from)
    target_pairs = [
        p for p in raw_target if p.get("source") != "arg_variety_boost"
    ]
    target_entropy = compute_arg_entropy(target_pairs) if target_pairs else seed_entropy

    if target_pairs:
        excluded = len(raw_target) - len(target_pairs)
        print()
        print(f"  loaded {len(target_pairs)} pairs from {args.entropy_from.name} "
              f"(validator's view, minus {excluded} prior arg_variety_boost emissions).")

    print()
    print("  Per-(tool, required-arg) entropy on TARGET (validator) pairs:")
    for (tool, arg), info in sorted(target_entropy.items()):
        flag = ""
        if info["total"] < MIN_PAIRS_TO_LEARN:
            flag = " too few pairs"
        elif info["ratio"] >= ENTROPY_SKIP:
            flag = " ok"
        elif info["ratio"] < ENTROPY_FLOOR:
            flag = " UNDER-COVERED"
        print(f"    {tool:28s} {arg:10s} "
              f"distinct={info['distinct']:4d} total={info['total']:4d} "
              f"ratio={info['ratio']:.3f}{flag}")

    # ---- Determine which slots get boosted.
    # Decision per user spec:
    #
    #   linux_brightness_set.step / linux_volume_set.step
    #     Boost iff the validator's view shows step uniqueness < 0.40.
    #     Step substitution operates on real seeds that contain a "by N
    #     percent" phrase; the step itself need not be populated as an arg
    #     in the seed (populate() will pick it up after substitution).
    #
    #   kde_dialog_confirm.prompt / kde_notifications_send (title,message)
    #     Always boost when the pool is THIN (< 200 pairs in the validator's
    #     view) — these tools' problem is too few pairs total, not low
    #     value entropy. The verb-swap rule emits more user phrasings while
    #     keeping target.arguments verbatim, so it can't damage entropy.
    #     If the pool is already healthy (>= 200), emit 0.
    #
    #   linux_service_status.name      -> emit 0 (no fabrication of names).
    #   kde_krunner_launch.app         -> emit 0 (already healthy).
    POOL_THIN = 200  # validator's per-tool cap; staying below means thin

    def step_under(tool: str) -> bool:
        info = target_entropy.get((tool, "step"))
        if not info:
            return True  # no step values present at all -> boost
        return info["ratio"] < ENTROPY_FLOOR

    def pool_thin(tool: str) -> bool:
        # Use any required arg's `total` as a proxy for pair count.
        total = 0
        for k in REQUIRED_ARGS.get(tool, []):
            info = target_entropy.get((tool, k))
            if info:
                total = max(total, info["total"])
        return total < POOL_THIN

    boost_brightness_step = step_under("linux_brightness_set")
    boost_volume_step     = step_under("linux_volume_set")
    boost_dialog_prompt   = pool_thin("kde_dialog_confirm")
    boost_notif_title     = pool_thin("kde_notifications_send")

    # linux_service_status: per-mandate, NEVER fabricate service names.
    svc_info = target_entropy.get(("linux_service_status", "name"))
    if svc_info:
        print(f"\n  linux_service_status.name = {svc_info['distinct']}/{svc_info['total']} "
              f"({svc_info['ratio']:.1%}) -> emit 0 (no fabrication of service names).")
    # kde_krunner_launch: explicit user mandate — DO NOT generate new app names.
    krun_info = target_entropy.get(("kde_krunner_launch", "app"))
    if krun_info:
        print(f"  kde_krunner_launch.app = {krun_info['distinct']}/{krun_info['total']} "
              f"({krun_info['ratio']:.1%}) -> emit 0 (per mandate; already healthy).")

    # ---- Pass 1: step substitution for brightness / volume.
    emitted: list[dict] = []
    examples: list[tuple[str, str, str]] = []  # (tool, seed_q, new_q)
    seed_count: Counter = Counter()    # tool -> # seeds we drew from
    emit_count: Counter = Counter()    # tool -> # pairs emitted

    seen_keys: set[tuple[str, str]] = set()
    # Pre-seed with seed (tool, query) tuples to avoid re-emitting the seed.
    for s in seeds:
        tool = (s.get("target") or {}).get("name")
        q = get_user_query(s).strip().lower()
        if tool and q:
            seen_keys.add((tool, q))

    distinct_step_emitted: dict[str, set[int]] = defaultdict(set)

    for s in seeds:
        tool = (s.get("target") or {}).get("name")
        if tool not in ("linux_brightness_set", "linux_volume_set"):
            continue
        if tool == "linux_brightness_set" and not boost_brightness_step:
            continue
        if tool == "linux_volume_set" and not boost_volume_step:
            continue
        seed_q = get_user_query(s)
        if not seed_q.strip():
            continue
        seed_args = (s.get("target") or {}).get("arguments") or {}
        original_step = seed_args.get("step")
        # Skip seeds where we don't have a step in the seed AND the user
        # query has no percent cue (nothing to substitute).
        subs = step_substitutions(seed_q, original_step)
        if not subs:
            continue
        seed_count[tool] += 1
        produced = 0
        for new_q, new_step in subs:
            if produced >= STEP_CAP:
                break
            key = (tool, new_q.strip().lower())
            if key in seen_keys:
                continue
            built = build_step_pair(s, new_q, new_step, original_step)
            if built is None:
                continue
            seen_keys.add(key)
            emitted.append(built)
            distinct_step_emitted[tool].add(new_step)
            emit_count[tool] += 1
            produced += 1
            if len(examples) < 8:
                examples.append((tool, seed_q, new_q))

    # ---- Pass 2: verb-swap for kde_dialog_confirm.
    if boost_dialog_prompt:
        for s in seeds:
            tool = (s.get("target") or {}).get("name")
            if tool != "kde_dialog_confirm":
                continue
            seed_q = get_user_query(s)
            if not seed_q.strip():
                continue
            seed_args = (s.get("target") or {}).get("arguments") or {}
            if not seed_args.get("prompt"):
                continue
            swaps = verb_swaps(seed_q, DIALOG_VERB_SYNONYMS, DIALOG_CAP)
            if not swaps:
                continue
            seed_count[tool] += 1
            for new_q, rule in swaps:
                key = (tool, new_q.strip().lower())
                if key in seen_keys:
                    continue
                built = build_verb_swap_pair(s, new_q, rule, tool)
                if built is None:
                    continue
                seen_keys.add(key)
                emitted.append(built)
                emit_count[tool] += 1
                if len(examples) < 12:
                    examples.append((tool, seed_q, new_q))

    # ---- Pass 3: verb-swap for kde_notifications_send.
    if boost_notif_title:
        for s in seeds:
            tool = (s.get("target") or {}).get("name")
            if tool != "kde_notifications_send":
                continue
            seed_q = get_user_query(s)
            if not seed_q.strip():
                continue
            seed_args = (s.get("target") or {}).get("arguments") or {}
            if not seed_args.get("title") or not seed_args.get("message"):
                continue
            swaps = verb_swaps(seed_q, NOTIF_VERB_SYNONYMS, NOTIF_CAP)
            if not swaps:
                continue
            seed_count[tool] += 1
            for new_q, rule in swaps:
                key = (tool, new_q.strip().lower())
                if key in seen_keys:
                    continue
                built = build_verb_swap_pair(s, new_q, rule, tool)
                if built is None:
                    continue
                seen_keys.add(key)
                emitted.append(built)
                emit_count[tool] += 1
                if len(examples) < 16:
                    examples.append((tool, seed_q, new_q))

    # ---- Deterministic ordering for idempotent output.
    emitted.sort(key=lambda p: (
        p["target"]["name"],
        get_user_query(p).lower(),
    ))

    # ---- Write output.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in emitted:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ---- Per-slot summary.
    print()
    print("  boosted slots:")
    if boost_brightness_step:
        print(f"    linux_brightness_set.step      seeds={seed_count.get('linux_brightness_set', 0)}  "
              f"emitted={emit_count.get('linux_brightness_set', 0):4d}  "
              f"distinct_step_values={sorted(distinct_step_emitted.get('linux_brightness_set', set()))}")
    if boost_volume_step:
        print(f"    linux_volume_set.step          seeds={seed_count.get('linux_volume_set', 0)}  "
              f"emitted={emit_count.get('linux_volume_set', 0):4d}  "
              f"distinct_step_values={sorted(distinct_step_emitted.get('linux_volume_set', set()))}")
    if boost_dialog_prompt:
        print(f"    kde_dialog_confirm.prompt      seeds={seed_count.get('kde_dialog_confirm', 0)}  "
              f"emitted={emit_count.get('kde_dialog_confirm', 0):4d}")
    if boost_notif_title:
        print(f"    kde_notifications_send         seeds={seed_count.get('kde_notifications_send', 0)}  "
              f"emitted={emit_count.get('kde_notifications_send', 0):4d}")

    # ---- Example seed -> emitted lines.
    print()
    print("  example (seed -> emitted) lines:")
    for tool, sq, nq in examples[:8]:
        print(f"    [{tool}]")
        print(f"      seed:    {sq}")
        print(f"      emitted: {nq}")

    print()
    print(f"  total emitted: {len(emitted)}")
    print(f"  output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
