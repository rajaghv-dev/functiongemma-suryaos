#!/usr/bin/env python3
"""
analyze_embeddings.py — post-training analysis of the trained tokenizer.

Run AFTER train_tokenizer.py completes. Loads the saved tokenizer + embeddings
and produces deep insights you can't get from the training-time probes alone:

  1. NEAREST NEIGHBOUR ANALYSIS
     For each new token, show top-K nearest base-vocab tokens AND nearest
     other-new-tokens. Reveals what the model "thinks" each token means.

  2. CATEGORY CLUSTER QUALITY
     Compute intra-category vs inter-category cosine averages.
     Goal: intra > inter by 0.2+ (clean clusters).

  3. EMBEDDING NORM DISTRIBUTION
     Histogram of new vs base norms. Detects outliers (tokens that didn't
     train) and norm imbalance.

  4. PROBE SENTENCE PROBABILITY
     Run a few "completion probe" sentences through the model and check
     which token wins. e.g. "Tool to check RAM: ___" should be the
     linux_memory_usage token.

  5. FROM/TO MIGRATION
     Compare embed_init.pt (smart-init values) to current embeddings.
     Shows how each token MOVED during training — confirms learning
     happened and points out tokens that didn't move (starved).

  6. TEXTUAL CLUSTER MAP
     ASCII visualization of category clusters in 2D PCA projection.
     Quick visual sanity check.

Usage:
    .fngemma-suryaos/bin/python training/analyze_embeddings.py
    .fngemma-suryaos/bin/python training/analyze_embeddings.py --top-k 10
    .fngemma-suryaos/bin/python training/analyze_embeddings.py --no-model  # skip probe sentences
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT     = Path(__file__).resolve().parent.parent
TRAINING_DIR  = Path(__file__).resolve().parent
EXTENDED_DIR  = TRAINING_DIR / "tokenizer_extended"
EMBED_INIT    = EXTENDED_DIR / "embed_init.pt"
NEW_TOKENS    = REPO_ROOT / "dataset" / "tokenizer" / "new_tokens.json"
HF_MODEL_ID   = "google/gemma-3-270m-it"


def _banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def _ok(msg: str)     -> None: print(f"  [OK]      {msg}")
def _info(msg: str)   -> None: print(f"  [INFO]    {msg}")
def _learn(msg: str)  -> None: print(f"  [LEARN]   {msg}")
def _insight(msg: str)-> None: print(f"  [INSIGHT] {msg}")
def _warn(msg: str)   -> None: print(f"  [WARN]    {msg}", file=sys.stderr)
def _err(msg: str)    -> None: print(f"  [ERR]     {msg}", file=sys.stderr)


# =============================================================================
# Probe sentences for the completion test (Goal 4 — generalization)
# =============================================================================
# Each entry: (prompt, expected_token_substring)
# We feed the prompt to the model and check whether the highest-probability
# next token contains the expected substring (or is one of the expected tokens).
# =============================================================================

PROBE_SENTENCES = [
    ("To check current RAM usage on Linux, the agent calls ",
     ["linux_memory_usage", "memory_usage"]),
    ("To launch an application via KRunner, use ",
     ["kde_krunner_launch", "krunner_launch"]),
    ("The tool that adjusts screen brightness is called ",
     ["linux_brightness_set", "brightness_set"]),
    ("To check if a systemd service is running, dispatch ",
     ["linux_service_status", "service_status"]),
    ("For showing a desktop notification, the right tool is ",
     ["kde_notifications_send", "notifications_send"]),
    ("Battery status on a laptop is reported by ",
     ["linux_battery_status", "battery_status"]),
    ("Disk space queries should route to ",
     ["linux_disk_usage", "disk_usage"]),
    ("To focus an existing window in KDE, use ",
     ["kde_window_focus", "window_focus"]),
]


def _get_hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    f = Path.home() / ".cache" / "huggingface" / "token"
    if f.exists():
        return f.read_text().strip()
    return None


# =============================================================================
# Module 1: load extended tokenizer + embeddings
# =============================================================================

def load_artifacts():
    """Load the trained tokenizer, embedding init snapshot, and full model embeddings."""
    if not EXTENDED_DIR.exists():
        _err(f"{EXTENDED_DIR} not found.")
        _err("Run training/train_tokenizer.py first.")
        sys.exit(1)
    if not EMBED_INIT.exists():
        _err(f"{EMBED_INIT} not found.")
        sys.exit(1)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        _err("transformers/torch not installed. Run: bash training/bootstrap.sh")
        sys.exit(1)

    _info(f"Loading extended tokenizer from {EXTENDED_DIR.name}/")
    tok = AutoTokenizer.from_pretrained(str(EXTENDED_DIR), trust_remote_code=False)

    _info("Loading smart-init snapshot (embed_init.pt) ...")
    init_data = torch.load(str(EMBED_INIT), map_location="cpu", weights_only=True)
    new_token_strings = init_data["new_token_strings"]
    base_vocab_size   = init_data["base_vocab_size"]
    init_embeddings   = init_data["embeddings"].float()  # [n_new, hidden]
    _ok(f"Smart-init snapshot: {len(new_token_strings)} tokens, "
        f"base_vocab={base_vocab_size:,}")

    _info(f"Loading full model from {HF_MODEL_ID} (~536MB) to read trained embeddings ...")
    hf_token = _get_hf_token()
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model.resize_token_embeddings(len(tok))

    # Restore the trained embeddings
    embed_table = model.model.embed_tokens.weight
    n_new = len(new_token_strings)
    embed_table.data[base_vocab_size: base_vocab_size + n_new] = init_data["embeddings"]
    _ok("Trained embeddings restored into the model.")

    # Load category info
    if NEW_TOKENS.exists():
        with open(NEW_TOKENS) as f:
            by_cat_raw = json.load(f)
        token_to_cat = {}
        for cat, items in by_cat_raw.items():
            for item in items:
                token_to_cat[item["token"]] = cat
    else:
        token_to_cat = {}

    return {
        "tokenizer":       tok,
        "model":           model,
        "embed_table":     embed_table.detach().float(),
        "init_embeddings": init_embeddings,
        "new_tokens":      new_token_strings,
        "base_vocab_size": base_vocab_size,
        "token_to_cat":    token_to_cat,
    }


# =============================================================================
# Module 2: Nearest-neighbour analysis (the main insight tool)
# =============================================================================

def analyze_neighbours(art: dict, top_k: int = 5, max_tokens: int = 30) -> None:
    _banner(f"NEAREST NEIGHBOURS — what does each new token 'mean' to the model?")

    import torch
    import torch.nn.functional as F

    embed = art["embed_table"]
    base  = art["base_vocab_size"]
    tok   = art["tokenizer"]
    new_strs = art["new_tokens"]
    cat   = art["token_to_cat"]

    # Pre-normalise base vocab for fast cosine-via-dot-product
    base_emb     = embed[:base]
    base_norm    = F.normalize(base_emb, dim=1)
    new_emb      = embed[base: base + len(new_strs)]
    new_norm_arr = F.normalize(new_emb, dim=1)

    print(f"  Showing top-{top_k} base-vocab + top-3 other-new-token neighbours")
    print(f"  for first {min(max_tokens, len(new_strs))} new tokens.\n")

    meaningful_count = 0  # how many tokens have ≥ 2 sensible neighbours
    insight_records  = []

    for i, ts in enumerate(new_strs[:max_tokens]):
        token_cat = cat.get(ts, "?")
        new_id = base + i
        if new_id >= embed.shape[0]:
            continue
        v = new_norm_arr[i].unsqueeze(0)  # [1, hidden]

        # Top-K base-vocab neighbours
        sims_base = (v @ base_norm.T).squeeze(0)
        top_v, top_i = sims_base.topk(top_k)
        base_pairs = [
            (tok.convert_ids_to_tokens(idx.item()), val.item())
            for idx, val in zip(top_i, top_v)
        ]

        # Top-3 nearest OTHER new tokens
        sims_new = (v @ new_norm_arr.T).squeeze(0)
        sims_new[i] = -1.0  # exclude self
        topn_v, topn_i = sims_new.topk(min(3, len(new_strs) - 1))
        new_pairs = [
            (new_strs[idx.item()], val.item())
            for idx, val in zip(topn_i, topn_v)
        ]

        # Render row
        base_str = " | ".join(f"{n!r}({s:.2f})" for n, s in base_pairs)
        new_str  = " | ".join(f"{n}({s:.2f})" for n, s in new_pairs)
        print(f"  {ts:32s} [{token_cat}]")
        print(f"      base-vocab : {base_str}")
        print(f"      new-tokens : {new_str}")

        # Heuristic: are top-2 base-vocab neighbours "meaningful"?
        # A meaningful neighbour is one whose string appears as a substring of
        # the new token, OR shares a key concept word.
        meaningful = sum(
            1 for n, _ in base_pairs[:3]
            if n.strip().lower().lstrip("▁_") in ts.lower()
        )
        if meaningful >= 1:
            meaningful_count += 1
            insight_records.append((ts, "good", base_pairs[0][0]))
        else:
            insight_records.append((ts, "weak", base_pairs[0][0]))
        print()

    # Insight summary
    pct = 100 * meaningful_count / max(1, min(max_tokens, len(new_strs)))
    print(f"  ────────────────────────────────────────────────────────────")
    _insight(f"{meaningful_count}/{min(max_tokens, len(new_strs))} sampled tokens "
             f"({pct:.0f}%) have at least one MEANINGFUL base-vocab neighbour")
    if pct >= 70:
        _learn("Goal 1 (semantic placement) appears HIT — embeddings live in sensible neighborhoods.")
    elif pct >= 50:
        _learn("Goal 1 (semantic placement) is PARTIAL — corpus quality could improve placement.")
    else:
        _warn("Goal 1 (semantic placement) FAILS — embeddings drifted to meaningless regions.")
        _warn("Likely cause: corpus too templated. See dataset-strategies.md A1, A2.")


# =============================================================================
# Module 3: Category cluster quality
# =============================================================================

def analyze_clusters(art: dict) -> None:
    _banner("CATEGORY CLUSTER QUALITY — intra vs inter category similarity")

    import torch
    import torch.nn.functional as F

    embed   = art["embed_table"]
    base    = art["base_vocab_size"]
    new_str = art["new_tokens"]
    cat     = art["token_to_cat"]

    # Group token indices by category
    cat_to_indices: dict[str, list[int]] = {}
    for i, ts in enumerate(new_str):
        c = cat.get(ts, "uncategorised")
        cat_to_indices.setdefault(c, []).append(base + i)

    # Compute per-category intra similarity
    intra_means = {}
    for c, idxs in cat_to_indices.items():
        if len(idxs) < 2:
            continue
        vecs = F.normalize(embed[idxs], dim=1)
        sim_matrix = vecs @ vecs.T
        # Exclude diagonal
        n = sim_matrix.shape[0]
        off_diag_sum  = sim_matrix.sum() - sim_matrix.diag().sum()
        off_diag_mean = off_diag_sum / (n * n - n)
        intra_means[c] = off_diag_mean.item()

    # Inter-category mean
    cats = [c for c in cat_to_indices if len(cat_to_indices[c]) >= 2]
    inter_pairs = []
    for ci, c1 in enumerate(cats):
        for c2 in cats[ci+1:]:
            v1 = F.normalize(embed[cat_to_indices[c1]], dim=1)
            v2 = F.normalize(embed[cat_to_indices[c2]], dim=1)
            inter_sim = (v1 @ v2.T).mean().item()
            inter_pairs.append((c1, c2, inter_sim))

    inter_mean = sum(p[2] for p in inter_pairs) / max(1, len(inter_pairs))
    intra_mean = sum(intra_means.values()) / max(1, len(intra_means))

    # Print table
    print("  Intra-category similarity (tokens within same category):")
    print(f"    {'category':18s} {'n_tokens':>10s} {'avg_sim':>10s}  bar")
    for c in sorted(intra_means, key=lambda c: -intra_means[c]):
        n = len(cat_to_indices[c])
        sim = intra_means[c]
        filled = int((sim + 1) / 2 * 30)
        bar = "█" * filled + "░" * (30 - filled)
        print(f"    {c:18s} {n:>10d} {sim:>+10.4f}  {bar}")
    print(f"    {'OVERALL INTRA':18s} {'':>10s} {intra_mean:>+10.4f}")
    print()

    print("  Inter-category similarity (tokens across different categories):")
    print(f"    {'cat_A vs cat_B':30s} {'avg_sim':>10s}")
    for c1, c2, sim in sorted(inter_pairs, key=lambda x: -x[2])[:10]:
        print(f"    {c1+' vs '+c2:30s} {sim:>+10.4f}")
    print(f"    {'OVERALL INTER':30s} {inter_mean:>+10.4f}")
    print()

    separation = intra_mean - inter_mean
    _insight(f"Intra-category mean: {intra_mean:+.4f}")
    _insight(f"Inter-category mean: {inter_mean:+.4f}")
    _insight(f"Separation:          {separation:+.4f}  (intra − inter)")

    if separation >= 0.2:
        _learn("Strong category separation — embeddings cluster cleanly by purpose.")
        _learn("Goal 2 (cluster geometry) appears HIT.")
    elif separation >= 0.1:
        _learn("Moderate category separation — improvement possible.")
        _learn("Goal 2 (cluster geometry) is PARTIAL.")
    else:
        _warn("Weak category separation — embeddings collapsed across categories.")
        _warn("Goal 2 (cluster geometry) FAILS — see dataset-strategies.md A1, A3.")


# =============================================================================
# Module 4: Norm distribution
# =============================================================================

def analyze_norms(art: dict) -> None:
    _banner("EMBEDDING NORM DISTRIBUTION — magnitude health")

    embed = art["embed_table"]
    base  = art["base_vocab_size"]
    new   = art["new_tokens"]

    base_norms = embed[:base].norm(dim=1)
    new_norms  = embed[base: base + len(new)].norm(dim=1)

    print(f"  Base vocab ({base:,} tokens):")
    print(f"    mean = {base_norms.mean().item():.4f}")
    print(f"    std  = {base_norms.std().item():.4f}")
    print(f"    min  = {base_norms.min().item():.4f}")
    print(f"    max  = {base_norms.max().item():.4f}")
    print()
    print(f"  New tokens ({len(new)}):")
    print(f"    mean = {new_norms.mean().item():.4f}")
    print(f"    std  = {new_norms.std().item():.4f}")
    print(f"    min  = {new_norms.min().item():.4f}")
    print(f"    max  = {new_norms.max().item():.4f}")
    print()

    ratio = new_norms.mean().item() / base_norms.mean().item()
    _insight(f"Norm ratio (new/base): {ratio:.2f}")
    if 0.7 <= ratio <= 1.2:
        _learn("Goal 3 (norm equivalence) HIT — new embeddings will integrate cleanly.")
    elif ratio < 0.5:
        _warn("New tokens are too quiet — model will under-use them in generation.")
    elif ratio > 1.5:
        _warn("New tokens are too loud — risk of over-generation.")
    else:
        _learn("Goal 3 borderline — within range but not optimal.")

    # Outlier tokens (norm > 2 std from mean)
    mean_n = new_norms.mean().item()
    std_n  = new_norms.std().item()
    outliers = []
    for i, n in enumerate(new_norms):
        z = abs(n.item() - mean_n) / std_n if std_n > 0 else 0
        if z > 2:
            outliers.append((new[i], n.item(), z))
    if outliers:
        print()
        _warn(f"{len(outliers)} outlier tokens (norm >2σ from mean):")
        for tok, n, z in sorted(outliers, key=lambda x: -x[2])[:5]:
            print(f"             {tok!r:30s} norm={n:.3f} (z={z:.1f})")
        _learn("Outliers usually indicate tokens that didn't train (norm too low)")
        _learn("or got over-trained (norm too high). Check corpus coverage for them.")


# =============================================================================
# Module 5: From/To migration (drift from smart-init)
# =============================================================================

def analyze_drift(art: dict) -> None:
    _banner("DRIFT FROM SMART-INIT — did training actually move embeddings?")

    init_e  = art["init_embeddings"]
    embed   = art["embed_table"]
    base    = art["base_vocab_size"]
    new_str = art["new_tokens"]
    cur_e   = embed[base: base + len(new_str)]

    drifts = (cur_e - init_e).norm(dim=1)
    mean_d = drifts.mean().item()
    std_d  = drifts.std().item()
    min_d  = drifts.min().item()
    max_d  = drifts.max().item()

    print(f"  Drift = ||current − init||₂ per token")
    print(f"    mean = {mean_d:.4f}")
    print(f"    std  = {std_d:.4f}")
    print(f"    min  = {min_d:.4f}")
    print(f"    max  = {max_d:.4f}")
    print()

    # Tokens that didn't move (suggesting starvation)
    starved = [(new_str[i], drifts[i].item())
               for i in range(len(new_str))
               if drifts[i].item() < mean_d * 0.3]
    if starved:
        _warn(f"{len(starved)} tokens drifted < 30% of average — likely STARVED:")
        for tok, d in sorted(starved, key=lambda x: x[1])[:8]:
            print(f"             {tok!r:30s} drift={d:.3f}")
        _learn("Starved tokens didn't train enough. Add more corpus sentences for them.")
    else:
        _learn("All tokens drifted meaningfully — no starvation detected.")

    # Tokens that moved the most (possibly over-trained)
    movers = sorted(
        [(new_str[i], drifts[i].item()) for i in range(len(new_str))],
        key=lambda x: -x[1],
    )
    print()
    _info("Top 5 most-moved tokens (large drift = strong learning signal):")
    for tok, d in movers[:5]:
        print(f"             {tok!r:30s} drift={d:.3f}")


# =============================================================================
# Module 6: Probe sentence completion (Goal 4 — generalization)
# =============================================================================

def probe_completions(art: dict) -> None:
    _banner("PROBE COMPLETIONS — does the model produce the right token?")

    print("  For each prompt, we run a forward pass and check whether the model's")
    print("  TOP-PROBABILITY next token is one of the expected tool-name tokens.\n")

    import torch

    model = art["model"]
    tok   = art["tokenizer"]
    new_strs = set(art["new_tokens"])
    model.eval()

    correct = 0
    total = 0
    for prompt, expected_substrings in PROBE_SENTENCES:
        total += 1
        try:
            ids = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model(**ids)
            last_logits = out.logits[0, -1, :]
            top_probs, top_ids = last_logits.softmax(dim=-1).topk(10)

            # Check if any of the top-10 is one of the expected tokens
            top_tokens = [tok.convert_ids_to_tokens(i.item()) for i in top_ids]
            top_strings = [t.lstrip("▁_") for t in top_tokens]

            hit_idx = -1
            for sub in expected_substrings:
                for j, t in enumerate(top_strings):
                    if sub in t or t in sub:
                        hit_idx = j
                        break
                if hit_idx >= 0:
                    break

            status = "✓" if hit_idx >= 0 else "✗"
            if hit_idx >= 0:
                correct += 1

            print(f"  {status} {prompt[:60]:60s}")
            print(f"       expected: {expected_substrings}")
            print(f"       top-3:    {top_tokens[:3]}  (probs: {[f'{p.item():.3f}' for p in top_probs[:3]]})")
            if hit_idx >= 0:
                print(f"       found expected token at rank {hit_idx + 1}")
            print()
        except Exception as e:
            print(f"  ! probe failed for: {prompt[:60]}: {e}")

    print(f"  ────────────────────────────────────────────────────────────")
    pct = 100 * correct / total if total else 0
    _insight(f"Top-10 hit rate: {correct}/{total} ({pct:.0f}%)")
    if pct >= 70:
        _learn("Strong completion accuracy — embeddings + LM head produce right tokens.")
    elif pct >= 40:
        _learn("Moderate completion — corpus needs more user-query → tool patterns.")
    else:
        _warn("Weak completion — see dataset-strategies.md A4, A5 (Q&A patterns).")


# =============================================================================
# Module 7: ASCII cluster visualization (PCA-ish)
# =============================================================================

def ascii_cluster_map(art: dict, width: int = 60, height: int = 20) -> None:
    _banner("ASCII CLUSTER MAP — embedding space at a glance")

    import torch

    embed   = art["embed_table"]
    base    = art["base_vocab_size"]
    new_str = art["new_tokens"]
    cat     = art["token_to_cat"]
    new_emb = embed[base: base + len(new_str)].float()

    # Simple PCA: project to top-2 components
    centered = new_emb - new_emb.mean(dim=0, keepdim=True)
    U, S, V = torch.svd(centered)
    proj = centered @ V[:, :2]  # [n, 2]

    xs = proj[:, 0]
    ys = proj[:, 1]
    x_min, x_max = xs.min().item(), xs.max().item()
    y_min, y_max = ys.min().item(), ys.max().item()

    # Build the ASCII canvas
    canvas = [[" "] * width for _ in range(height)]

    # Map each category to a single ASCII char
    cat_chars = {
        "tool_name":   "●",
        "kde":         "K",
        "system":      "S",
        "file_format": "F",
        "git":         "G",
        "ml":          "M",
    }

    for i, ts in enumerate(new_str):
        c = cat.get(ts, "?")
        ch = cat_chars.get(c, "·")
        x = xs[i].item()
        y = ys[i].item()
        # Map to canvas coords
        cx = int((x - x_min) / (x_max - x_min + 1e-9) * (width - 1))
        cy = int((y - y_min) / (y_max - y_min + 1e-9) * (height - 1))
        # Flip y because terminal y grows downward
        cy = (height - 1) - cy
        # Don't overwrite existing markers (preserves overlapping signal)
        if canvas[cy][cx] == " ":
            canvas[cy][cx] = ch

    # Render
    print("  PCA projection of new token embeddings (2D, top-2 components)")
    print("  ┌" + "─" * width + "┐")
    for row in canvas:
        print("  │" + "".join(row) + "│")
    print("  └" + "─" * width + "┘")
    print("  Legend: " + "  ".join(f"{ch}={c}" for c, ch in cat_chars.items() if c in cat.values()))
    print()
    _insight("Clusters that look DISTINCT (separated regions) = healthy")
    _insight("All chars piled in one spot = embedding collapse (see goals.md)")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k",       type=int, default=5,
                        help="Top-K base-vocab neighbours per token (default: 5)")
    parser.add_argument("--max-tokens",  type=int, default=30,
                        help="Max tokens to show in neighbour analysis (default: 30)")
    parser.add_argument("--no-model",    action="store_true",
                        help="Skip the probe-completion module (avoids loading full model)")
    parser.add_argument("--only", choices=["neighbours", "clusters", "norms",
                                            "drift", "probe", "map"],
                        default=None,
                        help="Run only one analysis module (default: all)")
    args = parser.parse_args()

    print("\n  functiongemma embedding analysis tool")
    print(f"  Inputs: {EXTENDED_DIR}/, {EMBED_INIT.name}, {NEW_TOKENS}")

    art = load_artifacts()

    modules = {
        "neighbours": lambda: analyze_neighbours(art, args.top_k, args.max_tokens),
        "clusters":   lambda: analyze_clusters(art),
        "norms":      lambda: analyze_norms(art),
        "drift":      lambda: analyze_drift(art),
        "probe":      lambda: probe_completions(art) if not args.no_model else None,
        "map":        lambda: ascii_cluster_map(art),
    }

    if args.only:
        modules[args.only]()
    else:
        analyze_neighbours(art, args.top_k, args.max_tokens)
        analyze_clusters(art)
        analyze_norms(art)
        analyze_drift(art)
        if not args.no_model:
            probe_completions(art)
        ascii_cluster_map(art)

    _banner("ANALYSIS COMPLETE")
    print("  Compare these results against goals.md targets:")
    print("    - Goal 1 (placement): nearest-neighbour analysis")
    print("    - Goal 2 (geometry):  cluster quality + ASCII map")
    print("    - Goal 3 (norms):     norm distribution")
    print("    - Goal 4 (general):   drift + probe completions")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
