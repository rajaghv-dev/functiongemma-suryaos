#!/usr/bin/env python3
"""
train_tokenizer.py — Extend Gemma 3 tokenizer with 319 domain tokens and
warm up their embeddings on domain corpus before LoRA fine-tuning.

Run this BEFORE finetune.py --mode train.

Phases:
  1. add    — load new_tokens.json, add all 319 tokens to tokenizer
  2. init   — initialize each new embedding as the avg of its subword parts
              (far better than random: "linux_memory_usage" starts near "linux"
               + "memory" + "usage" instead of random noise)
  3. train  — warm up new embedding rows on corpus.txt with base vocab frozen
              (only the 319 new rows learn; rest of model is untouched)
  4. save   — write extended tokenizer + embed_init.pt to tokenizer_extended/

finetune.py loads from tokenizer_extended/ when it exists, so LoRA training
starts from meaningful embeddings rather than random init.

Why this matters:
  Without pre-warming, the first 30–50% of LoRA training steps are spent
  pulling new token embeddings out of random initialization. Pre-warming
  means LoRA training focuses purely on routing logic from step 1.

Telemetry written to training/tokenizer_extended/train_log.jsonl:
  - Subword fragmentation before/after extension
  - Token coverage in corpus (flag any token appearing < 5 times)
  - Embedding norm distribution (new vs base vocab)
  - Cosine similarity for semantic probe pairs, tracked per epoch
  - Per-epoch training loss
  - Nearest base-vocab neighbors for each new token (post-training)

Usage:
  python3 training/train_tokenizer.py
  python3 training/train_tokenizer.py --epochs 3 --lr 5e-4
  python3 training/train_tokenizer.py --skip-corpus   # smart init only, no LM training
  python3 training/train_tokenizer.py --neighbors 5   # show top-5 nearest neighbors
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Fast dependency check — fail before doing any work
import importlib.util as _ilu
_MISSING = [pkg for pkg in ("torch", "transformers", "sentencepiece")
            if _ilu.find_spec(pkg) is None]
del _ilu
if _MISSING:
    _venv = Path(__file__).resolve().parent.parent / ".fngemma-suryaos" / "bin" / "python3"
    print(f"[ERR]  Missing packages: {', '.join(_MISSING)}", file=sys.stderr)
    print(f"[ERR]  Run:  bash training/bootstrap.sh", file=sys.stderr)
    print(f"[ERR]  Then: {_venv} training/train_tokenizer.py", file=sys.stderr)
    sys.exit(1)

REPO_ROOT     = Path(__file__).resolve().parent.parent
TRAINING_DIR  = Path(__file__).resolve().parent
TOK_DATA_DIR  = REPO_ROOT / "dataset" / "tokenizer"
NEW_TOKENS_JSON = TOK_DATA_DIR / "new_tokens.json"
CORPUS_FILE   = TOK_DATA_DIR / "corpus.txt"
OUTPUT_DIR    = TRAINING_DIR / "tokenizer_extended"
LOG_FILE      = OUTPUT_DIR / "train_log.jsonl"

# HF model to load embeddings from (same fallback as finetune.py)
HF_MODEL_ID   = "google/gemma-3-270m-it"
MODEL_HF_DIR  = TRAINING_DIR / "model_hf"

# Semantically related pairs to probe throughout training.
# Each tuple: (token_a, token_b, description)
# We track cosine similarity for each pair per epoch to verify embeddings
# converge toward semantically meaningful clusters.
PROBE_PAIRS = [
    # Same tool in two naming forms — should converge to high similarity
    ("linux_memory_usage", "memory_usage",      "same-tool forms"),
    ("linux_disk_usage",   "disk_usage",         "same-tool forms"),
    ("service_status",     "systemctl",          "tool ↔ CLI equiv"),
    ("krunner_launch",     "KRunner",            "tool ↔ KDE component"),
    # Related Linux tools — should end up in same region
    ("linux_memory_usage", "linux_disk_usage",   "sibling linux tools"),
    ("linux_metrics_summary", "linux_memory_usage", "metrics ↔ memory"),
    # Cross-domain should stay dissimilar
    ("linux_memory_usage", "brightness_set",     "cross-domain (expected low)"),
    # Tokens that co-occur in training context
    ("torch",              "transformers",        "co-occurring ML libs"),
    ("merge",              "commit",              "co-occurring git ops"),
    ("GGUF",               "ollama",              "co-occurring serving terms"),
]


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _ok(msg: str)   -> None: print(f"  [OK]   {msg}")
def _warn(msg: str) -> None: print(f"  [WARN] {msg}", file=sys.stderr)
def _err(msg: str)  -> None: print(f"  [ERR]  {msg}", file=sys.stderr)


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_entries: list[dict] = []

def _log(entry: dict) -> None:
    _log_entries.append(entry)


def _flush_log() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        for e in _log_entries:
            f.write(json.dumps(e) + "\n")


# ===========================================================================
# Phase 1: Add tokens
# ===========================================================================

def phase_add_tokens() -> tuple[list[str], "AutoTokenizer"]:
    """Load tokenizer, add 319 domain tokens, return (new_token_list, tokenizer)."""
    _banner("PHASE 1: Add domain tokens to tokenizer")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        _err("transformers not installed.")
        _err("Run:  bash training/bootstrap.sh")
        _err("  or: .fngemma-suryaos/bin/pip install -r training/requirements.txt")
        sys.exit(1)

    if not NEW_TOKENS_JSON.exists():
        _err(f"{NEW_TOKENS_JSON} not found. Run build_tokenizer_dataset.py first.")
        sys.exit(1)

    # Load new_tokens.json
    with open(NEW_TOKENS_JSON) as f:
        by_category: dict[str, list[dict]] = json.load(f)

    all_tokens = [entry["token"] for entries in by_category.values() for entry in entries]
    all_tokens = list(dict.fromkeys(all_tokens))  # deduplicate preserving order
    _ok(f"Loaded {len(all_tokens)} unique domain tokens from new_tokens.json")
    for cat, entries in sorted(by_category.items()):
        print(f"    {cat:20s} {len(entries):3d} tokens")

    # Load base tokenizer
    model_path = str(MODEL_HF_DIR) if (MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir())) else HF_MODEL_ID
    _ok(f"Loading tokenizer from {model_path} …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    except Exception as e:
        _err(f"Tokenizer load failed: {e}")
        sys.exit(1)

    base_vocab_size = len(tokenizer)
    _ok(f"Base vocabulary size: {base_vocab_size:,}")

    # Fragmentation probe BEFORE adding tokens
    frag_before = _measure_fragmentation(tokenizer, all_tokens)
    _ok(f"Fragmentation before extension: avg {frag_before['mean']:.1f} subwords/token "
        f"(max {frag_before['max']:.0f}, {frag_before['already_single']}/{len(all_tokens)} already single-token)")
    _log({"phase": "add_tokens", "event": "fragmentation_before", **frag_before})

    # Add tokens (skip any already in vocab)
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in all_tokens if t not in existing]
    if new_tokens:
        n_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
        _ok(f"Added {n_added} new tokens (skipped {len(all_tokens) - n_added} already in vocab)")
    else:
        _ok("All domain tokens already in vocabulary — none added")

    new_vocab_size = len(tokenizer)
    _ok(f"Extended vocabulary size: {new_vocab_size:,} (+{new_vocab_size - base_vocab_size})")

    # Fragmentation probe AFTER adding tokens
    frag_after = _measure_fragmentation(tokenizer, all_tokens)
    _ok(f"Fragmentation after extension:  avg {frag_after['mean']:.1f} subwords/token "
        f"(max {frag_after['max']:.0f}, {frag_after['already_single']}/{len(all_tokens)} now single-token)")
    _log({"phase": "add_tokens", "event": "fragmentation_after", **frag_after})

    # Coverage check: how many times does each token appear in corpus?
    _report_coverage(all_tokens)

    _log({
        "phase": "add_tokens",
        "base_vocab_size": base_vocab_size,
        "new_vocab_size": new_vocab_size,
        "n_new_tokens": n_added if new_tokens else 0,
        "categories": {k: len(v) for k, v in by_category.items()},
    })

    return new_tokens if new_tokens else [], tokenizer


def _measure_fragmentation(tokenizer, token_strings: list[str]) -> dict:
    """Count how many subword pieces each token string splits into."""
    counts = []
    already_single = 0
    worst = []

    for t in token_strings:
        ids = tokenizer.encode(t, add_special_tokens=False)
        n = len(ids)
        counts.append(n)
        if n == 1:
            already_single += 1
        if n > 5:
            worst.append((t, n))

    worst.sort(key=lambda x: -x[1])
    return {
        "mean": sum(counts) / len(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "already_single": already_single,
        "total": len(counts),
        "worst_10": worst[:10],
    }


def _report_coverage(token_list: list[str], min_occur: int = 5) -> None:
    """Check each token appears ≥ min_occur times in corpus.txt."""
    if not CORPUS_FILE.exists():
        _warn(f"Corpus not found at {CORPUS_FILE} — skipping coverage check")
        return

    corpus = CORPUS_FILE.read_text()
    weak: list[tuple[str, int]] = []
    zero: list[str] = []
    counts: dict[str, int] = {}

    for t in token_list:
        n = corpus.count(t)
        counts[t] = n
        if n == 0:
            zero.append(t)
        elif n < min_occur:
            weak.append((t, n))

    if zero:
        _warn(f"{len(zero)} tokens have 0 occurrences in corpus.txt:")
        for t in zero[:10]:
            print(f"      {t!r}")
        if len(zero) > 10:
            print(f"      ... and {len(zero)-10} more")

    if weak:
        _warn(f"{len(weak)} tokens have 1–{min_occur-1} occurrences in corpus.txt:")
        for t, n in sorted(weak, key=lambda x: x[1])[:10]:
            print(f"      {t!r}: {n}")
    else:
        _ok(f"All {len(token_list)} tokens appear ≥{min_occur} times in corpus.txt")

    _log({
        "phase": "coverage",
        "zero_count": len(zero),
        "weak_count": len(weak),
        "zero_tokens": zero[:20],
        "weak_tokens": [{"token": t, "count": n} for t, n in weak[:20]],
    })


# ===========================================================================
# Phase 2: Smart initialization
# ===========================================================================

def phase_smart_init(
    tokenizer,
    new_token_strings: list[str],
    base_vocab_size: int,
) -> "torch.Tensor":
    """Initialize new token embeddings as avg of their subword components.

    For "linux_memory_usage": average the embeddings of however many subwords
    the base tokenizer produced (e.g., ["linux", "_", "memory", "_", "usage"]).
    This is far better than random initialization.

    Returns a float32 tensor of shape [n_new_tokens, hidden_size].
    """
    _banner("PHASE 2: Smart embedding initialization")

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        _err("torch or transformers not installed.")
        _err("Run:  bash training/bootstrap.sh")
        sys.exit(1)

    model_path = str(MODEL_HF_DIR) if (MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir())) else HF_MODEL_ID
    _ok(f"Loading model embeddings from {model_path} (this takes 1-2 min) …")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    _ok(f"Model loaded in {_elapsed(t0)}")

    # Resize to include new tokens (adds zero-init rows for new tokens)
    model.resize_token_embeddings(len(tokenizer))

    embed_weight: torch.Tensor = model.model.embed_tokens.weight  # [vocab, hidden]
    hidden_size = embed_weight.shape[1]
    _ok(f"Embedding matrix: {embed_weight.shape[0]:,} × {hidden_size}")

    # Build a "clean" tokenizer that doesn't know about new tokens yet,
    # so we can get the base subword decomposition for each new token string.
    # We do this by directly calling the underlying SentencePiece model.
    # Since we can't easily reload the tokenizer without new tokens, we use
    # the vocabulary directly: for each new token string, tokenize it and
    # look up only IDs < base_vocab_size (original subwords).

    base_norm_mean = embed_weight[:base_vocab_size].norm(dim=1).mean().item()
    base_norm_std  = embed_weight[:base_vocab_size].norm(dim=1).std().item()
    _ok(f"Base vocab embedding norms: mean={base_norm_mean:.4f} ± std={base_norm_std:.4f}")

    _log({
        "phase": "smart_init",
        "base_vocab_size": base_vocab_size,
        "hidden_size": hidden_size,
        "base_norm_mean": base_norm_mean,
        "base_norm_std": base_norm_std,
    })

    # Compute subword averages for each new token
    n_smart = 0
    n_fallback = 0

    for i, token_str in enumerate(new_token_strings):
        new_id = base_vocab_size + i
        if new_id >= embed_weight.shape[0]:
            _warn(f"New token index {new_id} out of range — skipping {token_str!r}")
            continue

        # Tokenize the token string using base vocab (IDs < base_vocab_size)
        subword_ids = tokenizer.encode(token_str, add_special_tokens=False)
        # Only use IDs from the original vocabulary (not new tokens)
        base_ids = [sid for sid in subword_ids if sid < base_vocab_size]

        if base_ids:
            avg_embed = embed_weight[base_ids].mean(dim=0)
            embed_weight.data[new_id] = avg_embed
            n_smart += 1
        else:
            # Fallback: use the mean of the entire base vocabulary
            avg_embed = embed_weight[:base_vocab_size].mean(dim=0)
            embed_weight.data[new_id] = avg_embed
            n_fallback += 1

    _ok(f"Smart init: {n_smart} tokens (subword avg), {n_fallback} tokens (vocab mean fallback)")

    # Report new token embedding norms after smart init
    new_embeds = embed_weight[base_vocab_size: base_vocab_size + len(new_token_strings)]
    new_norm_mean = new_embeds.norm(dim=1).mean().item()
    new_norm_std  = new_embeds.norm(dim=1).std().item()
    _ok(f"New token embedding norms (post-init): mean={new_norm_mean:.4f} ± std={new_norm_std:.4f}")

    # Probe: cosine similarity for semantic pairs
    sim_before = _cosine_probe(tokenizer, embed_weight)
    _print_cosine_table(sim_before, label="cosine similarities (post smart-init)")

    _log({
        "phase": "smart_init",
        "event": "post_init",
        "n_smart": n_smart,
        "n_fallback": n_fallback,
        "new_norm_mean": new_norm_mean,
        "new_norm_std": new_norm_std,
        "cosine_probes": sim_before,
    })

    return model, embed_weight, base_vocab_size


# ===========================================================================
# Phase 3: Corpus training
# ===========================================================================

def phase_corpus_train(
    model,
    tokenizer,
    embed_weight: "torch.Tensor",
    base_vocab_size: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Train new embedding rows on corpus.txt; base vocab rows are frozen."""
    _banner("PHASE 3: Corpus warm-up training")

    try:
        import torch
    except ImportError:
        _err("torch not installed. Run:  bash training/bootstrap.sh")
        sys.exit(1)

    if not CORPUS_FILE.exists():
        _err(f"Corpus not found: {CORPUS_FILE}")
        sys.exit(1)

    sentences = [s.strip() for s in CORPUS_FILE.read_text().splitlines() if s.strip()]
    _ok(f"Corpus: {len(sentences)} sentences from {CORPUS_FILE}")

    # Freeze all model params except the embedding table
    for name, param in model.named_parameters():
        param.requires_grad_(name == "model.embed_tokens.weight")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _ok(f"Trainable parameters: {trainable:,} (embedding table only)")

    # Gradient hook: zero out base vocab rows so they cannot drift
    def _zero_base_rows(grad: "torch.Tensor") -> "torch.Tensor":
        g = grad.clone()
        g[:base_vocab_size] = 0.0
        return g

    hook_handle = embed_weight.register_hook(_zero_base_rows)

    optimizer = torch.optim.AdamW([embed_weight], lr=lr)

    # Tokenize corpus into batches
    max_len = 128
    tokenized = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    input_ids      = tokenized["input_ids"]       # [N, max_len]
    attention_mask = tokenized["attention_mask"]  # [N, max_len]
    n_sentences = input_ids.shape[0]

    n_batches = math.ceil(n_sentences / batch_size)
    _ok(f"Training: {n_sentences} sentences × {epochs} epochs, batch={batch_size} ({n_batches} steps/epoch)")
    _ok(f"Learning rate: {lr}")

    t_train = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        epoch_loss = 0.0
        model.train()

        # Shuffle sentence order each epoch
        perm = torch.randperm(n_sentences)
        input_ids_shuf = input_ids[perm]
        mask_shuf      = attention_mask[perm]

        for b in range(n_batches):
            start = b * batch_size
            end   = min(start + batch_size, n_sentences)
            ids   = input_ids_shuf[start:end]
            mask  = mask_shuf[start:end]

            # Shift for causal LM: labels = ids shifted left
            labels = ids.clone()
            # Mask out padding from loss
            labels[mask == 0] = -100

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss    = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                _warn(f"  Epoch {epoch} batch {b}: loss={loss.item():.4f} — skipping")
                continue

            optimizer.zero_grad()
            loss.backward()
            # Clamp gradient norms for stability
            torch.nn.utils.clip_grad_norm_([embed_weight], max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")

        # New token embedding stats this epoch
        new_embeds = embed_weight[base_vocab_size:].detach()
        new_norm_mean = new_embeds.norm(dim=1).mean().item()
        new_norm_std  = new_embeds.norm(dim=1).std().item()

        # Cosine similarity probes
        model.eval()
        sim_epoch = _cosine_probe(tokenizer, embed_weight.detach())
        model.train()

        _ok(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
            f"new_norm={new_norm_mean:.4f}±{new_norm_std:.4f}  "
            f"elapsed={_elapsed(t_epoch)}")
        _print_cosine_table(sim_epoch, label=f"  cosine sims epoch {epoch}")

        _log({
            "phase": "corpus_train",
            "epoch": epoch,
            "loss": avg_loss,
            "new_norm_mean": new_norm_mean,
            "new_norm_std": new_norm_std,
            "cosine_probes": sim_epoch,
        })

    hook_handle.remove()  # clean up gradient hook

    total_time = _elapsed(t_train)
    _ok(f"Corpus training complete in {total_time}")


# ===========================================================================
# Phase 4: Report + Save
# ===========================================================================

def phase_save(
    tokenizer,
    model,
    base_vocab_size: int,
    new_token_strings: list[str],
    n_neighbors: int,
) -> None:
    """Save extended tokenizer + pre-trained embeddings."""
    _banner("PHASE 4: Final report + save")

    try:
        import torch
    except ImportError:
        sys.exit(1)

    embed_weight: "torch.Tensor" = model.model.embed_tokens.weight.detach()

    # Final embedding health
    new_embeds  = embed_weight[base_vocab_size: base_vocab_size + len(new_token_strings)]
    base_embeds = embed_weight[:base_vocab_size]

    new_norm_mean  = new_embeds.norm(dim=1).mean().item()
    base_norm_mean = base_embeds.norm(dim=1).mean().item()
    _ok(f"Final embedding norms — new tokens: {new_norm_mean:.4f}  base vocab: {base_norm_mean:.4f}")

    sim_final = _cosine_probe(tokenizer, embed_weight)
    _print_cosine_table(sim_final, label="Final cosine similarities")

    # Nearest base-vocab neighbors for each new token
    if n_neighbors > 0:
        _report_nearest_neighbors(tokenizer, embed_weight, base_vocab_size, new_token_strings, n_neighbors)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(str(OUTPUT_DIR))
    _ok(f"Tokenizer saved to {OUTPUT_DIR}/")

    # Save only new embedding rows (don't duplicate the full model)
    new_embed_rows = embed_weight[base_vocab_size: base_vocab_size + len(new_token_strings)].clone()
    embed_path = OUTPUT_DIR / "embed_init.pt"
    torch.save({
        "new_token_strings": new_token_strings,
        "base_vocab_size": base_vocab_size,
        "embeddings": new_embed_rows,   # shape: [n_new_tokens, hidden_size]
    }, str(embed_path))
    _ok(f"Embeddings saved to {embed_path} ({embed_path.stat().st_size / 1e6:.1f} MB)")

    _log({
        "phase": "save",
        "output_dir": str(OUTPUT_DIR),
        "n_new_tokens": len(new_token_strings),
        "final_norm_new": new_norm_mean,
        "final_norm_base": base_norm_mean,
        "cosine_probes": sim_final,
    })

    _flush_log()
    _ok(f"Training log written to {LOG_FILE}")

    print(f"\n  {'='*60}")
    print(f"  Tokenizer extended saved to: {OUTPUT_DIR}")
    print(f"  finetune.py will auto-load from here when --mode train runs.")
    print(f"  {'='*60}")


def _cosine_probe(tokenizer, embed_weight: "torch.Tensor") -> dict:
    """Compute cosine similarity for each PROBE_PAIR. Returns {desc: sim}."""
    import torch
    import torch.nn.functional as F

    results = {}
    for token_a, token_b, desc in PROBE_PAIRS:
        id_a = tokenizer.convert_tokens_to_ids(token_a)
        id_b = tokenizer.convert_tokens_to_ids(token_b)
        unk  = tokenizer.unk_token_id

        if id_a == unk or id_b == unk:
            results[desc] = None  # token not in vocabulary
            continue
        if id_a >= embed_weight.shape[0] or id_b >= embed_weight.shape[0]:
            results[desc] = None
            continue

        ea = embed_weight[id_a].float()
        eb = embed_weight[id_b].float()
        sim = F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
        results[desc] = round(sim, 4)

    return results


def _print_cosine_table(sims: dict, label: str = "") -> None:
    if label:
        print(f"\n  {label}:")
    for desc, sim in sims.items():
        if sim is None:
            bar = "(token not found)"
        else:
            filled = int((sim + 1) / 2 * 20)  # map [-1,1] → [0,20] blocks
            bar = "█" * filled + "░" * (20 - filled)
            bar = f"{sim:+.4f}  {bar}"
        print(f"    {desc:40s} {bar}")


def _report_nearest_neighbors(
    tokenizer,
    embed_weight: "torch.Tensor",
    base_vocab_size: int,
    new_token_strings: list[str],
    k: int,
) -> None:
    """For each new token, print the k nearest base-vocab tokens by cosine sim."""
    import torch
    import torch.nn.functional as F

    print(f"\n  Nearest {k} base-vocab neighbors for new tokens:")

    base_embeds = embed_weight[:base_vocab_size].float()  # [base_vocab, hidden]
    base_norms  = F.normalize(base_embeds, dim=1)         # [base_vocab, hidden]

    # Batch process to avoid giant memory allocation
    for token_str in new_token_strings[:40]:  # cap at 40 to keep output manageable
        new_id = tokenizer.convert_tokens_to_ids(token_str)
        if new_id is None or new_id >= embed_weight.shape[0]:
            continue

        new_vec  = embed_weight[new_id].float().unsqueeze(0)  # [1, hidden]
        new_norm = F.normalize(new_vec, dim=1)                 # [1, hidden]
        sims     = (new_norm @ base_norms.T).squeeze(0)       # [base_vocab]

        top_vals, top_ids = sims.topk(k)
        neighbor_strs = [tokenizer.convert_ids_to_tokens(i.item()) for i in top_ids]

        pairs = " | ".join(f"{t!r}({v:.3f})" for t, v in zip(neighbor_strs, top_vals.tolist()))
        print(f"    {token_str:30s} → {pairs}")

    _log({
        "phase": "nearest_neighbors",
        "n_tokens_reported": min(40, len(new_token_strings)),
        "k": k,
    })


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int,   default=2,    help="Corpus training epochs (default: 2)")
    parser.add_argument("--lr",          type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--batch-size",  type=int,   default=16,   help="Batch size for corpus training (default: 16)")
    parser.add_argument("--skip-corpus", action="store_true",      help="Skip corpus training; only do smart init")
    parser.add_argument("--neighbors",   type=int,   default=5,    help="Nearest neighbors to show per new token (0 = skip)")
    args = parser.parse_args()

    t_total = time.time()

    print(f"\nfunctiongemma tokenizer extension")
    print(f"  New tokens:    {NEW_TOKENS_JSON}")
    print(f"  Corpus:        {CORPUS_FILE}")
    print(f"  Output:        {OUTPUT_DIR}")
    print(f"  Epochs:        {args.epochs if not args.skip_corpus else 'skipped (--skip-corpus)'}")
    print(f"  LR:            {args.lr}")

    # Phase 1: add tokens
    new_token_strings, tokenizer = phase_add_tokens()
    base_vocab_size = len(tokenizer) - len(new_token_strings)

    # Phase 2: smart init (loads model, initializes new embedding rows)
    model, embed_weight, base_vocab_size = phase_smart_init(
        tokenizer, new_token_strings, base_vocab_size
    )

    # Phase 3: corpus warm-up training
    if not args.skip_corpus:
        phase_corpus_train(
            model, tokenizer, embed_weight, base_vocab_size,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        )
    else:
        _ok("Skipping corpus training (--skip-corpus)")

    # Phase 4: final report + save
    phase_save(tokenizer, model, base_vocab_size, new_token_strings, n_neighbors=args.neighbors)

    _ok(f"Total time: {_elapsed(t_total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
