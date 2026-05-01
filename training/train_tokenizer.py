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
  Without pre-warming, the first 30-50% of LoRA training steps are spent
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

# ---------------------------------------------------------------------------
# Fast dependency check — fail with a helpful message before doing any work.
# importlib.util.find_spec() checks without importing, so this is instant.
# If packages are missing, the user sees exactly which ones and how to fix it.
# ---------------------------------------------------------------------------
import importlib.util as _ilu
_MISSING = [pkg for pkg in ("torch", "transformers", "sentencepiece")
            if _ilu.find_spec(pkg) is None]
del _ilu  # clean up the temporary import alias
if _MISSING:
    # Point at the venv python so the user runs the right interpreter
    _venv = Path(__file__).resolve().parent.parent / ".fngemma-suryaos" / "bin" / "python3"
    print(f"[ERR]  Missing packages: {', '.join(_MISSING)}", file=sys.stderr)
    print(f"[ERR]  Run:  bash training/bootstrap.sh", file=sys.stderr)
    print(f"[ERR]  Then: {_venv} training/train_tokenizer.py", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths — all resolved from this file's location so the script works from any
# working directory.
# ---------------------------------------------------------------------------
REPO_ROOT       = Path(__file__).resolve().parent.parent   # functiongemma-suryaos/
TRAINING_DIR    = Path(__file__).resolve().parent          # training/
TOK_DATA_DIR    = REPO_ROOT / "dataset" / "tokenizer"     # tokenizer corpus + token lists
NEW_TOKENS_JSON = TOK_DATA_DIR / "new_tokens.json"        # 319 domain tokens by category
CORPUS_FILE     = TOK_DATA_DIR / "corpus.txt"             # ~3800 sentences for embedding training
OUTPUT_DIR      = TRAINING_DIR / "tokenizer_extended"     # where we save the extended tokenizer
LOG_FILE        = OUTPUT_DIR / "train_log.jsonl"          # structured telemetry log

# If a local HF conversion already exists, use it (faster than HF Hub download).
# Otherwise fall back to downloading from the Hub (needs HF_TOKEN for gated models).
HF_MODEL_ID  = "google/gemma-3-270m-it"
MODEL_HF_DIR = TRAINING_DIR / "model_hf"

# ---------------------------------------------------------------------------
# Semantic probe pairs — tracked throughout training to verify that embeddings
# are converging toward meaningful clusters.
#
# Reading the output:
#   +1.0 = identical direction in embedding space (same meaning)
#    0.0 = orthogonal (unrelated)
#   -1.0 = opposite direction
#
# What we want to see:
#   same-tool forms        → should rise above +0.7 by end of training
#   sibling linux tools    → should stay in +0.4 to +0.7 range
#   cross-domain           → should stay below +0.3
#
# If cross-domain similarity is rising, the embeddings are collapsing (bad).
# If same-tool similarity isn't rising, the corpus templates need more variety.
# ---------------------------------------------------------------------------
PROBE_PAIRS = [
    # Two naming forms of the same tool — highest expected similarity
    ("linux_memory_usage",    "memory_usage",          "same-tool forms"),
    ("linux_disk_usage",      "disk_usage",            "same-tool forms"),
    # Tool name vs the CLI command it calls — should be nearby
    ("service_status",        "systemctl",             "tool vs CLI equiv"),
    ("krunner_launch",        "KRunner",               "tool vs KDE component"),
    # Sibling tools in the same category — should cluster together
    ("linux_memory_usage",    "linux_disk_usage",      "sibling linux tools"),
    ("linux_metrics_summary", "linux_memory_usage",    "metrics vs memory"),
    # Cross-domain pair — should stay dissimilar (low score expected)
    ("linux_memory_usage",    "brightness_set",        "cross-domain (expected low)"),
    # Co-occurring terms in training context — should pick up shared context
    ("torch",                 "transformers",          "co-occurring ML libs"),
    ("merge",                 "commit",                "co-occurring git ops"),
    ("GGUF",                  "ollama",                "co-occurring serving terms"),
]


# ---------------------------------------------------------------------------
# Terminal output helpers — consistent prefix makes log scanning easy
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    """Print a bold section header to mark phase boundaries in the terminal."""
    width = 72
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _ok(msg: str)   -> None: print(f"  [OK]   {msg}")
def _warn(msg: str) -> None: print(f"  [WARN] {msg}", file=sys.stderr)
def _err(msg: str)  -> None: print(f"  [ERR]  {msg}", file=sys.stderr)


def _elapsed(start: float) -> str:
    """Human-readable elapsed time string, e.g. '2m 14s'."""
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


# ---------------------------------------------------------------------------
# Narrator — emits intuitive plain-English commentary as training proceeds.
#
# The point: when you watch the terminal during a training run, you should
# learn what's happening at every micro-step, not just see numbers fly by.
# This class tracks history and emits messages like:
#   [LEARN]    Step 50: loss dropped 40% — embeddings settling into clusters
#   [INSIGHT]  Cosine 'memory_usage' vs 'linux_memory_usage' rose 0.32→0.71
#              The model now treats these as the same concept.
#   [WARN]     Gradient norm spiked to 1.8 — clipping prevented instability
# ---------------------------------------------------------------------------

class Narrator:
    """
    Live training commentary. Tracks loss/grad history and emits insights.

    Prefixes used:
      [LEARN]    The model just learned something concrete (loss milestone hit)
      [PROGRESS] Healthy ongoing improvement
      [INSIGHT]  A pattern was detected in the metrics
      [PLATEAU]  Loss has stopped improving — maybe done training
      [WARN]     Anomaly that needs attention
    """

    def __init__(self, name: str = "training"):
        self.name              = name
        self.loss_history      = []   # list of (step, loss) tuples
        self.grad_norm_history = []   # list of (step, grad_norm) tuples
        self.last_emit_step    = -100 # rate-limit narration
        self.milestones        = set() # avoid re-announcing thresholds

    def _emit(self, prefix: str, msg: str) -> None:
        """All narrations go through here so we can rate-limit/style consistently."""
        print(f"  [{prefix:8s}] {msg}")

    def step(self, step: int, total_steps: int, loss: float,
             grad_norm: Optional[float] = None, lr: Optional[float] = None) -> None:
        """
        Called every training step. Emits commentary on notable changes.
        Most steps don't trigger output — only when something interesting happens.
        """
        self.loss_history.append((step, loss))
        if grad_norm is not None and not math.isnan(grad_norm):
            self.grad_norm_history.append((step, grad_norm))

        # Very early steps: explain what's happening at the start
        if step == 1:
            self._emit("LEARN",
                f"Step 1/{total_steps}: initial loss={loss:.3f}. "
                f"Embeddings start near random — this is the baseline.")
            return

        # ---- Loss milestones ----
        # Crossing key loss thresholds tells the user the model has reached
        # a specific competence level.
        if loss < 3.0 and "loss_below_3" not in self.milestones:
            self._emit("LEARN",
                f"Step {step}: loss dropped below 3.0 (={loss:.3f}). "
                f"Model is starting to predict context tokens correctly.")
            self.milestones.add("loss_below_3")

        if loss < 1.5 and "loss_below_1_5" not in self.milestones:
            self._emit("LEARN",
                f"Step {step}: loss below 1.5 (={loss:.3f}). "
                f"New token embeddings are settling into meaningful clusters.")
            self.milestones.add("loss_below_1_5")

        if loss < 0.8 and "loss_below_08" not in self.milestones:
            self._emit("LEARN",
                f"Step {step}: loss below 0.8 (={loss:.3f}). "
                f"Embeddings are well-formed; further training has diminishing returns.")
            self.milestones.add("loss_below_08")

        # Rate-limit other narrations to every 10 steps so we don't flood
        if step - self.last_emit_step < 10:
            return

        # ---- Trend detection ----
        # Compare last 10 steps to previous 10 to detect rapid improvement,
        # plateau, or instability.
        if len(self.loss_history) >= 20:
            recent = [l for _, l in self.loss_history[-10:]]
            older  = [l for _, l in self.loss_history[-20:-10]]
            recent_avg = sum(recent) / len(recent)
            older_avg  = sum(older) / len(older)

            if older_avg > 0:
                pct = (older_avg - recent_avg) / older_avg * 100  # positive = improving

                if pct > 30:
                    self._emit("PROGRESS",
                        f"Step {step}: loss down {pct:.0f}% in last 10 steps "
                        f"(was {older_avg:.3f}, now {recent_avg:.3f}) — rapid learning")
                    self.last_emit_step = step
                elif pct > 10:
                    self._emit("PROGRESS",
                        f"Step {step}: loss down {pct:.0f}% — steady convergence")
                    self.last_emit_step = step
                elif abs(pct) < 2:
                    self._emit("PLATEAU",
                        f"Step {step}: loss stable at ~{recent_avg:.3f} "
                        f"(<2% change) — model has converged for this lr")
                    self.last_emit_step = step
                elif pct < -10:
                    self._emit("WARN",
                        f"Step {step}: loss INCREASED {-pct:.0f}% "
                        f"(was {older_avg:.3f}, now {recent_avg:.3f}) — "
                        f"check grad_norm and consider lowering lr")
                    self.last_emit_step = step

        # ---- Gradient stability ----
        if grad_norm is not None and grad_norm > 1.5:
            self._emit("WARN",
                f"Step {step}: grad_norm={grad_norm:.2f} exceeded clip threshold (1.0). "
                f"Gradient was clipped — preventing instability but slowing learning.")

    def epoch_end(self, epoch: int, total_epochs: int, loss: float,
                  prev_loss: Optional[float] = None,
                  new_norm_mean: Optional[float] = None,
                  base_norm_mean: Optional[float] = None) -> None:
        """Rich end-of-epoch interpretation."""
        print()  # blank line for readability
        if prev_loss is None:
            self._emit("LEARN",
                f"Epoch {epoch}/{total_epochs} done. Loss={loss:.4f}. "
                f"This is the baseline; we'll measure progress against this.")
        else:
            delta = prev_loss - loss
            pct   = delta / prev_loss * 100 if prev_loss > 0 else 0
            if delta > 0.1:
                self._emit("PROGRESS",
                    f"Epoch {epoch}/{total_epochs}: loss {prev_loss:.4f} -> {loss:.4f} "
                    f"({pct:+.0f}%). Embeddings continue to improve.")
            elif delta > 0.01:
                self._emit("PROGRESS",
                    f"Epoch {epoch}/{total_epochs}: loss {prev_loss:.4f} -> {loss:.4f} "
                    f"({pct:+.0f}%). Diminishing returns — close to convergence.")
            else:
                self._emit("PLATEAU",
                    f"Epoch {epoch}/{total_epochs}: loss barely changed "
                    f"({prev_loss:.4f} -> {loss:.4f}). "
                    f"More epochs unlikely to help; consider stopping or raising lr.")

        # Norm health check: new tokens should approach base vocab norm magnitude
        if new_norm_mean is not None and base_norm_mean is not None and base_norm_mean > 0:
            ratio = new_norm_mean / base_norm_mean
            if ratio < 0.5:
                self._emit("INSIGHT",
                    f"New token norms are only {ratio*100:.0f}% of base vocab. "
                    f"Embeddings are still 'quiet' — model may underweight them.")
            elif ratio > 1.5:
                self._emit("INSIGHT",
                    f"New token norms are {ratio*100:.0f}% of base vocab. "
                    f"Embeddings are 'loud' — may dominate attention; reduce lr next time.")
            else:
                self._emit("INSIGHT",
                    f"New token norms at {ratio*100:.0f}% of base vocab "
                    f"— in healthy range, embeddings will integrate cleanly with base model.")

    def cosine_change(self, label_changes: list[tuple[str, float, float]]) -> None:
        """Interpret cosine similarity changes between epochs.
        label_changes: list of (description, old_sim, new_sim).
        """
        for desc, old, new in label_changes:
            if old is None or new is None:
                continue
            delta = new - old
            if "same-tool" in desc and delta > 0.1:
                self._emit("LEARN",
                    f"'{desc}' similarity {old:+.2f} -> {new:+.2f}: "
                    f"model now treats these as the same concept.")
            elif "cross-domain" in desc and delta > 0.1:
                self._emit("WARN",
                    f"'{desc}' similarity {old:+.2f} -> {new:+.2f}: "
                    f"unrelated tools getting closer — possible embedding collapse.")
            elif abs(delta) > 0.15:
                arrow = "rising" if delta > 0 else "dropping"
                self._emit("INSIGHT",
                    f"'{desc}' similarity {arrow} fast: {old:+.2f} -> {new:+.2f}")


# ---------------------------------------------------------------------------
# Structured logging — all telemetry is buffered and flushed to train_log.jsonl
# at save time so we have a machine-readable record of every training run.
# ---------------------------------------------------------------------------
_log_entries: list[dict] = []

def _log(entry: dict) -> None:
    """Append one JSON entry to the in-memory log buffer."""
    _log_entries.append(entry)


def _flush_log() -> None:
    """Write all buffered log entries to train_log.jsonl (one JSON object per line)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        for e in _log_entries:
            f.write(json.dumps(e) + "\n")


# ===========================================================================
# Phase 1: Add tokens to the tokenizer vocabulary
# ===========================================================================

def phase_add_tokens() -> tuple[list[str], "AutoTokenizer"]:
    """
    Load the 319 domain tokens from new_tokens.json and add them to Gemma 3's
    tokenizer, then report before/after fragmentation and corpus coverage.

    Fragmentation explained:
      Without extension, 'linux_memory_usage' splits into 4-5 subword pieces.
      After adding it as a single token, it maps to exactly 1 ID.
      Fewer tokens per tool name = fewer positions the model must predict
      correctly = much easier to learn reliable dispatch.

    Returns: (list_of_newly_added_token_strings, extended_tokenizer)
    """
    _banner("PHASE 1: Add domain tokens to tokenizer")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        _err("transformers not installed.")
        _err("Run:  bash training/bootstrap.sh")
        _err("  or: .fngemma-suryaos/bin/pip install -r training/requirements.txt")
        sys.exit(1)

    if not NEW_TOKENS_JSON.exists():
        _err(f"{NEW_TOKENS_JSON} not found.")
        _err("Run build_tokenizer_dataset.py first to generate the token lists.")
        sys.exit(1)

    # Load the categorized token list.
    # Structure: {"tool_name": [{"token": "linux_memory_usage", ...}, ...], "kde": [...], ...}
    with open(NEW_TOKENS_JSON) as f:
        by_category: dict[str, list[dict]] = json.load(f)

    # Flatten to a single ordered list; deduplicate while preserving first-seen order.
    all_tokens = [entry["token"] for entries in by_category.values() for entry in entries]
    all_tokens = list(dict.fromkeys(all_tokens))
    _ok(f"Loaded {len(all_tokens)} unique domain tokens from new_tokens.json")

    # Print the breakdown by category so we can see what we're adding
    for cat, entries in sorted(by_category.items()):
        print(f"    {cat:20s} {len(entries):3d} tokens")

    # Prefer local model_hf/ if it exists (avoids HF Hub download).
    # model_hf/ is created by finetune.py --mode convert.
    model_path = (str(MODEL_HF_DIR)
                  if MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir())
                  else HF_MODEL_ID)
    _ok(f"Loading tokenizer from {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    except Exception as e:
        _err(f"Tokenizer load failed: {e}")
        sys.exit(1)

    base_vocab_size = len(tokenizer)
    _ok(f"Base vocabulary size: {base_vocab_size:,}")

    # ---- Fragmentation probe BEFORE adding tokens ----
    # This shows us how badly the base tokenizer splits our domain terms.
    # e.g. 'linux_memory_usage' → ['linux', '_', 'memory', '_', 'usage'] = 5 pieces
    frag_before = _measure_fragmentation(tokenizer, all_tokens)
    _ok(f"Fragmentation BEFORE extension: avg {frag_before['mean']:.1f} subwords/token "
        f"(max {frag_before['max']:.0f}, "
        f"{frag_before['already_single']}/{len(all_tokens)} already single-token)")
    _log({"phase": "add_tokens", "event": "fragmentation_before", **frag_before})

    # Show concrete examples so the user understands the problem visually.
    # Pick 5 representative tokens that demonstrate the fragmentation.
    print(f"\n  [DEMO] Without extension, the base tokenizer breaks up domain terms:")
    demo_tokens = [t for t in all_tokens if "_" in t or "." in t][:5]
    for t in demo_tokens:
        ids    = tokenizer.encode(t, add_special_tokens=False)
        pieces = [tokenizer.decode([i]) for i in ids]
        print(f"    {t!r:35s} -> {pieces}  ({len(ids)} tokens)")
    print(f"  [DEMO] The model has to predict each piece in order. With 4-5 pieces")
    print(f"  [DEMO] per tool name, even small per-token errors compound badly.\n")

    # Add all tokens that aren't already in the vocabulary.
    # special_tokens=False: treat as regular vocab (not EOS/BOS/PAD/etc.)
    existing  = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in all_tokens if t not in existing]
    if new_tokens:
        n_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
        _ok(f"Added {n_added} new tokens "
            f"(skipped {len(all_tokens) - n_added} already in vocab)")
    else:
        n_added = 0
        _ok("All domain tokens already in vocabulary — none added")

    new_vocab_size = len(tokenizer)
    _ok(f"Extended vocabulary size: {new_vocab_size:,} (+{new_vocab_size - base_vocab_size})")

    # ---- Fragmentation probe AFTER adding tokens ----
    # Should show 'already_single' close to total — most tokens now map to 1 ID.
    frag_after = _measure_fragmentation(tokenizer, all_tokens)
    _ok(f"Fragmentation AFTER extension:  avg {frag_after['mean']:.1f} subwords/token "
        f"(max {frag_after['max']:.0f}, "
        f"{frag_after['already_single']}/{len(all_tokens)} now single-token)")
    _log({"phase": "add_tokens", "event": "fragmentation_after", **frag_after})

    # Show same examples after extension — should be 1 token each
    print(f"\n  [DEMO] After extension, the same terms tokenize as a single ID:")
    for t in demo_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            print(f"    {t!r:35s} -> [{ids[0]}]  (1 token — single embedding to learn)")
        else:
            # Edge case: token was already in vocab as multi-piece (rare)
            print(f"    {t!r:35s} -> {ids}  ({len(ids)} tokens still)")
    print(f"  [DEMO] Now the model only needs to learn one embedding per tool name.")
    print(f"  [LEARN] Improvement: {frag_before['mean']:.1f} -> {frag_after['mean']:.1f} "
          f"avg pieces per token (lower = easier to learn).\n")

    # Verify the corpus covers every token enough times for embeddings to converge
    _report_coverage(all_tokens)

    _log({
        "phase":          "add_tokens",
        "base_vocab_size": base_vocab_size,
        "new_vocab_size":  new_vocab_size,
        "n_new_tokens":    n_added if new_tokens else 0,
        "categories":      {k: len(v) for k, v in by_category.items()},
    })

    return new_tokens if new_tokens else [], tokenizer


def _measure_fragmentation(tokenizer, token_strings: list[str]) -> dict:
    """
    For each token string, count how many subword pieces the tokenizer produces.

    A score of 1 means the token is a single vocab entry — ideal.
    A score of 5+ means heavy fragmentation — the model sees unrelated meanings.
    E.g. 'bluetooth' → ['blue', 'tooth'] conflates color + dental with wireless.
    """
    counts        = []
    already_single = 0
    worst          = []  # tokens with the most fragmentation

    for t in token_strings:
        # encode without BOS/EOS so we count only the actual token pieces
        ids = tokenizer.encode(t, add_special_tokens=False)
        n   = len(ids)
        counts.append(n)
        if n == 1:
            already_single += 1
        if n > 5:
            worst.append((t, n))  # flag heavily fragmented tokens

    worst.sort(key=lambda x: -x[1])  # worst first
    return {
        "mean":           sum(counts) / len(counts) if counts else 0,
        "max":            max(counts) if counts else 0,
        "already_single": already_single,
        "total":          len(counts),
        "worst_10":       worst[:10],  # top-10 most fragmented for inspection
    }


def _report_coverage(token_list: list[str], min_occur: int = 5) -> None:
    """
    Check that every token appears at least min_occur times in corpus.txt.

    Why min_occur=5?  Embedding training is gradient descent — a token that
    appears only once or twice will receive too few gradient updates to converge.
    Empirically, 5 occurrences gives embeddings that are directionally correct
    even if not fully converged.  Tokens appearing 0 times will stay at their
    smart-init value and receive no further training signal.
    """
    if not CORPUS_FILE.exists():
        _warn(f"Corpus not found at {CORPUS_FILE} — skipping coverage check")
        return

    corpus = CORPUS_FILE.read_text()  # load the full corpus as one string for counting
    weak: list[tuple[str, int]] = []
    zero: list[str]             = []

    for t in token_list:
        n = corpus.count(t)  # simple substring count across all sentences
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
        _warn("Re-run build_tokenizer_dataset.py to add these to the corpus.")

    if weak:
        _warn(f"{len(weak)} tokens have 1-{min_occur-1} occurrences (may not converge):")
        for t, n in sorted(weak, key=lambda x: x[1])[:10]:
            print(f"      {t!r}: {n}")
    else:
        _ok(f"All {len(token_list)} tokens appear >= {min_occur} times in corpus.txt")

    _log({
        "phase":        "coverage",
        "zero_count":   len(zero),
        "weak_count":   len(weak),
        "zero_tokens":  zero[:20],
        "weak_tokens":  [{"token": t, "count": n} for t, n in weak[:20]],
    })


# ===========================================================================
# Phase 2: Smart embedding initialization
# ===========================================================================

def phase_smart_init(
    tokenizer,
    new_token_strings: list[str],
    base_vocab_size: int,
) -> tuple:
    """
    Initialize each new token's embedding as the average of its subword pieces.

    Why not random initialization?
      Random init means the new embedding starts somewhere arbitrary in the
      640-dimensional space. The model has no idea what 'linux_memory_usage'
      means. The first N gradient steps teach it "this token exists" rather
      than "this token means dispatch to memory tools".

    Why subword average?
      'linux_memory_usage' → subwords ['linux', '_', 'memory', '_', 'usage']
      Those subwords have well-trained embeddings that already encode the
      concepts of "linux system", "memory", and "usage". Averaging them gives
      a starting point close to the right neighborhood in embedding space.

    This is the standard "mean subword embedding" initialization from the
    original BERT fine-tuning literature, adapted for tokenizer extension.

    Returns: (model, embed_weight_tensor, base_vocab_size)
    """
    _banner("PHASE 2: Smart embedding initialization")

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        _err("torch or transformers not installed.")
        _err("Run:  bash training/bootstrap.sh")
        sys.exit(1)

    model_path = (str(MODEL_HF_DIR)
                  if MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir())
                  else HF_MODEL_ID)

    # Detect GPU and choose the right dtype.
    # Using fp16/bf16 on GPU cuts memory by 2x vs float32 and speeds up init.
    # We use bfloat16 on datacenter Ampere cards (A100, H100) because their
    # bf16 tensor cores are faster and the wider exponent range is more stable.
    # Consumer RTX (30xx/40xx) use float16 — same memory savings, slightly
    # narrower exponent range is fine for embeddings.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute  = torch.cuda.get_device_capability(0)
        # Datacenter Ampere (compute >= 8.0) gets bf16; consumer RTX gets fp16
        _dc   = any(k in gpu_name for k in ("A100", "H100", "H200", "A40", "A10G"))
        dtype = torch.bfloat16 if (_dc and compute[0] >= 8) else torch.float16
        _ok(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM, compute {compute[0]}.{compute[1]}) "
            f"— dtype={dtype}")
    else:
        dtype = torch.float32  # float32 required for numerical stability on CPU
        _ok("No GPU detected — using CPU + float32 (expect 1-3 min load time)")

    _ok(f"Loading model from {model_path} (full 270M weights needed for embeddings) ...")
    t0 = time.time()

    # low_cpu_mem_usage=True: loads tensors one-by-one instead of all at once,
    # cutting peak RAM from ~4 GB to ~2 GB during the load phase.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    _ok(f"Model loaded in {_elapsed(t0)}")

    # Resize the embedding table to accommodate the new tokens.
    # This adds zero-initialized rows at positions [base_vocab_size:].
    # We will overwrite those rows with the subword averages below.
    model.resize_token_embeddings(len(tokenizer))

    # Direct reference to the embedding weight matrix: shape [vocab_size, hidden_size]
    # Modifying embed_weight.data also modifies the model's parameter in place.
    embed_weight: torch.Tensor = model.model.embed_tokens.weight
    hidden_size = embed_weight.shape[1]  # 640 for Gemma 3 270M
    _ok(f"Embedding matrix: {embed_weight.shape[0]:,} x {hidden_size} "
        f"({embed_weight.numel() * embed_weight.element_size() / 1e6:.0f} MB)")

    # Record the norm statistics of the BASE vocab embeddings.
    # These are the "healthy" norms that new token embeddings should approach.
    # If new norms are much smaller, the model will treat new tokens as noise.
    base_norm_mean = embed_weight[:base_vocab_size].norm(dim=1).mean().item()
    base_norm_std  = embed_weight[:base_vocab_size].norm(dim=1).std().item()
    _ok(f"Base vocab embedding norms — mean={base_norm_mean:.4f}  std={base_norm_std:.4f}")
    _ok(f"  (healthy range for new tokens after init: "
        f"[{base_norm_mean - base_norm_std:.3f}, {base_norm_mean + base_norm_std:.3f}])")

    _log({
        "phase":          "smart_init",
        "base_vocab_size": base_vocab_size,
        "hidden_size":     hidden_size,
        "base_norm_mean":  base_norm_mean,
        "base_norm_std":   base_norm_std,
    })

    # ---- Compute subword averages ----
    n_smart    = 0  # tokens initialized from their own subword pieces
    n_fallback = 0  # tokens that had no base-vocab subwords → use global mean

    for i, token_str in enumerate(new_token_strings):
        new_id = base_vocab_size + i  # row index for this new token in embed_weight

        if new_id >= embed_weight.shape[0]:
            _warn(f"Token index {new_id} out of range for {token_str!r} — skipping")
            continue

        # Tokenize using the EXTENDED tokenizer, but filter out any newly added IDs.
        # This gives us only the original base-vocab piece IDs for this token string.
        # e.g. 'linux_memory_usage' → [12345, 432, 6789, 432, 4567] (hypothetical)
        #   → only keep IDs < base_vocab_size (original vocab pieces)
        subword_ids = tokenizer.encode(token_str, add_special_tokens=False)
        base_ids    = [sid for sid in subword_ids if sid < base_vocab_size]

        if base_ids:
            # Mean of the subword embeddings — directionally near the right concept
            avg_embed = embed_weight[base_ids].mean(dim=0)
            embed_weight.data[new_id] = avg_embed
            n_smart += 1
        else:
            # Fallback: token string had no recognizable subwords in base vocab
            # (rare for ASCII tokens; could happen for very unusual Unicode tokens).
            # Use the global mean so at least the norm is in the right ballpark.
            avg_embed = embed_weight[:base_vocab_size].mean(dim=0)
            embed_weight.data[new_id] = avg_embed
            n_fallback += 1

    _ok(f"Smart init complete: {n_smart} via subword avg, {n_fallback} via global mean fallback")

    # Report new token norm distribution — should be close to base vocab norms
    new_embeds    = embed_weight[base_vocab_size: base_vocab_size + len(new_token_strings)]
    new_norm_mean = new_embeds.norm(dim=1).mean().item()
    new_norm_std  = new_embeds.norm(dim=1).std().item()
    _ok(f"New token embedding norms (post-init) — mean={new_norm_mean:.4f}  "
        f"std={new_norm_std:.4f}")

    # If new_norm_mean is much smaller than base_norm_mean, the model will treat
    # new tokens as "quiet signals" and may ignore them during inference.
    if new_norm_mean < base_norm_mean * 0.5:
        _warn(f"New token norms ({new_norm_mean:.3f}) are much smaller than base "
              f"({base_norm_mean:.3f}) — corpus training should fix this.")

    # Concrete demo: show what subwords each new token started from.
    # This makes "smart init" feel real: you see which base concepts the model
    # is averaging to seed each new domain token.
    print(f"\n  [DEMO] Smart init seeded each new token from its subword pieces:")
    sample_tokens = [t for t in new_token_strings if "_" in t][:5]
    for ts in sample_tokens:
        sub_ids   = tokenizer.encode(ts, add_special_tokens=False)
        base_only = [sid for sid in sub_ids if sid < base_vocab_size]
        pieces    = [tokenizer.decode([i]) for i in base_only]
        print(f"    {ts!r:35s} init = mean({pieces})")
    print(f"  [LEARN] Each new token starts in the right neighborhood —")
    print(f"  [LEARN] e.g. 'linux_memory_usage' is already near 'linux'+'memory'+'usage'")
    print(f"  [LEARN] in embedding space, not at random.\n")

    # Cosine similarity probe to verify the embeddings started in the right place.
    # At this point we expect moderate similarity (0.3–0.6) for same-tool forms
    # because the subword averaging captures shared concepts but isn't refined yet.
    sim_before = _cosine_probe(tokenizer, embed_weight)
    _print_cosine_table(sim_before, label="cosine similarities after smart-init (before corpus training)")
    _interpret_cosine_table(sim_before, phase_label="post smart-init")

    _log({
        "phase":          "smart_init",
        "event":          "post_init",
        "n_smart":        n_smart,
        "n_fallback":     n_fallback,
        "new_norm_mean":  new_norm_mean,
        "new_norm_std":   new_norm_std,
        "cosine_probes":  sim_before,
    })

    return model, embed_weight, base_vocab_size


# ===========================================================================
# Phase 3: Corpus warm-up training
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
    """
    Train ONLY the new embedding rows on corpus.txt using causal LM loss.

    Why is the base vocab frozen?
      The 256,000 base vocab embeddings are already excellent — they encode
      rich semantic information from pre-training on massive text. Allowing
      gradients to flow into them would corrupt that with our tiny corpus.
      We freeze them via a gradient hook that zeros out rows 0..base_vocab_size
      before every optimizer step.

    What does "causal LM loss" mean here?
      For each sentence, the model predicts token[i+1] given tokens[0..i].
      The cross-entropy loss penalizes wrong predictions. This forces the new
      embeddings to position themselves so that context words correctly predict
      the new token and vice versa.

    What should the loss curve look like?
      Epoch 1: 3.0 to 4.0 (embeddings are fresh from smart init, model is
               still learning the context patterns)
      Epoch 2: 1.5 to 2.5 (improving — embeddings settling into place)
      Epoch 3: 1.0 to 1.8 (converging — diminishing returns after this)
      If loss stays above 4.0, the lr is too high or the corpus has issues.
      If loss immediately drops to 0.5, the corpus has too little variety.
    """
    _banner("PHASE 3: Corpus warm-up training")

    try:
        import torch
    except ImportError:
        _err("torch not installed. Run:  bash training/bootstrap.sh")
        sys.exit(1)

    if not CORPUS_FILE.exists():
        _err(f"Corpus not found: {CORPUS_FILE}")
        _err("Run build_tokenizer_dataset.py to generate it.")
        sys.exit(1)

    # Get the device the model is actually on (cuda:0 or cpu)
    device = next(model.parameters()).device
    _ok(f"Training device: {device}")

    sentences = [s.strip() for s in CORPUS_FILE.read_text().splitlines() if s.strip()]
    _ok(f"Corpus: {len(sentences)} sentences from {CORPUS_FILE.name}")

    # ---- Freeze everything except the embedding table ----
    # We only want the new token rows to learn, so we disable gradients on all
    # other parameters. This also means the optimizer only tracks one tensor,
    # which saves memory significantly (no optimizer state for 268M frozen params).
    for name, param in model.named_parameters():
        param.requires_grad_(name == "model.embed_tokens.weight")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _ok(f"Trainable parameters: {trainable_params:,}  "
        f"(embedding table only; {trainable_params / 1e6:.1f}M of 268M total)")

    # ---- Gradient hook: protect base vocab from any drift ----
    # Even though embed_weight requires_grad=True (so new rows can learn),
    # the hook zeros out gradient rows 0..base_vocab_size before every step.
    # This ensures ONLY the new rows at indices base_vocab_size+ are updated.
    # Without this, the AdamW optimizer would update ALL embedding rows, slowly
    # corrupting the base vocab embeddings with gradients from our tiny corpus.
    def _zero_base_rows(grad: "torch.Tensor") -> "torch.Tensor":
        g = grad.clone()             # clone to avoid in-place modification on autograd tape
        g[:base_vocab_size] = 0.0   # zero gradients for original vocabulary rows
        return g

    hook_handle = embed_weight.register_hook(_zero_base_rows)
    _ok(f"Gradient hook registered — base {base_vocab_size:,} vocab rows are frozen")

    # AdamW with default betas (0.9, 0.999) — standard for embedding training.
    # The optimizer only tracks state for embed_weight (one tensor), so it's cheap.
    optimizer = torch.optim.AdamW([embed_weight], lr=lr)

    # ---- Pre-tokenize the entire corpus ----
    # Tokenize once upfront and keep as tensors on CPU.
    # We move to device batch-by-batch in the training loop to control GPU memory.
    # max_length=128: corpus sentences are short; 128 tokens captures all of them.
    # padding=True: pad to the longest sentence in the batch for batched forward pass.
    max_len = 128
    _ok(f"Tokenizing {len(sentences)} sentences (max_length={max_len}) ...")
    tokenized      = tokenizer(sentences, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_len)
    input_ids      = tokenized["input_ids"]       # shape: [N, max_len]
    attention_mask = tokenized["attention_mask"]  # shape: [N, max_len]; 1=real, 0=padding
    n_sentences    = input_ids.shape[0]

    n_batches = math.ceil(n_sentences / batch_size)
    total_steps = n_batches * epochs
    _ok(f"Training plan: {n_sentences} sentences x {epochs} epochs, "
        f"batch={batch_size} ({n_batches} steps/epoch, {total_steps} total)")
    _ok(f"Learning rate: {lr}  |  Gradient clip: 1.0  |  Optimizer: AdamW")
    print()

    # Initialize the narrator — it tracks history and emits insights live.
    # Also baseline a sample of new-token embeddings so we can measure how
    # far they have moved by the end of training.
    narrator = Narrator("tokenizer-corpus-train")
    initial_embeds = embed_weight[base_vocab_size:].detach().clone().cpu().float()

    t_train     = time.time()
    prev_epoch_loss = None  # for epoch-over-epoch comparison narration
    prev_sims       = None  # for cosine change interpretation

    global_step = 0

    for epoch in range(1, epochs + 1):
        t_epoch    = time.time()
        epoch_loss = 0.0
        model.train()

        # Shuffle sentence order each epoch so the model doesn't memorize
        # the corpus order — this improves generalization of the embeddings.
        perm           = torch.randperm(n_sentences)
        input_ids_shuf = input_ids[perm]
        mask_shuf      = attention_mask[perm]

        for b in range(n_batches):
            global_step += 1
            start = b * batch_size
            end   = min(start + batch_size, n_sentences)

            # Move batch to GPU (if available). Keeping data on CPU and moving
            # per-batch avoids OOM for large corpora on smaller GPUs.
            ids  = input_ids_shuf[start:end].to(device)
            mask = mask_shuf[start:end].to(device)

            # For causal LM loss, labels = input_ids shifted by 1.
            # The Transformers library handles the shift internally; we just
            # pass the same IDs as labels. We set -100 for padding positions
            # so the loss function (cross entropy) ignores them.
            labels = ids.clone()
            labels[mask == 0] = -100  # -100 = ignored index in CrossEntropyLoss

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss    = outputs.loss  # mean cross-entropy over non-ignored positions

            # Skip NaN/Inf losses (can happen with fp16 on overflow) — don't let
            # one bad batch corrupt the optimizer state.
            if torch.isnan(loss) or torch.isinf(loss):
                _warn(f"  Epoch {epoch} batch {b}: loss={loss.item()} — skipping bad batch")
                continue

            optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm to 1.0 to prevent large gradient updates
            # from destabilizing embeddings (common with fp16 and high lr)
            grad_norm_val = torch.nn.utils.clip_grad_norm_(
                [embed_weight], max_norm=1.0
            ).item()
            optimizer.step()
            # The gradient hook fires here, zeroing base vocab rows before
            # AdamW applies the update — so only new rows actually change.

            loss_val    = loss.item()
            epoch_loss += loss_val

            # ---- LIVE NARRATION ----
            # Tell the user what just happened in plain English. The narrator
            # rate-limits itself so the terminal isn't flooded.
            narrator.step(global_step, total_steps, loss_val,
                          grad_norm=grad_norm_val, lr=lr)

            # Log every step at the structured level (machine-readable)
            _log({
                "phase":     "corpus_train",
                "event":     "step",
                "epoch":     epoch,
                "step":      global_step,
                "loss":      loss_val,
                "grad_norm": grad_norm_val,
            })

        # ---- End of epoch: compute stats and probes ----
        avg_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")

        # Check how much the new embeddings have moved (norm = length of vector).
        # Growing norms mean embeddings are building stronger representations.
        # Flat norms mean the lr might be too low or the corpus is too repetitive.
        new_embeds    = embed_weight[base_vocab_size:].detach()
        new_norm_mean = new_embeds.norm(dim=1).mean().item()
        new_norm_std  = new_embeds.norm(dim=1).std().item()

        # How far have the new embeddings drifted from their smart-init values?
        # This shows actual learning happened (drift > 0) vs no learning (drift ≈ 0).
        drift = (new_embeds.cpu().float() - initial_embeds).norm(dim=1).mean().item()

        # Cosine similarity probes — track whether semantic clusters are forming
        model.eval()
        sim_epoch = _cosine_probe(tokenizer, embed_weight.detach())
        model.train()

        # Compare to base norms for the narrator's health interpretation
        base_norm_mean = embed_weight[:base_vocab_size].detach().norm(dim=1).mean().item()

        _ok(f"Epoch {epoch}/{epochs} stats  "
            f"loss={avg_loss:.4f}  "
            f"new_norm={new_norm_mean:.4f}+-{new_norm_std:.4f}  "
            f"drift_from_init={drift:.4f}  "
            f"time={_elapsed(t_epoch)}")
        _print_cosine_table(sim_epoch, label=f"  cosine similarities after epoch {epoch}")
        _interpret_cosine_table(sim_epoch, phase_label=f"epoch {epoch}/{epochs}")

        # Per-epoch narrator commentary
        narrator.epoch_end(epoch, epochs, avg_loss,
                           prev_loss=prev_epoch_loss,
                           new_norm_mean=new_norm_mean,
                           base_norm_mean=base_norm_mean)

        # Cosine change narration vs previous epoch
        if prev_sims is not None:
            changes = [
                (desc, prev_sims.get(desc), sim_epoch.get(desc))
                for desc in sim_epoch
            ]
            narrator.cosine_change(changes)

        prev_epoch_loss = avg_loss
        prev_sims       = sim_epoch

        _log({
            "phase":           "corpus_train",
            "event":           "epoch_end",
            "epoch":           epoch,
            "loss":            avg_loss,
            "new_norm_mean":   new_norm_mean,
            "new_norm_std":    new_norm_std,
            "drift_from_init": drift,
            "cosine_probes":   sim_epoch,
        })

    # Remove the gradient hook — good practice to avoid memory leaks if the
    # model is reused elsewhere in the same process.
    hook_handle.remove()
    _ok(f"Corpus training complete in {_elapsed(t_train)}")

    # Final summary of what the model has learned in this phase.
    print(f"\n  [SUMMARY] Tokenizer warm-up complete:")
    print(f"  [SUMMARY]   - {len(initial_embeds)} new token embeddings trained")
    print(f"  [SUMMARY]   - Base vocab ({base_vocab_size:,} embeddings) untouched")
    print(f"  [SUMMARY]   - Total trainable rows updated: {len(initial_embeds)} of "
          f"{embed_weight.shape[0]:,}")
    print(f"  [SUMMARY] What this means for finetune.py:")
    print(f"  [SUMMARY]   - Tool-name tokens now have meaningful starting embeddings")
    print(f"  [SUMMARY]   - LoRA training won't waste steps learning what tokens 'mean'")
    print(f"  [SUMMARY]   - Adapter can focus on routing logic from step 1\n")


# ===========================================================================
# Phase 4: Final report and save
# ===========================================================================

def phase_save(
    tokenizer,
    model,
    base_vocab_size: int,
    new_token_strings: list[str],
    n_neighbors: int,
) -> None:
    """
    Print a final embedding health report and save the extended tokenizer.

    Saved outputs (in training/tokenizer_extended/):
      tokenizer_config.json     — tokenizer type + special tokens
      tokenizer.model           — SentencePiece binary model
      special_tokens_map.json   — BOS/EOS/UNK/PAD mappings
      added_tokens.json         — the 319 new tokens we added
      embed_init.pt             — pre-trained embeddings for new tokens only
                                  (NOT the full model — just the new rows)
      train_log.jsonl           — structured telemetry from all phases

    finetune.py looks for tokenizer_extended/ at startup and uses it instead of
    the base tokenizer. It also loads embed_init.pt to restore the warm
    embeddings into the model's embedding table before LoRA training begins.
    """
    _banner("PHASE 4: Final report + save")

    try:
        import torch
    except ImportError:
        sys.exit(1)

    # Detach from autograd graph — we only need the values, not gradients
    embed_weight: "torch.Tensor" = model.model.embed_tokens.weight.detach()

    # ---- Final embedding health check ----
    # Compare new token norms to base vocab norms.
    # They should be within ~20% of each other after training.
    # If still much smaller, the lr was too low or corpus too small.
    new_embeds     = embed_weight[base_vocab_size: base_vocab_size + len(new_token_strings)]
    base_embeds    = embed_weight[:base_vocab_size]
    new_norm_mean  = new_embeds.norm(dim=1).mean().item()
    base_norm_mean = base_embeds.norm(dim=1).mean().item()
    ratio          = new_norm_mean / base_norm_mean if base_norm_mean > 0 else 0.0

    _ok(f"Final embedding norms — new: {new_norm_mean:.4f}  base: {base_norm_mean:.4f}  "
        f"ratio: {ratio:.2f}")
    if ratio < 0.5:
        _warn("New token norms are <50% of base vocab — embeddings may be too weak. "
              "Consider running more epochs or increasing --lr.")
    elif ratio > 2.0:
        _warn("New token norms are >2x base vocab — may cause instability during LoRA "
              "training. Consider reducing --lr or --epochs.")

    # Final cosine similarity table — shows the fully-trained embedding geometry
    sim_final = _cosine_probe(tokenizer, embed_weight)
    _print_cosine_table(sim_final, label="Final cosine similarities (post-training)")

    # ---- Nearest base-vocab neighbors ----
    # This shows what the model "thinks" each new token is similar to.
    # Good example: 'linux_memory_usage' → neighbors should include 'memory', 'RAM', 'usage'
    # Bad example:  'linux_memory_usage' → neighbors include 'blue', 'tooth' (wrong subwords)
    if n_neighbors > 0:
        _report_nearest_neighbors(
            tokenizer, embed_weight, base_vocab_size, new_token_strings, n_neighbors
        )

    # ---- Save tokenizer ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    _ok(f"Tokenizer saved to {OUTPUT_DIR}/")

    # ---- Save only the new embedding rows, not the full model ----
    # The full model is 270M params (~540 MB in fp16). We only need the
    # 319 new rows (~319 x 640 x 2 bytes = ~410 KB). finetune.py loads
    # the full base model and then writes just these rows into it.
    new_embed_rows = embed_weight[base_vocab_size:
                                  base_vocab_size + len(new_token_strings)].clone()
    embed_path = OUTPUT_DIR / "embed_init.pt"
    torch.save({
        "new_token_strings": new_token_strings,    # for validation at load time
        "base_vocab_size":   base_vocab_size,      # row offset to write to
        "embeddings":        new_embed_rows,        # shape: [n_new_tokens, hidden_size]
    }, str(embed_path))
    size_kb = embed_path.stat().st_size / 1e3
    _ok(f"Embeddings saved to {embed_path.name} ({size_kb:.0f} KB) — "
        f"shape: {list(new_embed_rows.shape)}")

    _log({
        "phase":           "save",
        "output_dir":      str(OUTPUT_DIR),
        "n_new_tokens":    len(new_token_strings),
        "final_norm_new":  new_norm_mean,
        "final_norm_base": base_norm_mean,
        "norm_ratio":      ratio,
        "cosine_probes":   sim_final,
    })

    # Write the complete telemetry log in one shot at the end
    _flush_log()
    _ok(f"Training log written to {LOG_FILE}")

    print(f"\n  {'='*60}")
    print(f"  Tokenizer extended saved to:  {OUTPUT_DIR}")
    print(f"  finetune.py will auto-load this when --mode train runs.")
    print(f"  Run next: python training/finetune.py --mode all")
    print(f"  {'='*60}")


# ===========================================================================
# Probe helpers — used across phases to track embedding quality
# ===========================================================================

def _cosine_probe(tokenizer, embed_weight: "torch.Tensor") -> dict:
    """
    Compute cosine similarity for every pair in PROBE_PAIRS.

    Cosine similarity measures the angle between two embedding vectors:
      cos(theta) = dot(a, b) / (|a| * |b|)
    It ignores magnitude and focuses on direction, which is what matters
    for meaning in embedding space.

    Returns: {description: similarity_score | None}
    None means one of the tokens wasn't found in the vocabulary.
    """
    import torch
    import torch.nn.functional as F

    results = {}
    for token_a, token_b, desc in PROBE_PAIRS:
        id_a = tokenizer.convert_tokens_to_ids(token_a)
        id_b = tokenizer.convert_tokens_to_ids(token_b)
        unk  = tokenizer.unk_token_id

        # If either token maps to UNK, it's not in the vocabulary
        if id_a == unk or id_b == unk:
            results[desc] = None
            continue
        # Guard against out-of-range indices (shouldn't happen after resize)
        if id_a >= embed_weight.shape[0] or id_b >= embed_weight.shape[0]:
            results[desc] = None
            continue

        # Upcast to float32 for accurate similarity — fp16 precision is fine
        # for training but cosine_similarity is more accurate in float32
        ea  = embed_weight[id_a].float()
        eb  = embed_weight[id_b].float()
        sim = F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
        results[desc] = round(sim, 4)

    return results


def _print_cosine_table(sims: dict, label: str = "") -> None:
    """
    Print cosine similarities as a bar chart for easy visual scanning.

    Bar chart mapping: similarity in [-1, 1] → bar width in [0, 20] blocks
      Full bar (20 blocks) = +1.0 (identical direction)
      Half bar (10 blocks) = 0.0 (orthogonal)
      Empty bar (0 blocks) = -1.0 (opposite direction)
    """
    if label:
        print(f"\n  {label}:")
    for desc, sim in sims.items():
        if sim is None:
            bar = "(token not in vocabulary)"
        else:
            # Map [-1, 1] to [0, 20] for block count
            filled = int((sim + 1) / 2 * 20)
            bar    = "█" * filled + "░" * (20 - filled)
            bar    = f"{sim:+.4f}  {bar}"
        print(f"    {desc:40s} {bar}")


def _interpret_cosine_table(sims: dict, phase_label: str = "") -> None:
    """
    Print plain-English interpretation of the cosine probe results.

    The point: numbers alone don't tell you if training is healthy.
    "0.42" means nothing without context — but "0.42 between two same-tool
    forms" means the model is starting to recognize them as related.
    """
    same_tool   = []  # values for "same-tool" pairs
    cross       = []  # values for cross-domain pairs
    for desc, sim in sims.items():
        if sim is None:
            continue
        if "same-tool" in desc:
            same_tool.append(sim)
        elif "cross-domain" in desc:
            cross.append(sim)

    print(f"  [INSIGHT] How to read these numbers ({phase_label}):")

    if same_tool:
        avg = sum(same_tool) / len(same_tool)
        if avg < 0.3:
            print(f"    same-tool forms avg sim = {avg:+.2f} — WEAK clustering. "
                  f"Different naming variants (e.g. 'memory_usage' vs 'linux_memory_usage')")
            print(f"    are still seen as unrelated. More corpus training will help.")
        elif avg < 0.6:
            print(f"    same-tool forms avg sim = {avg:+.2f} — moderate clustering. "
                  f"Model is starting to see naming variants as related concepts.")
        else:
            print(f"    same-tool forms avg sim = {avg:+.2f} — STRONG clustering. "
                  f"Model treats naming variants as essentially the same concept.")

    if cross:
        avg = sum(cross) / len(cross)
        if avg > 0.5:
            print(f"    cross-domain avg sim = {avg:+.2f} — DANGER: unrelated tools "
                  f"are too similar. Embeddings may be collapsing into one cluster.")
        elif avg > 0.3:
            print(f"    cross-domain avg sim = {avg:+.2f} — slight bleed; acceptable.")
        else:
            print(f"    cross-domain avg sim = {avg:+.2f} — clean separation. "
                  f"Different tool categories are well-separated in embedding space.")
    print()


def _report_nearest_neighbors(
    tokenizer,
    embed_weight: "torch.Tensor",
    base_vocab_size: int,
    new_token_strings: list[str],
    k: int,
) -> None:
    """
    For each new token, find the k most similar base-vocab tokens by cosine sim.

    This reveals what the model has learned to associate with each new token.
    Good output: 'linux_memory_usage' → 'memory', 'RAM', 'usage', 'free', 'MiB'
    Bad output:  'linux_memory_usage' → 'blue', 'tooth', 'dental' (wrong subwords)

    We cap at 40 tokens to keep the output readable; full results are in train_log.jsonl.
    """
    import torch
    import torch.nn.functional as F

    print(f"\n  Nearest {k} base-vocab neighbors for each new token:")
    print(f"  (shows what concepts the model associates with each new term)\n")

    # Pre-normalize all base vocab embeddings — cosine sim then reduces to
    # a simple dot product, which is much faster to compute in batch.
    base_embeds = embed_weight[:base_vocab_size].float()  # [base_vocab, hidden]
    base_norms  = F.normalize(base_embeds, dim=1)         # [base_vocab, hidden], unit vectors

    shown = 0
    for token_str in new_token_strings:
        if shown >= 40:  # cap output at 40 tokens to avoid flooding the terminal
            remaining = len(new_token_strings) - shown
            print(f"    ... ({remaining} more tokens in train_log.jsonl)")
            break

        new_id = tokenizer.convert_tokens_to_ids(token_str)
        if new_id is None or new_id >= embed_weight.shape[0]:
            continue

        # Normalize this token's embedding and compute similarity to all base tokens
        new_vec  = embed_weight[new_id].float().unsqueeze(0)  # [1, hidden]
        new_norm = F.normalize(new_vec, dim=1)                 # [1, hidden]
        sims     = (new_norm @ base_norms.T).squeeze(0)       # [base_vocab] dot products

        top_vals, top_ids = sims.topk(k)
        neighbors = [tokenizer.convert_ids_to_tokens(i.item()) for i in top_ids]

        # Format as: 'token' → 'neighbor1'(0.92) | 'neighbor2'(0.88) | ...
        pairs = " | ".join(f"{n!r}({v:.3f})"
                           for n, v in zip(neighbors, top_vals.tolist()))
        print(f"    {token_str:35s} -> {pairs}")
        shown += 1

    _log({
        "phase":              "nearest_neighbors",
        "n_tokens_reported":  shown,
        "k":                  k,
    })


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Corpus training epochs (default: 2). "
             "2 epochs is usually enough for convergence on a 3800-sentence corpus. "
             "Use 3+ if same-tool cosine similarities stay below 0.5 after training.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Learning rate for embedding training (default: 5e-4). "
             "Higher than LoRA lr because embeddings start from scratch. "
             "Reduce to 1e-4 if loss oscillates; increase to 1e-3 if loss barely moves.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Sentences per gradient step (default: 16). "
             "Increase for GPU (32-64 on 16GB VRAM). Reduce if OOM.",
    )
    parser.add_argument(
        "--skip-corpus", action="store_true",
        help="Skip Phase 3 corpus training; only do smart init (Phase 1+2). "
             "Use when you want fast initialization without full warm-up. "
             "Embeddings will be directionally correct but not yet converged.",
    )
    parser.add_argument(
        "--neighbors", type=int, default=5,
        help="Nearest base-vocab neighbors to show per new token (default: 5). "
             "Set to 0 to skip the neighbor report entirely.",
    )
    args = parser.parse_args()

    t_total = time.time()

    print(f"\nfunctiongemma tokenizer extension")
    print(f"  Input tokens:  {NEW_TOKENS_JSON}")
    print(f"  Input corpus:  {CORPUS_FILE}")
    print(f"  Output dir:    {OUTPUT_DIR}")
    print(f"  Corpus epochs: {args.epochs if not args.skip_corpus else 'skipped (--skip-corpus)'}")
    print(f"  LR:            {args.lr}")
    print(f"  Batch size:    {args.batch_size}")

    # --- Phase 1: add the 319 domain tokens to the tokenizer ---
    new_token_strings, tokenizer = phase_add_tokens()
    # base_vocab_size = extended size minus the tokens we just added
    base_vocab_size = len(tokenizer) - len(new_token_strings)

    # --- Phase 2: smart init — load model, initialize new embedding rows ---
    model, embed_weight, base_vocab_size = phase_smart_init(
        tokenizer, new_token_strings, base_vocab_size
    )

    # --- Phase 3: corpus warm-up (skip with --skip-corpus for quick testing) ---
    if not args.skip_corpus:
        phase_corpus_train(
            model, tokenizer, embed_weight, base_vocab_size,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        )
    else:
        _ok("Skipping corpus training (--skip-corpus flag set)")

    # --- Phase 4: final report + save everything to tokenizer_extended/ ---
    phase_save(
        tokenizer, model, base_vocab_size,
        new_token_strings, n_neighbors=args.neighbors,
    )

    _ok(f"Total elapsed: {_elapsed(t_total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
