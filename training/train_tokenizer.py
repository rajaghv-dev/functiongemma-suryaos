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


def _get_hf_token() -> Optional[str]:
    """
    Resolve the HuggingFace auth token from any standard location.

    Gemma 3 is a GATED model — you must accept the license at
    https://huggingface.co/google/gemma-3-270m-it before downloading.
    Once accepted, the Hub requires an authenticated request.

    We check, in order:
      1. HF_TOKEN              env var (preferred — set this in your shell)
      2. HUGGINGFACE_HUB_TOKEN  env var (legacy name; transformers also reads this)
      3. ~/.cache/huggingface/token  (written by `huggingface-cli login`)

    Returns the token string, or None if no token found anywhere.
    """
    # 1+2: environment variables
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if tok:
        return tok.strip()

    # 3: token file written by huggingface-cli login
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        try:
            t = token_file.read_text().strip()
            if t:
                return t
        except Exception:
            pass

    return None


def _print_hf_auth_help() -> None:
    """Print exact commands to set up HF authentication for gated models."""
    print()
    _err("Gemma 3 is a GATED model — needs HuggingFace authentication.")
    print()
    print("  Fix in 2 steps:")
    print("    1. Accept the license (one-time, in browser):")
    print("       https://huggingface.co/google/gemma-3-270m-it")
    print()
    print("    2. Set your token (pick ONE of these):")
    print("       a) export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx")
    print("          (then re-run this script in the same shell)")
    print()
    print("       b) huggingface-cli login")
    print("          (one-time; writes ~/.cache/huggingface/token)")
    print()
    print("    Get your token at: https://huggingface.co/settings/tokens")
    print("    (any 'read' scope token works)")
    print()

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
    # ── Same-tool forms — naming variants of one tool (HIGH expected) ──
    ("linux_memory_usage",    "memory_usage",          "same-tool forms"),
    ("linux_disk_usage",      "disk_usage",            "same-tool forms"),

    # ── Tool ↔ CLI/KDE equivalent (MODERATE expected) ──
    # NOTE: ⚠ BUG-005 fix applied here. The OLD probe used
    #   `service_status` / `krunner_launch` (without prefix) — those are now
    #   filtered OUT by build_tokenizer_dataset.py because they were generic
    #   single-tokens in base Gemma. We use the prefixed forms which ARE
    #   trainable new tokens. The CLI side (systemctl, KRunner) is base-vocab
    #   and frozen, but the tool side moves during training — so the probe
    #   is now ALIVE (cosine can change as the tool token learns).
    ("linux_service_status",  "systemctl",             "tool vs CLI equiv"),
    ("kde_krunner_launch",    "KRunner",               "tool vs KDE component"),

    # ── Sibling tools — same category, different routing (MODERATE) ──
    ("linux_memory_usage",    "linux_disk_usage",      "sibling linux tools"),
    ("linux_metrics_summary", "linux_memory_usage",    "metrics vs memory"),

    # ── Cross-domain — different categories (LOW expected) ──
    ("linux_memory_usage",    "linux_brightness_set",  "cross-domain (expected low)"),

    # ── BUG-005 REPLACEMENTS for the dead frozen-token probes ──
    # Old: ("torch","transformers"), ("merge","commit"), ("GGUF","ollama")
    # All three pairs had BOTH tokens in base vocab — frozen by gradient hook,
    # cosine couldn't move during training. Replaced with new-token pairs
    # that genuinely test what training is doing.
    ("linux_memory_usage",    "memory",                "new tool vs base concept"),
    ("kde_window_focus",      "kde_krunner_launch",    "kde sibling tools"),
    ("kde_dialog_confirm",    "linux_battery_status",  "cross-category kde vs linux"),
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

                # Concise messages that fit in ~80 columns; details on a
                # second line so users on narrow terminals can still read them.
                if pct > 30:
                    self._emit("PROGRESS",
                        f"Step {step}: loss -{pct:.0f}% in 10 steps  "
                        f"({older_avg:.2f}→{recent_avg:.2f}) — rapid learning")
                    self.last_emit_step = step
                elif pct > 10:
                    self._emit("PROGRESS",
                        f"Step {step}: loss -{pct:.0f}% — steady convergence")
                    self.last_emit_step = step
                elif abs(pct) < 2:
                    self._emit("PLATEAU",
                        f"Step {step}: loss {recent_avg:.3f} (±2%) — converged")
                    self.last_emit_step = step
                elif pct < -10:
                    self._emit("WARN",
                        f"Step {step}: loss +{-pct:.0f}% "
                        f"({older_avg:.2f}→{recent_avg:.2f}) — lower lr?")
                    self.last_emit_step = step

        # ---- Gradient stability ----
        # Rate-limit the warnings: only emit every 20 steps once we're past
        # the early-warmup phase (step > 30). Constant clipping above 1.0 is
        # NORMAL during corpus warm-up — embeddings are at random init and
        # the gradient is large until they settle. The user doesn't need
        # 200 lines of "clipped at 1.0" warnings.
        if grad_norm is not None:
            if grad_norm > 5.0:
                # Big spike — always emit, this matters
                self._emit("WARN", f"Step {step}: grad_norm={grad_norm:.1f} (clipped)")
            elif grad_norm > 1.5 and step <= 30:
                # First 30 steps — warn once or twice during cold start
                if step % 5 == 0:
                    self._emit("WARN",
                        f"Step {step}: grad_norm={grad_norm:.1f} "
                        f"(early-step clipping is normal)")
            # Beyond step 30 with grad < 5.0: stay silent — clipping is fine

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
    using_local = MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir())
    model_path  = str(MODEL_HF_DIR) if using_local else HF_MODEL_ID

    # Hub access requires a token for gated Gemma. Local files don't.
    hf_token = None if using_local else _get_hf_token()
    if not using_local:
        if hf_token:
            _ok(f"HuggingFace token found (length={len(hf_token)}) — authenticated download")
        else:
            _print_hf_auth_help()
            sys.exit(1)

    _ok(f"Loading tokenizer from {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
            token=hf_token,  # None for local files; explicit token for Hub
        )
    except Exception as e:
        _err(f"Tokenizer load failed: {e}")
        # Friendly hint when the failure looks like an auth problem
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            _print_hf_auth_help()
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

    using_local = MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir())
    model_path  = str(MODEL_HF_DIR) if using_local else HF_MODEL_ID
    hf_token    = None if using_local else _get_hf_token()  # gated-model auth

    # ---- BUG FIX (L13 postmortem) ----
    # Load a SECOND tokenizer instance — clean, with no added tokens.
    # Why: the `tokenizer` we received already had add_tokens() called on it.
    # Calling `tokenizer.encode("linux_memory_usage")` on it returns the new
    # token ID (e.g. 262148) because the added-tokens trie short-circuits the
    # subword splitter. We then filter `< base_vocab_size` and get an empty
    # list, falling back to the global mean for every token. Result: 251
    # identical embeddings, training collapses.
    #
    # The clean instance still has the original SentencePiece subword splitter,
    # so encode("linux_memory_usage") returns ["linux", "_", "memory", "_",
    # "usage"] (or whatever subwords the base vocab uses). We average THOSE.
    from transformers import AutoTokenizer
    _ok("Loading clean base tokenizer (for subword decomposition) ...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, token=hf_token,
    )

    # ── Detect GPU/torch mismatch (silent CPU fallback) ──────────────────
    # If nvidia-smi shows a GPU but torch.cuda.is_available() is False, the
    # CPU torch wheel was installed by mistake. Warn loudly with the fix
    # command — silent CPU fallback wastes hours of training time.
    if not torch.cuda.is_available():
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0 and r.stdout.strip():
                gpu_name_smi = r.stdout.strip().splitlines()[0]
                _warn("=" * 60)
                _warn("GPU/torch MISMATCH DETECTED")
                _warn("=" * 60)
                _warn(f"  nvidia-smi sees:  {gpu_name_smi}")
                _warn(f"  torch sees:       CPU only (torch={torch.__version__})")
                _warn("  This means a CPU PyTorch wheel is installed.")
                _warn("  Training WILL run on CPU — 8x slower than GPU.")
                _warn("")
                _warn("  Fix:  bash training/bootstrap.sh --reinstall")
                _warn("        (will swap CPU torch for cu121 GPU wheel)")
                _warn("=" * 60)
        except Exception:
            pass  # nvidia-smi not present → real CPU machine, no warning needed

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
        # BUG-006 FIX: use bf16 for ANY Ampere+ GPU, not just datacenter cards.
        # fp16's 5-bit exponent (max ~65504) overflows during attention softmax,
        # producing NaN on the first batch. bf16 has the same 8-bit exponent
        # as fp32, so gradients never overflow. RTX 30xx/40xx have hardware
        # bf16 tensor cores. See docs/bug-fixes.md BUG-006.
        if compute[0] >= 8:
            dtype = torch.bfloat16  # Ampere or newer → bf16 (stable)
        elif compute[0] >= 7:
            dtype = torch.float16   # Turing/Volta only → fp16 (no bf16 hw)
        else:
            dtype = torch.float32   # very old GPU → fp32 fallback
        _ok(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM, compute {compute[0]}.{compute[1]}) "
            f"— dtype={dtype}")
    else:
        dtype = torch.float32  # float32 required for numerical stability on CPU
        _ok("No GPU detected — using CPU + float32 (expect 1-3 min load time)")

    _ok(f"Loading model from {model_path} (full 270M weights needed for embeddings) ...")
    t0 = time.time()

    # low_cpu_mem_usage=True: loads tensors one-by-one instead of all at once,
    # cutting peak RAM from ~4 GB to ~2 GB during the load phase.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            token=hf_token,  # None for local files; explicit token for gated Hub model
        )
    except Exception as e:
        _err(f"Model load failed: {e}")
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            _print_hf_auth_help()
        sys.exit(1)
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

        # Tokenize using the CLEAN base tokenizer (loaded above) which doesn't
        # have the new tokens in its added-tokens trie. This forces the
        # SentencePiece subword splitter to actually run.
        # e.g. 'linux_memory_usage' → ['linux', '_', 'memory', '_', 'usage']
        #                          → [12345, 432, 6789, 432, 4567]
        # All resulting IDs are guaranteed < base_vocab_size by construction.
        subword_ids = base_tokenizer.encode(token_str, add_special_tokens=False)
        # Filter is now a no-op safety check (all should already be < base_vocab_size)
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
        # Use the CLEAN base tokenizer here too — same reason as the bug fix above
        sub_ids   = base_tokenizer.encode(ts, add_special_tokens=False)
        base_only = [sid for sid in sub_ids if sid < base_vocab_size]
        pieces    = [base_tokenizer.decode([i]) for i in base_only]
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

    # Initialize Prometheus Pushgateway pusher (silent no-op if not running).
    # All pushes are best-effort — never block training on a network call.
    try:
        from .metrics import MetricsPusher
    except ImportError:
        # Direct script invocation (no package context) — fall back to local import
        sys.path.insert(0, str(TRAINING_DIR))
        from metrics import MetricsPusher
    pusher = MetricsPusher(phase="tokenizer_corpus_train")

    t_train     = time.time()
    prev_epoch_loss = None  # for epoch-over-epoch comparison narration
    prev_sims       = None  # for cosine change interpretation

    global_step = 0
    consecutive_nans = 0  # early-abort counter (BUG-006 protection)

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
            # one bad batch corrupt the optimizer state. After 10 consecutive
            # NaN batches, abort early — the run is broken and continuing
            # wastes time. (Common cause: fp16 dtype on a model that needs bf16
            # — see BUG-006.)
            if torch.isnan(loss) or torch.isinf(loss):
                _warn(f"  Epoch {epoch} batch {b}: loss={loss.item()} — skipping bad batch")
                consecutive_nans = consecutive_nans + 1
                if consecutive_nans >= 10:
                    _err("=" * 60)
                    _err("ABORTING — 10 consecutive NaN batches.")
                    _err("=" * 60)
                    _err("This means the loss is exploding on every batch.")
                    _err("Most likely cause: fp16 numerical instability.")
                    _err(f"  Current dtype: {dtype}")
                    _err(f"  GPU compute capability: {compute if device == 'cuda' else 'N/A'}")
                    _err("")
                    _err("Fix:")
                    _err("  1. Make sure you have the latest code (BUG-006 fix):")
                    _err("       git pull")
                    _err("  2. Re-run after pulling — the dtype selection now")
                    _err("     defaults to bf16 on Ampere+ GPUs (RTX 30xx/40xx),")
                    _err("     which has 8-bit exponent like fp32 (no overflow).")
                    _err("  3. If the issue persists, try CPU mode:")
                    _err("     bash training/bootstrap.sh --with-cpu-fallback")
                    _err("     .fngemma-suryaos-cpu/bin/python training/train_tokenizer.py")
                    sys.exit(1)
                continue
            else:
                consecutive_nans = 0  # reset counter on a good batch

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

            # Push to Pushgateway every 5 steps to avoid spamming the network.
            # Prometheus scrapes Pushgateway every 5s anyway, so finer pushes
            # would get coalesced. 5 steps is a good balance of resolution and cost.
            if global_step % 5 == 0:
                pusher.push_step(
                    step=global_step, loss=loss_val,
                    lr=lr, grad_norm=grad_norm_val,
                    epoch=epoch + (b / n_batches),  # fractional epoch
                )

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
        _print_cosine_table(sim_epoch,
                            label=f"  cosine similarities after epoch {epoch}",
                            prev_sims=prev_sims)
        _interpret_cosine_table(sim_epoch,
                                phase_label=f"epoch {epoch}/{epochs}",
                                prev_sims=prev_sims)

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

        # Push tokenizer-phase epoch metrics to Pushgateway / Grafana
        pusher.push_cosine_probes(epoch=epoch, sims=sim_epoch)
        norm_ratio = new_norm_mean / base_norm_mean if base_norm_mean > 0 else 0.0
        pusher.push_norm_ratio(ratio=norm_ratio, drift=drift)

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


# Per-probe target ranges (low, high) sourced from goals.md.
# This is the canonical truth for what each probe should converge to.
PROBE_TARGETS: dict[str, tuple[float, float]] = {
    "same-tool forms":              (0.50, 0.80),  # cluster but don't collapse
    "tool vs CLI equiv":            (0.40, 0.70),  # related, not identical
    "tool vs KDE component":        (0.40, 0.70),  # related, not identical
    "sibling linux tools":          (0.30, 0.50),  # same domain, distinct
    "metrics vs memory":            (0.40, 0.60),  # umbrella vs specific
    "cross-domain (expected low)":  (0.00, 0.30),  # MUST separate
    "new tool vs base concept":     (0.40, 0.70),  # tool ↔ base concept
    "kde sibling tools":            (0.30, 0.50),  # KDE siblings
    "cross-category kde vs linux":  (0.00, 0.30),  # MUST separate
}


def _status_for(desc: str, sim: float) -> tuple[str, str]:
    """
    Returns (status_tag, hint_text) for a probe value vs its goal range.
    status_tag: ✓ HIT | ⚠ HIGH | ⚠ LOW  | ✗ HIGH | ✗ LOW
    """
    if desc not in PROBE_TARGETS:
        return ("?", "")
    lo, hi = PROBE_TARGETS[desc]
    if lo <= sim <= hi:
        return ("✓ HIT ", "in band")
    if sim > hi:
        gap = sim - hi
        if gap < 0.10:
            return ("⚠ HIGH", f"slightly above (need -{gap:.2f})")
        return ("✗ HIGH", f"way above (need -{gap:.2f}; clusters too tight)")
    # sim < lo
    gap = lo - sim
    if gap < 0.10:
        return ("⚠ LOW ", f"slightly below (need +{gap:.2f})")
    return ("✗ LOW ", f"way below (need +{gap:.2f}; not clustering)")


def _print_cosine_table(sims: dict, label: str = "",
                         prev_sims: Optional[dict] = None) -> None:
    """
    Print cosine probes with target band, current value, status, and Δ vs
    previous epoch.

    Layout (≤ 80 cols):
      probe_name          current  target     status   Δ_vs_prev  bar
    """
    if label:
        print(f"\n  {label}:")
    print(f"    {'probe':35s} {'curr':>5s}  {'target':<9s}  {'status':<8s} {'Δ':>6s}  bar")
    for desc, sim in sims.items():
        if sim is None:
            print(f"    {desc:35s}  (token not in vocab)")
            continue

        # Bar with goal band overlaid (▓ for goal range, ● for current)
        if desc in PROBE_TARGETS:
            lo, hi = PROBE_TARGETS[desc]
            target_str = f"{lo:.2f}-{hi:.2f}"
        else:
            lo, hi = -1.0, 1.0
            target_str = "any"

        # 20-char bar from -1..1; show goal band as ▓, others as ░, current as ●
        bar = list("░" * 20)
        lo_idx = max(0, int((lo + 1) / 2 * 20))
        hi_idx = min(20, int((hi + 1) / 2 * 20))
        for i in range(lo_idx, hi_idx):
            bar[i] = "▓"
        cur_idx = max(0, min(19, int((sim + 1) / 2 * 20)))
        bar[cur_idx] = "●"

        # Status + delta vs previous epoch
        status, _ = _status_for(desc, sim)
        if prev_sims is not None and prev_sims.get(desc) is not None:
            delta = sim - prev_sims[desc]
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "  —  "

        print(f"    {desc:35s} {sim:+.2f}  {target_str:<9s}  {status:<8s} {delta_str:>6s}  {''.join(bar)}")


def _interpret_cosine_table(sims: dict, phase_label: str = "",
                              prev_sims: Optional[dict] = None) -> None:
    """
    Per-probe insights with direction-of-travel commentary.

    For each probe, we say:
      - is it in target?
      - is it heading toward or away from target?
      - what does the user need to know about why?
    """
    print(f"\n  [INSIGHT] What the numbers tell us ({phase_label}):")

    aggregates = {"hit": 0, "high": 0, "low": 0, "?": 0}
    headed_toward = 0
    headed_away   = 0

    for desc, sim in sims.items():
        if sim is None or desc not in PROBE_TARGETS:
            continue

        status, hint = _status_for(desc, sim)
        if "HIT" in status:
            aggregates["hit"] += 1
            continue
        elif "HIGH" in status:
            aggregates["high"] += 1
        elif "LOW" in status:
            aggregates["low"] += 1

        # Direction of travel (only if we have a previous reading)
        direction = ""
        if prev_sims is not None and prev_sims.get(desc) is not None:
            delta = sim - prev_sims[desc]
            lo, hi = PROBE_TARGETS[desc]
            target_mid = (lo + hi) / 2
            # Positive movement = toward target if we're below; away if we're above
            if sim < lo and delta > 0.005:
                direction = " ↑ heading toward target"; headed_toward += 1
            elif sim > hi and delta < -0.005:
                direction = " ↓ heading toward target"; headed_toward += 1
            elif sim < lo and delta < -0.005:
                direction = " ↓ moving AWAY (worse)";   headed_away   += 1
            elif sim > hi and delta > 0.005:
                direction = " ↑ moving AWAY (worse)";   headed_away   += 1
            else:
                direction = " — stable"

        # Interpretation hint
        if "cross-domain" in desc or "cross-category" in desc:
            if "HIGH" in status:
                msg = "DANGER: unrelated tools collapsing together — corpus needs more contrastive examples"
            else:
                msg = "OK: tools well-separated in embedding space"
        elif "sibling" in desc:
            if "HIGH" in status:
                msg = "siblings indistinguishable — model can't route between them; add A3 contrastive lines"
            elif "LOW" in status:
                msg = "siblings too dispersed — share more contextual co-occurrence"
            else:
                msg = "siblings related-but-distinct — healthy"
        elif "same-tool" in desc:
            if "HIGH" in status:
                msg = "naming variants are clones — consider differentiating arg shapes per variant"
            elif "LOW" in status:
                msg = "naming variants seen as unrelated — add 'X and Y are the same tool' lines"
            else:
                msg = "variants cluster correctly"
        else:
            msg = hint

        print(f"    {desc:35s} {status} {direction}")
        print(f"      └─ {msg}")

    # Summary line
    total_targeted = sum(aggregates.values())
    print(f"\n  [SUMMARY] {aggregates['hit']}/{total_targeted} probes IN BAND  "
          f"|  {aggregates['high']} too high  |  {aggregates['low']} too low")
    if prev_sims is not None and (headed_toward + headed_away) > 0:
        print(f"            Direction: {headed_toward} heading toward target, "
              f"{headed_away} moving away")
        if headed_away > headed_toward:
            print(f"            ⚠ More probes moving AWAY than toward — training may be regressing.")
            print(f"              Consider stopping early if this persists.")
        elif headed_toward > 0:
            print(f"            ✓ Net progress toward goals — keep training.")

    # Top 1-2 actions
    top_offenders = sorted(
        [(d, s) for d, s in sims.items()
         if s is not None and d in PROBE_TARGETS
         and "HIT" not in _status_for(d, s)[0]],
        key=lambda x: abs(x[1] - sum(PROBE_TARGETS[x[0]]) / 2),
        reverse=True,
    )[:2]
    if top_offenders:
        print(f"\n  [ACTION] Biggest gaps to close:")
        for d, s in top_offenders:
            lo, hi = PROBE_TARGETS[d]
            mid = (lo + hi) / 2
            gap = s - mid
            print(f"    - {d:35s} current={s:+.2f}  needs {-gap:+.2f} to mid-target")
        print(f"    Recommended strategies: dataset-strategies.md A3 (contrastive),")
        print(f"    A2 (co-occurrence), A1 (more natural-language corpus).")
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
        "--epochs", type=int, default=5,
        help="Corpus training epochs (default: 5). "
             "Bumped from 2 after run #1 (see learnings.md L13) — loss was still "
             "trending down at epoch 2, embeddings hadn't converged. "
             "Use 3 for quick iteration; 7-10 for highest quality.",
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
    parser.add_argument(
        "--corpus", type=Path, default=None,
        help="Override corpus.txt path (default: dataset/tokenizer/corpus.txt). "
             "Use to point at an iter-specific corpus file.",
    )
    parser.add_argument(
        "--new-tokens", type=Path, default=None,
        help="Override new_tokens.json path (default: dataset/tokenizer/new_tokens.json).",
    )
    args = parser.parse_args()
    if args.corpus is not None:
        globals()["CORPUS_FILE"] = args.corpus.resolve()
        print(f"  [INFO] using --corpus override: {CORPUS_FILE}")
    if args.new_tokens is not None:
        globals()["NEW_TOKENS_JSON"] = args.new_tokens.resolve()
        print(f"  [INFO] using --new-tokens override: {NEW_TOKENS_JSON}")

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
