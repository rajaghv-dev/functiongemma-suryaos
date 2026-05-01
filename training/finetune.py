#!/usr/bin/env python3
"""
finetune_dispatch.py — Fine-tune functiongemma:270m for surya_agent tool dispatch.

OVERVIEW
========
functiongemma:270m is a Gemma 3 model (268M parameters, Q8_0 quantisation) that
already has "function calling" capability baked in by its original training.
However, it may pick the wrong tool when the user's phrasing is colloquial or
when tool names are domain-specific (e.g. "linux_metrics_summary").

This script teaches the model the *12 SuryaOS tools* via LoRA (Low-Rank
Adaptation) — a lightweight adapter technique that adds ~1–4 M trainable
parameters on top of the frozen base weights.  We never touch the base weights,
so the adapter can be merged later or discarded without harm.

WHY LoRA?
---------
Training all 268 M parameters on a CPU would take days and risk forgetting
general capabilities (catastrophic forgetting).  LoRA inserts small rank-8
matrices into the attention layers (q_proj, v_proj) — enough to steer the
model's routing decisions without rewriting its general language understanding.

WHY CPU-ONLY TRAINING?
----------------------
Intel Meteor Lake has no dedicated CUDA GPU.  PyTorch CPU inference and
training are fully supported; it is slow but perfectly functional for a small
model and a small dataset.  With 48 training examples and 3 epochs the full
training run takes roughly 15–25 minutes on this machine.

TRAINING DATA TARGET (scale note for future maintainers)
---------------------------------------------------------
Current scope   : 12 system tools,  500 examples target for v3
Future scope (v4): Code-compile, test, git, IDE workflows — 2000+ examples
                   needed once the agent handles development workflows.

PIPELINE STAGES (--mode)
-------------------------
  setup   → print the pip install commands; never auto-installs
  check   → verify deps, data file, and model blob
  convert → GGUF blob → HuggingFace safetensors in training/model_hf/
  train   → LoRA fine-tuning with SFTTrainer, saves to training/model_lora/
  export  → merge LoRA → base, quantise back to GGUF, write Modelfile
  all     → check → train → export  (skips convert if model_hf/ already exists)

Usage examples:
  python3 scripts/training/finetune_dispatch.py --mode setup
  python3 scripts/training/finetune_dispatch.py --mode check
  python3 scripts/training/finetune_dispatch.py --mode convert
  python3 scripts/training/finetune_dispatch.py --mode train
  python3 scripts/training/finetune_dispatch.py --mode export
  python3 scripts/training/finetune_dispatch.py --mode all
  python3 scripts/training/finetune_dispatch.py --mode train --epochs 5 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths — everything is relative to the repo root, which is two levels up
# from scripts/training/.
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parent.parent   # repo root
TRAINING_DIR = Path(__file__).resolve().parent          # training/
DATA_FILE    = ROOT / "dataset" / "dispatch_pairs.jsonl"
MODEL_HF_DIR = TRAINING_DIR / "model_hf"               # HF safetensors after convert
MODEL_LORA   = TRAINING_DIR / "model_lora"             # LoRA adapter after train
MODEL_MERGED = TRAINING_DIR / "model_merged"           # merged weights after export
MODELFILE    = TRAINING_DIR / "Modelfile"              # for `ollama create`
GGUF_OUT     = TRAINING_DIR / "functiongemma-suryaos-270m.gguf"
TOKENIZER_EXTENDED = TRAINING_DIR / "tokenizer_extended"  # output of train_tokenizer.py

# The exact blob path for functiongemma:270m on this machine.
# `ollama show functiongemma:270m --modelfile` shows which blob is the weights.
OLLAMA_BLOB = Path(
    "/usr/share/ollama/.ollama/models/blobs/"
    "sha256-415f8f959d807bd4d4da891f01225d7b330416947fb011a8473080ae4fd07885"
)

# HuggingFace Hub fallback if local model_hf/ conversion was skipped.
HF_MODEL_ID = "google/gemma-3-270m-it"

# Domain-specific tokens for the 12 SuryaOS tool names.
# Adding these ensures the tokenizer has a single token for each tool name,
# which helps the model learn to emit them reliably in the output JSON.
DOMAIN_TOKENS = [
    "metrics_summary",
    "volume_change",
    "network_status",
    "battery_status",
    "memory_usage",
    "disk_usage",
    "service_status",
    "krunner_launch",
    "window_focus",
    "brightness_set",
    "notifications_send",
    "ebpf_summary",
]

# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    """Print a clear section header."""
    width = 72
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}", file=sys.stderr)


def _err(msg: str) -> None:
    print(f"  [ERR] {msg}", file=sys.stderr)


def _elapsed(start: float) -> str:
    secs = int(time.time() - start)
    return f"{secs // 60}m {secs % 60}s"


def _detect_hardware() -> dict:
    """
    Inspect the available GPU (if any) and return a training profile.

    Why this matters:
      Using the wrong dtype or batch size causes either OOM errors or wastes
      10-30% of GPU utilization. This function makes those decisions once so
      every downstream function just reads from the returned dict.

    dtype selection:
      bfloat16 — Ampere datacenter (A100, H100): wider exponent range means
                 training is numerically stable even with large gradients. BF16
                 tensor cores on these chips are as fast as FP16.
      float16  — Consumer RTX 30xx/40xx, Turing (RTX 20xx), Tesla T4: slightly
                 narrower exponent range than bf16 but tensor cores are fast.
                 Works fine for LoRA adapters on small models.
      float32  — CPU fallback: the only stable option without FP16 hardware.
                 Much slower but numerically exact.

    Batch size heuristic:
      270M model weights in fp16 = ~540 MB
      LoRA adapter (r=8)         = ~4 MB
      Activations per sample (seq_len=512) = ~150-300 MB depending on layer count
      So for 16 GB VRAM: 16384 MB - 540 MB model - 200 MB overhead = ~15 GB for
      activations → batch 8 x 300 MB = ~2.4 GB → safely fits with headroom.
    """
    try:
        import torch
    except ImportError:
        # torch not installed — return safe CPU defaults
        return {"device": "cpu", "dtype": "float32", "fp16": False, "bf16": False,
                "batch_size": 1, "grad_accum": 4, "gpu_name": None, "vram_gb": 0.0}

    if not torch.cuda.is_available():
        # No CUDA-capable GPU found — fall back to CPU training
        return {"device": "cpu", "dtype": "float32", "fp16": False, "bf16": False,
                "batch_size": 1, "grad_accum": 4, "gpu_name": None, "vram_gb": 0.0}

    props    = torch.cuda.get_device_properties(0)  # device index 0 = first GPU
    gpu_name = props.name
    vram_gb  = props.total_memory / 1e9              # bytes -> GB
    compute  = (props.major, props.minor)            # e.g. (8, 6) for RTX 3080 Ti

    # Keywords that identify datacenter / server-class GPUs.
    # These prefer bf16; consumer GPUs (RTX, Quadro) work better with fp16.
    _dc_keywords  = ("A100", "H100", "H200", "A40", "A10G", "A10 ", "V100")
    is_datacenter = any(kw in gpu_name for kw in _dc_keywords)
    is_ampere_plus = compute[0] >= 8  # Ampere = compute 8.x; Hopper = 9.x

    use_bf16 = is_datacenter and is_ampere_plus   # bf16: stable + fast on datacenter
    use_fp16 = (not use_bf16) and compute[0] >= 7 # fp16: Volta (7.x) and newer consumer

    dtype = "bfloat16" if use_bf16 else ("float16" if use_fp16 else "float32")

    # Batch size scaled to VRAM — keeps GPU utilization high without OOM.
    # Model (fp16) ~540 MB; activations ~150-300 MB/sample at seq_len=512.
    if   vram_gb >= 40:  batch = 16  # A100 80 GB / H100 80 GB
    elif vram_gb >= 24:  batch = 12  # A100 40 GB / RTX 4090 24 GB
    elif vram_gb >= 16:  batch = 8   # RTX 3080 Ti 16 GB / RTX 4080 16 GB  ← this machine
    elif vram_gb >= 10:  batch = 4   # RTX 3080 10 GB / RTX 4070 12 GB
    elif vram_gb >= 8:   batch = 2   # RTX 3060 Ti 8 GB / Tesla T4 16 GB (conservative)
    else:                batch = 1   # anything smaller — single-sample to avoid OOM

    # Keep effective_batch = batch * grad_accum around 8.
    # Larger effective batch = smoother gradient estimates = more stable training.
    # But larger per-step batch = more memory. Grad accumulation simulates a
    # bigger batch by summing gradients over N micro-batches before updating.
    grad_accum = max(1, 8 // batch)

    return {
        "device":     "cuda",
        "dtype":      dtype,
        "fp16":       use_fp16,
        "bf16":       use_bf16,
        "batch_size": batch,
        "grad_accum": grad_accum,
        "gpu_name":   gpu_name,
        "vram_gb":    vram_gb,
        "compute":    compute,
    }


# ===========================================================================
# Dataset formatting helpers — module-level so DispatchCallback can reuse them
# ===========================================================================

def _fmt_example(ex: dict, tokenizer) -> Optional[str]:
    """
    Convert one dispatch_pairs.jsonl entry into a Gemma 3 SFT training string.

    Input  (ex):
      {
        "messages": [
          {"role": "system",  "content": "Call the right tool."},
          {"role": "user",    "content": "how much ram is used"}
        ],
        "tools":  [{ ...tool schema JSON... }],
        "target": {"name": "linux_memory_usage", "arguments": {}}
      }

    Output (Gemma 3 chat template string):
      <bos><start_of_turn>user
      Call the right tool.

      Available tools: [{...}]

      User request: how much ram is used
      <end_of_turn>
      <start_of_turn>model
      {"name":"linux_memory_usage","arguments":{}}
      <end_of_turn>

    SFTTrainer computes cross-entropy loss on the FULL sequence but the model
    is only evaluated on the model-turn tokens. The user-turn tokens are used
    as context (input) only, not penalized.

    Returns None for malformed examples (missing messages or target).
    """
    messages = ex.get("messages", [])
    tools    = ex.get("tools", [])
    target   = ex.get("target", {})
    if not messages or not target:
        return None

    # Extract system and user content from the message list
    system = next((m["content"] for m in messages if m["role"] == "system"),
                  "Call the right tool.")
    user   = next((m["content"] for m in messages if m["role"] == "user"), "")

    # Compact JSON serialization — no extra spaces — keeps sequences short
    tools_text  = json.dumps(tools,  separators=(",", ":"))
    target_json = json.dumps(target, separators=(",", ":"))

    turns = [
        # User turn: system instruction + tool schema + query all in one message
        {"role": "user",  "content": (f"{system}\n\n"
                                      f"Available tools: {tools_text}\n\n"
                                      f"User request: {user}")},
        # Model turn: the correct tool call as compact JSON
        {"role": "model", "content": target_json},
    ]
    try:
        # apply_chat_template produces the exact token sequence Gemma 3 expects
        # add_generation_prompt=False: include the model response so SFT can
        # compute loss on it — if True, the template would stop at "<start_of_turn>model"
        return tokenizer.apply_chat_template(
            turns, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback for tokenizers that don't support apply_chat_template
        return (f"<bos><start_of_turn>user\n{system}\n\n"
                f"Available tools: {tools_text}\n\n"
                f"User request: {user}\n<end_of_turn>\n"
                f"<start_of_turn>model\n{target_json}\n<end_of_turn>")


def _fmt_prompt_only(ex: dict, tokenizer) -> str:
    """
    Format only the USER-TURN of an example (no model response).

    Used by DispatchCallback to compute the length of the prompt prefix in
    tokens. The callback then sets labels=-100 for those prefix positions so
    the validation loss only measures how well the model generates the response,
    not how well it "memorized" the input.

    add_generation_prompt=True: appends "<start_of_turn>model\n" so the
    tokenized length includes the turn-start token (which is part of the input
    context, not the response we measure).
    """
    messages   = ex.get("messages", [])
    tools      = ex.get("tools", [])
    system     = next((m["content"] for m in messages if m["role"] == "system"),
                      "Call the right tool.")
    user       = next((m["content"] for m in messages if m["role"] == "user"), "")
    tools_text = json.dumps(tools, separators=(",", ":"))

    turns = [{"role": "user",
               "content": f"{system}\n\nAvailable tools: {tools_text}\n\nUser request: {user}"}]
    try:
        return tokenizer.apply_chat_template(
            turns, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (f"<bos><start_of_turn>user\n{system}\n\n"
                f"Available tools: {tools_text}\n\n"
                f"User request: {user}\n<end_of_turn>\n<start_of_turn>model\n")


# ===========================================================================
# Narrator — runtime commentary on what the LoRA adapter just learned
# ===========================================================================

class _LoRANarrator:
    """
    Live commentary on LoRA training progress.

    Unlike the silent JSONL log, this class WRITES TO THE TERMINAL with
    plain-English interpretation of every notable event:

      [LEARN]    Step 50: loss crossed 1.0 — model can now reproduce tool calls
      [PROGRESS] Step 100: loss down 35% in last 20 steps — healthy convergence
      [PLATEAU]  Step 200: loss stable at 0.4 — adapter has converged
      [WARN]     Step 75: grad_norm=2.1 — clipping prevented instability spike

    The narrator rate-limits itself (one emit per ~10 steps) so it doesn't
    flood the terminal during normal smooth training.
    """

    def __init__(self) -> None:
        self.loss_history     = []        # list of (step, loss)
        self.grad_history     = []        # list of (step, grad_norm)
        self.last_emit_step   = -100      # rate limiter
        self.milestones       = set()     # one-shot threshold announcements
        self.prev_epoch_per_tool = {}     # for cross-epoch tool comparison

    def _emit(self, prefix: str, msg: str) -> None:
        print(f"  [{prefix:8s}] {msg}")

    def step(self, step: int, loss: Optional[float],
             grad_norm: Optional[float], lr: Optional[float]) -> None:
        """Called from on_log. Emits commentary when the metrics shift meaningfully."""
        if loss is None:
            return

        self.loss_history.append((step, loss))
        if grad_norm is not None and not math.isnan(grad_norm):
            self.grad_history.append((step, grad_norm))

        # Always print the first sample to set the baseline
        if step == self.loss_history[0][0]:
            self._emit("LEARN",
                f"First sample at step {step}: loss={loss:.3f}. "
                f"This is what the adapter looks like before any real updates.")

        # ---- Loss milestones — these tell the user "the model just got X capability" ----
        if loss < 1.5 and "loss_below_1_5" not in self.milestones:
            self._emit("LEARN",
                f"Step {step}: loss below 1.5 (={loss:.3f}). "
                f"Adapter is starting to influence the response — model can produce "
                f"the JSON envelope ({{'name':...}}) reliably.")
            self.milestones.add("loss_below_1_5")

        if loss < 1.0 and "loss_below_1" not in self.milestones:
            self._emit("LEARN",
                f"Step {step}: loss crossed 1.0 (={loss:.3f}). "
                f"Model can predict the correct tool name for most training examples.")
            self.milestones.add("loss_below_1")

        if loss < 0.5 and "loss_below_05" not in self.milestones:
            self._emit("LEARN",
                f"Step {step}: loss below 0.5 (={loss:.3f}). "
                f"Adapter has learned the dispatch patterns and argument structures.")
            self.milestones.add("loss_below_05")

        if loss < 0.2 and "loss_below_02" not in self.milestones:
            self._emit("LEARN",
                f"Step {step}: loss below 0.2 (={loss:.3f}). "
                f"Near-perfect reproduction of training examples — watch for overfitting.")
            self.milestones.add("loss_below_02")

        # Rate-limit other narration to ~once per 10 steps
        if step - self.last_emit_step < 10:
            return

        # ---- Trend detection: improving, plateaued, diverging ----
        if len(self.loss_history) >= 20:
            recent = [l for _, l in self.loss_history[-10:]]
            older  = [l for _, l in self.loss_history[-20:-10]]
            r_avg  = sum(recent) / len(recent)
            o_avg  = sum(older)  / len(older)

            if o_avg > 0:
                pct = (o_avg - r_avg) / o_avg * 100   # positive = improving

                if pct > 25:
                    self._emit("PROGRESS",
                        f"Step {step}: loss down {pct:.0f}% (last 10 steps avg "
                        f"{o_avg:.3f} -> {r_avg:.3f}) — adapter is learning fast")
                    self.last_emit_step = step
                elif pct > 8:
                    self._emit("PROGRESS",
                        f"Step {step}: loss down {pct:.0f}% — steady convergence")
                    self.last_emit_step = step
                elif abs(pct) < 1.5:
                    self._emit("PLATEAU",
                        f"Step {step}: loss stable at {r_avg:.3f} "
                        f"({pct:+.1f}% over 10 steps) — adapter has converged")
                    self.last_emit_step = step
                elif pct < -8:
                    self._emit("WARN",
                        f"Step {step}: loss INCREASED {-pct:.0f}% "
                        f"({o_avg:.3f} -> {r_avg:.3f}) — possible overfitting; "
                        f"check next epoch's probe table for tool-specific damage")
                    self.last_emit_step = step

        # ---- Gradient anomalies ----
        if grad_norm is not None and grad_norm > 1.5:
            self._emit("WARN",
                f"Step {step}: grad_norm={grad_norm:.2f} above clip threshold (1.0). "
                f"Clipping activated — adapter wanted to make a big jump.")

    def tool_table(self, epoch: int, per_tool: dict) -> None:
        """
        Interpret the per-tool loss breakdown after each epoch's probe.

        Tells the user concretely:
          - which tools the adapter has mastered (low loss)
          - which tools are still struggling (high loss)
          - which tools improved most since last epoch
        """
        if not per_tool:
            return

        # Compute per-tool means and sort by performance
        means = {t: sum(v)/len(v) for t, v in per_tool.items()}

        # Mastered (loss < 0.5) vs struggling (loss > 1.5)
        mastered   = sorted([t for t, m in means.items() if m < 0.5])
        struggling = sorted([t for t, m in means.items() if m > 1.5])

        if mastered:
            self._emit("LEARN",
                f"Mastered tools (loss < 0.5): {', '.join(mastered)}")
            self._emit("LEARN",
                f"  Any user phrasing matching these tools should dispatch correctly now.")

        if struggling:
            self._emit("WARN",
                f"Still struggling (loss > 1.5): {', '.join(struggling)}")
            self._emit("WARN",
                f"  Add more dispatch_pairs.jsonl examples for these tools.")

        # Show biggest movers vs previous epoch (improvement or regression)
        if self.prev_epoch_per_tool:
            improvements = []
            for t, cur in means.items():
                prev = self.prev_epoch_per_tool.get(t)
                if prev is None or prev == 0:
                    continue
                delta = prev - cur
                if abs(delta) > 0.2:  # only mention meaningful changes
                    improvements.append((t, prev, cur, delta))

            improvements.sort(key=lambda x: -x[3])  # biggest improvement first
            if improvements:
                print(f"\n  [DELTA]    Largest changes vs previous epoch:")
                for t, prev, cur, delta in improvements[:5]:
                    arrow = "->"
                    sign  = "improved" if delta > 0 else "regressed"
                    print(f"             {t:<28s} {prev:.3f} {arrow} {cur:.3f}  "
                          f"({sign} {abs(delta):.3f})")

        self.prev_epoch_per_tool = means


# ===========================================================================
# DispatchCallback — telemetry and per-epoch accuracy probe
# ===========================================================================

try:
    from transformers import TrainerCallback as _TrainerCallbackBase
except ImportError:
    _TrainerCallbackBase = object  # fallback when transformers not yet installed


class DispatchCallback(_TrainerCallbackBase):
    """
    HuggingFace TrainerCallback wired into SFTTrainer for live telemetry.

    How TrainerCallback works:
      HuggingFace Trainer calls specific methods at specific training events.
      We override the ones we care about and ignore the rest.
      The Trainer passes the current model, state, and log dict to each hook.

    What this callback does:

    on_log  (fires every logging_steps, default=5):
      Captures the loss/lr/grad_norm dict that the Trainer computed and
      appends a JSON entry to training_log.jsonl. Also records RAM usage
      via psutil so you can spot memory leaks over time.

      Reading the log:
        {"event":"step", "step":10, "epoch":0.129, "loss":2.341,
         "learning_rate":0.000198, "grad_norm":0.42, "memory_mb":4821}
        loss:          cross-entropy; lower=better; healthy range 0.1-1.0 at end
        learning_rate: should decay from initial lr to ~0 over training
        grad_norm:     should be < 1.0 (we clip at 1.0); spike = instability
        memory_mb:     watch for steady growth = memory leak

    on_epoch_end  (fires once per epoch):
      Runs the "response-only loss probe": forward passes on 20 sampled
      examples with the prompt positions masked from the loss. Reports a
      per-tool table. Seeing which tools have high loss at the end of each
      epoch tells you which ones need more training examples.

      Reading the table:
        Tool                           n    mean loss    bar
        linux_memory_usage             3      0.2341     ████████████████░░░░
        kde_krunner_launch             2      1.8700     ███████░░░░░░░░░░░░░
        → krunner_launch still has high loss = model struggles with app launch queries
        → Add more app-launch examples to dataset/dispatch_pairs.jsonl

    on_train_end  (fires after the last training step):
      Runs one final probe for a clean end-of-training summary.
    """

    def __init__(self, probe_data: list, tokenizer, log_path: Path) -> None:
        # probe_data: list of (full_text_string, prompt_token_length, tool_name_string)
        # full_text       — the complete training example as a formatted string
        # prompt_token_len — number of tokens in the user-turn prefix (for masking)
        # tool_name       — expected tool name (for per-tool grouping in reports)
        self.probe_data = probe_data
        self.tokenizer  = tokenizer
        self.log_path   = log_path
        self._buf: list[dict] = []           # in-memory write buffer, flushed to disk each step
        self.narrator    = _LoRANarrator()   # live commentary engine
        self._last_loss  = None              # for narration of cumulative trends

    # ---- TrainerCallback hook: fires every logging_steps ----

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """Capture per-step metrics from the Trainer's internal log dict."""
        if not logs:
            return

        # psutil gives us the process RSS (resident set size) = actual RAM used.
        # Optional — the callback works without it, just with memory_mb=null.
        try:
            import psutil
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
        except ImportError:
            mem_mb = None  # psutil not installed — telemetry still works, just no memory

        entry = {
            "event":         "step",
            "step":          state.global_step,             # absolute step count
            "epoch":         round(state.epoch or 0, 3),    # fractional epoch (e.g. 1.5)
            "loss":          logs.get("loss"),               # training loss this window
            "learning_rate": logs.get("learning_rate"),      # current LR on the schedule
            "grad_norm":     logs.get("grad_norm"),          # gradient L2 norm pre-clip
            "memory_mb":     mem_mb,                         # process RAM in MB
        }
        self._buf.append(entry)
        self._flush()  # write immediately so we have logs even if training crashes

        # ---- LIVE NARRATION ----
        # Print plain-English interpretation of the metrics to the terminal so
        # the user can SEE what the adapter just learned without parsing JSON.
        self.narrator.step(
            step      = state.global_step,
            loss      = logs.get("loss"),
            grad_norm = logs.get("grad_norm"),
            lr        = logs.get("learning_rate"),
        )

    # ---- TrainerCallback hook: fires at end of each epoch ----

    def on_epoch_end(self, args, state, control, model=None, **kwargs) -> None:
        """Run the response-loss probe and print the per-tool table."""
        if model is None or not self.probe_data:
            return
        epoch = int(state.epoch or 0)
        self._run_probe(model, epoch)

    # ---- TrainerCallback hook: fires once after training completes ----

    def on_train_end(self, args, state, control, model=None, **kwargs) -> None:
        """Final probe after training — definitive end-of-run performance summary."""
        if model is None or not self.probe_data:
            return
        print("\n  [Callback] Final post-training probe:")
        self._run_probe(model, epoch=-1)  # epoch=-1 prints as "Final" in the table

    # ---- Internal probe runner ----

    def _run_probe(self, model, epoch: int) -> None:
        """
        Compute response-only loss for each probe example.

        Why "response-only" loss?
          If we computed loss on the full sequence (prompt + response), most of
          the loss would come from the model predicting the prompt tokens, which
          is irrelevant — the prompt is always given as input, not generated.
          We want to measure: given the right context, does the model produce
          the correct tool call?

        How it works:
          1. Tokenize the full example (prompt + response).
          2. Set labels=-100 for all prompt positions.
             CrossEntropyLoss skips positions where label=-100.
          3. Forward pass → loss only over response tokens.
          4. Group by tool name and average.

        This is a forward-pass-only probe (no generation), so it's fast:
          ~0.5s per example on GPU, ~2s on CPU.
          20 examples = ~10s GPU / ~40s CPU per epoch.
        """
        try:
            import torch
            from collections import defaultdict
        except ImportError:
            return

        tokenizer = self.tokenizer
        model.eval()  # disable dropout and batch norm in eval mode
        per_tool: dict[str, list[float]] = defaultdict(list)

        with torch.no_grad():  # no gradient computation needed — saves memory
            for full_text, prompt_len, tool_name in self.probe_data:
                try:
                    enc = tokenizer(
                        full_text, return_tensors="pt",
                        truncation=True, max_length=512,
                    )
                    labels = enc.input_ids.clone()
                    # Mask the prompt prefix: loss is computed ONLY on the model
                    # response (the tool-call JSON after <start_of_turn>model)
                    labels[0, :prompt_len] = -100
                    # Also mask padding tokens if any
                    if "attention_mask" in enc:
                        labels[enc.attention_mask == 0] = -100

                    out = model(**enc, labels=labels)
                    # Skip NaN/Inf (can happen with fp16 edge cases)
                    if not torch.isnan(out.loss) and not torch.isinf(out.loss):
                        per_tool[tool_name].append(out.loss.item())
                except Exception:
                    pass  # skip malformed examples silently

        model.train()  # return to training mode for the next step

        if not per_tool:
            return

        all_losses = [v for vals in per_tool.values() for v in vals]
        mean_all   = sum(all_losses) / len(all_losses)

        label = f"Epoch {epoch}" if epoch >= 0 else "Final"
        print(f"\n  [{label} probe — response loss per tool]")
        print(f"  How to read: lower loss = adapter has learned this tool.")
        print(f"               loss < 0.5 = mastered;  > 1.5 = still struggling.\n")
        print(f"  {'Tool':<30s} {'n':>4s}  {'mean loss':>10s}")
        print(f"  {'-'*30} {'-'*4}  {'-'*10}")
        for tool, losses in sorted(per_tool.items()):
            mean_loss = sum(losses) / len(losses)
            bar_len   = max(0, int(20 * (2.0 - mean_loss) / 2.0))  # rough visual
            bar       = "█" * bar_len + "░" * (20 - bar_len)
            # Append a status tag so the user sees state at a glance
            tag = "(mastered)"   if mean_loss < 0.5 else \
                  "(learning)"   if mean_loss < 1.5 else \
                  "(struggling)"
            print(f"  {tool:<30s} {len(losses):>4d}  {mean_loss:>10.4f}  {bar}  {tag}")
        print(f"  {'OVERALL':<30s} {len(all_losses):>4d}  {mean_all:>10.4f}")

        # Plain-English interpretation: which tools are mastered, which need work,
        # and how things moved since the previous epoch.
        self.narrator.tool_table(epoch, per_tool)

        self._buf.append({
            "event":    "epoch_probe",
            "epoch":    epoch,
            "per_tool": {k: {"mean": sum(v)/len(v), "n": len(v)} for k, v in per_tool.items()},
            "overall_mean_loss": mean_all,
        })
        self._flush()

    def _flush(self) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                for entry in self._buf:
                    f.write(json.dumps(entry) + "\n")
            self._buf.clear()
        except Exception:
            pass


# ===========================================================================
# MODE: setup
# ===========================================================================

def mode_setup() -> None:
    """Print environment setup commands — never auto-installs anything."""
    _banner("SETUP — Python environment for CPU fine-tuning")

    venv = Path(__file__).resolve().parent.parent / ".fngemma-suryaos"
    pip  = venv / "bin" / "pip"
    py   = venv / "bin" / "python3"
    req  = Path(__file__).resolve().parent / "requirements.txt"

    print(f"""
The project uses a dedicated virtual environment at:
  {venv}

━━━ Quickstart (recommended) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Run the bootstrap script — creates venv if missing, installs everything:
  bash training/bootstrap.sh

━━━ Manual steps ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # 1. Create the venv (skip if .fngemma-suryaos/ already exists):
  python3 -m venv {venv}

  # 2. Install all dependencies (CPU PyTorch + HuggingFace stack):
  {pip} install -r {req}

  # 3. Activate (optional — scripts can be run with the full path too):
  source {venv}/bin/activate

━━━ Verify ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  {py} -c "import torch;        print('torch',        torch.__version__,        '| CUDA:', torch.cuda.is_available())"
  {py} -c "import transformers; print('transformers', transformers.__version__)"
  {py} -c "import peft;         print('peft',         peft.__version__)"
  {py} -c "import trl;          print('trl',          trl.__version__)"
  {py} -c "import sentencepiece; print('sentencepiece OK')"

━━━ After setup, run in order ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  {py} training/train_tokenizer.py        # extend tokenizer + warm up embeddings
  {py} training/finetune.py --mode check  # verify environment
  {py} training/finetune.py --mode all    # full pipeline: check→train→export
""")


# ===========================================================================
# MODE: check
# ===========================================================================

def mode_check() -> bool:
    """Verify all requirements are met before training."""
    _banner("CHECK — verifying environment")
    ok = True

    # --- Python packages ---
    required_packages = {
        "torch":        "torch",
        "transformers": "transformers",
        "datasets":     "datasets",
        "peft":         "peft",
        "trl":          "trl",
        "gguf":         "gguf",
        "sentencepiece": "sentencepiece",
    }

    print("\n  [dependencies]")
    for display_name, import_name in required_packages.items():
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            _ok(f"{display_name} {ver}")
        except ImportError:
            _err(f"{display_name} not installed — run --mode setup")
            ok = False

    # --- hardware ---
    print("\n  [hardware]")
    try:
        import torch
        hw = _detect_hardware()

        _ok(f"CPU threads: {torch.get_num_threads()}")
        try:
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            _ok(f"System RAM: {mem_bytes / 1e9:.1f} GB")
        except Exception:
            pass

        if hw["device"] == "cuda":
            _ok(f"GPU:         {hw['gpu_name']}")
            _ok(f"VRAM:        {hw['vram_gb']:.1f} GB")
            _ok(f"Compute cap: {hw['compute'][0]}.{hw['compute'][1]}")
            _ok(f"CUDA:        {torch.version.cuda}")
            _ok(f"Training:    dtype={hw['dtype']}  "
                f"fp16={hw['fp16']}  bf16={hw['bf16']}")
            _ok(f"Auto batch:  per_device={hw['batch_size']}  "
                f"grad_accum={hw['grad_accum']}  "
                f"effective={hw['batch_size'] * hw['grad_accum']}")
        else:
            _warn("No CUDA GPU detected — training on CPU (slow)")
            _warn("  Install CUDA PyTorch: bash training/bootstrap.sh")
            _ok(f"Training:    dtype=float32  batch=1  grad_accum=4")
    except ImportError:
        pass

    # --- Training data ---
    print("\n  [training data]")
    if DATA_FILE.exists():
        lines = DATA_FILE.read_text().strip().splitlines()
        n = len(lines)
        _ok(f"{DATA_FILE} — {n} lines")
        if n < 20:
            _warn(f"Only {n} examples — model may not generalise well. "
                  "Target 500+ for reliable dispatch. Run generate_pairs.py --mode augment.")
        elif n < 100:
            _warn(f"{n} examples is marginal. Consider augmenting to 500+.")
        # Quick schema validation on first line
        try:
            first = json.loads(lines[0])
            assert "messages" in first and "target" in first and "tools" in first
            _ok("Schema validation passed (sample line)")
        except Exception as e:
            _err(f"Schema validation failed: {e}")
            ok = False
    else:
        _err(f"{DATA_FILE} not found — run scripts/training/generate_pairs.py first")
        ok = False

    # --- Ollama blob ---
    print("\n  [model blob]")
    if OLLAMA_BLOB.exists():
        size_mb = OLLAMA_BLOB.stat().st_size / 1e6
        _ok(f"GGUF blob found: {OLLAMA_BLOB.name} ({size_mb:.0f} MB)")
    else:
        _warn(f"GGUF blob not found at expected path:\n       {OLLAMA_BLOB}")
        _warn("--mode convert will fall back to HuggingFace Hub download.")

    # --- model_hf/ ---
    print("\n  [converted model]")
    if MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir()):
        files = list(MODEL_HF_DIR.iterdir())
        _ok(f"model_hf/ present ({len(files)} files) — --mode train can use this")
    else:
        _warn(f"model_hf/ not found — --mode train will download from HF Hub ({HF_MODEL_ID})")

    # --- model_lora/ ---
    print("\n  [LoRA adapter]")
    if MODEL_LORA.exists() and any(MODEL_LORA.iterdir()):
        _ok(f"model_lora/ present — --mode export can proceed")
    else:
        _warn("model_lora/ not found — run --mode train first before --mode export")

    print()
    if ok:
        _ok("All critical checks passed.")
    else:
        _err("Some checks FAILED — address errors above before proceeding.")

    return ok


# ===========================================================================
# MODE: convert
# ===========================================================================

def mode_convert() -> None:
    """Convert the Ollama GGUF blob to HuggingFace safetensors format.

    WHY THIS STEP?
    --------------
    Ollama stores models as GGUF files (a single binary blob with all tensor
    data + metadata).  HuggingFace's transformers library expects models in its
    own format: a config.json + model.safetensors + tokenizer files.

    The `gguf` Python package can read GGUF files.  We use it to:
      1. Inspect the metadata (architecture, vocab size, head counts …)
      2. Extract the weight tensors
      3. Write them out in HF safetensors format with a matching config.json

    NOTE: The gguf package's conversion utilities are still evolving.  If the
    automated conversion fails, the fallback is to download google/gemma-3-270m-it
    directly from HuggingFace Hub (requires internet + HF token for gated models).
    Gemma 3 270M IT is gated, so you need to accept the licence at:
      https://huggingface.co/google/gemma-3-270m-it
    and set HF_TOKEN in your environment.
    """
    _banner("CONVERT — GGUF blob → HuggingFace safetensors")

    MODEL_HF_DIR.mkdir(parents=True, exist_ok=True)

    if not OLLAMA_BLOB.exists():
        _warn(f"GGUF blob not found at {OLLAMA_BLOB}")
        _warn("Falling back to HuggingFace Hub download …")
        _hf_download_fallback()
        return

    try:
        import gguf
    except ImportError:
        _err("gguf package not installed — run --mode setup first")
        sys.exit(1)

    _ok(f"Reading GGUF from {OLLAMA_BLOB}")
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Strategy: use gguf.GGUFReader to extract metadata + tensors, then use
    # the transformers auto-conversion path (llama.cpp convert_hf_to_gguf.py
    # approach in reverse).
    #
    # For Gemma 3 the tensor names follow the standard HF naming convention,
    # so we can map them back directly.
    # -----------------------------------------------------------------------

    try:
        reader = gguf.GGUFReader(str(OLLAMA_BLOB), mode="r")
    except Exception as e:
        _err(f"Failed to open GGUF: {e}")
        _warn("Falling back to HuggingFace Hub download …")
        _hf_download_fallback()
        return

    # Print a summary of what's inside the GGUF
    print("\n  GGUF metadata:")
    metadata: dict = {}
    for field in reader.fields.values():
        key = field.name
        # Each field has parts (type + data); grab the first data element
        if hasattr(field, "parts") and field.parts:
            try:
                val = field.parts[field.data[0]][0]
                # Convert numpy scalar to Python type
                if hasattr(val, "item"):
                    val = val.item()
                metadata[key] = val
                if key.startswith("general.") or key.startswith("gemma"):
                    print(f"    {key}: {val}")
            except Exception:
                pass

    # Check we actually have Gemma 3 weights
    arch = metadata.get("general.architecture", "unknown")
    if "gemma" not in str(arch).lower():
        _warn(f"Unexpected architecture: {arch}. Proceeding anyway.")

    # -----------------------------------------------------------------------
    # Build Gemma 3 config.json from GGUF metadata.
    # These fields map directly from GGUF keys to HF config keys.
    # -----------------------------------------------------------------------

    # Helper: get int metadata with fallback
    def _meta_int(key: str, default: int) -> int:
        return int(metadata.get(key, default))

    n_layers       = _meta_int("gemma3.block_count",                  18)
    hidden_size    = _meta_int("gemma3.embedding_length",            1152)
    # 270M model uses 640 embedding_length per `ollama show`
    hidden_size    = _meta_int("gemma3.embedding_length",             640)
    n_heads        = _meta_int("gemma3.attention.head_count",         8)
    n_kv_heads     = _meta_int("gemma3.attention.head_count_kv",     4)
    intermediate   = _meta_int("gemma3.feed_forward_length",        3072)
    head_dim_val   = _meta_int("gemma3.attention.key_length",         256)
    vocab_size     = _meta_int("tokenizer.ggml.tokens",            256000)
    rms_norm_eps   = float(metadata.get("gemma3.attention.layer_norm_rms_epsilon", 1e-6))
    ctx_length     = _meta_int("gemma3.context_length",            32768)

    config = {
        "architectures":             ["Gemma3ForCausalLM"],
        "model_type":                "gemma3",
        "hidden_size":               hidden_size,
        "intermediate_size":         intermediate,
        "num_hidden_layers":         n_layers,
        "num_attention_heads":       n_heads,
        "num_key_value_heads":       n_kv_heads,
        "head_dim":                  head_dim_val,
        "hidden_act":                "gelu_pytorch_tanh",
        "max_position_embeddings":   ctx_length,
        "rms_norm_eps":              rms_norm_eps,
        "vocab_size":                vocab_size,
        "torch_dtype":               "float32",
        "transformers_version":      "4.40.0",
        "use_cache":                 True,
    }

    config_path = MODEL_HF_DIR / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    _ok(f"Wrote config.json (hidden={hidden_size}, layers={n_layers}, heads={n_heads})")

    # -----------------------------------------------------------------------
    # Extract tensors and write safetensors.
    # The GGUF tensor names use llama.cpp convention; map to HF names.
    # -----------------------------------------------------------------------

    import numpy as np

    # Gemma 3 GGUF → HF name mapping (partial — covers the main weight types)
    # GGUF key pattern          → HF key pattern
    #   token_embd.weight       → model.embed_tokens.weight
    #   blk.N.attn_q.weight     → model.layers.N.self_attn.q_proj.weight
    #   blk.N.attn_k.weight     → model.layers.N.self_attn.k_proj.weight
    #   blk.N.attn_v.weight     → model.layers.N.self_attn.v_proj.weight
    #   blk.N.attn_output.weight→ model.layers.N.self_attn.o_proj.weight
    #   blk.N.ffn_gate.weight   → model.layers.N.mlp.gate_proj.weight
    #   blk.N.ffn_up.weight     → model.layers.N.mlp.up_proj.weight
    #   blk.N.ffn_down.weight   → model.layers.N.mlp.down_proj.weight
    #   blk.N.attn_norm.weight  → model.layers.N.input_layernorm.weight
    #   blk.N.ffn_norm.weight   → model.layers.N.post_feedforward_layernorm.weight
    #   output_norm.weight      → model.norm.weight
    #   output.weight           → lm_head.weight

    def _gguf_to_hf_name(gguf_name: str) -> Optional[str]:
        name = gguf_name
        if name == "token_embd.weight":
            return "model.embed_tokens.weight"
        if name == "output_norm.weight":
            return "model.norm.weight"
        if name == "output.weight":
            return "lm_head.weight"
        if name.startswith("blk."):
            parts = name.split(".")
            layer_num = parts[1]
            rest = ".".join(parts[2:])
            prefix = f"model.layers.{layer_num}."
            mapping = {
                "attn_q.weight":       "self_attn.q_proj.weight",
                "attn_k.weight":       "self_attn.k_proj.weight",
                "attn_v.weight":       "self_attn.v_proj.weight",
                "attn_output.weight":  "self_attn.o_proj.weight",
                "ffn_gate.weight":     "mlp.gate_proj.weight",
                "ffn_up.weight":       "mlp.up_proj.weight",
                "ffn_down.weight":     "mlp.down_proj.weight",
                "attn_norm.weight":    "input_layernorm.weight",
                "ffn_norm.weight":     "post_feedforward_layernorm.weight",
                # Gemma 3 has pre/post attn norm variants in some configs
                "attn_post_norm.weight": "post_attention_layernorm.weight",
                "ffn_post_norm.weight":  "post_feedforward_layernorm.weight",
            }
            hf_rest = mapping.get(rest)
            return prefix + hf_rest if hf_rest else None
        return None

    # Dequantise Q8_0 tensors to float32 for safetensors.
    # Q8_0: each block of 32 values has a float16 scale + 32 int8 values.
    # dequantised_value = scale * int8_value

    def _dequantize_q8_0(data: np.ndarray, shape: tuple) -> np.ndarray:
        """Dequantise a flat Q8_0 byte buffer to float32."""
        # Block layout: 2 bytes (float16 scale) + 32 bytes (int8 values)
        BLOCK_SIZE  = 32
        BYTES_BLOCK = 2 + BLOCK_SIZE  # 34 bytes per block
        n_elements  = 1
        for d in shape:
            n_elements *= d
        n_blocks = n_elements // BLOCK_SIZE

        raw = data.tobytes() if not isinstance(data, bytes) else data
        raw = np.frombuffer(raw, dtype=np.uint8)

        scales = np.zeros(n_blocks, dtype=np.float32)
        weights = np.zeros(n_blocks * BLOCK_SIZE, dtype=np.float32)

        for b in range(n_blocks):
            offset = b * BYTES_BLOCK
            # Scale: little-endian float16
            scale = np.frombuffer(raw[offset:offset+2], dtype=np.float16)[0].astype(np.float32)
            # Quantised int8 values
            qs = raw[offset+2 : offset+2+BLOCK_SIZE].view(np.int8).astype(np.float32)
            weights[b*BLOCK_SIZE : (b+1)*BLOCK_SIZE] = scale * qs

        return weights.reshape(shape)

    print("\n  Extracting and converting tensors (this may take a few minutes) …")

    state_dict: dict[str, "torch.Tensor"] = {}
    skipped = 0
    converted = 0

    try:
        import torch
    except ImportError:
        _err("torch not installed — cannot extract tensors. Run --mode setup.")
        sys.exit(1)

    for tensor in reader.tensors:
        hf_name = _gguf_to_hf_name(tensor.name)
        if hf_name is None:
            skipped += 1
            continue

        shape = tuple(reversed(tensor.shape.tolist()))  # GGUF stores dims reversed
        raw_data = tensor.data

        # Most quantisation types can be handled by gguf's own dequant
        try:
            # Try gguf's built-in dequantisation if available
            if hasattr(tensor, "numpy"):
                arr = tensor.numpy().astype(np.float32)
            else:
                # Fallback: try to interpret as float32 or dequant Q8_0 manually
                quant_type = int(tensor.tensor_type)
                # gguf.GGMLQuantizationType.Q8_0 == 8
                if quant_type == 8:
                    arr = _dequantize_q8_0(raw_data, shape)
                elif quant_type == 0:  # F32
                    arr = raw_data.reshape(shape).astype(np.float32)
                elif quant_type == 1:  # F16
                    arr = raw_data.view(np.float16).reshape(shape).astype(np.float32)
                else:
                    _warn(f"  Unsupported quant type {quant_type} for {tensor.name}, skipping")
                    skipped += 1
                    continue
            state_dict[hf_name] = torch.from_numpy(arr)
            converted += 1
        except Exception as e:
            _warn(f"  Failed to convert {tensor.name}: {e}")
            skipped += 1

    _ok(f"Converted {converted} tensors, skipped {skipped}")

    if converted == 0:
        _err("No tensors were extracted. The GGUF format may be incompatible.")
        _warn("Falling back to HuggingFace Hub download …")
        _hf_download_fallback()
        return

    # Write safetensors
    try:
        from safetensors.torch import save_file as st_save
        st_path = MODEL_HF_DIR / "model.safetensors"
        st_save(state_dict, str(st_path))
        size_mb = st_path.stat().st_size / 1e6
        _ok(f"Saved {st_path} ({size_mb:.0f} MB)")
    except ImportError:
        # Fallback: save as .pt (works without safetensors package)
        import torch
        pt_path = MODEL_HF_DIR / "pytorch_model.bin"
        torch.save(state_dict, str(pt_path))
        size_mb = pt_path.stat().st_size / 1e6
        _ok(f"Saved {pt_path} ({size_mb:.0f} MB) [safetensors not installed, used .bin]")

    # Copy tokenizer files from the base Ollama model if possible.
    # The tokenizer is stored in the params blob alongside the weights.
    # For Gemma 3, we write a minimal tokenizer_config.json that points to the
    # HF tokenizer, which will be auto-downloaded by transformers.
    tokenizer_config = {
        "model_type":                 "gemma3",
        "tokenizer_class":            "GemmaTokenizerFast",
        "bos_token":                  "<bos>",
        "eos_token":                  "<eos>",
        "unk_token":                  "<unk>",
        "pad_token":                  "<pad>",
        "add_bos_token":              True,
        "add_eos_token":              False,
        "clean_up_tokenization_spaces": False,
        # transformers will pull the sentencepiece model from here
        "auto_map": {"AutoTokenizer": "transformers.GemmaTokenizerFast"},
    }
    (MODEL_HF_DIR / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2)
    )
    _ok("Wrote tokenizer_config.json")

    print(f"\n  Total time: {_elapsed(t0)}")
    _ok(f"Conversion complete → {MODEL_HF_DIR}")
    print("\n  NOTE: The tokenizer's sentencepiece model (tokenizer.model) will be")
    print("  downloaded automatically when --mode train first loads the tokenizer.")
    print(f"  Alternatively, copy it from {HF_MODEL_ID} on HuggingFace Hub.")


def _hf_download_fallback() -> None:
    """Download google/gemma-3-270m-it from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        try:
            from transformers.utils.hub import cached_file
        except ImportError:
            _err("Neither huggingface_hub nor transformers is installed.")
            _err(f"Manually download {HF_MODEL_ID} to {MODEL_HF_DIR}")
            sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        _warn("HF_TOKEN not set. If the model is gated you must set it:")
        _warn("  export HF_TOKEN=hf_...")
        _warn("  See: https://huggingface.co/google/gemma-3-270m-it")

    _ok(f"Downloading {HF_MODEL_ID} from HuggingFace Hub → {MODEL_HF_DIR} …")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=str(MODEL_HF_DIR),
            token=hf_token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
        )
        _ok(f"Downloaded to {MODEL_HF_DIR}")
    except Exception as e:
        _err(f"Download failed: {e}")
        _err("Resolve HF token / network issue, then re-run --mode convert")
        sys.exit(1)


# ===========================================================================
# MODE: train
# ===========================================================================

def mode_train(epochs: int = 3, lr: float = 2e-4, batch_size: int = 0,
               grad_accum: int = 0) -> None:
    """LoRA fine-tuning with SFTTrainer — GPU or CPU.

    WHAT HAPPENS HERE
    -----------------
    1. Detect hardware — GPU (RTX/A100/H100) or CPU; choose dtype + batch size.
    2. Load tokenizer — from tokenizer_extended/ if train_tokenizer.py has run,
       otherwise adds 12 DOMAIN_TOKENS to base Gemma 3 tokenizer.
    3. Load base model — fp16/bf16 on GPU, float32 on CPU.
    4. Restore embeddings — loads embed_init.pt warm embeddings when available.
    5. Apply LoRA — rank-8 adapter on q_proj + v_proj.
    6. Format dataset — convert dispatch_pairs to Gemma 3 chat template strings.
    7. Train with SFTTrainer + DispatchCallback — per-step telemetry and
       per-epoch per-tool accuracy probe.
    8. Save adapter — adapter_model.safetensors + adapter_config.json.

    TIMING ESTIMATES
    ----------------
    RTX 3080 Ti 16GB (fp16):   ~1–3 min for 77 examples, 3 epochs
    CPU (Intel Meteor Lake):   ~15–25 min for 48 examples, 3 epochs

    LoRA HYPERPARAMETERS
    --------------------
    r=8      : rank — controls adapter capacity. 8 is standard for small tasks.
    alpha=16 : scaling factor α/r = 2.0. Higher alpha = larger effective LR.
    dropout=0.05 : light regularisation to prevent overfitting on small data.
    target_modules: q_proj, v_proj — attention projections that control tool
                    routing decisions. k_proj omitted (encodes content, not routing).
    """
    _banner("TRAIN — LoRA fine-tuning (GPU/CPU auto-detected)")
    t0 = time.time()

    # -- Import heavy deps here so setup/check still work without them --
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        _err(f"Missing dependency: {e}")
        _err("Run:  bash training/bootstrap.sh")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 0. Detect hardware — GPU wins; CPU is fallback
    # -----------------------------------------------------------------------

    hw = _detect_hardware()

    if hw["device"] == "cuda":
        _ok(f"GPU: {hw['gpu_name']}  ({hw['vram_gb']:.1f} GB VRAM)")
        _ok(f"     dtype={hw['dtype']}  fp16={hw['fp16']}  bf16={hw['bf16']}")
    else:
        _ok("No GPU detected — training on CPU (slow; run bootstrap.sh on a GPU machine)")

    # Resolve batch / grad_accum: 0 means "auto-detect from hardware"
    effective_batch      = batch_size  if batch_size  > 0 else hw["batch_size"]
    effective_grad_accum = grad_accum  if grad_accum  > 0 else hw["grad_accum"]
    _ok(f"     batch={effective_batch}  grad_accum={effective_grad_accum}  "
        f"effective={effective_batch * effective_grad_accum}")

    torch_dtype = (torch.bfloat16 if hw["bf16"] else
                   torch.float16  if hw["fp16"] else
                   torch.float32)
    device_map  = "auto" if hw["device"] == "cuda" else "cpu"

    # -----------------------------------------------------------------------
    # 1. Decide which model to load: local model_hf/ or HF Hub
    # -----------------------------------------------------------------------

    if MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir()):
        model_path = str(MODEL_HF_DIR)
        _ok(f"Loading model from local model_hf/ ({model_path})")
    else:
        model_path = HF_MODEL_ID
        _ok(f"model_hf/ not found — loading from HuggingFace Hub: {model_path}")
        _warn("This requires internet access and acceptance of Gemma licence.")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            _warn("Set HF_TOKEN if the model is gated.")

    # -----------------------------------------------------------------------
    # 2. Load tokenizer — prefer pre-trained extension from train_tokenizer.py
    # -----------------------------------------------------------------------

    # If train_tokenizer.py has already run, load the extended tokenizer that
    # has all 319 domain tokens with warmed-up embeddings.  Otherwise fall back
    # to adding the 12 core DOMAIN_TOKENS to the base tokenizer.
    if TOKENIZER_EXTENDED.exists() and any(TOKENIZER_EXTENDED.iterdir()):
        tok_source = str(TOKENIZER_EXTENDED)
        _ok(f"Loading pre-trained extended tokenizer from {TOKENIZER_EXTENDED}/")
        _ok("  (run training/train_tokenizer.py first if this is the initial run)")
    else:
        tok_source = model_path
        _warn("tokenizer_extended/ not found — loading base tokenizer + adding 12 DOMAIN_TOKENS")
        _warn("  Run python3 training/train_tokenizer.py first for better embedding init.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=False)
    except Exception as e:
        _err(f"Tokenizer load failed: {e}")
        _err("If using local model_hf/, ensure tokenizer.model is present.")
        _err(f"If missing, copy from HF Hub: huggingface-cli download {HF_MODEL_ID} tokenizer.model")
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        _warn("pad_token was None; set to eos_token")

    # Only add DOMAIN_TOKENS if we didn't load the full extended tokenizer
    if tok_source == model_path:
        existing  = set(tokenizer.get_vocab().keys())
        new_tokens = [t for t in DOMAIN_TOKENS if t not in existing]
        if new_tokens:
            n_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
            _ok(f"Added {n_added} domain tokens to tokenizer: {new_tokens[:3]}…")
        else:
            n_added = 0
            _ok("All domain tokens already in vocabulary — none added")
    else:
        n_added = 0  # already added by train_tokenizer.py

    print(f"  Vocabulary size: {len(tokenizer)} tokens")

    # -----------------------------------------------------------------------
    # 3. Load base model in float32 on CPU
    # -----------------------------------------------------------------------

    load_desc = f"{torch_dtype} on {device_map}"
    _ok(f"Loading base model ({load_desc}) — this takes 1–2 minutes …")
    load_start = time.time()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        _err(f"Model load failed: {e}")
        sys.exit(1)

    _ok(f"Model loaded in {_elapsed(load_start)}")
    n_params = sum(p.numel() for p in model.parameters())
    _ok(f"Total parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Resize embedding table to match the extended vocabulary.
    # When loading from tokenizer_extended/, the vocab is already larger than
    # the base model — resize and then restore warm embeddings from embed_init.pt.
    base_vocab_size = model.config.vocab_size
    if len(tokenizer) > base_vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        _ok(f"Resized embeddings: {base_vocab_size:,} → {len(tokenizer):,} tokens")

        # Restore pre-trained embeddings for new tokens if embed_init.pt exists
        embed_init_path = TOKENIZER_EXTENDED / "embed_init.pt"
        if embed_init_path.exists():
            try:
                saved = torch.load(str(embed_init_path), map_location="cpu", weights_only=True)
                pre_trained: torch.Tensor = saved["embeddings"]   # [n_new, hidden]
                saved_base_size: int      = saved["base_vocab_size"]
                embed_table = model.model.embed_tokens.weight
                # Sanity-check dimensions before writing
                n_new = len(tokenizer) - saved_base_size
                if pre_trained.shape == (n_new, embed_table.shape[1]):
                    embed_table.data[saved_base_size: saved_base_size + n_new] = pre_trained
                    _ok(f"Loaded pre-trained embeddings for {n_new} new tokens from {embed_init_path.name}")
                else:
                    _warn(f"embed_init.pt shape {pre_trained.shape} doesn't match "
                          f"expected ({n_new}, {embed_table.shape[1]}) — using random init")
            except Exception as e:
                _warn(f"Could not load embed_init.pt: {e} — using random init")
        else:
            _warn("embed_init.pt not found — new token embeddings start from random init")
            _warn("  (run python3 training/train_tokenizer.py to fix this)")
    elif n_added > 0:
        # Fallback for the 12-token DOMAIN_TOKENS case
        model.resize_token_embeddings(len(tokenizer))
        _ok(f"Resized embeddings to {len(tokenizer):,} tokens (random init for new tokens)")

    # -----------------------------------------------------------------------
    # 4. Apply LoRA — inject trainable adapter matrices into attention layers
    # -----------------------------------------------------------------------
    #
    # How LoRA works:
    #   For each target weight matrix W (shape [out, in]), LoRA adds two small
    #   matrices A (shape [r, in]) and B (shape [out, r]) where r << min(out,in).
    #   The effective weight becomes:  W + (lora_alpha/r) * B @ A
    #   During training: W stays frozen; only A and B are updated.
    #   During inference: B @ A can be merged into W (merge_and_unload) so there
    #   is zero overhead — the adapter "disappears" into the base weights.
    #
    # Why q_proj and v_proj (not k_proj or mlp)?
    #   q_proj = query projection: controls WHAT the model pays attention to.
    #            Changing q_proj steers which context tokens influence the output.
    #            This is the key lever for changing "which tool to call".
    #   v_proj = value projection: controls WHAT information is extracted when
    #            attending. Adapting this lets the model extract tool-relevant
    #            features (e.g. "memory" → dispatch to linux_memory_usage).
    #   k_proj = key projection: encodes where context tokens "are" in the
    #            attention key space. This encodes content/position, not routing.
    #            Leaving it frozen preserves the model's positional understanding.
    #   mlp    = fully-connected layers: general reasoning + factual knowledge.
    #            We don't need to change general reasoning for tool dispatch.
    #
    # Why rank r=8?
    #   The rank r controls how many "directions" the adapter can change.
    #   r=1  → extremely limited, may not learn dispatch at all
    #   r=8  → standard for small fine-tuning tasks; ~4M extra params on 268M base
    #   r=32 → used for larger tasks; 4x more params, 4x more memory
    #   r=8 is the right choice here: 12 tools is a small task with clear patterns.
    #
    # Why lora_alpha=16 with r=8?
    #   The effective update scale = lora_alpha / r = 16 / 8 = 2.0
    #   This scales the adapter output before adding to the frozen weight.
    #   Convention: set alpha = 2*r for a scale of 2.0 (empirically robust).
    #   Higher alpha → adapter has more influence → faster learning but more risk
    #   of overwriting base knowledge (catastrophic forgetting).
    #
    # Why lora_dropout=0.05?
    #   Dropout randomly zeros adapter activations during training (5% of them).
    #   On small datasets (< 1000 examples), this prevents memorization of
    #   training examples. At 0.05 it's light enough to not hurt convergence.
    #
    # Expected print_trainable_parameters output:
    #   trainable params: ~4,194,304 || all params: ~272,000,000
    #   trainable%: ~0.35% (the LoRA adapter is <1% of total params)

    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,  # we're doing causal language modeling
        r              = 8,                    # adapter rank (see above)
        lora_alpha     = 16,                   # scale = alpha/r = 2.0 (see above)
        lora_dropout   = 0.05,                 # light regularization (see above)
        target_modules = ["q_proj", "v_proj"], # attention query + value (see above)
        bias           = "none",               # don't adapt bias terms → fewer params
        inference_mode = False,                # keep adapter in training mode
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # prints trainable% — expect ~0.3-0.5%

    # -----------------------------------------------------------------------
    # 5. Format the dataset into Gemma 3 chat template strings
    # -----------------------------------------------------------------------

    _ok(f"Loading training data from {DATA_FILE} …")
    raw_lines = DATA_FILE.read_text().strip().splitlines()
    _ok(f"  {len(raw_lines)} examples")

    def _format_example(line: str) -> Optional[str]:
        """Convert one dispatch_pairs JSONL line to a training string.

        INPUT FORMAT (from generate_pairs.py):
          {
            "messages": [
              {"role": "system",  "content": "Call the right tool."},
              {"role": "user",    "content": "how much ram is used"}
            ],
            "tools": [{ <tool JSON schema> }],
            "target": {"name": "linux_memory_usage", "arguments": {}}
          }

        OUTPUT FORMAT (Gemma 3 function-call style):
          We use the Gemma 3 chat template via tokenizer.apply_chat_template,
          which produces:

            <bos><start_of_turn>user
            [system prompt + tool schemas]
            <end_of_turn>
            <start_of_turn>model
            {"name": "linux_memory_usage", "arguments": {}}
            <end_of_turn>

        The SFTTrainer trains the model to generate the "model" turn given
        the "user" turn.  This is the standard supervised fine-tuning (SFT)
        loss: cross-entropy on the output tokens only (DataCollatorForSeq2Seq
        masks the input tokens' loss contribution).

        WHY INCLUDE TOOL SCHEMA IN THE PROMPT?
        The schema tells the model the available tool names and their
        parameter shapes.  This mirrors inference time, where the agent also
        passes the schema.  Without this, the model only learns names without
        understanding their interface.
        """
        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            return None

        messages = ex.get("messages", [])
        tools    = ex.get("tools", [])
        target   = ex.get("target", {})

        if not messages or not target:
            return None

        # Build the user-facing content: system prompt + tool schema + user query
        system_content = next(
            (m["content"] for m in messages if m["role"] == "system"),
            "Call the right tool."
        )
        user_content = next(
            (m["content"] for m in messages if m["role"] == "user"),
            ""
        )

        # Format tool schemas as a compact JSON block in the prompt.
        # This is simpler than using the Gemma 3 native function-call tokens,
        # which vary across Ollama builds and may not be in the HF tokenizer.
        tools_text = json.dumps(tools, separators=(",", ":"))

        # The expected model output is the target tool call as JSON.
        # Keep it compact — the model should learn to emit exactly this.
        target_json = json.dumps(target, separators=(",", ":"))

        # Build as a two-turn conversation: user prompt → model response
        combined_messages = [
            {
                "role": "user",
                "content": (
                    f"{system_content}\n\n"
                    f"Available tools: {tools_text}\n\n"
                    f"User request: {user_content}"
                ),
            },
            {
                "role": "model",
                "content": target_json,
            },
        ]

        # Apply Gemma 3 chat template.
        # add_generation_prompt=False because we include the response in the
        # "model" turn — we want the full sequence for SFT loss computation.
        try:
            text = tokenizer.apply_chat_template(
                combined_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: manual template (some older transformers versions)
            text = (
                f"<bos><start_of_turn>user\n"
                f"{system_content}\n\n"
                f"Available tools: {tools_text}\n\n"
                f"User request: {user_content}\n"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
                f"{target_json}\n"
                f"<end_of_turn>"
            )

        return text

    formatted = []
    skipped = 0
    for line in raw_lines:
        text = _format_example(line)
        if text:
            formatted.append({"text": text})
        else:
            skipped += 1

    if skipped:
        _warn(f"Skipped {skipped} malformed examples")

    _ok(f"Formatted {len(formatted)} training examples")

    # Sanity-print one example so you can verify the format looks correct
    if formatted:
        print("\n  --- Sample training string (first example, truncated to 300 chars) ---")
        sample = formatted[0]["text"]
        print(f"  {sample[:300]}…" if len(sample) > 300 else f"  {sample}")
        print("  ---")

    train_dataset = Dataset.from_list(formatted)

    # -----------------------------------------------------------------------
    # 5b. Build probe dataset for DispatchCallback
    #     We pick up to 20 examples uniformly spread across the training set.
    #     For each, we record: (full_text, prompt_token_length, tool_name).
    #     The callback computes response-only loss at the end of each epoch.
    # -----------------------------------------------------------------------

    probe_data: list = []
    step = max(1, len(raw_lines) // 20)
    for i in range(0, len(raw_lines), step):
        line = raw_lines[i].strip()
        if not line:
            continue
        try:
            ex = json.loads(line)
            full_text  = _fmt_example(ex, tokenizer)
            prompt_txt = _fmt_prompt_only(ex, tokenizer)
            tool_name  = ex["target"]["name"]
            if full_text and prompt_txt and tool_name:
                prompt_len = len(tokenizer(prompt_txt, add_special_tokens=False)["input_ids"])
                probe_data.append((full_text, prompt_len, tool_name))
        except Exception:
            pass

    _ok(f"Probe dataset: {len(probe_data)} examples covering "
        f"{len({d[2] for d in probe_data})} distinct tools")

    dispatch_callback = DispatchCallback(
        probe_data=probe_data,
        tokenizer=tokenizer,
        log_path=MODEL_LORA / "training_log.jsonl",
    )

    # -----------------------------------------------------------------------
    # 6. Configure training
    # -----------------------------------------------------------------------

    # WHY THESE HYPERPARAMETERS?
    #
    # batch_size=1       : CPU RAM constraint. The model + gradients + optimizer
    #                      state fit in ~8 GB at batch=1. Larger batches would
    #                      OOM or swap heavily.
    #
    # grad_accum=4       : Simulates a batch of 4 without extra RAM. Gradients
    #                      from 4 micro-batches are accumulated before each weight
    #                      update. Effective batch size = 1 × 4 = 4.
    #
    # learning_rate=2e-4 : Standard for LoRA. The adapter is small, so it can
    #                      tolerate a slightly higher LR than full fine-tuning.
    #
    # warmup_ratio=0.1   : First 10% of steps use a linear LR warmup to avoid
    #                      gradient spikes at the start.
    #
    # weight_decay=0.01  : L2 regularisation on adapter weights. Keeps them small.
    #
    # max_seq_length=512 : Truncate sequences at 512 tokens. The tool schema +
    #                      user query + tool call typically fit in 150–250 tokens.
    #                      Capping at 512 saves memory and time.
    #
    # save_steps=50      : Save a checkpoint every 50 steps. With 48 examples
    #                      and batch=1, each epoch is 48 steps, so we save at
    #                      the end of each epoch.

    MODEL_LORA.mkdir(parents=True, exist_ok=True)

    # Telemetry log path
    log_path = MODEL_LORA / "training_log.jsonl"
    if log_path.exists():
        log_path.unlink()  # start fresh each run
    _ok(f"Telemetry log: {log_path}")

    # Estimate training time for the user
    n_steps = (len(formatted) * epochs) // max(1, effective_batch * effective_grad_accum)
    sec_per_step = 0.3 if hw["device"] == "cuda" else 5.0   # rough GPU vs CPU estimate
    print(f"\n  Estimated training steps: {n_steps}")
    print(f"  Rough timing @ ~{sec_per_step}s/step ({hw['device'].upper()}): "
          f"~{int(n_steps * sec_per_step // 60)} minutes")
    print(f"  (Actual time depends on sequence length and GPU/CPU speed)\n")

    # gradient_checkpointing trades compute for memory: instead of storing all
    # intermediate activations for backprop, they are recomputed on the backward
    # pass. On GPU this frees ~30% VRAM at ~15% extra compute — worth it.
    # On CPU it's slower and RAM isn't usually the bottleneck, so we skip it.
    use_gc = hw["device"] == "cuda"

    # adamw_torch_fused: PyTorch 2.0+ fused CUDA kernel for AdamW — combines
    # the element-wise operations into one GPU kernel call, ~10-20% faster.
    # Only available on CUDA; CPU must use the standard adamw_torch.
    optim_str = "adamw_torch_fused" if hw["device"] == "cuda" else "adamw_torch"

    try:
        # SFTConfig (trl >= 0.8) is the modern interface; falls back to TrainingArguments
        training_args = SFTConfig(
            output_dir                  = str(MODEL_LORA),  # where checkpoints are saved
            num_train_epochs            = epochs,            # full passes over the dataset

            # Batch size and accumulation: set from _detect_hardware() above
            per_device_train_batch_size = effective_batch,
            gradient_accumulation_steps = effective_grad_accum,
            # effective_batch_size = per_device * grad_accum (= 8 for RTX 3080 Ti)

            # Learning rate: 2e-4 is the standard LoRA LR.
            # Higher than full fine-tuning because adapters are small and need
            # more aggressive updates. Range: 1e-4 (conservative) to 5e-4 (aggressive).
            learning_rate               = lr,

            # Warmup: linear ramp from 0 to lr over first 10% of steps.
            # Prevents large gradient updates at the start when the adapter weights
            # are randomly initialized and may point in chaotic directions.
            warmup_ratio                = 0.1,

            # Weight decay: L2 penalty on adapter weights (not on base model).
            # Keeps adapter weights from growing too large → prevents overfitting.
            weight_decay                = 0.01,

            # Log metrics every 5 steps → DispatchCallback.on_log fires every 5 steps
            logging_steps               = 5,

            # Save a checkpoint every 50 steps; keep at most 2 (older ones deleted)
            save_steps                  = 50,
            save_total_limit            = 2,

            # Mixed precision: set from _detect_hardware()
            # fp16=True on RTX 30xx/40xx, bf16=True on A100/H100, both False on CPU
            fp16                        = hw["fp16"],
            bf16                        = hw["bf16"],

            optim                       = optim_str,       # fused AdamW on GPU
            gradient_checkpointing      = use_gc,          # recompute activations on GPU

            # 0 workers = dataset loaded in main process; avoids forking issues
            # on Windows/WSL where multiprocessing fork can hang
            dataloader_num_workers      = 0,

            # Disable external reporters (WandB, TensorBoard, etc.)
            # Our DispatchCallback handles all logging to training_log.jsonl
            report_to                   = "none",

            # Max sequence length: tool schema + query + response = ~200-400 tokens.
            # 512 gives headroom without wasting memory on padding.
            max_seq_length              = 512,
            dataset_text_field          = "text",   # column name in our Dataset dict

            # Packing: disabled. Packing concatenates multiple short sequences to
            # fill max_seq_length, which improves GPU utilization on large datasets.
            # On our small dataset it makes loss masking harder and can confuse the
            # model by mixing tool-call boundaries between examples.
            packing                     = False,
        )
    except TypeError:
        # Older trl versions (< 0.8) don't have SFTConfig — fall back to TrainingArguments
        _warn("SFTConfig not found in this trl version — using TrainingArguments")
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            output_dir                  = str(MODEL_LORA),
            num_train_epochs            = epochs,
            per_device_train_batch_size = effective_batch,
            gradient_accumulation_steps = effective_grad_accum,
            learning_rate               = lr,
            warmup_ratio                = 0.1,
            weight_decay                = 0.01,
            logging_steps               = 5,
            save_steps                  = 50,
            save_total_limit            = 2,
            fp16                        = hw["fp16"],
            bf16                        = hw["bf16"],
            optim                       = optim_str,
            gradient_checkpointing      = use_gc,
            dataloader_num_workers      = 0,
            report_to                   = "none",
        )

    # -----------------------------------------------------------------------
    # 7. Create SFTTrainer and kick off training
    # -----------------------------------------------------------------------
    #
    # SFTTrainer (Supervised Fine-Tuning Trainer) from the TRL library wraps
    # HuggingFace Trainer with a few extras:
    #   - Understands dataset_text_field and handles tokenization automatically
    #   - Supports packing (disabled here) and response-template masking
    #
    # We pre-applied the chat template ourselves (_format_example above), so we
    # pass raw text strings and tell SFTTrainer which column to read via
    # dataset_text_field="text".
    #
    # DispatchCallback is registered here so the Trainer will call its hooks
    # at every logging step and at the end of each epoch.
    #
    # What happens during trainer.train():
    #   For each step:
    #     1. Sample a mini-batch of examples
    #     2. Forward pass → compute cross-entropy loss on model-response tokens
    #     3. Backward pass → compute gradients for LoRA A, B matrices only
    #        (base model weights have requires_grad=False → no gradients flow to them)
    #     4. (every grad_accum steps) optimizer step → update A and B
    #     5. (every logging_steps steps) DispatchCallback.on_log → write to jsonl
    #   For each epoch end:
    #     DispatchCallback.on_epoch_end → run response-loss probe → print table
    #   After all epochs:
    #     DispatchCallback.on_train_end → final probe

    try:
        trainer = SFTTrainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_dataset,
            tokenizer       = tokenizer,
            callbacks       = [dispatch_callback],
        )
    except TypeError as e:
        _warn(f"SFTTrainer signature mismatch ({e}) — trying without tokenizer/callbacks arg")
        trainer = SFTTrainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_dataset,
        )
        # Register callback manually if SFTTrainer accepted it
        try:
            trainer.add_callback(dispatch_callback)
        except Exception:
            _warn("Could not register DispatchCallback — per-epoch probes disabled")

    _ok("Starting training …")
    print(f"  {len(formatted)} examples × {epochs} epochs = ~{len(formatted)*epochs} steps total")
    print(f"  Per-epoch probe: response-only loss on {len(probe_data)} examples")
    print(f"  Telemetry: {log_path}\n")

    try:
        train_result = trainer.train()
        _ok(f"Training complete in {_elapsed(t0)}")
        _ok(f"Final loss: {train_result.training_loss:.4f}")
        _ok(f"Telemetry written to {log_path}  ({log_path.stat().st_size if log_path.exists() else 0} bytes)")
    except KeyboardInterrupt:
        _warn("Training interrupted by user. Saving partial checkpoint …")
    except Exception as e:
        _err(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 8. Save LoRA adapter (not the full model — just the small adapter)
    # -----------------------------------------------------------------------

    _ok(f"Saving LoRA adapter to {MODEL_LORA} …")
    try:
        model.save_pretrained(str(MODEL_LORA))
        tokenizer.save_pretrained(str(MODEL_LORA))
        _ok("Adapter saved")
    except Exception as e:
        _err(f"Save failed: {e}")
        sys.exit(1)

    # Print what was saved
    saved_files = list(MODEL_LORA.iterdir())
    print(f"\n  Saved files ({len(saved_files)}):")
    for f in sorted(saved_files):
        size = f.stat().st_size / 1e6 if f.is_file() else 0
        print(f"    {f.name}  ({size:.1f} MB)" if size > 0.1 else f"    {f.name}")

    print(f"\n  Total training time: {_elapsed(t0)}")
    print(f"\n  Next step: python3 {__file__} --mode export")


# ===========================================================================
# MODE: export
# ===========================================================================

def mode_export() -> None:
    """Merge LoRA adapter into base weights, convert to GGUF, write Modelfile.

    STEPS
    -----
    1. Load base model + LoRA adapter from model_lora/
    2. Merge adapter weights into base (adapter_model × merge_and_unload)
    3. Save merged HF model to model_merged/
    4. Convert merged model → GGUF using the gguf package
    5. Write training/Modelfile
    6. Print the `ollama create` command
    """
    _banner("EXPORT — merge LoRA + convert to GGUF + write Modelfile")
    t0 = time.time()

    if not MODEL_LORA.exists() or not any(MODEL_LORA.iterdir()):
        _err(f"model_lora/ not found or empty at {MODEL_LORA}")
        _err("Run --mode train first.")
        sys.exit(1)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as e:
        _err(f"Missing dependency: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1+2. Load base + LoRA, then merge
    # -----------------------------------------------------------------------

    # Decide base model path (same logic as train)
    if MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir()):
        base_path = str(MODEL_HF_DIR)
    else:
        base_path = HF_MODEL_ID

    _ok(f"Loading base model from {base_path} …")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        _err(f"Base model load failed: {e}")
        sys.exit(1)

    _ok(f"Loading LoRA adapter from {MODEL_LORA} …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_LORA))
        model = PeftModel.from_pretrained(base_model, str(MODEL_LORA))
    except Exception as e:
        _err(f"Adapter load failed: {e}")
        sys.exit(1)

    # Resize embeddings if vocabulary was extended during training
    if len(tokenizer) > base_model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        _ok(f"Resized embeddings to {len(tokenizer)}")

    # merge_and_unload: multiplies LoRA matrices (A × B) into the base weight
    # matrices, then removes the LoRA hooks.  Result: a plain nn.Module with
    # slightly different weights — no LoRA overhead at inference time.
    _ok("Merging LoRA weights into base model (merge_and_unload) …")
    merged = model.merge_and_unload()
    _ok(f"Merge complete in {_elapsed(t0)}")

    # -----------------------------------------------------------------------
    # 3. Save merged HF model
    # -----------------------------------------------------------------------

    MODEL_MERGED.mkdir(parents=True, exist_ok=True)
    _ok(f"Saving merged model to {MODEL_MERGED} …")
    merged.save_pretrained(str(MODEL_MERGED), safe_serialization=True)
    tokenizer.save_pretrained(str(MODEL_MERGED))
    _ok("Merged model saved")

    # -----------------------------------------------------------------------
    # 4. Convert merged model to GGUF
    # -----------------------------------------------------------------------

    # Strategy: try two approaches in order of preference:
    #   A. llama.cpp convert_hf_to_gguf.py (highest quality, proper quant support)
    #   B. gguf Python package (simpler, F32 or F16 only)
    #
    # If llama.cpp is not installed, fall back to the gguf package.

    _ok("Converting merged model to GGUF …")

    converted_to_gguf = False

    # Approach A: llama.cpp converter script
    llama_cpp_convert = _find_llama_cpp_converter()
    if llama_cpp_convert:
        _ok(f"Found llama.cpp converter: {llama_cpp_convert}")
        cmd = [
            sys.executable, llama_cpp_convert,
            str(MODEL_MERGED),
            "--outfile", str(GGUF_OUT),
            "--outtype", "q8_0",   # match original quantisation
        ]
        _ok(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            _ok(f"GGUF written to {GGUF_OUT}")
            converted_to_gguf = True
        else:
            _warn(f"llama.cpp converter failed:\n{result.stderr}")

    if not converted_to_gguf:
        # Approach B: gguf Python package — write F16 GGUF manually
        _ok("Falling back to gguf Python package (F16 precision) …")
        try:
            converted_to_gguf = _export_gguf_python(merged, tokenizer)
        except Exception as e:
            _warn(f"GGUF Python export failed: {e}")
            _warn("The merged HF model is still available at:")
            _warn(f"  {MODEL_MERGED}")
            _warn("You can convert it manually with llama.cpp:")
            _warn(f"  python3 convert_hf_to_gguf.py {MODEL_MERGED} --outtype q8_0")

    # -----------------------------------------------------------------------
    # 5. Write Modelfile
    # -----------------------------------------------------------------------

    _ok(f"Writing Modelfile to {MODELFILE} …")

    if converted_to_gguf and GGUF_OUT.exists():
        from_line = f"FROM {GGUF_OUT}"
    else:
        # Fallback: reference merged HF path (won't work with `ollama create`
        # directly but is a useful placeholder)
        from_line = f"# GGUF conversion failed — use llama.cpp to convert:\n# {MODEL_MERGED}"

    modelfile_content = f"""# Modelfile for functiongemma:270m-suryaos
# Generated by finetune_dispatch.py on {time.strftime('%Y-%m-%d')}
#
# This model is functiongemma:270m fine-tuned with LoRA on SuryaOS tool dispatch.
# Training data: {DATA_FILE}
# Training scope: 12 system tools (v3: 500 examples target)
# Future (v4): 2000+ examples when agent handles code/test/git/IDE workflows.

{from_line}

# Use the same template as the original functiongemma:270m
TEMPLATE {{{{ .Prompt }}}}
RENDERER functiongemma
PARSER functiongemma

# Keep the original inference parameters
PARAMETER top_k 64
PARAMETER top_p 0.95

# System prompt for tool dispatch
SYSTEM \"""You are a tool dispatcher for SuryaOS. Given the user's request and available tools, call the correct tool with appropriate arguments. Respond only with the tool call JSON.\"""
"""

    MODELFILE.write_text(modelfile_content)
    _ok(f"Modelfile written to {MODELFILE}")

    # -----------------------------------------------------------------------
    # 6. Print the ollama create command
    # -----------------------------------------------------------------------

    print(f"\n  Total export time: {_elapsed(t0)}")
    print()
    print("  ================================================================")
    print("  To register the fine-tuned model with Ollama, run:")
    print()
    print(f"    ollama create functiongemma:270m-suryaos -f {MODELFILE}")
    print()
    print("  Then test it:")
    print("    ollama run functiongemma:270m-suryaos 'how much ram is used'")
    print("  ================================================================")


def _find_llama_cpp_converter() -> Optional[str]:
    """Search common locations for llama.cpp's convert_hf_to_gguf.py."""
    candidates = [
        # System-wide installs / git clones in standard locations
        "/usr/local/lib/llama.cpp/convert_hf_to_gguf.py",
        "/opt/llama.cpp/convert_hf_to_gguf.py",
        str(Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"),
        str(Path.home() / "src" / "llama.cpp" / "convert_hf_to_gguf.py"),
        # In PATH (pip-installed llama-cpp-python may install helper scripts)
        shutil.which("convert_hf_to_gguf.py") or "",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


def _export_gguf_python(model, tokenizer) -> bool:
    """Write a minimal GGUF file using the gguf Python package.

    This produces a F16 GGUF (not Q8_0) — suitable for testing but larger
    than the original.  Use llama.cpp convert_hf_to_gguf.py for production.

    Returns True if successful.
    """
    try:
        import gguf
        import torch
        import numpy as np
    except ImportError as e:
        _warn(f"Cannot export GGUF: {e}")
        return False

    writer = gguf.GGUFWriter(str(GGUF_OUT), "gemma3")

    # Write architecture metadata
    config = model.config
    writer.add_architecture()
    writer.add_name("functiongemma-suryaos-270m")
    writer.add_description("functiongemma:270m fine-tuned on SuryaOS tool dispatch (LoRA v3)")

    # Gemma 3 metadata fields
    try:
        writer.add_context_length(getattr(config, "max_position_embeddings", 32768))
        writer.add_embedding_length(getattr(config, "hidden_size", 640))
        writer.add_block_count(getattr(config, "num_hidden_layers", 18))
        writer.add_feed_forward_length(getattr(config, "intermediate_size", 3072))
        writer.add_head_count(getattr(config, "num_attention_heads", 8))
        writer.add_head_count_kv(getattr(config, "num_key_value_heads", 4))
        writer.add_layer_norm_rms_eps(getattr(config, "rms_norm_eps", 1e-6))
        writer.add_vocab_size(len(tokenizer))
    except Exception as e:
        _warn(f"Some metadata fields not written: {e}")

    # Write tensors as F16
    _ok("Writing tensors to GGUF (F16) …")
    n_written = 0
    for name, param in model.named_parameters():
        try:
            # Convert HF name → GGUF name (reverse of convert step)
            gguf_name = _hf_to_gguf_name(name)
            if gguf_name is None:
                continue
            arr = param.detach().float().numpy().astype(np.float16)
            writer.add_tensor(gguf_name, arr)
            n_written += 1
        except Exception as e:
            _warn(f"  Failed to write {name}: {e}")

    _ok(f"Wrote {n_written} tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = GGUF_OUT.stat().st_size / 1e6
    _ok(f"GGUF file: {GGUF_OUT} ({size_mb:.0f} MB) [F16 — use llama.cpp for Q8_0]")
    return True


def _hf_to_gguf_name(hf_name: str) -> Optional[str]:
    """Map a HuggingFace parameter name to llama.cpp GGUF convention."""
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    if hf_name == "lm_head.weight":
        return "output.weight"
    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".")
        # model.layers.N.sub.module.weight
        layer_num = parts[2]
        rest = ".".join(parts[3:])
        prefix = f"blk.{layer_num}."
        reverse_map = {
            "self_attn.q_proj.weight":              "attn_q.weight",
            "self_attn.k_proj.weight":              "attn_k.weight",
            "self_attn.v_proj.weight":              "attn_v.weight",
            "self_attn.o_proj.weight":              "attn_output.weight",
            "mlp.gate_proj.weight":                 "ffn_gate.weight",
            "mlp.up_proj.weight":                   "ffn_up.weight",
            "mlp.down_proj.weight":                 "ffn_down.weight",
            "input_layernorm.weight":               "attn_norm.weight",
            "post_feedforward_layernorm.weight":    "ffn_norm.weight",
            "post_attention_layernorm.weight":      "attn_post_norm.weight",
        }
        gguf_rest = reverse_map.get(rest)
        return prefix + gguf_rest if gguf_rest else None
    return None


# ===========================================================================
# MODE: all
# ===========================================================================

def mode_all(epochs: int, lr: float) -> None:
    """Run check → train → export (convert is skipped if model_hf/ exists)."""
    _banner("ALL — running full pipeline: check → train → export")

    # Step 1: check
    ok = mode_check()
    if not ok:
        _err("Check failed — fix the issues above before continuing.")
        sys.exit(1)

    # Step 2: convert only if model_hf/ doesn't exist
    if not MODEL_HF_DIR.exists() or not any(MODEL_HF_DIR.iterdir()):
        _ok("model_hf/ not found — running convert step …")
        mode_convert()
    else:
        _ok("model_hf/ exists — skipping convert (use --mode convert to force)")

    # Step 3: train
    mode_train(epochs=epochs, lr=lr)

    # Step 4: export
    mode_export()

    _banner("ALL DONE")
    print(f"\n  Fine-tuned model Modelfile: {MODELFILE}")
    print(f"  Register with: ollama create functiongemma:270m-suryaos -f {MODELFILE}")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["setup", "check", "convert", "train", "export", "all"],
        help=(
            "setup: print install commands | "
            "check: verify environment | "
            "convert: GGUF→HF safetensors | "
            "train: LoRA fine-tuning | "
            "export: merge+GGUF+Modelfile | "
            "all: check→train→export"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3). "
             "With 48 examples, 3 epochs ≈ 15–25 min on CPU.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for LoRA training (default: 2e-4). "
             "Range: 1e-4 to 5e-4 for LoRA adapters.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size (default: 1 for CPU memory constraints).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4, effective batch = 4).",
    )

    args = parser.parse_args()

    if args.mode == "setup":
        mode_setup()
    elif args.mode == "check":
        mode_check()
    elif args.mode == "convert":
        mode_convert()
    elif args.mode == "train":
        mode_train(
            epochs     = args.epochs,
            lr         = args.lr,
            batch_size = args.batch_size,
            grad_accum = args.grad_accum,
        )
    elif args.mode == "export":
        mode_export()
    elif args.mode == "all":
        mode_all(epochs=args.epochs, lr=args.lr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
