# Bug fixes — what broke, why it broke, how we fixed it

> A running log of bugs caught during real training runs, with the
> intuition behind each fix. The goal is that someone reading this
> 6 months from now can understand what we *learned*, not just what
> code we changed.

This file is structured oldest → newest. Each bug entry has:
- **Symptom** — what the terminal showed
- **Root cause** — the actual mechanism
- **Mental model** — the intuition for why it broke
- **Fix** — what code changed
- **Validation** — how we know it's fixed
- **Lesson** — what to remember next time

---

## BUG-001 — Smart-init bug (May 2026)

### Symptom

After running `train_tokenizer.py` for the first time on RTX 3080 Ti
(18 minutes, 2 epochs), the output looked superficially OK:

```
[OK]  Smart init complete: 0 via subword avg, 251 via global mean fallback
[OK]  New token embedding norms (post-init) — mean=0.4962  std=0.0000
```

`std=0.0000` — every new token's embedding had **identical magnitude**.
Not "very similar" — *byte-for-byte the same vector*, 251 times over.

Downstream consequences:
- Every cosine similarity probe came back at +0.92 (including pairs
  that should have been near zero, like `linux_memory_usage` vs
  `brightness_set`)
- Loss plateaued at 6.7 from step 30 onwards — never broke through
- All 251 tokens' nearest neighbours were the same set:
  `<image_soft_token>`, plus various Kannada and Tamil tokens
- Sustained gradient norms in the 4-7 range (we clip at 1.0), meaning
  the optimiser was fighting the bad init the entire 18 minutes

### Root cause

```python
# This was supposed to give us subword IDs from the BASE vocabulary:
subword_ids = tokenizer.encode(token_str, add_special_tokens=False)
base_ids    = [sid for sid in subword_ids if sid < base_vocab_size]
```

The `tokenizer` we received had already had `add_tokens()` called on it
in Phase 1. HuggingFace tokenizers maintain an "added tokens trie" that
gets *checked first* during `encode()` — before the SentencePiece
subword splitter ever runs.

So `tokenizer.encode("linux_memory_usage")` returned `[262148]`, not
`["linux", "_", "memory", "_", "usage"]`. The trie matched the whole
string and short-circuited.

The filter `< base_vocab_size` then dropped that single new ID, leaving
`base_ids = []`. The fallback fired — for every single token —
assigning `embed_weight[:base_vocab_size].mean(dim=0)` to all 251 new
rows. That's the global vocabulary centroid.

So all 251 embeddings became the same vector. Identical clones.

### Mental model

Think of it like asking a translator to translate a word, but the word
is already in their dictionary as a registered phrase. They reply
"that's just X — index 262148" instead of breaking it down into the
syllables that compose it.

We needed the *naive* translator who doesn't know any of these new
words yet, so they're forced to spell them out phonetically. That's
the "clean" base tokenizer.

### Why this also explains the weird symptoms

- **Cosine sim ≈ 1 everywhere**: `cos(v, v) = 1` by definition. All
  vectors were the same vector.
- **Plateau at loss 6.7**: with all tokens at the same point, the
  language model can never assign different probabilities to different
  tokens given the same context — the cross-entropy floor is high.
- **Same nearest neighbours for every new token**: they all live at
  the same point, so they share the same neighbours. That point
  happens to be near `<image_soft_token>` because the multilingual
  vocab centroid lands there.
- **Gradient norms 4-7 throughout**: gradients flow through the LM
  head (which is tied to the embedding matrix). If many tokens want
  similar updates but the embedding matrix is tied to itself in a
  pathological way, the gradient amplifies. Combine with cold init
  norms half of base vocab, and you get persistent large gradients.

### Fix

`training/train_tokenizer.py` Phase 2:

```python
# Load a SECOND tokenizer instance — clean, with no added tokens.
from transformers import AutoTokenizer
base_tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=False, token=hf_token,
)
# ...later...
subword_ids = base_tokenizer.encode(token_str, add_special_tokens=False)
```

Five effective lines of code. The clean base tokenizer doesn't have
the new tokens in its trie, so `encode()` falls through to the
SentencePiece subword splitter. We get back actual subword IDs and
the average is meaningful.

### Validation

After re-running with the fix, expect to see:
- `Smart init complete: 251 via subword avg, 0 via global mean fallback`
- `New token embedding norms — mean=~0.99 std=0.05`-ish (close to base)
- Cosine probe: `same-tool forms` similar (~0.4-0.7), `cross-domain` low (~0.1-0.3)
- Demo block: `init = mean(['linux', 'memory', 'usage'])` instead of `mean([])`
- Gradient norms in the 0.3-1.0 range, occasional spikes only at start

### Lesson

**Tokenizer state is not commutative**. Calling `add_tokens()` mutates
the tokenizer in place; subsequent `encode()` calls will reflect that
state. If you need pre-mutation behaviour, you need a separate
instance — `copy.deepcopy()` is too brittle for a HuggingFace
tokenizer (rust-backed, has unpicklable state). Reload from the same
path is the safe pattern.

This bug was completely silent at the API level — `encode()` returned
a perfectly valid integer list. No exception, no warning, no log
message. The only signal was a single `std=0.0000` in the telemetry
output, and we caught it because we'd added that telemetry line.
**Telemetry that exposes invariants of correctness is more valuable
than telemetry that just records numbers.**

---

## BUG-002 — HuggingFace gated-model auth (May 2026)

### Symptom

```
[ERR]  Tokenizer load failed: You are trying to access a gated repo.
401 Client Error.
Cannot access gated repo for url
  https://huggingface.co/google/gemma-3-270m-it/resolve/main/config.json.
```

Even though the user had a paid HF account.

### Root cause

`AutoTokenizer.from_pretrained()` and `AutoModelForCausalLM.from_pretrained()`
were called without an explicit `token=` parameter. HuggingFace claims
to auto-detect from `HF_TOKEN` env var, but that auto-detection failed
silently in this environment (likely because the env var was set in a
new shell after the script was launched, or because a different env
mechanism took priority).

### Mental model

"Auto-detect" is a polite word for "best effort with silent fallback".
For gated models, **explicit token passing is mandatory**, not optional.

### Fix

Added `_get_hf_token()` helper to both `train_tokenizer.py` and
`finetune.py`. Resolves token from three sources in order:

1. `HF_TOKEN` env var (preferred)
2. `HUGGINGFACE_HUB_TOKEN` env var (legacy)
3. `~/.cache/huggingface/token` file (from `huggingface-cli login`)

Every `from_pretrained()` call now receives `token=hf_token` explicitly.
On 401/403/"gated" errors, we print exact 2-step recovery instructions
inline rather than a generic stack trace.

### Validation

```
[OK]  HuggingFace token found (length=37) — authenticated download
[OK]  Loading tokenizer from google/gemma-3-270m-it ...
config.json: 1.35kB ...
```

If the token is missing or wrong, the error message points at
exactly the two commands needed to fix it (license URL + token export).

### Lesson

**Don't trust framework auto-detection for security-critical state.**
Read it explicitly, log that you found it, and pass it explicitly down
the call stack. The cost is a few lines of code; the benefit is that
auth failures are diagnosable in one read of the terminal output.

---

## BUG-003 — Bootstrap installs CPU torch on GPU machine (May 2026)

### Symptom

After running `bootstrap.sh`, training scripts ran without errors but
all training happened on CPU — 30× slower than expected on the RTX 3080
Ti box. `torch.cuda.is_available()` returned `False`.

### Root cause

`requirements.txt` originally had:

```
torch>=2.2.0
torchvision
torchaudio
--extra-index-url https://download.pytorch.org/whl/cpu
```

The `--extra-index-url` line correctly pointed at CPU wheels — so even
on a GPU machine, pip would happily install the CPU torch (which IS a
valid torch, just without CUDA support). No error, no warning.

A second source of the same bug: when running `bootstrap.sh` a second
time on a machine where CPU torch was already installed, pip's
"already satisfied" logic refused to swap to the GPU wheel because
"torch" the package name was technically present.

### Mental model

pip thinks at the package-name level. To pip, `torch==2.5.1+cpu` and
`torch==2.5.1+cu121` are *the same package* — same name, same version
number. The `+cpu` / `+cu121` is a "local version" suffix and pip's
resolver doesn't gate on it.

Two different CUDA wheels can coexist on PyPI under one name; you have
to **uninstall first** and force-install from the right index URL.

### Fix

Two changes to `bootstrap.sh`:

1. Remove `torch` / `torchvision` / `torchaudio` from `requirements.txt`
   entirely. They are now installed by `bootstrap.sh` directly with the
   right `--index-url`, chosen from GPU detection.

2. When the existing torch is the wrong variant (CPU on GPU box),
   uninstall it first before reinstalling:

```bash
if "$PYTHON" -c "import torch" 2>/dev/null; then
    "$PIP" uninstall -y torch torchvision torchaudio
fi
"$PIP" install torch torchvision torchaudio --index-url "$TORCH_INDEX"
```

### Validation

```
[OK]   torch 2.5.1+cu121 | CUDA: True
[OK]   GPU device: NVIDIA GeForce RTX 3080 Ti Laptop GPU
[OK]   GPU VRAM: 16.0 GB
```

If `CUDA: True` shows up after bootstrap, the right wheel landed.

### Lesson

**Default behaviour quietly works on the wrong device.** CPU torch
running on a GPU machine is the worst-case silent regression: nothing
fails, training just takes 30× longer. The only way to catch it is an
explicit verify step that prints `CUDA: True/False` and an idempotent
reinstall path that swaps the variant.

The lesson generalises: any time a library has multiple build variants
under the same package name (CUDA torch, MKL numpy, GPU TensorFlow),
you cannot rely on `pip install -r requirements.txt` to land the right
one. Detection + explicit installation is the only safe pattern.

---

## BUG-004 — Bootstrap install always re-runs even when satisfied (May 2026)

### Symptom

Re-running `bootstrap.sh` on a fully-installed machine took 5+ minutes
because pip walked through every package, doing dependency resolution
and verifying versions even when nothing needed to change.

### Root cause

The script always called `pip install -r requirements.txt`
unconditionally. Pip's "already satisfied" logic skips actual
downloads but still does a full resolution pass.

### Fix

Three `[SKIP]` early-exits in `bootstrap.sh`:

1. **Step 4 (PyTorch)** — Skip if `import torch` works AND (on GPU
   machines) `torch.cuda.is_available()` is True
2. **Step 5 (requirements.txt)** — Skip if every name in
   `REQUIRED_IMPORTS` is importable
3. **Step 6 (bitsandbytes)** — Skip if `import bitsandbytes` works

Plus a `--reinstall` escape-hatch flag that bypasses all three skips.

### Validation

A re-run of `bootstrap.sh` on a fully-installed machine now completes
in ~3 seconds and prints:

```
[bootstrap]   [SKIP]  PyTorch 2.5.1+cu121 with CUDA 12.1 already installed
[bootstrap]   [SKIP]  All 12 required packages already importable
[bootstrap]   [SKIP]  bitsandbytes 0.43.1 already installed (QLoRA ready)
```

### Lesson

**Idempotent setup scripts are a force multiplier.** Once a setup
script is fast on re-run, you start running it more often — for
sanity checking, for onboarding, for CI smoke tests. The 5-line cost
of state checks pays back many times over.

---

## BUG-005 — Probe pairs measure frozen tokens (May 2026) ✓ FIXED in iter #3

### Symptom

Run #2 cosine probes for three pairs returned **identical** values
across all 5 epochs:

```
co-occurring ML libs                     +0.2937   (epoch 1)
co-occurring ML libs                     +0.2937   (epoch 2)
co-occurring ML libs                     +0.2937   (epoch 3)
co-occurring ML libs                     +0.2937   (epoch 4)
co-occurring ML libs                     +0.2937   (epoch 5)
```

Same exact-to-4-digits value for `co-occurring git ops` and
`co-occurring serving terms`. Other probes moved naturally.

### Root cause

The probe pairs were:
- `("torch", "transformers", "co-occurring ML libs")`
- `("merge", "commit", "co-occurring git ops")`
- `("GGUF", "ollama", "co-occurring serving terms")`

Both tokens in each pair are **already in the base Gemma vocabulary**
(they're common English/CS words). Our training only updates new
token embedding rows (rows 262145..262396); the base vocab rows are
explicitly *frozen* via the gradient hook in `phase_corpus_train`:

```python
def _zero_base_rows(grad):
    g = grad.clone()
    g[:base_vocab_size] = 0.0   # base vocab gradients zeroed every step
    return g
```

So the embeddings of `torch`, `transformers`, `merge`, etc. literally
cannot change during our training. Their cosine similarity is fixed at
whatever the pre-trained Gemma model assigned to them.

### Mental model

The probe was supposed to test "does corpus training cluster co-occurring
domain terms?" But for that to work, at least one token in each pair
must be a token *we're actually training*. The probe authors (us, in a
previous session) assumed all probe pairs would change; we forgot that
half of them reference frozen rows.

A useful probe must be able to move when the thing it measures is being
trained. A frozen-token probe is dead instrumentation.

### Fix

Update `PROBE_PAIRS` in `train_tokenizer.py` to ensure every pair
includes **at least one new token** (one that's in our `new_tokens.json`
list, hence trainable):

```python
PROBE_PAIRS = [
    # NEW vs NEW (same-tool variants — high expected after training)
    ("linux_memory_usage", "memory_usage", "same-tool forms"),
    ("linux_disk_usage",   "disk_usage",   "same-tool forms"),
    # NEW vs BASE (testing if new tokens learn association with related concepts)
    ("linux_memory_usage", "memory",       "new tool vs base concept"),
    ("krunner_launch",     "KRunner",      "new tool vs base name"),
    # NEW vs NEW (sibling tools — moderate similarity expected)
    ("linux_memory_usage", "linux_disk_usage", "sibling linux tools"),
    # NEW vs NEW (cross-domain — should stay low)
    ("linux_memory_usage", "brightness_set", "cross-domain (expected low)"),
    # ... DELETE the torch/transformers, merge/commit, GGUF/ollama pairs
]
```

### Validation

After the fix, all probe pairs should have at least one trainable
token. Across epochs, every pair should show non-zero movement (positive
or negative — both are valid signals). No pair should be locked at
exactly the same 4-digit value across multiple epochs.

### Lesson

**Telemetry must be testable against its own assumptions.** A probe
that always returns the same value is signalling its own brokenness, not
the system's state. Add a sanity check: at training start, verify that
each probe involves at least one trainable parameter.

This bug is a sibling of BUG-001 — both come from forgetting which
tokens are/aren't trainable in our setup. The general lesson: when you
have a "trainable subset" inside a larger frozen system, every piece
of code that *measures* the trainable subset must verify it's actually
looking at the right subset.

### Resolution (iteration #3, commit 89e0f4d)

`PROBE_PAIRS` in `train_tokenizer.py` updated. Three frozen-token pairs
removed and replaced with pairs that include at least one trainable new
token. Specifically:

| Removed (frozen) | Added (live) | Why the new pair works |
|---|---|---|
| `(torch, transformers)` | `(linux_memory_usage, memory)` | new tool ↔ base concept |
| `(merge, commit)` | `(kde_window_focus, kde_krunner_launch)` | KDE sibling tools |
| `(GGUF, ollama)` | `(kde_dialog_confirm, linux_battery_status)` | cross-category KDE vs Linux |

After fix: every probe has at least one new (trainable) token, so cosine
moves during training and gives live signal.

---

## BUG-006 — fp16 NaN explosion on RTX 30xx/40xx (May 2026) ✓ FIXED

### Symptom

After enabling GPU training (BUG-003 fix), every batch on RTX 3080 Ti
produced `loss=nan`:

```
[OK]   Training device: cuda:0
[OK]   Trainable parameters: 167,841,920
[WARN] Epoch 1 batch 0: loss=nan — skipping bad batch
[WARN] Epoch 1 batch 1: loss=nan — skipping bad batch
[WARN] Epoch 1 batch 2: loss=nan — skipping bad batch
... (every batch for 5 epochs)
[OK]   Epoch 1/5 stats  loss=0.0000  drift_from_init=0.0000  time=0m 12s
```

`drift_from_init=0.0000` is the smoking gun: every batch was skipped, so
no optimizer step ever happened, so embeddings never moved. The 12-second
"epoch" was just NaN forward-passes.

Cosine probes after epoch 1 still show smart-init values unchanged
(0.93 same-tool, 0.86 cross-domain) because no learning occurred.

### Root cause

`_detect_hardware()` selected `dtype=torch.float16` for RTX 30xx based
on a "datacenter cards only get bf16" heuristic:

```python
_dc_keywords  = ("A100", "H100", "H200", ...)
is_datacenter = any(kw in gpu_name for kw in _dc_keywords)
use_bf16 = is_datacenter and is_ampere_plus
use_fp16 = not use_bf16   # → True for RTX 3080 Ti
```

The heuristic was wrong. **All Ampere+ GPUs (compute ≥ 8.0) have hardware
bf16 tensor cores**, including RTX 30xx (8.6) and RTX 40xx/Ada (8.9).
The "datacenter only" rule was outdated 2022-era guidance.

The actual problem: **fp16 has only 5-bit exponent (max ~65504)**.
Gemma 3's attention softmax values can exceed 65504, producing inf →
NaN propagates through the loss computation. fp16 is a precision format
designed for inference, not training.

**bf16** has the **same 8-bit exponent as fp32** (~3.4×10³⁸ max). It
has less precision (7-bit mantissa vs fp16's 10-bit) but precision
doesn't matter when the alternative is NaN. Modern frameworks
(transformers, peft, trl) all default to bf16 on any Ampere+ GPU.

### Mental model

Think of fp16 as a tool with a tight tolerance — useful when you know
the values stay in range, deadly when one batch hits the ceiling.
bf16 trades precision for headroom: same dynamic range as fp32, just
fewer significant digits. For training, you almost never miss the
precision; you very often need the headroom.

The "datacenter vs consumer" distinction stopped mattering with Ampere
in 2020. Both classes have full bf16 hardware. The distinction only
exists in old NVIDIA marketing.

### Fix

`_detect_hardware()` in `finetune.py` and the equivalent block in
`train_tokenizer.py phase_smart_init()` updated:

```python
# OLD
_dc = any(k in gpu_name for k in ("A100", "H100", ...))
use_bf16 = is_datacenter and is_ampere_plus

# NEW (BUG-006 fix)
is_ampere_plus = compute[0] >= 8
use_bf16 = is_ampere_plus      # ANY Ampere+ → bf16 (stable)
use_fp16 = (not use_bf16) and compute[0] >= 7  # Turing/Volta only
```

Plus an **early-abort guard**: if 10 consecutive batches return NaN, the
training script bails with a clear error message pointing at the fix
command. No more 40-minute silent NaN runs.

### Validation

After the fix, RTX 3080 Ti reports:
```
GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU (16.0 GB, compute 8.6) — dtype=torch.bfloat16
```

And training produces real loss values (~6-8 → < 4) with non-zero drift.

### Lesson

**Heuristics that name specific products age badly.** "Datacenter cards
get the good thing, consumer cards get the lesser thing" was true in
the V100/T4 era; not since 2020. When the line stops being technically
meaningful, it stops protecting users.

The safer pattern is **capability-based selection**: read the actual
hardware feature flag (`compute[0] >= 8` → bf16 hardware exists) instead
of pattern-matching the marketing name. That way new cards are
auto-supported the day they ship.

This bug also illustrates why the **early-abort** is critical. Without
it, a user's training run "succeeds" (exit code 0) with embeddings
that never moved — a worst-case silent failure. With the guard,
broken runs fail fast and loud.

---

## BUG-007 — torch_dtype deprecation + grad warning flood (May 2026) ✓ FIXED

### Symptom

Two minor but annoying issues that surfaced after the GPU/bf16 fix:

```
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
```

And every grad-clip event flooding the terminal:
```
[WARN    ] Step 273: grad_norm=35.75 exceeded clip threshold (1.0). Gradient was clipped — preventing instability but slowing learning.
[WARN    ] Step 283: grad_norm=36.25 exceeded clip threshold (1.0). Gradient was clipped — preventing instability but slowing learning.
[WARN    ] Step 293: grad_norm=54.00 exceeded clip threshold (1.0). Gradient was clipped — preventing instability but slowing learning.
... (200+ lines)
```

### Root cause

1. **transformers 4.46+** renamed `torch_dtype=` to `dtype=` on
   `from_pretrained`. We had four callsites with the old name.
2. The Narrator emitted a `[WARN]` for *every* batch where `grad_norm > 1.5`.
   During corpus warm-up, embeddings are at random init, so every early
   batch has grad > 1.5 → 200+ identical-looking warnings drowning out
   the useful narration.
3. The warning text was 117 characters wide — wrapped on most terminals
   and wasted vertical space.

### Fix

**For (1):** `sed -i 's/torch_dtype=/dtype=/g'` across all training scripts.
Four sites: `train_tokenizer.py:766`, `finetune.py:1484`, `finetune.py:2042`,
`analyze_embeddings.py:145`.

**For (2)** — rate-limit grad warnings:

```python
# OLD: every grad > 1.5 warned
if grad_norm is not None and grad_norm > 1.5:
    self._emit("WARN", f"Step {step}: grad_norm={grad_norm:.2f} exceeded clip threshold (1.0). Gradient was clipped — preventing instability but slowing learning.")

# NEW: tiered + rate-limited + concise
if grad_norm > 5.0:
    # Big spike — always emit (this matters)
    self._emit("WARN", f"Step {step}: grad_norm={grad_norm:.1f} (clipped)")
elif grad_norm > 1.5 and step <= 30:
    # Cold-start clipping — warn every 5 steps for first 30 only
    if step % 5 == 0:
        self._emit("WARN", f"Step {step}: grad_norm={grad_norm:.1f} (early-step clipping is normal)")
# Beyond step 30 with grad < 5.0: stay silent — clipping is fine
```

**For (3)** — narrowed all narrator messages to ≤ 80 columns. Examples:

| Before (117 cols) | After (~60 cols) |
|---|---|
| `Step 273: loss stable at ~6.155 (<2% change) — model has converged for this lr` | `Step 273: loss 6.155 (±2%) — converged` |
| `Step 273: grad_norm=35.75 exceeded clip threshold (1.0). Gradient was clipped — preventing instability but slowing learning.` | `Step 273: grad_norm=35.7 (clipped)` |
| `Step 273: loss down 35% in last 10 steps (was 8.50, now 5.55) — rapid learning` | `Step 273: loss -35% in 10 steps (8.50→5.55) — rapid learning` |

### Validation

Re-running training on RTX 3080 Ti now produces:
- No deprecation warning at model load
- Clean log without grad-norm flood
- Lines fit on standard 80-column terminals

### Lesson

**Verbose telemetry erodes its own usefulness.** A warning that fires
for every batch becomes noise — users learn to ignore the prefix and
miss the *actual* anomalies when they appear. The right pattern:
- ALWAYS emit for genuinely abnormal events (grad > 5.0)
- Rate-limit for known-but-tolerable events (grad 1.5-5.0 during warmup)
- Stay silent for normal operation

And: keep terminal lines ≤ 80 columns. Most users have wider screens
but log files, ssh sessions, paste-into-issues, and small-screen
review all benefit from narrow lines. There's no upside to width.

---

## Open bugs / known issues

These are caught but not yet fixed. Tracked here so we don't forget.

### KNOWN-001 — Embedding norm imbalance after training

After tokenizer warm-up, new token embedding norms are ~58% of base
vocab norms (0.58 vs 1.00). This biases the LM head against generating
new tokens (because logits = `hidden @ embed.T`, and smaller-norm
embeddings produce smaller logits).

Strategy in [tokenizer-improvements.md](tokenizer-improvements.md) Tier
4.4: rescale new embeddings to base-norm magnitude after smart init,
or use multivariate-normal init that matches base distribution
exactly.

### KNOWN-002 — Templated corpus produces monotonous gradients

Identified in [tokenizer-improvements.md](tokenizer-improvements.md)
Tier 2. The corpus is 70% rotated templates; embeddings move in
lockstep rather than spreading into a useful geometry. Needs
corpus-generator overhaul.

### KNOWN-003 — `arg_value` token category starved

Tokens like `up`, `down`, `active`, `inactive` appear in too few
contexts to converge. See Tier 2.5.

---

## How to add a new bug entry

1. Reproduce the symptom and capture exact terminal output
2. Identify the root cause (not just the symptom)
3. Write the **mental model** section in 2-3 sentences — this is the
   most valuable part for future readers
4. Document the fix with code snippets
5. State **validation** — how someone else would confirm it's fixed
6. Extract a **lesson** that generalises beyond this specific bug

The mental model + lesson are the parts that remain valuable when
the codebase has moved on. The code-level fix tends to bit-rot.
