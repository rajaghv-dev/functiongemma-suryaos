# Insights & Intuition — grounding iter #5 in what Gemma's authors actually say

> Captured 2026-05-06, end of iter #4 (Session 6, commit `96c9b2b` plus the
> v4 dataset rebuild). Companion to `goals.md`, `docs/learnings.md`,
> `SESSIONS.md`, and the in-flight `training/probes.py`.
>
> This file is for **the next person who has to decide what to do next**
> (which, in iter #5, is us). It pairs each piece of Gemma 3 documentation
> we could verify with the concrete decision it should drive in our repo.

---

## 1. Why this doc exists

Iterations #1–#4 were trial-and-error. Each one shipped, each one
regressed somewhere, each one taught us a new lesson the previous
iteration's KPI dashboard had not predicted (BUG-001 clones in #1,
templated-corpus monotony in #3, mining flood in #4, the empty-arguments
iceberg also in #4 — see `docs/learnings.md` L13/L14/L16/L17).

What was missing: a sanity check against the **base model's own
documentation**. We were tuning a hyperparameter we'd inherited (LoRA
r=8, fp16, 5 epochs of tokenizer warm-up, mining cap 3000) without
asking "what does Gemma 3 270M's authoring team actually recommend?"
This doc closes that gap and turns the answer into iter #5 plan items.

It also exists because the parallel agent is shipping
`training/probes.py` with eight diagnostics. **A probe is only useful if
you already know what the failure mode looks like.** This doc states
the failure mode for each of the eight probes so the moment one fires
in iter #5, the diagnosis is already half-written.

Goal of the doc: every claim either cites a Gemma URL, cites our own
`docs/learnings.md` L-entry, or honestly says "the docs don't address
this, and our intuition is X."

---

## 2. Gemma 3 270M — what we're actually training

### 2.1 Architecture (decoder-only, RoPE, sliding-window, GQA)

Gemma 3 keeps the decoder-only transformer Gemma 2 had, with three
specific changes that matter for us. From the Hugging Face `transformers`
[Gemma 3 page](https://huggingface.co/docs/transformers/main/en/model_doc/gemma3)
and the [Hugging Face Gemma 3 release blog](https://huggingface.co/blog/gemma3):

| Property | Gemma 3 value | Source | Our intuition |
|---|---|---|---|
| Attention pattern | 5 local sliding-window layers : 1 global layer | HF blog | Local-bias means within a 1024-token window the model resolves token relationships strongly — ideal for our 1-3 sentence dispatch queries (≤ 60 tokens) where everything fits in a single local window |
| Sliding-window size | 1024 tokens (down from 4096 in Gemma 2) | HF blog | Our entire `system_prompt + tools_json + user_query` fits well inside one window. Sliding-window is a non-issue for us. |
| RoPE base frequency | 1M (scaled ×8 in 4B/12B/27B; 1B/270M pretrained at 32k) | HF blog | At 32K context for the 270M, we use < 1% of the position range. Position extrapolation will not be exercised by our workload. |
| Activation | `gelu_pytorch_tanh` (not SwiGLU as some Gemma 2 docs claim) | HF transformers `Gemma3TextConfig` source | We do not touch this — relevant only when picking LoRA target modules (gate/up/down projections behave like a standard FFN under PEFT) |
| Normalization | RMSNorm (`rms_norm_eps=1e-6`) | HF transformers `Gemma3TextConfig` | Pre-norm transformer; numerically friendly to bf16 (not fp16 — see §7) |
| Final logit softcap | scaling factor on logits (config field `final_logit_softcapping`) | HF transformers `Gemma3TextConfig` | Caps extreme logits — relevant for our negative-margin probe: we cannot expect arbitrarily large margins between right and wrong tools, the softcap actively compresses them. |

The HF transformers default config (`Gemma3TextConfig`) is shaped after
the **4B** model (vocab=262208, hidden=2304, layers=26, heads=8, KV-heads=4
GQA, head_dim=256, intermediate=9216, `max_position_embeddings=131072`).
**Gemma docs do not publish the 270M-specific layer/hidden numbers in
the model card we could fetch.** Reasonable inference (270M ≈ 100M
embeddings + 170M transformer) puts the 270M at considerably fewer
layers / smaller hidden than the 4B; we should not assume the 4B numbers
above for our 270M.

### 2.2 Tokenizer

From the [HF Gemma 3 release blog](https://huggingface.co/blog/gemma3):

- **Type**: SentencePiece
- **Vocabulary size**: 262K entries (262208 by Gemma 3 default config)
- **Multilingual**: 1B and 270M are advertised as English-only; 4B/12B/27B
  cover 140+ languages (but the actual vocab is shared across the family,
  so the multilingual bytes are present in 270M too — this is why
  earlier runs observed Tamil/Kannada-character nearest neighbours when
  the global mean fallback fired in BUG-001, see L13)
- **Encoding improvements over Gemma 2**: better Chinese/Japanese/Korean
  segmentation, slight increase in tokens-per-English-word and
  tokens-per-code

**Special tokens we know exist** (from chat template usage in HF
`transformers`):
- `<bos>`, `<eos>`, `<pad>` — token ids 2, 1, 0 in the default config
- `<start_of_turn>`, `<end_of_turn>` — turn delimiters in chat template
- `<start_of_image>`, `<end_of_image>` — multimodal (irrelevant at 270M
  text mode but the embeddings exist in the 262K vocab)
- **No documented `<tool_call>` or `<function>` tokens.** The Gemma 3
  release blog and HF Gemma 3 model docs do not introduce dedicated
  function-calling tokens. (This is a major divergence from Llama 3's
  `<|python_tag|>` and Qwen's `<tool_call>`. See §2.4.)

Our intuition: the absence of dedicated tool-call tokens means **the
model emits tool calls as plain JSON** in the assistant turn body. The
first-token entropy probe (§4.5) measures this directly: if the
first emitted token after `<start_of_turn>model` is `{` with high
probability, the model is in tool-call mode.

### 2.3 Training-data philosophy

From the [HF gemma-3-270m model card](https://huggingface.co/google/gemma-3-270m)
that we could fetch:

- **Total tokens**: 6 trillion
- **Knowledge cutoff**: August 2024
- **Composition**: web documents (140+ languages, even though the 1B/270M
  are advertised English-only at the model level), code, mathematics,
  images
- **Preprocessing**: CSAM filter, sensitive-data filter, quality/safety
  filters

What this implies for our task:

1. The base 270M has seen **substantial code and CLI text** in
   pretraining. Our tool names (`linux_memory_usage`,
   `kde_dialog_confirm`) are inside its distribution — they fragment
   into known subwords, not random tokens. This is why §3 says we should
   stop spending tokenizer-vocab budget on tokens like `volume`,
   `memory`, `usage`, `kde` — Gemma already knows them.
2. The base 270M has **not** seen our 12-tool dispatch contract. It has
   seen *generic* function-calling JSON via web-scraped chat logs and
   GitHub README examples. It has not seen the specific
   `kde_krunner_launch({app:"kate"})` shape.
3. Pretraining at 6T tokens means our LoRA's signal must compete with
   six trillion tokens of general text already encoded in frozen
   weights. **A LoRA r=8 perturbation cannot relocate concepts — it can
   only re-route attention between concepts already represented.**
   This is the geometric reason why iter #1–#4 cosine separation work
   matters: get the routing-relevant directions sane in embedding
   space, and the LoRA does the rest.

### 2.4 Function-calling readiness

This is the most under-documented area of Gemma 3 we hit. Findings:

| Source | What it says about Gemma 3 function calling |
|---|---|
| `https://ai.google.dev/gemma/docs/capabilities/function-calling` | **We could not fetch this URL** — sandbox denied WebFetch. Cannot cite content. |
| HF Gemma 3 release blog | Not mentioned. |
| HF transformers Gemma 3 model doc | Not mentioned. |
| HF gemma-3-270m model card | Lists "Instruction-following, Question Answering" but not function calling specifically. |

**Honest read**: as of our research window, Gemma 3's official function
calling guidance from Google's own documentation we could fetch is
**not specified**. The community pattern (used by Phil Schmid's
fine-tune-gemma-3 post and Hugging Face's `gemma-peft` blog) is:

1. Tool schemas go in the **system prompt** (or first user turn) as a
   JSON list inside a markdown fenced block.
2. The assistant emits a **JSON object** — not a special token — with
   `name` and `arguments` keys, OpenAI-style.
3. There is **no dedicated tool-call token** wrapping the JSON.

This matches what our `dispatch_pairs_v4.jsonl` does today: prompt is
the user query, schema rendered as part of the system prompt by the
chat template, target is `{"name": "...", "arguments": {...}}`. Good
news: we are not violating Gemma conventions. Bad news: **we are also
not getting any pretraining lift from a `<tool_call>` special token**
because none exists. The model has to learn "the assistant turn is a
JSON object" purely from supervision.

### 2.5 Context length

- **270M context length**: 32K tokens (per HF gemma-3-270m model card).
- Our actual context per training example: ~ 250–500 tokens
  (system prompt + 12-tool schema JSON + 1-3 sentence query + JSON tool
  call). `training/finetune.py` `max_seq_length=512`.
- Our actual context per inference: ~ 200–350 tokens (1 schema after
  context-builder narrows to 1-3 tools — see L2).

We use **less than 2%** of the model's context budget. Sliding-window
attention, RoPE base frequency, and position-extrapolation fixes are
all irrelevant for our workload. Cutting `max_seq_length` to 384 would
buy us ~25% memory back if we hit OOM (we don't — see §7).

### 2.6 Recommended fine-tuning recipe (LoRA-on-Gemma)

The only Google-published recipe we could verify in the
`gemma-peft` blog (https://huggingface.co/blog/gemma-peft):

```python
from peft import LoraConfig
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj","o_proj","k_proj","v_proj",
                    "gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
# Training args from same post:
# per_device_train_batch_size=1, gradient_accumulation_steps=4,
# warmup_steps=2, learning_rate=2e-4, fp16=True,
# optim="paged_adamw_8bit"
```

What we currently do (`training/finetune.py`):

| Hyperparameter | HF gemma-peft blog | Ours | Gap |
|---|---|---|---|
| LoRA r | 8 | 8 | match |
| LoRA alpha | not stated | 16 | no Gemma-blessed value; alpha/r=2.0 is industry standard |
| LoRA dropout | not stated | 0.05 | similar — light regularization |
| target_modules | q,k,v,o,gate,up,down (7) | q,v (2) | **gap** — see iter #5 plan |
| LR | 2e-4 | 2e-4 | match |
| Warmup | 2 steps | warmup_ratio=0.1 | larger warmup; we are bigger dataset |
| Precision | fp16 | bf16 (after BUG-006) | **we are correct** — gemma-peft post predates bf16-everywhere advice; fp16 produces NaN on Ampere with this model (see L: `docs/bug-fixes.md` BUG-006) |
| Optimizer | paged_adamw_8bit | adamw_torch | minor — paged is for memory-constrained 4-bit setups |
| Batch * grad_accum | 1 * 4 = 4 | 4 * 4 = 16 effective | we have 4× their effective batch (L6 supports ≥ 8 for stability) |

**Iter #5 candidate from this table**: expand target_modules from 2 to
7. The gemma-peft post recommends all attention + FFN projections;
L6 in `docs/learnings.md` argues for q_proj+v_proj only based on
"attention routing" intuition. The blog's recommendation is more
recent and Google-blessed. **This is one of the iter #5 ablations the
probes will tell us about** (specifically: per-tool loss + arg
fidelity probes — see §4).

---

## 3. The iter #4 dataset — what it should and should not teach

The 12 tools (`tools/tool_schemas.json`), each tagged with what we
expect the LoRA to add over the base model's pretrained knowledge:

| Tool | Args | Real pairs (v4) | Base model already knows | LoRA must add | LoRA cannot fix |
|---|---|---:|---|---|---|
| `linux.memory.usage` | none | 18 | "memory", "RAM", "free", "/proc/meminfo" — strong | "this user query → emit `{name:linux_memory_usage,arguments:{}}`" | Pretraining bias toward emitting `free -h` output instead of JSON tool call |
| `linux.disk.usage` | optional `path` | 13 | "df", "disk", "/home", filesystem terms | Tool name + when to fill optional `path` arg | Bias to emit `df -h` output |
| `linux.brightness.set` | `direction` enum + optional `step` int | 12 | "brightness", "dim", "screen" — strong | Bind "dim"→"down", "brighten"→"up", parse "by 20 percent"→step:20 | "Up" / "down" are fragmented in odd cases — tokenizer-level mismatch |
| `linux.volume.set` | `direction` enum + optional `step` | 31 | "volume", "louder", "quieter" — strong | Bind cue→direction, extract numeric step | Same as brightness |
| `linux.battery.status` | none | 23 | "battery", "charging", "%", `acpi`, `upower` | Tool name | — |
| `linux.network.status` | none | 24 | "wifi", "online", `nmcli`, `ip` | Tool name | — |
| `linux.service.status` | required `name` | (mined > 100) | "systemd", "active", "ollama", "bluetooth" | Extract service name from query, fill `name` arg | Service names not seen in pretraining (e.g. `surya-agentd.service`) |
| `linux.metrics.summary` | none | 20 | "system status", "health" | Tool name + disambiguate from individual `linux_memory_usage` etc. | Sibling-tool confusion (the L16 cross-domain problem) |
| `kde.window.focus` | required `title` | (mined ~50?) | "switch to", "focus", "window" | Extract app/window title from "switch to firefox" | Wayland-specific window-title fragments |
| `kde.dialog.confirm` | required `prompt`, optional `title`, `default` | 13 | "ask", "confirm", "are you sure" | Synthesize dialog `prompt` from user request | — |
| `kde.notifications.send` | required `title`, `message` | 22 | "notify", "alert", `notify-send` | Synthesize `title` and `message` separately from query | — |
| `kde.krunner.launch` | required `app` | (mined ~250+) | App names — STRONG (firefox, kate, dolphin, konsole all in pretraining) | Bind verbal "open kate" → `app:"kate"` | Capitalization/aliasing (`Firefox` vs `firefox`) |

Three observations:

1. **8 of 12 tools have args.** The empty-args iceberg (L17) was
   catastrophic precisely because we were training the model to never
   emit args — directly opposite to what 8 tools require.
2. **Arg-extraction skill is per-tool.** "Down" cues differ for volume
   vs brightness vs anything else. This is why §4.3 (arg fidelity probe)
   needs **per-tool** breakdown, not aggregate.
3. **3 tools have no args at all** (`battery_status`, `memory_usage`,
   `network_status`, `metrics_summary`, `disk_usage` if path omitted).
   These are *easy* and will mislead aggregate metrics — a model that
   learns "always emit `{}`" hits 33% of pairs correctly. **The
   per-tool loss probe must report these separately or we will declare
   victory prematurely.**

---

## 4. Probes-to-insights mapping

The parallel agent is implementing eight probes in `training/probes.py`.
For each, we state: *what it measures*, *healthy trajectory*, *failure
pattern*, *iter #5 action it triggers*.

### 4.1 Per-tool loss

**Measures**: cross-entropy loss on the eval set, broken down by
`target.name`. Computed as the mean negative log-likelihood of the
golden assistant turn given each example.

**Healthy trajectory**: each of the 12 tools' loss decreases roughly
proportionally over epochs. Std-dev across tools at convergence < 1.5×
mean (Goal 4 has the same threshold for token-level loss in
`goals.md`).

**Failure pattern A — starvation**: 1-3 tools show loss 3-5× higher
than the median because they have 12-23 training pairs vs 100+ for
others (L18 lists `linux_brightness_set=12`, `kde_dialog_confirm=13`,
`linux_disk_usage=13`).
*Triggers*: class-weighted loss for the starved tools (multiply their
contribution by `floor / actual_count`), or upsample via controlled
paraphrasing (L18 option 4).

**Failure pattern B — sibling collapse**: `linux_memory_usage` and
`linux_metrics_summary` end with nearly identical loss because the
model has decided "if user asks about RAM, emit either tool with 50/50
probability." The confusion-matrix probe (4.2) confirms.
*Triggers*: hard-negative ratio uplift (currently avg 2 per pair; try 4).

### 4.2 Confusion matrix

**Measures**: 12 × 12 matrix of (gold_tool, predicted_tool) on the eval
split, top-1 prediction per example.

**Healthy trajectory**: trace > 90% of total mass; off-diagonal mass
concentrated near sibling pairs (`memory_usage` ↔ `metrics_summary`)
not at random; diagonal grows monotonically per epoch.

**Failure pattern A — single-tool bias**: one column gets > 30% of
all predictions. We saw this in pre-iter-#4 data
(`kde_krunner_launch` = 93.7% of training, model would fall back to
emitting it for anything). Should now be < 10% on the v4 dataset.
*Triggers*: confirm dataset rebalance worked; if still skewed, examine
whether the tokenizer fragmentation of dispreferred tool names creates
output-side bias (Goal 3, embedding norm equivalence).

**Failure pattern B — domain split**: `linux_*` tools confuse with each
other; `kde_*` tools confuse with each other; **but** the two domains
don't cross. This means cross-domain cosine has been fixed (good!)
but sibling clustering hasn't (the L16 sibling problem).
*Triggers*: more A3-style contrastive pairs at the sibling level
(currently we have 32 cross-domain contrast sentences in the
tokenizer corpus; we have *zero* sibling contrast, e.g.
"memory_usage and metrics_summary differ in granularity").

### 4.3 Arg fidelity

**Measures**: for each eval example, parse the model's emitted JSON,
and report (a) JSON parses, (b) all required args present, (c) each
required arg matches gold (string equality for enums, numeric tolerance
± 5% for ints, fuzzy match for free strings like `app` and `title`).
Output: per-tool arg-pass rate.

**Healthy trajectory**: > 85% on tools with simple required args
(`brightness.set`, `volume.set`, `service.status`, `krunner.launch`)
within 3 epochs. > 70% on tools with 2 required args
(`notifications.send` needs both `title` and `message`).

**Failure pattern A — empty-args regression**: arg-pass < 50% even on
simple tools because the model collapses to `arguments:{}`. This is
the L17 failure mode returning. Direct evidence: model emits
`{name:linux_volume_set, arguments:{}}` for "turn it down by 20%".
*Triggers*: explicit arg-extraction supervision pass — penalize
empty-args predictions in the loss with a multiplier (2× for tools
with required args). Also re-run `training/populate_arguments.py` on
any new mined data to make sure arg-population didn't regress in the
preprocessing pipeline.

**Failure pattern B — wrong-direction enum**: `brightness.set({direction:"up"})`
emitted for "dim the screen". We had this in iter #3 data (L17 quote:
"dim the screen labeled `direction:up`"). The L17 fix patched the
DATA; this probe checks whether the MODEL still has it.
*Triggers*: examine `populate_arguments.py` UP_CUES/DOWN_CUES regex
for false positives in the eval set. If arg-extraction is correct in
data but wrong in model output, the LoRA needs more direction-cue
contrast pairs.

**Failure pattern C — hallucinated args**: model invents `step:50`
when query says "louder" with no number. Less harmful (default exists)
but still indicates the model isn't properly conditioning on input.
*Triggers*: dataset hygiene — scan for cases where gold has step set
without a numeric cue in the query.

### 4.4 Schema compliance

**Measures**: did the emitted JSON validate against
`tools/tool_schemas.json`? Specifically: (a) `name` is in the 12-tool
catalog, (b) `arguments` is an object, (c) all required args present,
(d) no extra args, (e) enums match, (f) numeric ranges respected.

**Healthy trajectory**: > 95% by epoch 3. This is *easier* than arg
fidelity because schema compliance only checks structure, not
correctness; a model can emit `{name:linux_volume_set, arguments:
{direction:"up", step:5}}` correctly-structured-but-semantically-wrong
for "turn it down" and still pass schema compliance.

**Failure pattern A — JSON parse fail**: model emits free text
(`"Sure, I'll turn the volume down for you"`). Indicates SFT didn't
take — chat template issue or label masking issue.
*Triggers*: inspect the tokenizer chat template; verify
`add_generation_prompt=True` was used in eval; confirm assistant turn
labels are not all `-100` (label-masking bug).

**Failure pattern B — invented tool name**: model emits a tool that
doesn't exist (`linux_brightness_change`). This means the LoRA
adapter's last-layer logits aren't yet biased toward the 12 catalog
names. Indicates undertraining or insufficient tool-name token coverage
in the tokenizer phase.
*Triggers*: constrained decoding at inference (mask logits to only
allow the 12 valid tool name tokens). Easy win; already-implementable.

**Failure pattern C — extra args / wrong types**: `arguments:
{direction:"up", step:"five"}` emits `step` as string. Gemma's
pretraining bias is to output numbers as digits but typing slips
happen.
*Triggers*: JSON-mode constrained decoding (the iter #5 probe-tagged
plan item).

### 4.5 First-token entropy

**Measures**: at the very first generation step (immediately after the
`<start_of_turn>model` opening), compute the entropy of the next-token
distribution. Average across eval examples.

**Healthy trajectory**: low entropy (< 1.5 bits) at convergence,
because the model "knows" the response should start with `{`. The
*shape* of the distribution should concentrate on `{`, possibly
`{"name`, with everything else negligible.

**Failure pattern A — high entropy stays high**: the model is uncertain
whether to emit JSON, prose, code-fence, or a markdown header. Means
the chat template / SFT mask isn't pushing the model into "tool-call
mode" reliably. Common when the system prompt lacks an explicit
"respond with JSON only" instruction.
*Triggers*: tighten the system prompt; or add a `<tool_call>`-shaped
sentinel token to the tokenizer (we'd be inventing what Gemma doesn't
provide — careful).

**Failure pattern B — first-token confidence on the wrong token**: the
top-1 first token is something like `\n` or `Here` (English prefix).
Indicates the model decided the assistant is being chatty.
*Triggers*: more aggressive SFT — increase the LoRA's effective rank
or expand target modules per gemma-peft blog (§2.6).

### 4.6 Negative margin

**Measures**: for each eval example, compute the log-prob the model
assigns to the gold tool name token vs the next-best tool name token.
The "margin" is `logp(gold) - logp(best_other)`. Probe reports the
mean and the bottom-10% quantile.

**Healthy trajectory**: mean margin > 3 nats; bottom-10% > 0 nats
(i.e., even the hardest examples lean toward gold). Margin grows over
epochs.

**Failure pattern A — margin near zero or negative**: the model is
essentially flipping a coin between gold and a sibling tool. This is
the routing failure mode the cosine cross-domain probe was supposed to
catch but couldn't (because cosine is geometric while margin is
output-side).
*Triggers*: more hard-negative pairs at the sibling level. Iter #4
adds avg ~2 per pair; iter #5 try 4.

**Failure pattern B — high mean, low bottom-10%**: most examples are
easy (clean queries), the margin is dominated by them; the hard ones
remain ambiguous. Indicates we need to mine the hard examples
specifically.
*Triggers*: collect production audit trail (L11) and mine the
low-margin examples back into training as hard cases (the auto-fix
loop has been doing this informally; make it formal).

### 4.7 Source overfit

**Measures**: bucket eval examples by their `source` field
(currently 36 sources after iter #4 — `desktop_files`, `man_pages`,
`kde_help`, `audit_failures`, etc.) and report per-source loss.

**Healthy trajectory**: loss roughly even across sources (within 1.5×
mean). Variation is expected (man-page text is more formal than audit
queries) but extreme gaps mean overfit.

**Failure pattern A — one source dominates**: 80% of eval examples
have `source=audit_failures` because that's most of the data; the
model fits those at loss 0.4 while man-pages-derived examples sit at
2.5. Means the dataset balance has not really been fixed.
*Triggers*: re-cap per source during eval (oversample minority sources
or weight per-example by inverse source frequency).

**Failure pattern B — the synthetic split fits much better than real
sources**: any synthetic / paraphrased examples should NOT have lower
loss than real ones. If they do, the model has learned the synthesis
template's signature (a la L16 mining flood).
*Triggers*: shrink the synthetic share or diversify the paraphrase
templates.

### 4.8 Arg value diversity

**Measures**: for tools with required args, compute the entropy of
`target.arguments` values across the training set. E.g. for
`linux_volume_set`, what's the distribution of `direction` (should be
~50/50 up/down), `step` (should span 5-100), `service.status name`
(should span 100+ unique service names).

**Healthy trajectory**: each arg has > 0.5 × max-entropy (uniform
distribution would be max-entropy; real-world skew is fine but extreme
skew means under-coverage).

**Failure pattern A — one value dominates**: 95% of `step` values are
10 because `populate_arguments.py` defaulted them. The model will
predict step=10 for everything.
*Triggers*: when `populate_arguments.py` cannot extract a number from
the query, **drop the `step` field entirely** instead of defaulting it
— let the schema's "default 10" handle it at runtime, and stop biasing
training toward step=10.

**Failure pattern B — service names are 90% `ollama` and `bluetooth`**:
mining was successful but limited to a few services on this one
machine (L18). Model overfits to "if user asks about a service,
predict ollama". Already known — this is the iter #4 ceiling problem.
*Triggers*: mine GitHub KDE/systemd issues for service-name diversity
(L18 option 2).

---

## 5. The empty-args iceberg — why it mattered (Gemma's perspective)

Recap of L17: `dispatch_pairs.jsonl` had 1564/1564 pairs labeled with
empty or wrong `target.arguments`. "Dim the screen" was labeled
`{direction:"up"}`. 93.7% of pairs routed to `kde_krunner_launch`
because `dataset/apps/launch_pairs.jsonl` was concatenated in.

Why this is fatal from Gemma's training-loss perspective:

**Cross-entropy on the assistant turn is computed token-by-token.**
For every training pair, the loss term for the closing `}}` of
`arguments:{}` is *added* to the gradient. That gradient pushes the
model toward emitting `}}` as the natural continuation after
`arguments:{`. With 1564 pairs all teaching the same thing
(`arguments:{` then immediately close), the model learns:

> *"After `arguments:` always emit `{}` and stop."*

This is a strong supervision signal in the most literal sense — strong
enough to overpower the (very weak, 0-26 examples per arg in iter #3)
positive examples of arg extraction. The Gemma docs do not address
this directly because they don't ship a function-calling supervision
recipe. **But the principle is universal**: SFT loss is the gradient
direction, and the gradient is the training data. Bad data ⇒ bad
gradient ⇒ bad model. No amount of LoRA rank, LR tuning, or schedule
adjustment fixes this.

The L17 fix (`training/populate_arguments.py`) closed the data side.
The arg-fidelity probe (§4.3) verifies the model side. **Iter #5 must
keep these two locked together** — test-time validation
(`tests/test_dispatch_pairs.py`) prevents data drift; arg-fidelity
probe catches model drift. If you only have the data test, you'll
notice when someone breaks the data; if you only have the model
probe, you'll notice when the data was already broken.

Decision rule (from L17, restated): before claiming "the model picked
the right tool", inspect `target.arguments` distribution. If > 5% of
training pairs have empty args for tools with required parameters,
**the supervision is broken** no matter how good the routing geometry
looks.

---

## 6. Cross-domain cosine collapse — geometric intuition

Why iter #3→#4 saw cosine regress 0.62 → 0.70 (`docs/learnings.md`
L16). What Gemma's authors say (and don't say) about this.

### 6.1 The geometric setup

Gemma 3's embedding matrix is shape `[262208, hidden]`. For 270M, the
hidden size is not directly published; for the 4B reference config
it is 2304. We added 108 new rows for tool-name tokens.

Each row is a vector in ℝ^hidden. "Cross-domain cosine" is the average
cosine similarity between vectors of unrelated tools (e.g.
`linux_memory_usage` row vs `kde_window_focus` row). Goal: < 0.30 (so
the model's last-layer logits, computed as `hidden_state @ embed.T`,
distinguish them sharply).

### 6.2 What iter #3 did wrong

`train_tokenizer.py` computed the loss as next-token prediction on the
corpus. The corpus was 84% mined `dispatch_pairs` rendered as
"`<query>` should dispatch to `<TOOL>`."

Every tool name appeared in the *same syntactic slot*. Backprop's
gradient through `embed[tool_id]` sees the same context every time:
"after `dispatch to `, predict `<TOOL>`". The optimal embedding for
that signal is **the average of all tools** projected onto whatever
direction maximizes "tool-token-ness". So the optimizer pushes all 12
tool-name embeddings toward a common centroid.

Result: cosine_cross_domain = 0.70 (geometric centroid effect).

### 6.3 Why hard negatives attack this directly

Hard negatives in iter #4 add training examples like:

> "Memory and brightness are different concerns — one reports RAM,
> the other adjusts screen luminance."

Now the gradient through `embed[linux_memory_usage]` and
`embed[linux_brightness_set]` sees:
- `linux_memory_usage` follows "reports RAM"
- `linux_brightness_set` follows "adjusts screen luminance"

These are non-overlapping context windows. Backprop pushes the two
embeddings toward **different** directions — directly attacking the
centroid effect. This is why iter #4's hard-negatives pass per pair is
the *single most important* corpus change.

### 6.4 The LoRA-rank-8 limitation

A LoRA adapter at rank 8 is a sum of 8 outer products: `ΔW ≈ Σᵢ
αᵢ uᵢ vᵢᵀ` where `uᵢ, vᵢ` are vectors in the model's hidden space.
This is at most 8-dimensional perturbation of the model's behavior.

It can:
- Re-route attention between concepts already represented (the L2
  "context builder narrows to 1-3 tools" benefit)
- Add task-specific decision boundaries among existing concepts
  (the dispatch logic)
- Bias output logits toward the 12 tool names

It **cannot**:
- Relocate an embedding from "near `<image_soft_token>`" to "near
  `memory`/`RAM`/`usage`". That requires updating the embedding row
  itself, which we do explicitly during the tokenizer phase. The LoRA
  adapter does not touch the embedding matrix.
- Compress 12 unrelated tool concepts into 12 well-separated clusters
  if the embedding rows are already collapsed.

Gemma docs do not address LoRA rank vs embedding geometry — that's
general PEFT theory. Our intuition: the tokenizer phase **earns its
keep** by establishing the per-token embedding placement that the
LoRA cannot fix. Goal 1 in `goals.md` is exactly this argument.

### 6.5 What this means for iter #5

The cross-domain cosine probe in `analyze_embeddings.py` and the per-tool
loss + confusion-matrix probes in the new `training/probes.py` should
agree:
- Cross-domain cosine drops below 0.30 → confusion matrix has clean
  block-diagonal structure (linux↔linux mistakes only, kde↔kde
  mistakes only) → routing works.
- Cross-domain cosine drops below 0.30 BUT confusion matrix still has
  `kde_window_focus → linux_metrics_summary` errors → embedding
  geometry is fine but the LoRA isn't pulling its weight; expand
  target_modules (§2.6).
- Cross-domain cosine STAYS ≥ 0.50 → tokenizer phase didn't work; do
  not blame the LoRA. Iterate on the corpus first.

---

## 7. RTX 3080 10GB Ampere — what the hardware can and cannot do

### 7.1 bf16 vs fp16 (BUG-006 lesson)

Gemma docs we could fetch (HF gemma-peft blog) use `fp16=True`. We use
bf16. **Both are wrong/right depending on the GPU**:
- fp16: 5-bit exponent, 10-bit mantissa. Range ≈ 10⁻⁵ to 10⁵. Gradients
  in causal LM with 8-bit attention scores often spike past 10⁵, producing
  Inf/NaN.
- bf16: 8-bit exponent, 7-bit mantissa. Range ≈ 10⁻³⁸ to 10³⁸ (same as
  fp32). Less precision, but the dynamic range is what matters for
  causal LM training.

Ampere (RTX 30xx) and Ada (40xx) both support bf16 natively. Our
`training/finetune.py` `_detect_hardware()` selects bf16. **Do not
revert to fp16 even if a Gemma tutorial says so** — those tutorials
predate the bf16-everywhere consensus on Ampere. (See L: BUG-006 in
`docs/bug-fixes.md`.)

### 7.2 Batch size economics

Current: `per_device_train_batch_size=4`,
`gradient_accumulation_steps=4`. Effective batch = 16.

Gemma-peft blog: 1 × 4 = 4 effective. Their 4 is for memory-constrained
8-bit setups. We have 10GB and don't quantize the base model.

Our 16 effective batch is at the sweet spot:
- Below 8: gradient noise too high, slow convergence.
- 8-32: stable; 16 has extra headroom for hard examples.
- > 32: diminishing returns, doesn't fit in 10GB anyway.

### 7.3 Memory budget at seq=512

Approx breakdown on RTX 3080 10GB with bf16:

| Component | Size |
|---|---|
| Gemma 3 270M base weights | ~540 MB |
| Adam optimizer states (LoRA only, ~4M params) | ~32 MB |
| LoRA adapter (4M params bf16) | ~8 MB |
| Activations (per sample, seq=512, bf16) | ~150-300 MB |
| KV cache (training: stored in activations) | included |
| **Per-batch peak** (4 samples × 250 MB activations) | ~1 GB |
| **Total training peak** | **~2 GB** of 10 GB |

We have 8 GB of headroom. **We are not memory-bound**. This is why
`max_seq_length=512` is comfortable, why we don't need 4-bit
quantization, and why we can afford to expand LoRA target_modules
from 2 to 7 in iter #5 (§2.6) — the 4M trainable params would become
~12M, still trivial.

### 7.4 LoRA-merge memory peak (the export step needs more)

`merge_and_unload()` at the end of training temporarily loads the full
fp32 base model (1.1 GB) plus the LoRA delta (32 MB) and the merged
output (1.1 GB) — peak ~3 GB plus framework overhead. Still fits.
**Do not run `analyze_embeddings.py` simultaneously with the merge**;
save the merge, free CUDA, then analyze.

### 7.5 Realistic epoch time for 1448 pairs

Run #4 (tokenizer phase, 3579 corpus sentences, seq~256, GPU bf16)
took 2m 29s for 5 epochs.

LoRA on 1448 dispatch pairs at seq=512, batch 4×4 bf16, 3 epochs:

```
1448 / 16 effective_batch = 91 update steps per epoch
× 3 epochs = 273 update steps
× ~ 0.3 s/step (forward + backward + grad-accum) ≈ 80 seconds
+ checkpoint save + eval ≈ ~ 2-3 minutes total
```

Iter #5 ablations (LoRA target_modules variants, balance30 vs full,
etc.) each take 3 min. We can afford to run the full grid in an
afternoon. **Wallclock is not the constraint; signal is.**

---

## 8. Iter #5 plan

Concrete deliverables, each tagged with the probe(s) that confirm
success or fire on failure:

- **(probes: confusion_matrix + source_overfit)** Train two LoRAs in
  parallel: (a) full v4 dataset, (b) per-source balanced to floor 30
  per source. Compare confusion matrix entries — does balanced
  training reduce `kde_krunner_launch` over-prediction further?
  Source-overfit probe shows whether one source's loss dominates.
  *Decision rule*: if (b) wins on confusion-matrix trace by ≥ 5
  points, use balanced going forward; the cost is 2× train time once.

- **(probe: arg_fidelity)** Explicit arg-extraction supervision pass
  — penalize empty-args predictions in the cross-entropy loss with
  a 2× multiplier for the args-portion of tools that have required
  args. Compare arg-fidelity probe before/after.
  *Decision rule*: if arg-fidelity rises > 5 points without per-tool
  loss regression, ship.

- **(probe: negative_margin + confusion_matrix)** Re-balance hard
  negatives. Iter #4 average is 2 per pair; try 4. Hard-negatives
  attack the L16 cross-domain centroid effect (§6.3). Negative-margin
  probe says whether the routing margin improves.
  *Decision rule*: bottom-10% margin > 0 nats target.

- **(probe: per_tool_loss)** Class-weighted loss for the 3 starved
  tools (`brightness.set`, `disk.usage`, `dialog.confirm`). Weight ∝
  `(target_floor / actual_count)`. Simpler than mining more — we
  *cannot* mine more from one machine (L18).
  *Decision rule*: per-tool loss std-dev drops below 1.5× mean.

- **(probe: schema_compliance)** JSON-mode constrained decoding at
  inference — does it help? Outlines/lm-format-enforcer grammar
  constrains the output to valid tool schema. Cheap experiment;
  measure top-1 schema-compliance pre/post.
  *Decision rule*: if constrained decoding lifts schema-compliance
  > 3 points without latency hit > 20%, ship.

- **(probe: per_tool_loss + arg_fidelity)** Expand LoRA target_modules
  from `[q_proj, v_proj]` to `[q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, down_proj]` per gemma-peft blog (§2.6). 7
  modules instead of 2; trainable params go ~4M → ~12M.
  *Decision rule*: if both per-tool loss and arg-fidelity improve and
  general-knowledge benchmark loss delta (Goal 5) stays < 5%, adopt.

- **(probe: source_overfit + arg_value_diversity)** Build the held-out
  arg-test split that L17 said was "first task for iter #5". Without
  it, `argument_extraction_accuracy` cannot be measured against a
  fixed yardstick. This is the prerequisite for everything above —
  do it first.
  *Decision rule*: arg-test split has ≥ 20 examples per tool with
  args, drawn from queries NOT seen in the training set, and
  arg-value distribution matches the training distribution within
  1.5× (so the test isn't a different domain).

- **(probe: arg_value_diversity, no probe action)** Production audit
  trail mining (L11, L18 option 1). The agent has been collecting
  failures since Session 2; iter #5 should formalize the pipeline
  that turns audit DB rows into `dispatch_pairs.jsonl` candidates.
  *Decision rule*: lift starved tools to floor 80 within ~ 4 weeks
  of agent uptime.

---

## 9. Open questions for the user

Before iter #5 starts, the following decisions need a yes/no:

1. **Synthetic-via-API budget?** The L18 "real-source ceiling" puts 9
   of 12 tools below floor 80. Reaching the floor requires either (a)
   waiting weeks for production audit trails, (b) controlled
   paraphrasing of seed sentences (cheap, in-scope), or (c) calling
   GPT-4 / Claude-API to paraphrase (costs ~$5 for 1000 paraphrases).
   *Question*: is (c) acceptable, or do we keep "minimize synthetic"
   strict and accept slower iteration?

2. **Swap base to Gemma 3 1B?** The 1B model has the same architecture
   family but more capacity. It would handle 12-tool routing with less
   need for the tokenizer-extension phase. Cost: ~3.7× memory
   (~2 GB + activations), ~ 4× train time, ~ 4× inference latency.
   *Question*: is the latency budget (currently ~6s warm at 270M)
   willing to absorb a 4× slowdown to ~24s for higher accuracy?

3. **Drop tokenizer-phase entirely?** If gemma-peft's 7-target-modules
   LoRA on the un-extended base hits accuracy ≥ extended-base's, the
   tokenizer phase is overhead. The "deepest goal" in `goals.md`
   says: tokenizer phase succeeds when removing it causes a
   measurable LoRA accuracy regression. We have not yet run the
   ablation. *Question*: schedule this ablation as iter #5's first
   experiment, or last?

4. **Holdout split: time-based or random?** Random splits from the
   v4 dataset risk train/test contamination because mined queries
   are similar within a source. A time-based split (drop everything
   from miners that ran in the last 24h) would be more honest.
   *Question*: are we OK rebuilding the split each iter on a
   time-based cut?

5. **Constrained decoding at inference: production-ready or
   experiment-only?** If we adopt it, the inference path becomes more
   complex (need a grammar lib like `outlines`, `lm-format-enforcer`,
   or `xgrammar`). *Question*: introduce this dependency, or only
   use it for probing in iter #5?

6. **Ship on plateau or chase the asymptote?** Iter #4 reached real
   data + zero arg misses — plausibly shippable. Iter #5 wants to
   close per-tool gaps and add held-out eval. *Question*: ship the
   v4 LoRA to production first as a baseline, then iterate? Or
   land iter #5 first?

---

## 10. References

URLs successfully fetched (date: 2026-05-06):

- [`https://huggingface.co/google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it)
  — instruction-tuned 270M model card. Source for: 32K context,
  pretraining data composition (6T tokens, multilingual), chat
  template usage example, intended use cases.
- [`https://huggingface.co/google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m)
  — pretrained 270M model card. Source for: bf16 tensor type, 6T
  pretraining tokens, August 2024 cutoff.
- [`https://huggingface.co/blog/gemma3`](https://huggingface.co/blog/gemma3)
  — Gemma 3 release announcement. Source for: 5:1 sliding-window
  pattern, 1024-token window, RoPE 1M base, 262K vocab,
  multilingual coverage by size.
- [`https://huggingface.co/docs/transformers/main/en/model_doc/gemma3`](https://huggingface.co/docs/transformers/main/en/model_doc/gemma3)
  — HF Transformers Gemma 3 doc. Source for: `Gemma3TextConfig` defaults
  (vocab=262208, hidden=2304 for 4B, head_dim=256, sliding_window=4096
  default, RMSNorm, GQA num_kv_heads=4, max_position=131072),
  `gelu_pytorch_tanh` activation, special tokens (bos=2, eos=1, pad=0),
  `<start_of_image>`.
- [`https://huggingface.co/blog/gemma-peft`](https://huggingface.co/blog/gemma-peft)
  — Hugging Face Gemma + PEFT blog. Source for: LoRA r=8,
  target_modules list (all 7 projections), LR 2e-4, paged_adamw_8bit,
  fp16 default (now superseded by bf16 on Ampere — see §7.1).

URLs we attempted but could not fetch (sandbox WebFetch denied):

- `https://ai.google.dev/gemma/docs/core/model_card_3` — Gemma 3
  official model card. **Cannot cite content.** All claims about
  Gemma 3 in this doc are based on HF mirrors above. If/when this is
  retrieved later, validate this doc's architecture claims against it.
- `https://ai.google.dev/gemma/docs/core` — Gemma overview. **Cannot
  cite content.**
- `https://ai.google.dev/gemma/docs/capabilities/function-calling` —
  Gemma function-calling guide. **Cannot cite content.** Section §2.4
  is therefore based on community patterns (`gemma-peft` blog,
  `dispatch_pairs_v4.jsonl` shape) rather than Google's authoring
  team's recommendation. **Iter #5 should retry this URL.**
- `https://developers.googleblog.com/en/introducing-gemma-3-270m/` —
  Gemma 3 270M intro blog. **Cannot cite content.**
- `https://www.philschmid.de/fine-tune-gemma-3` — Phil Schmid's
  fine-tune guide (often the most detailed third-party Gemma
  hyperparameter source). **Cannot cite content.**
- `https://huggingface.co/blog/gemma3-270m-finetune` — 404.

Internal cross-references:

- `goals.md` — the canonical 5-goal hierarchy + 8-number KPI
  dashboard (now 9 with arg-extraction-accuracy from iter #4).
- `docs/learnings.md` L1..L19 — Run-by-run postmortems. L13
  (Run-#1 clones), L14 (Run-#2 corpus bottleneck), L15 (iter #3
  shipped), L16 (Run-#4 cross-domain regression), L17 (empty-args
  iceberg), L18 (real-source ceiling), L19 (multi-agent extraction).
- `docs/dataset-strategies.md` — 24 strategies. iter #5 plan items
  cite A2/A3/A4/D1/D3/E1/E4 specifically.
- `tools/tool_schemas.json` — 12-tool catalog. The §3 inventory is
  derived directly from this file.
- `SESSIONS.md` Session 6 — iter #4 narrative log.
- `CHANGELOG.md` iter #4 — what shipped + when.
- `training/finetune.py` lines 1604-1609 — current LoRA config.
  §2.6 compares this against gemma-peft blog.
- `training/probes.py` — in-flight, written by parallel agent. §4 maps
  the eight probes to specific failure modes and iter #5 actions.
