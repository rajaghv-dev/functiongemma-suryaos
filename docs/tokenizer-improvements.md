# Tokenizer training — improvement strategies

> Companion to [learnings.md L13](learnings.md). Captures what we identified
> as the levers for improving tokenizer training quality after the first
> (broken) run revealed the smart-init bug.

This document is a **strategy register**, not a build plan. Most items
require dataset work or experimentation, not just code changes.

---

## Tier 1 — Bugs blocking progress (must fix first)

These are non-negotiable. Until they ship, dataset improvements have nothing
to land on.

| # | Issue | Strategy |
|---|---|---|
| 1.1 | Smart init returns 0 valid subwords for every token | Tokenize via a separate base-tokenizer instance (loaded fresh, no `add_tokens()`) so the encode path doesn't short-circuit through the added-tokens trie. |
| 1.2 | Default `--epochs 2` too few; loss still trending at end | Bump default to 5. With proper init, more epochs become meaningfully different. |
| 1.3 | Stale embed_init.pt left from broken runs | Document a clean-run procedure: `rm -rf training/tokenizer_extended/` before each new run during iteration. |
| 1.4 | Initial gradient spike (norm 8-12) on cold init | After 1.1 lands, embeddings start near base norm magnitude → much smaller gradients. If still spiking, switch to cosine LR schedule with longer warmup. |

## Tier 2 — Corpus improvements (highest leverage)

The corpus is currently 70% rotated templates like `"Call {token} to handle
this request"`. That teaches *"this token exists"* but very little about
what it *means*. Ranked by impact:

### 2.1. Co-occurrence is everything

Two tokens become similar in vector space **only when they appear in the
same context**. Currently:

- `service_status` and `systemctl` rarely appear in the same sentence
- `linux_memory_usage` and `memory_usage` appear in different templates

**Strategy:** rewrite the corpus generator so each sentence intentionally
co-occurs related tokens. Examples:

```
"systemctl is the CLI for service_status"
"memory_usage and linux_memory_usage refer to the same operation"
"To launch via krunner_launch, use the KRunner UI or kstart6"
```

This is the single biggest lever for clustering quality.

### 2.2. Replace templates with natural language

70% of the corpus is monotonous repetition. Templates produce monotonous
gradients → embeddings move in lockstep. Natural text varies, so embeddings
spread out into a useful geometry.

Sources to pull real text from:

- man pages: `man systemctl`, `man free`, `man df`, `man pactl`
- KDE documentation excerpts mentioning Plasma, KRunner, KWin, Dolphin
- `dataset/dispatch_pairs.jsonl` user queries reused as corpus sentences
- README sections from peft, trl, transformers, llama.cpp (for ML tokens)
- GitHub PR titles / commit message bodies (for git tokens)
- LibreOffice / Okular / Inkscape help pages (for file-format tokens)

### 2.3. Hard contrastive examples

The cross-domain probe wasn't supposed to be high — it was supposed to learn
*separation*. Add explicit contrastive lines:

```
"linux_memory_usage reports RAM; linux_disk_usage reports storage. Different."
"service_status checks systemd state; window_focus controls KDE windows. Unrelated."
```

These force the model to learn what tokens are *not* related, which is what
keeps the geometry clean.

### 2.4. Per-token coverage way higher than 5×

5 occurrences/token converges noisily. Real foundation models use
100-1000× per token. For a domain corpus, **20-50× is reasonable**.

| Current | Target |
|---|---|
| 5+ occurrences/token | 20-30+ occurrences/token |
| 3849 sentences | 10-15k sentences |
| `min_occur=5` in build_tokenizer_dataset.py | `min_occur=20` |

This goal alone forces 2.1 and 2.2 to happen.

### 2.5. Distribution audit

| Category | Count | Comment |
|---|---|---|
| `file_format` | 81 | Probably overrepresented |
| `tool_name` | 68 | Core focus, fine |
| `git` | 41 | Reasonable |
| `ml` | 41 | Reasonable |
| `system` | 33 | OK |
| `kde` | 24 | OK |
| `v4_workflow` | 16 | Future-only — could defer |
| `arg_value` | 15 | Starved — only a handful of contexts |

`arg_value` tokens (`up`, `down`, `active`, `inactive`) are starved. They
appear in too few contexts to converge. Either delete them or 5× their
corpus presence.

## Tier 3 — `new_tokens.json` improvements

### 3.1. Drop tokens that already fragment to 1 in base vocab

68 of 319 tokens were already single-token in base Gemma. Adding them as
new IDs is wasted work — the *original* IDs already have rich pre-trained
embeddings from web-scale pretraining. Wasting an embedding row on
`"compile"` (which Gemma already knows) is strictly worse than leaving it
alone.

**Strategy:** filter `new_tokens.json` to only tokens with fragmentation ≥ 2.

### 3.2. Audit cross-domain noise

Tokens like `"active"`, `"inactive"`, `"up"`, `"down"` are extremely
generic English words. Adding them as domain tokens detaches them from
their natural context.

**Strategy:** keep these as base subwords; only add genuinely
domain-specific compound tokens.

### 3.3. Hierarchical decomposition

Instead of:
```
linux_memory_usage, system_memory_usage, volume_memory_usage  (3 tokens)
```

Use:
```
linux_, system_, volume_      (3 prefix tokens)
memory_usage                  (1 base token)
```

The model composes at runtime: `linux_` + `memory_usage`. Trades 4 token
slots for tokens that can be *generated combinatorially*.

### 3.4. Argument-pattern tokens

No tokens for `--no-pager`, `--quiet`, common option flags. These show up
constantly in CLI contexts. Adding 10-20 high-frequency option flags
would help embedding quality more than 80 file-extension tokens.

## Tier 4 — Training procedure

### 4.1. What to train

Currently we train *only* the embedding rows for new tokens (151M params,
gradient-hooked to keep the base vocab frozen).

Options to expand:

| Option | Effect | Cost |
|---|---|---|
| Embedding rows only (current) | Fast, isolated | Limited expressivity |
| Embedding + lm_head | If untied — output projection adapts to new tokens | Same as current (Gemma 3 ties them, so it's automatic) |
| Embedding + first-layer attention | First layer sees the embedding directly | +5M params |
| Full embedding LoRA (rank 16) | Learns row-row interactions | +8M params, more flexible |

Note: Gemma 3 has `tie_word_embeddings=True` — `lm_head` is the same
weight as `embed_tokens`. Training one trains both.

### 4.2. Better LR schedule

Current: constant LR with brief warmup. Better:

- Cosine decay from `lr` → `lr/10` over the run
- Longer warmup (10% → 20%) to handle the cold-start gradient spike
- Lower starting LR (5e-4 → 2e-4) once smart init works (smaller updates needed)

### 4.3. Joint training instead of two-phase

The "warm up tokenizer first, then LoRA" pipeline is one valid choice,
not the only one.

Alternative: skip Phase 3 entirely; do smart init only (instant); run
finetune.py LoRA with embeddings unfrozen alongside the LoRA adapter.
1 training run instead of 2; embeddings learn in the same gradient loop
as routing logic.

Tradeoff: 2-phase isolates concerns and produces a reusable
`tokenizer_extended/` artifact; joint training is faster end-to-end.

### 4.4. Smarter initialization

Beyond subword averaging:

| Init scheme | What it does |
|---|---|
| Subword average (Tier 1.1 fix) | Mean of [linux, memory, usage] embeddings |
| Multivariate normal | HuggingFace `mean_resizing=True` default — fits a Gaussian to the existing embedding distribution |
| Semantic neighbour | For "linux_memory_usage", look up "memory" + "RAM" + "usage" + "free" by hand |
| Frequency-weighted average | Subword average weighted by inverse token frequency |

Could combine: subword avg + small Gaussian noise so identical-subword
tokens (e.g. `linux_memory_usage` vs `system_memory_usage`) start at
slightly different points and can diverge.

## Tier 5 — Validation (so you know when it works)

We currently have no held-out check. Three cheap signals:

### 5.1. Token-completion probe

Pick 20 sentences like `"To check RAM use the tool ___"` and see if the
model's argmax for the blank is `linux_memory_usage`. Run before/after
training; report top-1 / top-5 accuracy.

### 5.2. Held-out cosine pairs

Add 10 PROBE_PAIRS the corpus *doesn't* directly co-occur — held-out test
of whether embeddings generalised vs memorised.

### 5.3. Downstream task probe (the real test)

After tokenizer phase, run inference on 20 dispatch examples and check
tool-name token probability *without* any LoRA training yet. Better
embeddings should already lift this baseline.

## Highest-leverage moves, ranked

If only three things ever ship:

1. **Tier 1.1 — fix smart init bug.** 5 lines of code, unblocks everything else.
2. **Tier 2.1 + 2.2 — replace half the templated corpus with co-occurrence-rich natural text.** Biggest single quality jump, dataset overhaul.
3. **Tier 5.3 — downstream task probe.** Without this, we can't know whether any improvement actually helped the real task.

Everything else is incremental on top of these three.

---

## Decision log for this iteration

What we're doing **now** (committed):
- Tier 1.1: smart-init bug fix
- Tier 1.2: bump default `--epochs 2 → 5`
- Tier 1.3: document clean-run procedure

What we're **deferring** to later iterations:
- All Tier 2 work (needs corpus generator overhaul)
- Tier 3.1-3.4 (needs build_tokenizer_dataset.py rework)
- Tier 4.2-4.4 (after re-running with bug fix to see new baseline)
- Tier 5 (after we have a stable training pipeline to compare against)

Re-run procedure after each iteration:
```bash
rm -rf training/tokenizer_extended/
.fngemma-suryaos/bin/python training/train_tokenizer.py
# compare cosine probes + nearest neighbours vs prior run
```
