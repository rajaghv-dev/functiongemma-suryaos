# Goals — what success looks like for this project

> The canonical reference for what we're trying to achieve, where we are
> now, and how we get from one to the other. Captured 2026-05-02 after
> tokenizer training run #3, updated after iteration #3 dataset overhaul.

If a doc disagrees with this one, this one wins. Update this when goals
change.

**Status as of Run #4 (commit 275bef2):**
- Tokens pruned 319 → 108 (filter via fragmentation)
- Corpus rewritten 3849 templated → 3579 curated
- BUG-001..007 all fixed (smart-init, HF auth, GPU/torch, idempotent
  bootstrap, frozen probes, fp16 NaN, deprecation+log flood)
- GPU training now works on RTX 3080 Ti via bf16 (~2.5 min vs 41 min CPU)
- **Run #4 result: REGRESSION on cross-domain** (0.62 → 0.70).
  Mining flood (3000 sentences in same grammatical slot) outweighed the
  32 contrastive lines and pulled tools together more, not apart.
  See L16 postmortem in docs/learnings.md.
- Iter #4 plan: cut mining 3000→500, expand contrastive 32→300,
  add varied-position sentences.

---

## TL;DR

| Question | Answer |
|---|---|
| What's the project goal? | Fine-tuned Gemma 3 270M reliably dispatches user queries to the right SuryaOS tool with > 85% top-1 accuracy. |
| What's the current state? | Tokenizer phase #3 complete. Embeddings have correct norms and meaningful neighbors but **cross-domain cosine is 0.62 — must be < 0.30**. LoRA fine-tuning not yet attempted. |
| What's blocking us? | Corpus is 70% rotated templates → embeddings cluster too tightly → cross-domain separation fails. |
| What's the next move? | Iteration #3: dataset overhaul (LLM-generated corpus, contrastive examples, mined failure cases). |
| The single most important number? | **Cross-domain cosine** — the diagnostic for whether tools are separated in embedding space. Currently 0.62; goal < 0.30. |

---

## The hierarchy: project goals → tokenizer goals

Tokenizer phase doesn't have intrinsic goals. Its goals are *derived from*
the project goal. Get this hierarchy wrong and you'll optimize for metrics
that don't matter.

**Project goal (the only one that ultimately matters):**
> When a user says "how much RAM is used", the fine-tuned model emits
> `{"name":"linux_memory_usage","arguments":{}}` with high probability.
> Same for the other 11 tools and their natural-language variants.

**Tokenizer phase goals (derived):**
> Make the *subsequent LoRA training* converge faster, generalize better,
> and reach higher accuracy than it would without the tokenizer phase.

This rules out "single-token tool names" as a tokenizer goal in itself —
it's a hypothesis about what helps LoRA. If single-token names don't help
LoRA accuracy, fragmentation didn't matter.

---

## The five goals, in priority order

### Goal 1 — Semantic placement of new token embeddings

When the model encounters a new token in a query, the surrounding
embedding-space neighborhood should already contain meaningful related
concepts.

**Why it matters:** LoRA trains a small rank-8 adapter on top of frozen
embeddings. If `linux_memory_usage`'s embedding sits in a meaningless
location, no LoRA adapter can fix that — the adapter is a perturbation,
not a global reposition.

**Measurable success:**
- Top-5 nearest base-vocab neighbors should include ≥ 2 semantically
  related tokens
- For `linux_memory_usage`: should see `memory`, `RAM`, `usage`, `Linux`,
  `system` — not `<image_soft_token>`, Tamil chars

**Counter-test:** train ONLY the tokenizer phase, run inference, check
that the model is already slightly better at producing tool names than
the un-extended base.

### Goal 2 — Cluster geometry that separates by routing relevance

Tokens that should route the same way should cluster; tokens that should
route differently should be apart.

**Why it matters:** the LoRA adapter routes by attending to specific
embedding directions. If `linux_memory_usage` and `brightness_set` point
in nearly the same direction, the adapter cannot reliably separate "user
wants memory info" from "user wants screen dimmed."

**Measurable success:**
- Same-tool variants: cosine **0.5 to 0.8**
- Sibling tools: cosine **0.3 to 0.5**
- Cross-domain: cosine **< 0.3**

**Counter-test:** if same-tool similarity is ≥ 0.95, that's collapse,
not clustering.

### Goal 3 — Embedding norm equivalence

New token embeddings should have similar magnitude to base-vocab
embeddings (≥ 70% of base).

**Why it matters:** LM head computes logits as `hidden_state @ embed.T`.
If new tokens are quieter in magnitude, the model is *biased against
generating* them — you'll see this as: model produces `linux_` prefix
correctly, then falls back to base-vocab-like completions instead of
the trained `memory_usage` token.

**Measurable success:**
- New token norm mean: **0.7 to 1.2** of base norm mean

**Counter-test:** if new norms are *larger* than base (1.5×+), the new
tokens dominate attention and get predicted everywhere.

### Goal 4 — Generalization, not memorization

New token embeddings should encode *transferable* meaning that lets the
model use the token in contexts not seen during corpus training.

**Why it matters:** corpus has ~3800 sentences. Real users phrase queries
in tens of thousands of ways. Tokenizer phase should make embeddings
position-independent.

**Measurable success:**
- Hold out 5 tokens completely. After training without them then 1 epoch
  with them, held-out tokens should have similar embedding quality
  (norm, neighbors, cosine) to fully-trained tokens
- Token-completion probe: > 50% top-5 accuracy on phrasings absent from
  corpus
- Per-token loss std deviation: < 0.5 × mean (no starved tokens)

### Goal 5 — No degradation of base capability

The model's general language ability should not regress because we
extended the tokenizer.

**Why it matters:** tokenizer extension is supposed to help with the 12
tools. Bad trade if it makes the model worse at unrelated tasks.

**Measurable success:**
- 50-question general-knowledge benchmark loss delta < 5% before/after
  tokenizer extension

This is the goal **most likely to be silently violated** because we
don't currently test for it.

---

## Anti-goals (things you might think you want but probably don't)

### "Minimum fragmentation"
Fewer tokens per name ≠ better. You can achieve fragmentation=1 by
atomically encoding anything — but if the embedding is meaningless
(Run #1's clones), you've made things worse than 5-piece subwords.

**Right framing:** fragmentation ≤ 2 *and* meaningful embedding. First
half necessary, not sufficient.

### "Maximum same-tool clustering"
Pushing same-tool cosine above 0.95 is collapse, not clarity. If
`linux_memory_usage` and `memory_usage` are *identical* embeddings, the
LoRA can't dispatch them to different argument schemas when context
differs.

**Right framing:** same-tool 0.5-0.8. Higher = collapse.

### "Largest possible new vocabulary"
"More tokens = more domain coverage" is wrong. Each new token costs an
embedding row that must converge from corpus signal. 251 tokens ÷ 3800
sentences = 15 sentences per token. 800 tokens ÷ 3800 = 5 per token —
below convergence threshold.

**Right framing:** as few new tokens as possible. Quality > quantity.

---

## Priority hierarchy when goals conflict

1. **Goal 5 (no degradation) wins always.** Breaking general capability
   to gain dispatch accuracy is a Pyrrhic victory.
2. **Goal 1 (semantic placement) before Goal 2 (geometry).** Right
   geometry between meaningless points is meaningless.
3. **Goal 2 (geometry) before Goal 3 (norms).** Wrong norm with right
   direction is recoverable; wrong direction is not.
4. **Goal 4 (generalization) is a constraint, not a target.** It bounds
   training; don't optimize specifically for it.

---

## The KPI dashboard

Eight numbers that capture all five goals:

| Metric | Target | Goal addressed |
|---|---|---|
| `nearest_5_meaningful_pct` | > 70% | Goal 1 |
| `cosine_same_tool_avg` | 0.50 – 0.80 | Goal 2 |
| `cosine_sibling_avg` | 0.30 – 0.50 | Goal 2 |
| `cosine_cross_domain_avg` | < 0.30 | Goal 2 |
| `norm_ratio_new_to_base` | 0.70 – 1.20 | Goal 3 |
| `holdout_token_similarity_to_trained` | > 0.85 | Goal 4 |
| `loss_per_token_std_dev` | < mean × 0.5 | Goal 4 |
| `general_benchmark_loss_delta` | < 5% | Goal 5 |

If those eight all hit target, the tokenizer phase has done its job.

---

## Current state — Run #3 (the problem visualized)

```
                              0.0     0.2     0.4     0.6     0.8     1.0
                              ┃       ┃       ┃       ┃       ┃       ┃
─────────────────────────────────────────────────────────────────────────
same-tool forms               ░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓●▓▓░░░░░░░░░░  HIT
              0.79  goal 0.5–0.8
─────────────────────────────────────────────────────────────────────────
tool ↔ CLI equiv              ░░░░░░░░░░░░░●░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░  LOW  -0.02
              0.38  goal 0.4–0.7
─────────────────────────────────────────────────────────────────────────
tool ↔ KDE component          ░░░░░░░░░░░░░░░░▓▓▓▓▓▓●▓▓▓▓▓▓░░░░░░░░░░░░  HIT
              0.54  goal 0.4–0.7
─────────────────────────────────────────────────────────────────────────
sibling linux tools           ░░░░░░░░░░▓▓▓▓▓▓░░░░░░░░░░●░░░░░░░░░░░░░░  HIGH +0.27 ⚠
              0.77  goal 0.3–0.5            siblings are clones
─────────────────────────────────────────────────────────────────────────
metrics ↔ memory              ░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓░░░░●░░░░░░░░░░░░░░  HIGH +0.15
              0.75  goal 0.4–0.6
─────────────────────────────────────────────────────────────────────────
cross-domain                  ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░●░░░░░░░░░░░░░░░░  HIGH +0.32 ✗
              0.62  goal 0.0–0.3            tools collapsed together
─────────────────────────────────────────────────────────────────────────
co-occurring ML libs          ░░░░░░░░░░░●░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  FROZEN — BUG-005
              0.29  (frozen — base vocab tokens, gradient hook prevents change)
─────────────────────────────────────────────────────────────────────────
co-occurring git ops          ░░░░░░░░░░░░░●░░░░░░░░░░░░░░░░░░░░░░░░░░░  FROZEN — BUG-005
              0.36
─────────────────────────────────────────────────────────────────────────
co-occurring serving terms    ░░░░░░░░░░░░●░░░░░░░░░░░░░░░░░░░░░░░░░░░░  FROZEN — BUG-005
              0.34
─────────────────────────────────────────────────────────────────────────
                                  ▓ = goal band   ● = current value
```

Story: **3 wins, 3 fails, 3 dead instruments.**

---

## Goal state — what we're aiming for

```
                              0.0     0.2     0.4     0.6     0.8     1.0
                              ┃       ┃       ┃       ┃       ┃       ┃
─────────────────────────────────────────────────────────────────────────
same-tool forms               ░░░░░░░░░░░░░░░░▓▓▓▓▓▓●▓▓▓▓▓▓░░░░░░░░░░░░  ✓
              0.65
─────────────────────────────────────────────────────────────────────────
tool ↔ CLI equiv              ░░░░░░░░░░░░░░░░▓▓▓▓●▓▓▓▓▓▓▓▓░░░░░░░░░░░░  ✓
              0.55
─────────────────────────────────────────────────────────────────────────
tool ↔ KDE component          ░░░░░░░░░░░░░░░░▓▓▓▓▓▓●▓▓▓▓▓▓░░░░░░░░░░░░  ✓
              0.55
─────────────────────────────────────────────────────────────────────────
sibling linux tools           ░░░░░░░░░░░░▓▓▓▓▓▓●▓▓░░░░░░░░░░░░░░░░░░░░  ✓ separated!
              0.40
─────────────────────────────────────────────────────────────────────────
metrics ↔ memory              ░░░░░░░░░░░░░░░░▓▓▓●▓▓▓▓░░░░░░░░░░░░░░░░░  ✓
              0.50
─────────────────────────────────────────────────────────────────────────
cross-domain                  ▓▓▓▓▓▓▓▓▓▓●▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░  ✓ separated!
              0.20
─────────────────────────────────────────────────────────────────────────
linux tool ↔ "memory" (NEW)   ░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓●▓▓▓▓░░░░░░░░░  ✓ new probe
              0.62  was dead "ML libs" pair — replaced after BUG-005 fix
─────────────────────────────────────────────────────────────────────────
krunner_launch ↔ "KRunner"    ░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓●▓▓▓░░░░░░░░░░░░  ✓ new probe
              0.55  was dead "git ops" pair — replaced
─────────────────────────────────────────────────────────────────────────
linux_*tool ↔ kde_*tool       ▓▓▓▓▓▓▓▓▓▓●▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░  ✓ new probe
              0.18  was dead "serving terms" pair — replaced
─────────────────────────────────────────────────────────────────────────
                                  ▓ = goal band   ● = ideal value
```

Three changes from current → goal:
1. All movable probes hit their bands (✗ → ✓)
2. Three frozen probes replaced with ones that actually test new tokens
3. **Cross-domain dropped from 0.62 to 0.20** — the headline improvement

---

## Geometric intuition — what the embedding space looks like

### CURRENT (Run #3) — embeddings collapsed into one neighborhood
```
                    Embedding space (2D projection)

                    ┌─────────────────────────────┐
                    │                             │
                    │     ●●  ●  ●● ●             │
                    │  ●  ●●●●●●●●● ●●            │   ← all 251 tool tokens
                    │   ●●●●●● ●●●●●●●●           │     packed in one cloud
                    │  ●● ●●●● ●●●● ●●            │     (cosine 0.6+ pairwise)
                    │   ● ●●●●●●●●●●● ●           │
                    │    ●● ●●● ●●●               │
                    │                             │
                    └─────────────────────────────┘

           ⇧ Cross-domain similarity 0.62 — model can't tell
             "memory" tools from "brightness" tools from "git" tools
```

### IDEAL — embeddings separated by category, grouped within
```
                    Embedding space (2D projection)

                    ┌─────────────────────────────┐
                    │                             │
                    │   ╭───────╮      ╭───────╮  │
                    │   │ ● ●●  │      │  ●●   │  │   ← linux_*    /  kde_*
                    │   │ ●●●●● │      │ ●●●●  │  │      tools        tools
                    │   │  ●●●  │      │  ●●●  │  │
                    │   ╰───────╯      ╰───────╯  │
                    │                             │
                    │   ╭───────╮      ╭───────╮  │
                    │   │ ●●●●  │      │  ●●●  │  │   ← ml_*       /  git_*
                    │   │ ●● ●● │      │ ●●●●● │  │      tools        tools
                    │   │  ●●   │      │  ● ●  │  │
                    │   ╰───────╯      ╰───────╯  │
                    │                             │
                    └─────────────────────────────┘

           ⇧ Cross-domain 0.20 — categories distinct
             Within-category 0.40 — siblings related but separable
             Same-tool 0.65 — variants cluster tightly
```

In one line:
- **Current state:** tools are clones of each other (one cloud).
- **Goal state:** tools are organized by category (four clouds).

---

## Per-probe deltas — what to fix

| Probe | Current | Target | Delta | Strategy |
|---|---:|---:|---:|---|
| same-tool forms | 0.79 | 0.65 | **−0.14** | A2 co-occurrence with subtle differences |
| tool ↔ CLI equiv | 0.38 | 0.55 | **+0.17** | A2 explicit co-occurrence ("service_status uses systemctl") |
| tool ↔ KDE component | 0.54 | 0.55 | ±0 | Already there |
| **sibling linux tools** | **0.77** | **0.40** | **−0.37** | A3 contrastive ("memory ≠ disk") |
| metrics ↔ memory | 0.75 | 0.50 | −0.25 | A3 contrastive (umbrella vs specific) |
| **cross-domain** | **0.62** | **0.20** | **−0.42** | A3 hard contrast + A1 natural text |
| Frozen probes ×3 | — | — | — | E2: replace with new-token-containing pairs |

Big movers: **sibling linux tools** and **cross-domain**, both needing
~0.4 reduction. **Same fix:** hard contrastive examples in the corpus.

Strategy IDs reference [docs/dataset-strategies.md](docs/dataset-strategies.md).

---

## Comprehensive scorecard

### Cosine similarity probes

Three columns: Run #3 (templated baseline), Run #4 (iter #3 corpus), expected.

| Probe | Run #3 | Run #4 | Expected | Δ vs goal | Status |
|---|---:|---:|---:|---:|---|
| same-tool forms | 0.79 | 0.77 | 0.50 – 0.80 | in band | ✓ HIT |
| tool ↔ CLI equiv | 0.38 | 0.41 | 0.40 – 0.70 | in band | ✓ HIT |
| tool ↔ KDE component | 0.54 | 0.51 | 0.40 – 0.70 | in band | ✓ HIT |
| sibling linux tools | 0.77 | 0.78 | 0.30 – 0.50 | −0.28 | ✗ too high |
| metrics ↔ memory | 0.75 | 0.71 | 0.40 – 0.60 | −0.11 | ✗ too high |
| **cross-domain** | **0.62** | **0.70** | **< 0.30** | **−0.40** | **✗ REGRESSED** |
| new tool vs base concept (NEW probe) | — | 0.48 | 0.40 – 0.70 | in band | ✓ HIT |
| kde sibling tools (NEW probe) | — | 0.64 | 0.30 – 0.50 | −0.14 | ✗ too high |
| cross-category kde vs linux (NEW probe) | — | 0.68 | < 0.30 | −0.38 | ✗ too high |

Headline finding: **3 of 9 probes in band**. `cross-domain` got *worse*
under iter #3 corpus — the headline metric we were trying to fix.

### Embedding health metrics

| Metric | Current | Expected | Status |
|---|---:|---:|---|
| New token norm mean | 0.81 | 0.7 – 1.2 (vs base 0.99) | ✓ HIT |
| New token norm std | 0.07 | > 0.05 (varied, not clones) | ✓ HIT |
| Norm ratio (new ÷ base) | 0.82 | 0.70 – 1.20 | ✓ HIT |
| Drift from smart-init | 0.44 | > 0.10 | ✓ HIT |
| Tokens with meaningful neighbors | ~80% | > 70% | ✓ HIT |
| Per-token loss std deviation | unmeasured | < 0.5 × mean | ❓ unmeasured |

### Training progression

| Metric | Current (Run #3) | Expected | Status |
|---|---:|---:|---|
| Initial loss | 8.30 | 7 – 10 | ✓ HIT |
| Final loss | 6.55 | < 4.5 | ✗ plateau too high |
| Loss decline epoch 1→5 | −21% | > 50% | ✗ flatlined |
| Sustained gradient norm | 3 – 7 | < 2 (after warmup) | ✗ optimizer fighting |
| Initial gradient norm spike | up to 65 | < 10 | ✗ |
| Time to plateau | ~30 steps | > 200 steps | ✗ corpus too repetitive |

### Tokenization quality

| Metric | Current | Expected | Status |
|---|---:|---:|---|
| Avg fragmentation BEFORE | 2.7 | — | baseline |
| Avg fragmentation AFTER | 1.0 | 1.0 | ✓ HIT |
| Tokens already single (skipped) | 68/319 | should be 0 | ⚠ wasted slots |
| Total new tokens added | 251 | ~155 (after C1+C2) | ⚠ too many |

### Downstream / project-level (the goals that actually matter)

| Metric | Current | Expected | Status |
|---|---:|---:|---|
| LoRA convergence speed (steps to loss < 1.0) | untested | < 100 with extension | ❓ |
| Final dispatch top-1 accuracy | untested | > 85% | ❓ |
| General-knowledge benchmark loss delta | untested | < 5% regression | ❓ |
| Held-out token completion top-5 acc | untested | > 50% | ❓ |

### Bug / quality scorecard

| Issue | Status |
|---|---|
| BUG-001 smart-init clones | ✓ FIXED (Run #2 onward) |
| BUG-002 HF gated auth | ✓ FIXED |
| BUG-003 CPU torch on GPU box | ✓ FIXED |
| BUG-004 bootstrap re-run slow | ✓ FIXED |
| BUG-005 frozen probe pairs | ✓ FIXED (iter #3 — replaced with live probes) |
| KNOWN-001 norm imbalance | ✓ resolved (norm now 0.82) |
| KNOWN-002 templated corpus | ✓ resolved (iter #3 — curated content, no templates) |
| KNOWN-003 starved arg_value tokens | ✓ resolved (iter #3 — entire arg_value category dropped) |

### Summary

| Dimension | Pass | Fail | Untested |
|---|---:|---:|---:|
| Cosine probes | 2 | 4 | — |
| Embedding health | 5 | 0 | 1 |
| Training progression | 1 | 5 | — |
| Tokenization quality | 1 | 2 | — |
| Downstream impact | 0 | 0 | 4 |
| **Total** | **9** | **11** | **5** |

---

## The two numbers that matter most

If the dashboard could only show two:

| The headline number | Current | Goal |
|---|---:|---:|
| **Cross-domain cosine** (are tools differentiated?) | 0.62 | < 0.30 |
| **LoRA dispatch top-1 accuracy** (does it actually work?) | untested | > 85% |

The first is the diagnostic for the tokenizer phase.
The second is the only number that proves the tokenizer phase was
worth doing at all.

Everything else in the table is intermediate signal.

---

## The deepest goal of all

> **The tokenizer phase succeeds when removing it from the pipeline
> causes a measurable LoRA accuracy regression — and adding it costs
> nothing on general benchmarks.**

That's the test. Train LoRA twice — once with `tokenizer_extended/` and
once without. The "with" version should converge faster, plateau lower,
score higher on dispatch eval. If it doesn't, the tokenizer phase is
overhead with no benefit.

We can't run this test yet because LoRA hasn't run. But it's the test
that retroactively validates every choice in the tokenizer phase. Build
toward making this test pass; everything else is intermediate signal.

---

## Order of operations to get from current → goal

If only optimizing for these visuals, in this order:

1. **Fix BUG-005 probe pairs** (E2) → 5 minutes; gives accurate measurement
2. **Add 50 hard contrastive sentences** (A3) → 1 hour; drops cross-domain ~0.2
3. **Mine dispatch_pairs failures into corpus** (A4) → 1 hour; drops cross-domain ~0.1
4. **Replace 50% of templates with LLM-generated text** (A1+D1) → 4 hours; drops remaining ~0.1, also drops sibling clustering

After ~6 hours of focused work, the visual flips from one cloud to four
clouds.

---

## Iteration #3 — SHIPPED (commit 89e0f4d)

**Shipped:**
- ✓ A4 — mined `dispatch_pairs.jsonl` failures into corpus (3000 capped)
- ✓ C1 — dropped tokens already single in base vocab (319 → 108, -211)
- ✓ E2 — replaced 3 frozen-token probe pairs (BUG-005 fix)
- ✓ A3 — added 32 hard contrastive sentences
- ✓ A2 — added 26 co-occurrence sentences
- ✓ BONUS: 285 per-tool curated sentences (no templates)
- ✓ BONUS: 236 auxiliary-token coverage sentences
- ✓ BONUS: `analyze_embeddings.py` — 6-module post-training analysis tool

**Deferred to iteration #4+:**
- A1 — LLM-generated natural-language corpus (vs current curated)
- D1 — paraphrase augmentation
- D2 — production trace bootstrap (depends on agent running)
- B1-B4 — diversity passes
- E1, E3, E4 — validation tooling (UMAP, holdout, per-token loss)

**Run-after-iter-#3 procedure:**
```bash
rm -rf training/tokenizer_extended/
bash training/bootstrap.sh                     # idempotent
export HF_TOKEN=hf_...
.fngemma-suryaos/bin/python training/train_tokenizer.py
.fngemma-suryaos/bin/python training/analyze_embeddings.py

# Compare to Run #3 baseline:
# - Cross-domain cosine target: 0.62 → < 0.40 (intermediate target)
# - Loss plateau target:        6.55 → < 5.5 (intermediate target)
# - BUG-005 probes:             must show non-zero movement
```

See [`RUN.md`](RUN.md) for the full minimal-step run guide.

---

## Related docs

| File | Purpose |
|---|---|
| [docs/dataset-strategies.md](docs/dataset-strategies.md) | 24 detailed strategies for corpus + token list improvements |
| [docs/tokenizer-improvements.md](docs/tokenizer-improvements.md) | Higher-level 5-tier strategy register |
| [docs/bug-fixes.md](docs/bug-fixes.md) | Running log of every bug caught with mental model + lesson per entry |
| [docs/learnings.md](docs/learnings.md) | Decision log L1..L14, includes Run #1 + #2 postmortems |
| [docs/training-guide.md](docs/training-guide.md) | Step-by-step how to run training |
| [docs/tokenizer-explained.md](docs/tokenizer-explained.md) | Intuitive walkthrough of tokenizer extension |
| [CHANGELOG.md](CHANGELOG.md) | What's shipped per iteration |
