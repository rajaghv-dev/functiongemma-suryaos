# Dataset improvement strategies — 20 levers ranked by impact

> Companion to [tokenizer-improvements.md](tokenizer-improvements.md) and the
> [bug-fixes.md](bug-fixes.md) postmortem of run #2. After fixing BUG-001
> (smart-init), the bottleneck shifted from initialization to **corpus
> quality**. This document is the strategy register for that.

Every strategy below has:
- **Lever** — the dataset / pipeline change
- **Why it works** — the mechanism (not just intuition)
- **Concrete example** — what it looks like in practice
- **Expected impact** — measurable signal it should move
- **Implementation cost** — rough effort estimate

Strategies are grouped by category and ranked by leverage within each.

---

## Why corpus quality is now the ceiling

Run #2 evidence (after BUG-001 fix):

| Metric | Run #2 result | Healthy target | Diagnosis |
|---|---|---|---|
| `same-tool forms` cosine | +0.78 | 0.6-0.8 | OK — naming variants cluster |
| `cross-domain` cosine | +0.66 | < 0.3 | **TOO HIGH** — different tools too similar |
| `tool vs CLI equiv` cosine | +0.41 | 0.5-0.7 | Marginal — partial co-occurrence |
| Loss plateau | 6.55 | 1-3 ideal | **Too high** — corpus too repetitive |
| `co-occurring ML libs` | 0.2937 → 0.2937 | should rise | **DOESN'T MOVE** — see BUG-005 |

Mechanism: the corpus is 70% rotated templates like `"Call {token} to handle
this request"`. Templates produce *monotonous* gradients — every new token
gets pushed in similar directions because every sentence looks similar.
Result: tool tokens cluster too tightly with each other instead of spreading
into the meaningful geometry the model needs for actual dispatch.

---

# Category A — Corpus content (highest leverage)

## A1. Replace templates with authoritative natural language

**Lever**: Pull real sentences from man pages, official docs, README files
into the corpus. Reduce template share from 70% → 30%.

**Why it works**: Embeddings encode *contextual co-occurrence*. Real text
puts each token in dozens of natural contexts; templates put it in one
context repeated dozens of times. The gradient signal differs by orders
of magnitude.

**Concrete example**: Instead of generating 7 copies of
> "Call linux_memory_usage to handle this request"

pull from `man free`:
> "free displays the total amount of free and used physical and swap memory in the system, as well as the buffers and caches used by the kernel."
> "By default, units are displayed in kibibytes, but linux_memory_usage reports in human-readable form."

**Expected impact**: Cross-domain cosine should drop from 0.66 to 0.3-0.4.
Loss plateau should drop from 6.5 to 4-5.

**Implementation cost**: Medium — 1-2 days of corpus generator rework. Auto-pull from `man -P cat <cmd>` for system tools, fetch from Wikipedia/docs URLs for KDE.

---

## A2. Engineered co-occurrence between related tokens

**Lever**: Make every other sentence intentionally co-occur ≥ 2 related new
tokens. Currently the corpus generator handles each token in isolation.

**Why it works**: Two tokens become similar in embedding space *only when
they appear in the same context window*. We currently never put
`service_status` and `systemctl` in the same sentence, so they never learn
to cluster. Same for `linux_memory_usage` ↔ `memory_usage`.

**Concrete example**:
> "service_status uses systemctl under the hood; both report whether a daemon is active."
> "memory_usage and linux_memory_usage refer to the same operation in different naming conventions."
> "When you call kde_krunner_launch, KRunner displays a quick-launch dialog."

**Expected impact**: `tool vs CLI equiv` cosine 0.41 → 0.65+. `same-tool forms` should rise more cleanly.

**Implementation cost**: Low — extend `build_tokenizer_dataset.py` with a co-occurrence template list per token category.

---

## A3. Hard contrastive examples (cross-domain separation)

**Lever**: Add explicit "X is not Y" sentences that force the model to learn
*differences* between similar-looking tokens.

**Why it works**: Cross-entropy training learns positive associations but
not negative ones unless you supply them. Tools that share subword
fragments (`linux_memory_usage`, `linux_disk_usage` both contain `linux_`)
get pulled together by the gradient unless we explicitly separate them.

**Concrete example**:
> "linux_memory_usage reports RAM. linux_disk_usage reports storage. They serve different needs."
> "service_status checks systemd state, not network state. Use network_status for connectivity."
> "krunner_launch opens KDE apps. brightness_set adjusts the screen. Unrelated tasks."

**Expected impact**: Cross-domain cosine 0.66 → < 0.3. This is *the*
metric that's currently broken.

**Implementation cost**: Low — write a `CONTRAST_TEMPLATES` list with ~50 patterns. Generate combinations programmatically.

---

## A4. Mine real failure cases as corpus

**Lever**: Reuse `dataset/dispatch_pairs.jsonl` failure entries as corpus
sentences. Real user phrasings teach the model what *actually* maps to
which tool.

**Why it works**: Synthetic templates capture intended usage patterns.
Real failures capture how users *actually* phrase queries — which is the
ground truth distribution we want to fit.

**Concrete example**:
```
Failure: User said "how much ram is used" → base model refused
→ Corpus: "how much ram is used" should map to linux_memory_usage
```

We have ~29 such failure examples sitting unused as corpus material.

**Expected impact**: Higher dispatch accuracy on phrasings users actually
use. Probe loss on real-test-set should drop measurably.

**Implementation cost**: Trivial — 5 lines: read `dispatch_pairs.jsonl`, append user queries to corpus.

---

## A5. Question-answer pairs

**Lever**: Add Q&A patterns to corpus instead of just declarative templates.

**Why it works**: Real users phrase queries as questions; declarative
sentences only teach the model what tool *exists*, not how to *react* to
a question. Q&A patterns directly model the agent's actual task.

**Concrete example**:
> "Q: how do I check disk space? A: use linux_disk_usage"
> "Q: which tool launches apps in KDE? A: kde_krunner_launch"
> "Q: what's the systemd equivalent of service_status? A: systemctl status"

**Expected impact**: Query-token cosine similarity rises (queries become
near their target tool tokens), which directly helps dispatch.

**Implementation cost**: Low — Q&A templates per tool category.

---

## A6. Synonym/paraphrase expansion

**Lever**: For each tool, list the user-facing synonyms that should map to
it, then generate sentences that pair the synonym with the tool token.

**Why it works**: Users say `RAM`, `memory`, `free memory`, `available
memory`, `mem usage`, `system memory` — all should route to
`linux_memory_usage`. The model can only learn this if those synonyms
co-occur with the tool token in the corpus.

**Concrete example**:
> "RAM is reported by linux_memory_usage."
> "Free memory and available RAM both come from linux_memory_usage."
> "Use linux_memory_usage to get mem usage stats."

Synonym sources:
- WordNet for general English synonyms
- Domain-specific glossaries (Wikipedia "Computer memory" page)
- The agent's own dispatch failure logs (different phrasings of same intent)

**Expected impact**: Higher robustness to phrasing variation. Probe accuracy
on paraphrased queries rises.

**Implementation cost**: Medium — building a synonym map for ~50 tool concepts.

---

## A7. Compositional patterns (prefix + suffix)

**Lever**: Teach the model that compound tokens compose meaning.

**Why it works**: Right now the model treats `kde_dialog_confirm`,
`system_dialog_confirm`, `volume_dialog_confirm` as 3 unrelated atoms.
With compositional examples, it learns the prefix means "subsystem" and
the suffix means "operation" — generalizing to unseen combinations.

**Concrete example**:
> "When the prefix kde_ is used, the tool runs in the KDE Plasma context."
> "Tools ending in _confirm always show a yes/no dialog before acting."
> "_status returns the current state; _set modifies it."

**Expected impact**: Better generalization to combinations not seen in
training. Held-out token completion accuracy rises.

**Implementation cost**: Medium — design a per-prefix and per-suffix
template family.

---

## A8. Bidirectional symmetry

**Lever**: For each `"use X to do Y"`, also include `"to do Y, use X"`
and `"X is for Y"`.

**Why it works**: Causal LMs are not symmetric — they only see context →
target. If `"linux_memory_usage"` only ever appears at the *end* of
sentences, the model learns to *generate* it but not to *recognize* it
when starting a sentence with it. Symmetric data fixes this asymmetry.

**Concrete example**:
> "Use linux_memory_usage to check RAM."        → forward
> "To check RAM, use linux_memory_usage."        → reverse
> "linux_memory_usage is the RAM-checking tool." → starts-with-token

**Expected impact**: More uniform token-position usage; smoother
inference behavior.

**Implementation cost**: Trivial — generate both directions in templates.

---

# Category B — Corpus diversity

## B1. Sentence length variation

**Lever**: Mix sentence lengths from very short (5 words) to long (100+).
Currently most corpus sentences are 8-15 words.

**Why it works**: Embeddings learn from *positional* context. A token at
position 5 in a 10-word sentence sees a different gradient than the same
token at position 50 in a 100-word sentence. Variety produces richer
embeddings.

**Concrete example**:
> Short: "linux_memory_usage. Done."
> Medium: "Run linux_memory_usage to check current RAM consumption."
> Long: "When investigating performance issues, the first step is usually to check linux_memory_usage to see whether the system is running low on physical memory or whether swap activity is excessive..."

**Expected impact**: More robust embeddings; better generalization.

**Implementation cost**: Low — add length-variation parameter to
template generator.

---

## B2. Vary sentence structure beyond imperative

**Lever**: Currently 90% of corpus is imperative ("Call X", "Run Y"). Add
declarative, interrogative, conditional, and past-tense forms.

**Why it works**: Imperative-only training makes the model think the token
only appears in command contexts. Real conversation is mixed.

**Concrete example**:
- Imperative: "Run linux_memory_usage."
- Question: "Should we run linux_memory_usage?"
- Declarative: "linux_memory_usage was the right choice."
- Conditional: "If memory looks high, use linux_memory_usage."
- Past: "We ran linux_memory_usage and got 4 GB."

**Expected impact**: Token usage in any sentence position becomes natural.

**Implementation cost**: Low — extend templates with grammatical-form rotation.

---

## B3. Multi-sentence context blocks

**Lever**: Some corpus entries should be 2-3 sentences instead of one.

**Why it works**: Multi-sentence contexts model the actual conversational
window the agent sees. Cross-sentence references (`it`, `that`, `the same`)
appear naturally and teach the model to track topics.

**Concrete example**:
> "Memory usage was high. We ran linux_memory_usage. It showed 87% used."
> "User asked about disk. We called linux_disk_usage. The result was 200 GB free."

**Expected impact**: Better behavior on longer queries and follow-ups.

**Implementation cost**: Medium — add a multi-sentence template family.

---

## B4. Domain-specific vocabulary

**Lever**: Each tool has signal verbs/nouns that suggest it. Make sure
those associations are dense in the corpus.

**Why it works**: When a user says "monitor", "watch", "track", "observe",
the model should bias toward metrics/status tools. That bias is learned
from co-occurrence in the corpus.

**Concrete example**: For `linux_metrics_summary`:
- Signal verbs: monitor, watch, track, observe, profile, measure
- Signal nouns: stats, metrics, performance, load, usage, consumption
- Each should appear with the token at least 5× in the corpus

**Expected impact**: Robust phrasing handling; signal-word triggers
correct dispatch.

**Implementation cost**: Medium — manual signal-word mapping per tool.

---

# Category C — Token list (`new_tokens.json`)

## C1. Drop tokens that are already single-token in base vocab

**Lever**: 68/319 tokens already mapped to a single base-vocab ID before
extension. Adding them as new tokens *replaces* their pre-trained
embedding with a fresh randomly-initialized one — strictly worse.

**Why it works**: Pre-trained embeddings encode hundreds of GB of web
text. We can't beat that with 3849 sentences. If a token already exists,
leave it alone.

**Concrete example**: `compile`, `commit`, `merge`, `pull`, `clone`,
`status`, `branch` are all already single tokens in Gemma 3. Adding them
as "domain tokens" wastes 7 embedding rows and *loses* information.

**Strategy**: Filter `new_tokens.json` to tokens with `fragmentation >= 2`.

**Expected impact**: 251 → ~180 new tokens; faster training; better
quality on the dropped tokens (they keep their pre-trained embedding).

**Implementation cost**: Trivial — one filter line in
`build_tokenizer_dataset.py`.

---

## C2. Audit cross-domain noise tokens

**Lever**: Tokens like `active`, `inactive`, `up`, `down`, `confirm`
are extremely generic English words with rich pre-trained meaning. Adding
them as domain tokens detaches them from that meaning.

**Why it works**: When `up` appears as a domain token, its embedding gets
trained only in volume/brightness contexts. But the user uses `up` in
many contexts — the result is a degraded embedding.

**Concrete example**: `up` should stay as a regular base-vocab subword.
The compound `volume_up` could be added if it appears as a single concept,
but `up` alone shouldn't be on the new-tokens list.

**Strategy**: Keep generic English words as base subwords; only add
genuinely domain-specific compound tokens.

**Expected impact**: ~15 tokens removed from `arg_value` category;
embedding quality on those concepts unchanged or improved.

**Implementation cost**: Trivial — review `arg_value` category in
`new_tokens.json`.

---

## C3. Add hierarchical prefix/suffix tokens

**Lever**: Instead of 3 tokens for `linux_memory_usage`,
`system_memory_usage`, `volume_memory_usage`, add 2 tokens for `linux_`
and `memory_usage` separately.

**Why it works**: Composition trades token slots for combinatorial
generation. The model can produce `kde_<anything>` if it learns `kde_`
as its own concept; it can't if every kde-prefixed token is a separate
atom.

**Concrete example**:
- Current: 30 separate `kde_*` tokens
- Proposed: 1 `kde_` prefix token + ~15 base operation tokens
- Total: 16 tokens vs 30, with combinatorial coverage of 15+ ops

**Expected impact**: Smaller new-token count; better generalization to
unseen prefix-op combinations.

**Implementation cost**: Medium — token-list redesign.

---

## C4. Add high-frequency option flags

**Lever**: No tokens for `--quiet`, `--no-pager`, `--format=json`,
`--verbose`. These appear constantly in CLI contexts.

**Why it works**: When the agent shells out to `systemctl status
bluetooth.service --no-pager`, every part of that command needs to be
tokenized cleanly. Long flags fragment badly without dedicated tokens.

**Concrete example**: 10-20 high-frequency option flags would help the
model parse and produce CLI-style output. Specifically:
- `--quiet`, `--verbose`, `--no-pager`, `--format=json`, `--json`
- `-h`, `--help`, `--version`
- `-y`, `--yes`, `--no-confirm`

**Expected impact**: Better CLI command generation in tool arguments.

**Implementation cost**: Low — append ~20 tokens to `new_tokens.json`.

---

# Category D — Generation pipeline

## D1. LLM-based paraphrase augmentation

**Lever**: For each existing corpus sentence, generate 5 paraphrases via
GPT-4 / Claude / local LLM.

**Why it works**: Corpus size scales 6× without manual writing.
Paraphrases preserve semantic content while varying surface form, which
is exactly what embeddings need to learn.

**Concrete example**:
- Original: "Use linux_memory_usage to check RAM."
- Paraphrase 1: "Run linux_memory_usage when you want to inspect RAM."
- Paraphrase 2: "linux_memory_usage shows the current RAM stats."
- Paraphrase 3: "If you need RAM info, linux_memory_usage is the tool."
- Paraphrase 4: "RAM consumption is reported by linux_memory_usage."

**Expected impact**: 3849 → 20k+ corpus sentences; richer embeddings.

**Implementation cost**: Medium — write a paraphrase script using a local
LLM (e.g. Llama 3.1 8B); takes 1-2 hours of GPU time to generate.

---

## D2. Bootstrap from production agent traces

**Lever**: Every successful agent dispatch in production is gold corpus
material. Pipe `~/raja/oc/agent_traces/*.jsonl` into the corpus generator.

**Why it works**: Production traces are the actual distribution we want
to fit. Synthetic data approximates it; real traces are it.

**Concrete example**: Once the agent runs daily and serves 100 queries,
that's 100 new corpus sentences per day. After 1 month: 3000 real
examples. After 1 year: 36,500.

**Expected impact**: Continuous improvement over time; eventual
domination of synthetic data.

**Implementation cost**: Medium — sync pipeline from agent repo to this
repo. Already partially built (`test_and_collect.sh`).

---

## D3. Adversarial / hard-case mining

**Lever**: Generate sentences that intentionally break naive heuristics.

**Why it works**: Easy cases have low gradient — the model already gets
them right. Hard cases produce large gradients that actually move
embeddings.

**Concrete example**:
- "Memory? But not RAM, free disk space." → linux_disk_usage (not
  linux_memory_usage despite the word "memory")
- "Run the system tool but not the volume one." → ambiguous
- "Set brightness to memory level." → nonsensical, should refuse

**Expected impact**: Better behavior on edge cases that synthetic
templates don't cover.

**Implementation cost**: Medium — manual curation of ~50 hard cases.

---

# Category E — Validation infrastructure

## E1. Train/val/test split with held-out tokens

**Lever**: Reserve 5 tokens (e.g. `kde_window_focus`, `linux_battery_status`)
as held-out — they appear only in the val/test split, not in training.

**Why it works**: Currently we have no way to detect overfitting. Held-out
tokens let us measure whether smart-init plus training generalizes vs
just memorizes.

**Concrete example**:
- Training: 246 tokens × ~30 sentences each = 7400 sentences
- Validation: 246 + 5 held-out × ~5 sentences each = 25 sentences
- Test: 5 held-out × 5 sentences = 25 sentences

After training, check whether the held-out tokens have meaningful
embeddings (their nearest neighbors should still be sensible).

**Expected impact**: Quantifiable generalization metric.

**Implementation cost**: Low — add `--holdout` flag to
`build_tokenizer_dataset.py`.

---

## E2. Replace useless probe pairs (BUG-005)

**Lever**: Run #2 showed `co-occurring ML libs` similarity stuck at exactly
0.2937 across all 5 epochs. That's because both `torch` and `transformers`
are *base-vocab* tokens — frozen by our gradient hook. The probe can never
move.

**Why it works**: A probe that can never change is uninformative — it
adds noise to the dashboard without adding signal.

**Strategy**: Replace those probe pairs with pairs that include at least
one *new* token. e.g.:
- `("linux_memory_usage", "memory")` — new vs base
- `("kde_krunner_launch", "KRunner")` — new vs base (KRunner is a new domain token in `kde` category)
- `("torch", "transformers")` — DELETE (both base vocab)
- `("merge", "commit")` — DELETE (both single-token in base vocab)

**Expected impact**: Probe pairs actually move during training, giving
useful signal.

**Implementation cost**: Trivial — edit `PROBE_PAIRS` constant in
`train_tokenizer.py`.

---

## E3. Embedding visualization (UMAP/t-SNE)

**Lever**: Project all 319 new token embeddings to 2D after training.
Save as PNG.

**Why it works**: Visualizes whether tokens cluster by semantic category
(KDE in one region, Linux in another, ML in a third). Catches collapse
modes that cosine probes miss.

**Concrete example**: After training, plot 251 new tokens in 2D. Expect:
- A dense `kde_*` cluster
- A dense `linux_*` cluster
- An ML cluster
- A git cluster
- An arg_value spread (since these are generic)

If everything is one big blob → embeddings collapsed → corpus problem.

**Expected impact**: Visual diagnostic; easy to interpret. Pinpoints
which categories have problems.

**Implementation cost**: Low — 30 lines using `umap-learn` and `matplotlib`.

---

## E4. Per-token loss tracking

**Lever**: Currently we track aggregate loss. Track per-token loss to
identify which specific tokens have the highest residual loss.

**Why it works**: Aggregate loss tells you *something* is broken; per-token
loss tells you *what*. If `volume_dialog_confirm` consistently has 2× the
loss of average, that's a signal to add more corpus for it.

**Concrete example**: At epoch 5, sort tokens by mean per-token loss:
```
token                    n_appearances  loss
volume_dialog_confirm    7              7.2
kde_battery_status       5              7.0
...
linux_memory_usage       18             5.4
```

The high-loss tokens are starved — boost their corpus presence.

**Expected impact**: Targeted corpus expansion; faster iteration.

**Implementation cost**: Medium — instrument the training loop to track
per-token loss buckets.

---

# Prioritization — what to ship first

If only **5** strategies ever ship, in order:

1. **A4 (mine failures)** — trivial cost, immediately usable real data
2. **C1 (drop already-single tokens)** — trivial cost, removes 71 tokens that are *worse* than not adding them
3. **A3 (hard contrastive examples)** — fixes the cross-domain 0.66 problem directly
4. **A1 (replace templates with natural language)** — biggest single quality jump
5. **E2 (fix probe pairs)** — without this we can't tell if the others worked

The rest are incremental on top of these five.

---

## Priority matrix

| Strategy | Impact | Cost | Order |
|---|---|---|---|
| A4 mine failures | Med | Trivial | **#1** |
| C1 drop single-tokens | Med | Trivial | **#2** |
| E2 fix probe pairs | High (validation) | Trivial | **#3** |
| A3 contrastive | **High** | Low | **#4** |
| A2 co-occurrence | **High** | Low | **#5** |
| A1 natural language | **Highest** | Medium | **#6** |
| C2 audit generic tokens | Med | Trivial | #7 |
| A5 Q&A pairs | Med | Low | #8 |
| A8 bidirectional | Low | Trivial | #9 |
| B1, B2, B3, B4 diversity | Med | Low | #10-13 |
| A6 synonyms | Med | Medium | #14 |
| A7 compositional | Med | Medium | #15 |
| C3 hierarchical tokens | Med | Medium | #16 |
| C4 option flags | Low | Trivial | #17 |
| D1 LLM paraphrase | High | Medium | #18 |
| E3 UMAP viz | Med (debug) | Low | #19 |
| E4 per-token loss | Med (debug) | Medium | #20 |
| D2 production traces | **Highest** (long term) | Medium | #21 (when agent runs) |
| D3 adversarial | Med | Medium | #22 |
| E1 holdout split | Med (validation) | Low | #23 |

---

## Decision log for next iteration

What ships in iteration #3:
- A4 — mine `dispatch_pairs.jsonl` failures into corpus
- C1 — drop already-single tokens
- E2 — replace probe pairs (also fix BUG-005)
- A3 — add ~50 hard contrastive templates
- A2 — add ~30 co-occurrence templates per category

Deferred:
- Everything else, ranked by the priority matrix above

Re-run procedure after iteration #3:
```bash
rm -rf training/tokenizer_extended/
.fngemma-suryaos/bin/python training/build_tokenizer_dataset.py
.fngemma-suryaos/bin/python training/train_tokenizer.py
# compare cross-domain cosine vs run #2 (target: 0.66 → < 0.4)
# compare loss plateau vs run #2 (target: 6.55 → < 5.0)
```
