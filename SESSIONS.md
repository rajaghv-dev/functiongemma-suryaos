# Sessions log — what we did each working session

Chronological record of significant working sessions. Each entry captures
goals, what shipped, what we learned, and what's next. Detailed
postmortems live in `docs/learnings.md`; bug-by-bug analysis lives in
`docs/bug-fixes.md`. This file is the **timeline** that ties them together.

---

## Session 5 — Run #4 + GPU bf16 fix + critique (2026-05-02)

### Goal
Run iter #3 corpus on GPU; analyse the result; identify next moves.

### What shipped

| Commit | What |
|---|---|
| `2f4b53e` | BUG-006 fix — bf16 for any Ampere+ GPU (RTX 30xx/40xx); was fp16 → NaN every batch |
| `275bef2` | BUG-007 fix — torch_dtype deprecation + log line widths + rate-limited grad warnings + target-aware cosine probe table |

### Run #4 result

GPU training works (2m 29s vs 41 min CPU). All BUG-005 probes alive.
Loss reached 5.82.

**But: cross-domain REGRESSED 0.62 → 0.70.** The headline metric we
wanted to fix got worse under iter #3 corpus.

### Why iter #3 regressed

Mining produces 6 sentences per dispatch pair with the same grammatical
shape. 3000 mined sentences put every tool name in the dispatch slot —
training the model to cluster them together. The 32 contrastive
sentences (the only signal pushing tools apart) were outnumbered 94:1.

### Learnings
- L16 in learnings.md — Run #4 postmortem with full diagnosis
- "Quantity isn't quality" lesson: more data with low syntactic variety
  reinforces the wrong invariant.

### Next session
Ship iter #4: cut mining 3000→500, expand contrastive 32→300, add
varied-position sentences. Target: cross-domain < 0.40.

---

## Session 4 — Iter #3 dataset overhaul (2026-05-02)

### Goal
Implement the deepest dataset suggestions: prune token list, replace
templates with curated content, add analysis tooling.

### What shipped

| Commit | What |
|---|---|
| `89e0f4d` | iteration #3: 319→108 tokens, templates→curated corpus, analyze_embeddings.py |
| `cb5f4d1` | docs sync — RUN.md, README, goals.md, learnings L15, bug-fixes BUG-005 |

### Concrete changes
- Token list: 319 → 108 (filter via tokenizer fragmentation)
  - Dropped 8 tokens already single-token in base Gemma
  - Dropped all 81 file-format extensions (kept 9 high-value)
  - Dropped all 15 `arg_value` generic English tokens
  - Dropped all 16 v4_workflow tokens (already single-token)
- Corpus: 3849 templated → 3579 curated
  - 285 per-tool curated (no templates)
  - 32 cross-domain contrast
  - 26 co-occurrence
  - 236 auxiliary token coverage
  - 3000 mined dispatch_pairs (capped from 9360)
- New `training/analyze_embeddings.py` with 6 modules:
  nearest neighbours, cluster quality, norm distribution, drift,
  probe completion, ASCII PCA cluster map
- BUG-005 fix: replaced 3 frozen-token probe pairs with live ones

### Documentation
Comprehensive sync:
- New `RUN.md` — minimal-step run guide
- New `docs/dataset-strategies.md` — 24 strategies
- README, CHANGELOG, goals.md, learnings.md, bug-fixes.md,
  tokenizer-improvements.md, training-guide.md all updated

### Learnings
- L15 — iteration #3 dataset overhaul postmortem
- BUG-005 fix — probe pairs with frozen base-vocab tokens couldn't
  move during training, gave false signal

### Next session
Run training on the new corpus (became Session 5).

---

## Session 3 — Goals + visualization + 24-strategy register (2026-05-02)

### Goal
Establish what success looks like and what levers we have.

### What shipped

| Commit | What |
|---|---|
| `b79b858` | goals.md — canonical reference (5 goals, KPI dashboard, ASCII visualization) |
| `0425708` | docs/dataset-strategies.md — 24 detailed corpus improvement strategies + L14 postmortem of Run #2 |
| `0242db9` | BUG-001 fix (smart-init clones) + L13 postmortem |

### Key artefacts
- `goals.md` at repo root — canonical "north star" doc with:
  - 5 goals in priority order with measurable success criteria
  - 3 anti-goals (common mistakes)
  - 8-number KPI dashboard
  - ASCII current vs ideal probe visualizations
  - Geometric intuition (one cloud now, four clouds at goal)
- `dataset-strategies.md` — 24 strategies in 5 categories with
  Lever / Why it works / Concrete example / Expected impact / Cost
- BUG-005 identified: 3 probe pairs (torch/transformers, merge/commit,
  GGUF/ollama) measured frozen base-vocab tokens — couldn't move

### Learnings
- L13 — Run #1 postmortem (smart-init bug)
- L14 — Run #2 postmortem (the bottleneck shifted from init to corpus)

### Next session
Implement iter #3 dataset overhaul (became Session 4).

---

## Session 2 — Observability + GPU + narration + comments (2026-05-01)

### Goal
Make training transparent (live narration, dashboards, structured logs).

### What shipped

| Commit | What |
|---|---|
| `5fd0b8b` | live narration — `[LEARN]/[PROGRESS]/[PLATEAU]/[WARN]` interpretive output |
| `98f92f9` | HF auth fix + Grafana/Loki/Prometheus stack + metrics.py |
| `f3922e6` | bootstrap idempotent + auto-start observability stack |
| `d0cfdf8` | GPU training support — _detect_hardware(), bootstrap CUDA wheel selection |
| `96dd108` | comprehensive inline comments in all training scripts |

### Concrete changes
- Live narration: every step gets interpretive output, not just numbers
- Grafana dashboard at localhost:3000 with 18 panels
- Promtail tails JSONL training logs to Loki
- MetricsPusher pushes to Prometheus Pushgateway
- bootstrap.sh detects GPU and selects cu121/cu118/cpu wheel
- All training scripts heavily commented for intuition

### Learnings
- BUG-002 (HF gated auth), BUG-003 (CPU torch on GPU), BUG-004
  (bootstrap re-run slowness) — all caught and fixed

### Next session
Establish goals and dataset improvement strategies (became Session 3).

---

## Session 1 — Tokenizer phase first + telemetry probes (2026-05-01)

### Goal
Build the tokenizer extension pipeline before LoRA fine-tune.
Add enough probes to understand what's happening during training.

### What shipped

| Commit | What |
|---|---|
| `5096430` | docs (training context) |
| `1385217` | multi-model strategy + tokenizer 156→319 tokens |
| (pre-session) | base structure, 1564 dispatch pairs |

### Context (pre-session 1)
The repo started with:
- `functiongemma:270m` base model (Gemma 3, gated)
- 1564 dispatch_pairs.jsonl (real failure cases + curated)
- 156-token tokenizer extension dataset

### What changed in session 1
- New `training/train_tokenizer.py` — phase 1-4 (add tokens, smart init,
  corpus warm-up, save) — separate phase before LoRA
- Heavy probe instrumentation: per-step loss, grad norms, cosine
  probes per epoch, embedding drift tracking
- Token count grown to 319 (added KDE/system/git/file_format/ML
  categories)

### Learnings
- The smart-init logic had a critical bug — see Session 3 (BUG-001)
- Run #1 produced 251 identical embedding clones. Diagnosed in
  Session 3.

---

## Pattern across sessions

Every session has been **diagnosis-driven**: a training run produces
output, we critique it, identify the bottleneck, ship a fix, repeat.

```
Session 1 (build infrastructure)
   ↓
Session 2 (make it transparent — observability + narration)
   ↓ Run #1 → BUG-001 (clones)
Session 3 (establish goals, document strategies, fix BUG-001)
   ↓ Run #2 → bottleneck = corpus quality (L14)
Session 4 (iter #3 — token pruning + curated corpus + analyze tool)
   ↓ Run #3 → cross-domain 0.62 (still high)
Session 5 (GPU enabled via bf16; Run #4 → REGRESSED to 0.70)
   ↓ root cause: mining flood, contrast drowned 94:1
Session 6 (planned: iter #4 — cut mining, expand contrast)
```

Each session moved one component from "broken" to "working" or "working
but flawed". The rate of finding bugs is decreasing — early sessions
caught structural bugs (BUG-001 clone), recent ones catch quality
issues (corpus mix).

---

## How to use this file

When starting a new session:
1. Read the most recent entry to know where we left off
2. Check the "Next session" line — that's the planned starting point
3. Read the corresponding `learnings.md` L-entry for context
4. Read `goals.md` to remember what we're optimising for
5. Check `bug-fixes.md` for the latest bugs and their lessons

When ending a session:
1. Add a new entry at the TOP (newest first)
2. Cross-reference learnings.md L-entry and CHANGELOG.md commits
3. State the "Next session" intent so it's not lost
