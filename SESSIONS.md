# Sessions log — what we did each working session

Chronological record of significant working sessions. Each entry captures
goals, what shipped, what we learned, and what's next. Detailed
postmortems live in `docs/learnings.md`; bug-by-bug analysis lives in
`docs/bug-fixes.md`. This file is the **timeline** that ties them together.

---

## Session 6 — Iter #4 function-calling dataset rebuild (2026-05-06)

### Goal
Pivot from tokenizer-corpus rebalancing to function-calling dataset
reconstruction. Diagnose why Run #4 regressed; fix the deeper
supervision failure; mine real signal from the live machine instead of
synthesizing more text.

### What shipped

| File | Purpose |
|---|---|
| `tests/test_corpus_balance.py` | Validates `dataset/tokenizer/corpus.txt` — per-tool count, position diversity, contrastive count, co-occurrence, token coverage. Exits non-zero on threshold failure. |
| `tests/test_dispatch_pairs.py` | Validates `dataset/dispatch_pairs*.jsonl` — per-tool count, query-phrasing diversity, arg-value coverage, source mix, schema consistency. |
| `training/populate_arguments.py` | Per-tool extractors that fix wrong/empty `target.arguments` from the user query. Direction cues (UP_CUES/DOWN_CUES regex), KNOWN_SERVICES dict, app/title extraction, dialog prompt synthesis. 1538/1564 existing pairs resolved. |
| `training/build_real_dataset.py` | Orchestrator. Loads `dispatch_pairs.jsonl` + `dataset/real_sources/*.jsonl`, applies `populate_arguments(force=True)`, drops incomplete-arg rows, dedupes by (tool, lower(query)), per-tool caps, writes `dataset/dispatch_pairs_v4.jsonl`. |
| `training/mine_kde_machine.py` (Agent 3) | 9 mines from this live KDE 6.6.4 box → 517 raw pairs (.desktop, systemctl, qdbus6, /proc, /sys, nmcli, pactl, journalctl). Honest [SKIP] for Wayland-blocked window-list and missing bash_history. |
| `training/mine_kf5book.py` (Agent 2) | KF5 docs miner. Honest 0 pairs — kf5book covers KArchive/KAuth/etc., none mapping to our KRunner/KWin/KNotifications/KMessageBox tools. Script ready for follow-up with the right repos. |
| `training/mine_man_pages.py` (Agent 4) | 35 pairs from man pages of free, df, top, vmstat, sar, upower, pactl, amixer, nmcli, ip, systemctl, journalctl, notify-send. SEE ALSO blocks → real sibling-negative pairs (the G4 fix the corpus never had). |
| `training/mine_krunner_kde_config.py` (Agent 5) | 102 pairs from kglobalshortcutsrc, kxmlgui RC, recently-used.xbel, ksysguard XML, ps-derived running KDE apps. KRunner state file empty on this box (logged honestly). |
| `training/mine_kde_help.py` (Agent 6) | 174 pairs from `--help` output of kdialog/krunner/kstart/dolphin/kate/konsole/notify-send + `/proc/mounts` + `/sys/class/power_supply/ADP1` + `/etc/systemd/system/<target>.target.wants/` (117 real service names). |

### Concrete changes

**The empty-arguments iceberg.** Investigation revealed
`dataset/dispatch_pairs.jsonl` had 1564/1564 pairs with empty or wrong
`target.arguments`. "Dim the screen" was labeled `direction:"up"`.
1466/1564 (93.7%) routed to `kde_krunner_launch` because
`dataset/apps/launch_pairs.jsonl` was concatenated in. The tokenizer
corpus imbalance from L16 was a symptom; the supervision failure was
the disease.

**Multi-agent extraction.** 6 sub-agents in parallel: Agent 1 ranked
per-tool variety and recommended a build order; Agents 2-6 each wrote
one miner script. Agents 4 and 5 hit a sandbox `python3` restriction —
their scripts were syntactically valid and run by the orchestrator
post-handoff.

### Final dataset stats (`dataset/dispatch_pairs_v4.jsonl`)

- 776 pairs across 12 tools and 31 sources (was 1564 pairs, 2 sources, 93.7% one tool)
- 4 tools meet floor of 80; 9 tools below floor (worst: linux_brightness_set 12, linux_disk_usage 13, kde_dialog_confirm 13)
- Distribution went from 93.7% top tool → 26% top tool

### Documentation
- `docs/learnings.md` L17, L18, L19 added (empty-args iceberg / real-source ceiling / multi-agent robustness)
- `CHANGELOG.md` Iteration #4 entry
- `goals.md` iter #4 banner + new "argument extraction accuracy" KPI
- `README.md` pointer to multi-source miners

### Learnings
- L17 — the empty-arguments iceberg: even perfect tokenizer + perfect
  routing cannot produce `linux_volume_set({direction:"down", step:20})`
  if every example said `arguments={}`. Arg-extraction is independently
  more important than the cosine geometry the project was tracking.
- L18 — real-source mining has a hard ceiling on a single machine: 1
  backlight, 2 mountpoints, 4 power supplies, no `~/.bash_history`.
  Reaching the floor of 80 per tool from real signal requires another
  data source class or controlled paraphrasing of seed sentences.
- L19 — multi-agent extraction is robust to partial sandbox failures:
  agents WRITE scripts, host EXECUTES. Validate-only when execution is
  blocked; don't retry into a wall.

### Next session
Iter #5: lift the 9 starved tools to floor=80 via (a) production audit
trail mining, (b) GitHub issues for KDE projects, or (c) controlled
seed-sentence paraphrasing. Build a held-out arg-test split to measure
the new "argument extraction accuracy" KPI.

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
