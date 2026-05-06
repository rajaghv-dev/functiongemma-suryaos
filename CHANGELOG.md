# Changelog

All notable changes to the functiongemma-suryaos training repo.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## Iteration #4 — function-calling dataset rebuild from real sources (2026-05-06)

### Pivot
Iter #4 was scoped as a tokenizer-corpus rebalance (cut mining 3000→500,
expand contrast 32→300 per L16). While sourcing the new contrast lines
the user surfaced a deeper bug: `dataset/dispatch_pairs.jsonl` had
**1564/1564 pairs with empty or wrong `target.arguments`** ("dim the
screen" labeled `direction:"up"`) and 1466/1564 (93.7%) routed to
`kde_krunner_launch` because `dataset/apps/launch_pairs.jsonl` was
concatenated in. Iter #4 pivoted to function-calling dataset
reconstruction from real machine sources, minimize synthetic, multi-agent.

### Added — 8 new scripts
- `training/populate_arguments.py` — per-tool argument extractors
  (UP_CUES/DOWN_CUES regex, KNOWN_SERVICES dict, app/title extraction,
  dialog prompt synthesis). Resolved 1538/1564 of the existing pairs.
- `training/build_real_dataset.py` — orchestrator. Loads
  `dispatch_pairs.jsonl` + every `dataset/real_sources/*.jsonl`,
  applies `populate_arguments(force=True)`, drops incomplete-arg rows,
  dedupes by (tool, lower(query)), per-tool caps, writes
  `dataset/dispatch_pairs_v4.jsonl`.
- `training/mine_kde_machine.py` — 9 mines from the live KDE 6.6.4 box
  (.desktop files, systemctl, qdbus6, /proc/meminfo,
  /sys/class/power_supply, /sys/class/backlight, nmcli, pactl,
  journalctl). 517 raw pairs. Honest [SKIP] for window-list (Wayland
  blocks `loadDeclarativeScript`) and history (no `~/.bash_history`).
- `training/mine_kf5book.py` — KF5 docs miner. **Honest 0 pairs** —
  the kf5book repo covers KArchive/KAuth/KConfig/KI18n/KIdleTime/
  KItemModels/Sonnet/ThreadWeaver, none mapping to our KRunner/KWin/
  KNotifications/KMessageBox tools. Script ready for follow-up with
  the right repos (plasma-workspace, frameworks/knotifications, etc.).
- `training/mine_man_pages.py` — 35 real pairs from man pages of
  free, df, top, vmstat, sar, upower, pactl, amixer, nmcli, ip,
  systemctl, journalctl, notify-send. SEE ALSO blocks → real
  sibling-negative pairs.
- `training/mine_krunner_kde_config.py` — 102 real pairs from
  kglobalshortcutsrc, kxmlgui RC files, recently-used.xbel, ksysguard
  XML, ps-derived running KDE apps.
- `training/mine_kde_help.py` — 174 real pairs from `--help` output of
  kdialog/krunner/kstart/dolphin/kate/konsole/notify-send +
  `/proc/mounts` + `/sys/class/power_supply/ADP1` +
  `/etc/systemd/system/<target>.target.wants/` (117 real service names).

### Added — 2 new tests
- `tests/test_corpus_balance.py` — validates
  `dataset/tokenizer/corpus.txt`. Reports per-tool count, position
  diversity, contrastive count, co-occurrence, token coverage. Exits
  non-zero on threshold failure.
- `tests/test_dispatch_pairs.py` — validates
  `dataset/dispatch_pairs*.jsonl`. Reports per-tool count,
  query-phrasing diversity (lead-words, length stdev, bigram diversity),
  arg-value coverage, source mix, schema consistency.

### Final dataset (`dataset/dispatch_pairs_v4.jsonl`)
- **776 pairs across 12 tools and 31 sources** (was 1564 pairs, 2 sources, 93.7% one tool)
- Distribution: 93.7% top tool → 26% top tool
- 4 tools meet floor of 80; **9 tools below floor** (linux_volume_set 31,
  linux_network_status 24, linux_battery_status 23,
  kde_notifications_send 22, linux_metrics_summary 20,
  linux_memory_usage 18, kde_dialog_confirm 13, linux_disk_usage 13,
  linux_brightness_set 12).

### Multi-agent involvement
6 sub-agents spawned in parallel. Agent 1 (results-analysis) read all
docs and dispatch_pairs.jsonl, identified the empty-args iceberg,
produced the build order. Agents 2-6 each wrote one miner. Agents 4
and 5 hit a sandbox `python3` restriction; their scripts were
syntactically valid and run by the orchestrator post-handoff (see L19).

### Documentation
- `docs/learnings.md` — L17 (empty-args iceberg), L18 (real-source
  ceiling), L19 (multi-agent + sandbox pattern)
- `goals.md` — iter #4 banner + new "argument extraction accuracy" KPI
- `SESSIONS.md` — Session 6 entry
- `README.md` — iter #4 pointer to multi-source miners

---

## [Unreleased] — Run #4 + bug fixes for GPU training

### Fixed — BUG-006 (fp16 NaN explosion on RTX 30xx/40xx)
- `_detect_hardware()` was selecting `torch.float16` for consumer Ampere GPUs
  based on a "datacenter only gets bf16" heuristic. Result: every batch
  produced `loss=nan` because Gemma 3's attention softmax overflows fp16's
  5-bit exponent (max ~65504).
- Fix: use `torch.bfloat16` for ANY Ampere+ GPU (compute ≥ 8). bf16 has
  the same 8-bit exponent as fp32, so gradients never overflow.
- Added: early-abort if 10 consecutive NaN batches detected — bails with
  fix command instead of running through the entire corpus.

### Fixed — BUG-007 (deprecation + log flood)
- `torch_dtype=` deprecated in transformers 4.46+ → renamed to `dtype=`
  in 4 call sites (train_tokenizer.py, finetune.py × 2, analyze_embeddings.py)
- Grad-norm warnings flooded the terminal during cold start (200+ identical
  lines per epoch). Now tiered: always emit on grad > 5.0, every 5 steps
  for 1.5-5.0 in first 30 steps, silent otherwise.
- Narrator messages compressed to ≤ 80 columns for narrow terminals.

### Added — target-aware cosine probe table
- New `PROBE_TARGETS` dict at module level — single source of truth for
  goal ranges per probe (sourced from goals.md).
- `_print_cosine_table` now shows: current value, target band, status
  (✓ HIT / ⚠ HIGH/LOW / ✗ HIGH/LOW), Δ vs previous epoch, bar with
  goal band as ▓ and current as ●.
- `_interpret_cosine_table` now produces:
  - Per-probe insight with direction-of-travel arrows (↑ heading toward,
    ↓ moving away)
  - [SUMMARY] line: "X/9 probes IN BAND | Y too high | Z too low"
  - [ACTION] block listing 2 biggest gaps with strategy IDs

### Added — bootstrap.sh self-sustaining mode
- STEP 0 pre-flight checks: Python version (≥ 3.10), disk space (≥ 3 GB),
  network reachability, HF_TOKEN presence, Docker daemon. Each non-fatal
  warning carries the exact fix command.
- STEP 4 post-install verify: actually runs `torch.zeros(2).cuda() + 1`
  to confirm GPU operations work. If GPU detected but verification fails,
  does force-reinstall recovery.
- New flag: `--with-cpu-fallback` creates secondary `.fngemma-suryaos-cpu/`
  venv with pure CPU torch — both venvs coexist for fallback usage.
- Final summary table: PASS/WARN/FAIL per component (PyTorch / HF stack /
  Dashboard / HF token), copy-paste-ready next steps, troubleshooting
  reference.

### Run #4 result (commit 275bef2)
- GPU training works: 2m 29s on RTX 3080 Ti (was 41 min on CPU)
- Loss: 8.3 → 5.82 (better than Run #3's 6.55)
- Cosine probes:
  - 3/9 in band (same-tool, tool↔CLI, tool↔KDE, new-vs-base)
  - 6/9 too high (sibling, metrics-vs-memory, cross-domain ×2,
    kde sibling, cross-category)
  - **Cross-domain REGRESSED** 0.62 → 0.70 (worse than Run #3)
- Diagnosis: dispatch_pair mining (3000 sentences) put every tool name
  in identical grammatical slot → reinforced clustering, drowning the
  32 contrastive sentences 94:1.
- See learnings.md L16 for full postmortem.

### Iter #4 plan (deferred)
- Cut mining cap 3000 → 500
- Auto-generate ~300 cross-category contrastive sentences
- Add ~100 varied-position sentences (tool names as subjects/objects,
  not only as dispatch targets)
- Goal: cross-domain 0.70 → < 0.40 (better than Run #3 baseline)

---

## [Unreleased] — iteration #3 (dataset overhaul + analysis tooling)

### Changed — token list pruned 319 → 108 (66% reduction)
- `build_tokenizer_dataset.py` rewritten to filter via tokenizer fragmentation
- Dropped 8 candidates that already tokenize as 1 piece in base Gemma
  (HEAD, origin, upstream, transformers, trl, etc.) — re-adding them as
  new tokens would replace pre-trained embeddings with random init
- Dropped 70+ redundant file-format extensions (.png/.jpg/.gif/...) that
  never drove different routing — kept only ~10 high-value ones
- Dropped all 15 generic English `arg_value` tokens (active/inactive/up/
  down) that corrupt their pre-trained meaning when re-added
- Dropped all 16 v4_workflow tokens (already single-token in base)

### Changed — corpus rewritten (templates → curated content)
- REMOVED 7 rotated templates × 251 tokens = monotonous gradient signal
- ADDED 285 per-tool curated sentences (varied phrasings, descriptions,
  CLI co-occurrence, naming variants, contrastive)
- ADDED 32 cross-domain contrastive sentences ("X is RAM, Y is disk")
- ADDED 26 co-occurrence sentences ("tool wraps CLI")
- ADDED 236 auxiliary-token coverage sentences (KWin, Klipper, qdbus6,
  GGUF — each got zero corpus mentions before)
- ADDED 3000 mined dispatch_pairs lines (capped from 9360 to avoid
  drowning auxiliary tokens)
- Total: 3579 unique sentences (was 3849 templated). 15% multi-tool
  co-occurrence (was 0%).
- `CORE_TOOLS` dict introduced as single source of truth for all 12
  core tools with category, description, cli_equiv, user_synonyms,
  contrasts_with, key_concept, naming_variants

### Added — `training/analyze_embeddings.py`
Post-training analysis tool with six modules:
- **Nearest neighbours** — top-K base + new neighbours per token, auto-detects
  meaningful (substring match)
- **Category cluster quality** — intra vs inter cosine with PASS/FAIL on Goal 2
- **Embedding norm distribution** — outlier detection (norms > 2σ)
- **Drift from smart-init** — flags starved tokens (drift < 30% of mean)
- **Probe sentence completion** — Goal 4 generalization test (top-10 hit rate)
- **ASCII cluster map** — 2D PCA projection rendered as text

Each module ends with `[LEARN]` / `[INSIGHT]` commentary referencing goals.md.

### Fixed — BUG-005 (probe pairs measured frozen tokens)
- `PROBE_PAIRS` in `train_tokenizer.py` updated. Three frozen-token pairs
  replaced:
  - `(torch, transformers)` → `(linux_memory_usage, memory)`
  - `(merge, commit)` → `(kde_window_focus, kde_krunner_launch)`
  - `(GGUF, ollama)` → `(kde_dialog_confirm, linux_battery_status)`
- All probes now have ≥ 1 trainable token, so cosine moves during training.

### Added — `RUN.md`
Top-level minimal-step run guide. Covers: bootstrap, auth, training,
analysis, watching live in Grafana, troubleshooting, and "what success
looks like" reference.

### Documentation refresh
- `goals.md` cross-referenced from README quickstart and Documentation index
- `docs/bug-fixes.md` BUG-005 marked FIXED with resolution note
- `docs/learnings.md` L15 added — iteration #3 dataset overhaul postmortem
- `docs/dataset-strategies.md` strategies marked SHIPPED (A4, A3, A2, C1, E2)
- README.md repo layout, dataset table, quickstart, and documentation
  index all updated to reflect iteration #3 file structure

---

## [Unreleased] — first training iteration + observability + bug fixes

### Fixed
- **BUG-001 — Smart-init produces 251 identical embeddings** (`train_tokenizer.py`).
  `tokenizer.encode(token_str)` was being called on the already-extended tokenizer,
  so the added-tokens trie short-circuited subword decomposition. Every new token
  fell through to the global-mean fallback, producing 251 identical clones with
  `std=0.0000`. Fixed by loading a separate clean base tokenizer for subword
  lookup. See [docs/bug-fixes.md](docs/bug-fixes.md) BUG-001 for full diagnosis.
- **BUG-002 — Gated Gemma 3 fails to download even with paid HF account.**
  Added `_get_hf_token()` helper that resolves token from `HF_TOKEN`,
  `HUGGINGFACE_HUB_TOKEN`, or `~/.cache/huggingface/token`, then passes it
  explicitly to every `from_pretrained()` call. On 401/403, prints exact
  2-step recovery (license URL + token export).
- **BUG-003 — CPU torch silently installed on GPU machines.** Removed torch
  from `requirements.txt`; `bootstrap.sh` now installs the right wheel based
  on detected CUDA version. Uninstalls existing CPU torch before installing
  GPU variant.
- **BUG-004 — bootstrap.sh slow on re-runs.** Added `[SKIP]` early-exits for
  each install step when packages are already importable. New flags:
  `--no-dashboard`, `--dashboard-only`, `--reinstall`.

### Changed
- `train_tokenizer.py` default epochs: 2 → 5 (loss was still trending down
  at epoch 2 in run #1; see learnings.md L13).
- `bootstrap.sh` now also brings up the Grafana/Loki/Prometheus stack as
  the final step (idempotent — skipped if all 5 containers are running).
- `requirements.txt` removed torch entirely; bootstrap installs it with
  the correct CUDA wheel based on detection.

### Added — observability stack
- **`training/observability/`** — full local Grafana / Loki / Prometheus stack
  via `docker compose`. Pre-provisioned dashboard with 18 panels covering
  loss curve, lr schedule, gradient norm, memory, per-tool loss heatmap,
  cosine probe evolution, embedding norm ratio, drift, live event log tail,
  and per-epoch probe results. Multi-run comparison via `run_id` selector.
- **`training/metrics.py`** — `MetricsPusher` class that pushes time-series
  metrics to Prometheus Pushgateway from training scripts. Gracefully
  no-ops if `prometheus_client` is missing or Pushgateway is unreachable.
- **Promtail config** — auto-tails `training/*/train_log.jsonl` files
  and ships them to Loki with phase/event/epoch labels.

### Added — live training narration
- `train_tokenizer.py` prints intuitive interpretation during corpus
  training: `[LEARN]` for milestones, `[PROGRESS]` for steady convergence,
  `[PLATEAU]` when loss stabilises, `[WARN]` for gradient anomalies.
  Concrete before/after fragmentation demos.
- `finetune.py` `DispatchCallback` writes structured JSONL telemetry plus
  per-step terminal narration. Per-epoch probe table now annotates each
  tool with `(mastered)` / `(learning)` / `(struggling)` status tags.
- `[DELTA]` block after each epoch shows the 5 biggest tool-loss changes
  vs the previous epoch.

### Added — GPU training support
- `_detect_hardware()` auto-selects bf16 on A100/H100, fp16 on RTX
  30xx/40xx/T4, float32 on CPU. Auto batch-size scaling by VRAM:
  16GB → batch=8 (RTX 3080 Ti), 10GB → batch=4, 40GB+ → batch=16.
- `bootstrap.sh` detects CUDA via nvcc + nvidia-smi, picks `cu121` for
  CUDA 12.x, `cu118` for CUDA 11.x, `cpu` otherwise.
- `bitsandbytes` installed automatically on GPU systems (for QLoRA).

### Added — tokenizer warm-up phase
- **`training/train_tokenizer.py`** — new standalone script that runs
  BEFORE the LoRA fine-tune. Adds 319 domain tokens, smart-initialises
  new embeddings as the average of their subword pieces (after BUG-001
  fix), and trains only the new embedding rows on `corpus.txt` while
  keeping the base 262K vocab embeddings frozen via gradient hook.
- Output: `training/tokenizer_extended/` with extended tokenizer +
  `embed_init.pt` containing the warmed-up embeddings.
- `finetune.py` auto-detects this directory and loads the warmed-up
  embeddings into the model before starting LoRA training.

### Added — comprehensive documentation
- **`docs/bug-fixes.md`** — running log of every bug caught during
  real training runs. Each entry has Symptom → Root cause → Mental
  model → Fix → Validation → Lesson. Currently logs BUG-001 through
  BUG-005 plus 3 KNOWN issues.
- **`docs/tokenizer-improvements.md`** — strategy register identifying
  5 tiers of improvements (bugs, corpus, token list, training procedure,
  validation) with prioritised recommendations.
- **`docs/dataset-strategies.md`** — 20+ detailed dataset improvement
  strategies in 5 categories (corpus content, diversity, token list,
  generation pipeline, validation). Each strategy has Lever, Why it
  works, Concrete example, Expected impact, Implementation cost. Plus
  a priority matrix for the next iteration.
- **`docs/learnings.md` L13/L14** — postmortems of training runs #1
  and #2. L13 covers the smart-init clone bug; L14 covers the corpus-
  quality bottleneck that became visible after fixing L13.
- Inline comments throughout `train_tokenizer.py`, `finetune.py`, and
  `bootstrap.sh` explaining the WHY at every decision point.

### Identified — BUG-005 in `train_tokenizer.py` PROBE_PAIRS
- 3 of 9 cosine-probe pairs (`torch`/`transformers`, `merge`/`commit`,
  `GGUF`/`ollama`) measure tokens that are in the base Gemma vocabulary.
  These are frozen by our gradient hook and cannot move during training.
- Run #2 confirmed: those probes returned identical values to 4 decimal
  places across all 5 epochs. They are dead instrumentation.
- Fix scheduled for iteration #3: replace with pairs that include at
  least one new token (full diagnosis in bug-fixes.md BUG-005).

---

## [Unreleased] — pre-training baseline

### Added
- **Apps catalog** (`dataset/apps/`): 110 applications in 7 categories
  - 18 open-source browsers (Firefox, Brave, Chromium, LibreWolf, Tor,
    Vivaldi, Falkon, Qutebrowser, Pale Moon, Waterfox, Floorp, IceCat,
    Otter, Midori, Ungoogled-Chromium, Epiphany, Min, Nyxt)
  - 22 KDE core apps (Dolphin, Kate, Konsole, Spectacle, Gwenview, etc.)
  - 18 KDE utilities (Krita, Kdenlive, KDevelop, Yakuake, KStars, etc.)
  - 14 development tools (VS Code, VSCodium, Qt Creator, Postman, DBeaver)
  - 12 office apps (LibreOffice, Thunderbird, Joplin, Logseq, Obsidian)
  - 16 media apps (VLC, GIMP, Inkscape, Blender, OBS, Audacity)
  - 10 communication apps (Signal, Element, Telegram, Bitwarden, KeePassXC)
- **Tokenizer dataset** (`dataset/tokenizer/`): 156 atomic tokens + 1605-sentence corpus
  - 68 tool name forms (dot/underscore/short × 12 tools)
  - 24 KDE concepts (Plasma, KWin, Akonadi, qdbus6, kstart5)
  - 33 Linux/system terms (systemd, pipewire, NetworkManager, BAT0, wlo1)
  - 16 v4 workflow tokens (compile, commit, push, pytest)
  - 15 enum value tokens (up, down, active, connected)
- **Test harness** (in companion `oc` repo): 205 use cases across 13 categories
  - basic / multi_arg / service_alias / app_alias / casing / typo / negation
  - plain_chat / ambiguous / compound / user_green / admin_yellow / admin_red
  - browser / dev_app / kde_core / media / office / comm / v4_chain
- **Auto-fix loop**: extends tool YAMLs with retrieval-miss queries automatically
- **Documentation**:
  - `docs/training-guide.md` — step-by-step GPU/CPU training
  - `docs/scenarios.md` — 205 test cases catalog
  - `docs/policy-tiers.md` — green/yellow/red enforcement
  - `docs/integration.md` — deployment back to `~/raja/oc`
  - `docs/test-results.md` — current baseline
  - `docs/architecture.md` — system design (context builder, fine-tune)
  - `docs/v4-roadmap.md` — chain-of-task plans
- **Build scripts**:
  - `training/build_apps_catalog.py` — reproducible app catalog
  - `training/build_tokenizer_dataset.py` — reproducible tokenizer dataset
  - `training/finetune.py` — convert / train / export pipeline
  - `training/generate.py` — augment dispatch pairs

### Dataset growth log

| Date | Dispatch pairs | Tokenizer tokens | Apps | Trigger |
|---|---|---|---|---|
| 2026-05-01 init | 48 | — | — | yaml examples only |
| 2026-05-01 (+failures) | 77 | — | — | first user test session |
| 2026-05-01 (+augment) | 461 | — | — | qwen3:0.6b paraphraser |
| 2026-05-01 (+tokenizer) | 461 | 156 | — | tokenizer corpus generated |
| 2026-05-01 (+apps) | **1564** | 156 | **110** | apps catalog merged |

### Notes

- Base model (`functiongemma:270m`) has hardcoded refusals for ram/bluetooth
  queries that prompt engineering cannot remove. Fine-tuning is the documented
  path forward.
- L1 retrieval test results: 84% pass (rest are v2/v3 tools or correctly
  denied destructive queries).
- Two related repos:
  - [`rajaghv-dev/suryaos-opencode`](https://github.com/rajaghv-dev/suryaos-opencode) — full agent stack
  - [`rajaghv-dev/kde-oc`](https://github.com/rajaghv-dev/kde-oc) — KDE actions catalog (reference)

---

## [v0.1.0-init] — 2026-05-01

### Added
- Initial repo structure
- Base dataset extracted from SuryaOS tool YAMLs (48 dispatch pairs)
- Stub training script (`training/finetune.py`)
- README + dataset/README + docs/architecture + docs/v4-roadmap

### Removed
- Zoho-related references (per user direction — scope is KDE desktop only)

---

## Versioning policy

- **vX.Y.Z** — major.minor.patch (semver)
- **major** bumps when fine-tune changes input format or token vocabulary
  (existing inference code needs updates)
- **minor** bumps when dataset grows by ≥500 pairs or new categories added
- **patch** bumps for bug fixes, documentation updates, dataset cleanup

The first release tag (`v1.0.0`) will be cut after a successful fine-tune
run on the RTX 3080 box, verified via L3 test pass rate ≥80%.
