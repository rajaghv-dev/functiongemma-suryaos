# Changelog

All notable changes to the functiongemma-suryaos training repo.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
