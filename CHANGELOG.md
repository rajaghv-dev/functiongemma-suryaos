# Changelog

All notable changes to the functiongemma-suryaos training repo.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
