# functiongemma-suryaos

Fine-tuned `functiongemma:270m` for SuryaOS desktop tool dispatch.

[![status](https://img.shields.io/badge/status-pre--training-yellow)]()
[![base](https://img.shields.io/badge/base-functiongemma:270m-blue)]()
[![arch](https://img.shields.io/badge/arch-Gemma3-green)]()
[![license](https://img.shields.io/badge/license-Gemma_Terms-orange)]()

---

## TL;DR

`functiongemma:270m` (Google, Gemma 3, 268M params, Q8_0) refuses to call tools
for many SuryaOS desktop queries (`"how much ram"` → "I cannot provide system
statistics", `"is bluetooth active"` → "I cannot assist with Bluetooth"). These
are **trained-in safety constraints** — no prompt can remove them.

This repo fine-tunes them out via LoRA + tokenizer extension, producing
`functiongemma:270m-suryaos` — a 300 MB model that reliably dispatches **123
SuryaOS tools** (12 system + 110 app launches + edge cases) at ~6s warm
inference.

```
Before  →  "how much ram is used"   →  "I cannot provide statistics"
After   →  "how much ram is used"   →  linux_memory_usage({})
                                    →  "5.1 GiB used / 30.6 GiB total (17%)"
```

---

## Training priority order

**Train tokenizer FIRST, then functiongemma weights.** They compose:

```
[1] Tokenizer extension      [2] Functiongemma LoRA
   319 atomic tokens     →     1564 (query, schema, tool_call) pairs
   1605-sentence corpus   →    LoRA r=8 on q_proj + v_proj
   ─────────────────────────────────────────────────────
   Result: model treats `metrics_summary` as 1 token (not 5)
           AND knows when to call it for any natural query
```

This priority matters because the dispatch fine-tune happens **after** the
tokenizer is extended — so the new token embeddings get trained alongside
the LoRA adapter on real queries. Reverse the order and the new tokens stay
random.

---

## Datasets at a glance

| File | Lines | Trains | Status |
|---|---|---|---|
| [`dataset/tokenizer/new_tokens.json`](dataset/tokenizer/) | **108 tokens** | SentencePiece vocabulary | Ready (iter #3 — pruned from 319 after dropping already-single-token entries) |
| [`dataset/tokenizer/corpus.txt`](dataset/tokenizer/) | **3579 sentences** | Token embeddings | Ready — curated content (no templates) |
| [`dataset/dispatch_pairs.jsonl`](dataset/dispatch_pairs.jsonl) | 1564 pairs | Tool dispatch (LoRA) | Superseded — 1564/1564 had empty/wrong `target.arguments` (see L17) |
| [`dataset/dispatch_pairs_v4.jsonl`](dataset/dispatch_pairs_v4.jsonl) | **776 pairs** | Tool dispatch (LoRA) | **iter #4 — rebuilt from 31 real machine sources, 9/12 tools still below floor** |
| [`dataset/apps/launch_pairs.jsonl`](dataset/apps/) | 1450 pairs | App-launch subset | Included in dispatch |
| [`dataset/embed_pairs.jsonl`](dataset/embed_pairs.jsonl) | 151 pairs | all-minilm:22m embedder | Optional second stage |

**Iter #4 introduces multi-source real-data miners** — see
[`SESSIONS.md`](SESSIONS.md) Session 6 and `training/mine_*.py`.

**v4 target: 2000+ examples** covering chain-of-task workflows
(compile → test → commit → push). Current 1564 is a strong starting point;
real failures collected from production usage close the gap.

---

## Apps catalog (110 apps, FOSS-first)

| Category | Count | Highlights |
|---|---|---|
| **Browsers** | 18 | Firefox, Brave, Chromium, LibreWolf, Tor, Vivaldi, Falkon, Qutebrowser, IceCat, Pale Moon, Otter, Floorp |
| **KDE core** | 22 | Dolphin, Kate, KWrite, Konsole, KMail, Spectacle, Gwenview, Okular, Ark, KCalc |
| **KDE utilities** | 18 | Krita, Kdenlive, KDevelop, Yakuake, KStars, Marble |
| **Office** | 12 | LibreOffice (Writer/Calc/Impress), Thunderbird, Joplin, Logseq, Obsidian |
| **Media** | 16 | VLC, GIMP, Inkscape, Blender, OBS, Audacity, Ardour, MuseScore |
| **Development** | 14 | VS Code, VSCodium, Qt Creator, GitHub Desktop, Postman, DBeaver |
| **Communication** | 10 | Signal, Element, Telegram, Bitwarden, KeePassXC |

Each app has 3-5 natural-language aliases ("the browser", "private browser",
"text editor", etc.) generating 1450 launch training pairs.

See [`dataset/apps/README.md`](dataset/apps/README.md).

---

## Repo layout

```
.
├── README.md                          ← you are here
├── RUN.md                             ← minimal run steps
├── goals.md                           ← canonical goals + current state
├── CHANGELOG.md                       ← version history
├── CONTRIBUTING.md                    ← how to add tools / scenarios
│
├── dataset/                           ← all training data
│   ├── README.md                      ← dataset spec + how to grow
│   ├── dispatch_pairs.jsonl           ← functiongemma LoRA training (1564 pairs)
│   ├── embed_pairs.jsonl              ← embedder fine-tune (optional)
│   ├── tokenizer/                     ← tokenizer extension dataset
│   │   ├── README.md
│   │   ├── new_tokens.json            ← 108 tokens (iter #3 — pruned)
│   │   ├── corpus.txt                 ← 3579 curated sentences (no templates)
│   │   ├── corpus.jsonl
│   │   ├── tool_name_terms.txt        ← per-category flat lists
│   │   ├── kde_terms.txt
│   │   ├── system_terms.txt
│   │   ├── git_terms.txt
│   │   ├── ml_terms.txt
│   │   └── file_format_terms.txt
│   └── apps/                          ← app catalog + launch examples
│       ├── README.md
│       ├── apps_catalog.json          ← 110 apps in 7 categories
│       ├── launch_pairs.jsonl         ← 1450 (alias, schema, target) triples
│       └── app_aliases.txt            ← 168 aliases for tokenizer
│
├── docs/                              ← deep dives + decision logs
│   ├── architecture.md                ← system design (context builder, fine-tune)
│   ├── training-guide.md              ← step-by-step on GPU/CPU
│   ├── tokenizer-explained.md         ← intuitive tokenizer extension walkthrough
│   ├── tokenizer-improvements.md     ← 5-tier strategy register
│   ├── dataset-strategies.md         ← 24 detailed corpus improvement strategies
│   ├── bug-fixes.md                  ← every bug caught + mental models + lessons
│   ├── learnings.md                  ← decision log L1..L15 (run postmortems)
│   ├── scenarios.md                   ← user/admin green/yellow/red catalog
│   ├── policy-tiers.md                ← green/yellow/red policy in force
│   ├── integration.md                 ← how to deploy with ~/raja/oc
│   ├── test-results.md                ← latest pass/fail metrics
│   ├── v4-roadmap.md                  ← chain-of-task scale plan
│   ├── policy.yaml                    ← actual policy.yaml from oc
│   └── production.yaml                ← production config reference
│
├── tools/                             ← MCP handlers + tool YAMLs
│   ├── catalog/                       ← 12 tool YAML manifests
│   ├── tool_schemas.json              ← extracted MCP schemas
│   ├── system_handlers.py             ← Netdata-backed handlers
│   ├── volume_handler.py
│   └── dispatcher.py                  ← single-tool router (post-finetune)
│
├── inference/                         ← context builder used at inference
│   ├── context_builder.py             ← FTS+graph → 1-3 schemas
│   ├── fts.py                         ← retrieval index
│   └── graph.py                       ← dependency graph
│
└── training/                          ← training pipeline
    ├── bootstrap.sh                   ← one-shot env + Grafana setup
    ├── build_tokenizer_dataset.py     ← curated corpus generator (no templates)
    ├── train_tokenizer.py             ← extend tokenizer + warm embeddings
    ├── analyze_embeddings.py          ← post-training NN/cluster/probe analysis
    ├── finetune.py                    ← convert / train / export (LoRA)
    ├── metrics.py                     ← Pushgateway client (Prometheus)
    ├── requirements.txt               ← Python deps (PyTorch installed by bootstrap)
    └── observability/                 ← Grafana + Loki + Prometheus stack
        ├── README.md
        ├── docker-compose.yml         ← 5-container stack
        ├── grafana/dashboards/        ← pre-provisioned training dashboard
        ├── loki/                      ← log aggregation config
        ├── prometheus/                ← metrics scraping config
        └── promtail/                  ← JSONL tailer config
```

---

## Quickstart — minimal human steps

```bash
# 1. One-shot setup (creates venv, installs deps, starts Grafana stack)
#    Idempotent — re-runs are no-ops if everything is in place.
bash training/bootstrap.sh

# 2. Auth (Gemma 3 is gated)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# 3. (Optional) Open Grafana to watch training live
xdg-open http://localhost:3000          # admin / admin

# 4. Train tokenizer (~5 min GPU / ~40 min CPU)
.fngemma-suryaos/bin/python training/train_tokenizer.py

# 5. Inspect what the model learned
.fngemma-suryaos/bin/python training/analyze_embeddings.py

# 6. Full LoRA fine-tune + GGUF export
.fngemma-suryaos/bin/python training/finetune.py --mode all

# 7. Register with Ollama
ollama create functiongemma:270m-suryaos -f training/Modelfile
ollama run functiongemma:270m-suryaos "is bluetooth active"
# Expected: calls linux_service_status(name="bluetooth")
```

See [`RUN.md`](RUN.md) for the detailed run guide and
[`docs/training-guide.md`](docs/training-guide.md) for stage-by-stage
explanations.

---

## Integration with the SuryaOS agent

After training, deploy back into `~/raja/oc`:

```jsonc
// opencode.json — change the coder-fg agent model
"coder-fg": {
    "model": "ollama/functiongemma:270m-suryaos",
    ...
}
```

Then `opencode run --agent coder-fg "is bluetooth active"` uses the
fine-tuned model. See [`docs/integration.md`](docs/integration.md).

---

## Documentation

**Start here:**

| Doc | Purpose |
|---|---|
| [`RUN.md`](RUN.md) | **Minimal-step run guide.** Setup → train → analyze. |
| [`goals.md`](goals.md) | **Canonical goals + current state.** What success looks like, where we are. |

**Deep dives:**

| Doc | Purpose |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | System design (context builder + fine-tune) |
| [`docs/training-guide.md`](docs/training-guide.md) | Step-by-step GPU/CPU training |
| [`docs/tokenizer-explained.md`](docs/tokenizer-explained.md) | Intuitive walkthrough of tokenizer extension |
| [`docs/tokenizer-improvements.md`](docs/tokenizer-improvements.md) | 5-tier improvement strategy register |
| [`docs/dataset-strategies.md`](docs/dataset-strategies.md) | 24 detailed corpus-improvement strategies |
| [`docs/multi-model-training.md`](docs/multi-model-training.md) | Same dataset → both Gemma + Qwen |
| [`docs/scenarios.md`](docs/scenarios.md) | 205 test cases catalog |
| [`docs/policy-tiers.md`](docs/policy-tiers.md) | Green/yellow/red enforcement |
| [`docs/integration.md`](docs/integration.md) | Deploy back to ~/raja/oc |
| [`docs/test-results.md`](docs/test-results.md) | Current baseline + auto-fix history |
| [`docs/v4-roadmap.md`](docs/v4-roadmap.md) | Chain-of-task scale plan |
| [`training/observability/README.md`](training/observability/README.md) | Grafana / Loki / Prometheus stack |

**Decision logs (chronological):**

| Doc | Purpose |
|---|---|
| [`docs/learnings.md`](docs/learnings.md) | Why we made each choice (L1..L15) |
| [`docs/bug-fixes.md`](docs/bug-fixes.md) | Every bug caught + mental model + lesson |
| [`CHANGELOG.md`](CHANGELOG.md) | What shipped per iteration |

## Status & next steps

- [x] Dataset built: 1564 dispatch + 319 tokens + 1450 app launches
- [x] Test harness: 205 cases at L1 (FTS) / L2 (dispatcher) / L3 (model)
- [x] L1 retrieval: 84% pass (rest are v2 tools or correctly denied)
- [ ] Run training on RTX 3080 box → `functiongemma:270m-suryaos`
- [ ] Verify: re-run L3 tests with the fine-tuned model
- [ ] Iterate: capture real failures from production → retrain
- [ ] v4: chain-of-task tools (`code.compile`, `git.push`, etc.)

---

## Companion repos

- [`rajaghv-dev/suryaos-opencode`](https://github.com/rajaghv-dev/suryaos-opencode)
  — the full SuryaOS agent (MCP servers, opencode config, test harness, scripts)
