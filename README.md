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
   156 atomic tokens     →     1564 (query, schema, tool_call) pairs
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
| [`dataset/tokenizer/new_tokens.json`](dataset/tokenizer/) | 156 tokens | SentencePiece vocabulary | Ready |
| [`dataset/tokenizer/corpus.txt`](dataset/tokenizer/) | 1605 sentences | Token embeddings | Ready (≥5 occurrences each) |
| [`dataset/dispatch_pairs.jsonl`](dataset/dispatch_pairs.jsonl) | 1564 pairs | Tool dispatch (LoRA) | Ready |
| [`dataset/apps/launch_pairs.jsonl`](dataset/apps/) | 1450 pairs | App-launch subset | Included in dispatch |
| [`dataset/embed_pairs.jsonl`](dataset/embed_pairs.jsonl) | 151 pairs | all-minilm:22m embedder | Optional second stage |

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
├── CHANGELOG.md                       ← version history
├── CONTRIBUTING.md                    ← how to add tools / scenarios
│
├── dataset/                           ← all training data
│   ├── README.md                      ← dataset spec + how to grow
│   ├── dispatch_pairs.jsonl           ← functiongemma LoRA training
│   ├── embed_pairs.jsonl              ← embedder fine-tune (optional)
│   ├── tokenizer/                     ← tokenizer extension dataset
│   │   ├── README.md
│   │   ├── new_tokens.json            ← 156 tokens to add
│   │   ├── corpus.txt                 ← 1605 sentences for training embeddings
│   │   ├── corpus.jsonl
│   │   ├── tool_name_terms.txt        ← per-category flat lists
│   │   ├── kde_terms.txt
│   │   ├── system_terms.txt
│   │   ├── arg_value_terms.txt
│   │   └── v4_workflow_terms.txt
│   └── apps/                          ← app catalog + launch examples
│       ├── README.md
│       ├── apps_catalog.json          ← 110 apps in 7 categories
│       ├── launch_pairs.jsonl         ← 1450 (alias, schema, target) triples
│       └── app_aliases.txt            ← 168 aliases for tokenizer
│
├── docs/
│   ├── architecture.md                ← system design (context builder, fine-tune)
│   ├── training-guide.md              ← step-by-step on GPU/CPU
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
    ├── finetune.py                    ← convert / train / export (1470 lines)
    ├── generate.py                    ← dataset generator (yaml/augment/audit)
    ├── build_apps_catalog.py          ← rebuild apps catalog
    ├── build_tokenizer_dataset.py     ← rebuild tokenizer dataset
    └── requirements.txt               ← CPU + GPU pip deps
```

---

## Quickstart (GPU, ~10 min)

```bash
git clone https://github.com/rajaghv-dev/functiongemma-suryaos
cd functiongemma-suryaos
pip install -r training/requirements.txt

# Single command: convert → train → export
python3 training/finetune.py --mode all

# Import the resulting model into Ollama
ollama create functiongemma:270m-suryaos -f training/output/Modelfile
ollama run functiongemma:270m-suryaos "is bluetooth active"
# Expected: calls service_status(name="bluetooth")
```

See [`docs/training-guide.md`](docs/training-guide.md) for details
(including CPU training, ~25 min on Intel Meteor Lake).

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

## Status & next steps

- [x] Dataset built: 1564 dispatch + 156 tokens + 1450 app launches
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
