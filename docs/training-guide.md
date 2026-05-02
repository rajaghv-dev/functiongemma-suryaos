# Training guide

How to train `functiongemma:270m-suryaos` from a clean clone.

For the **minimal** version (4 commands, no explanation), see [`../RUN.md`](../RUN.md).

This guide explains *why* each step exists and how to debug when something
goes wrong.

---

## Pipeline overview

Three Python scripts, run in order:

```
                                    ┌──────────────────────┐
[1] build_tokenizer_dataset.py  →   │ dataset/tokenizer/   │
    (generate 108 tokens +          │   new_tokens.json    │
     3579-line curated corpus)      │   corpus.txt         │
                                    └──────────┬───────────┘
                                               ▼
                                    ┌──────────────────────┐
[2] train_tokenizer.py          →   │ training/            │
    (extend Gemma's tokenizer +     │   tokenizer_extended/│
     warm up new embeddings)        │   embed_init.pt      │
                                    └──────────┬───────────┘
                                               ▼
                                    ┌──────────────────────┐
[3] finetune.py --mode all      →   │ training/            │
    (LoRA fine-tune dispatch +      │   model_lora/        │
     export to GGUF)                │   functiongemma-...  │
                                    │   .gguf              │
                                    └──────────────────────┘

Plus:
    analyze_embeddings.py           (post-training analysis of [2])
```

---

## Stage 0 — Setup (one-time)

```bash
bash training/bootstrap.sh
```

What it does (idempotent — re-runs are no-ops if everything is in place):
1. Detects GPU/CUDA, picks the right PyTorch wheel (cu121 / cu118 / cpu)
2. Creates `.fngemma-suryaos/` venv if missing
3. Installs torch with the right wheel (uninstalls wrong-variant first)
4. Installs requirements.txt (transformers, peft, trl, datasets, etc.)
5. Installs bitsandbytes for QLoRA on GPU systems
6. Brings up Grafana / Loki / Prometheus stack via docker compose
7. Verifies every critical import and prints the dashboard URLs

Alternatives:
- `bash training/bootstrap.sh --no-dashboard` — skip the docker stack
- `bash training/bootstrap.sh --reinstall` — force reinstall

After bootstrap, set the HuggingFace token (Gemma 3 is gated):
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```
Get yours at https://huggingface.co/settings/tokens.

---

## Stage 1 — Build the tokenizer dataset

```bash
.fngemma-suryaos/bin/python training/build_tokenizer_dataset.py
```

What it produces:
- `dataset/tokenizer/new_tokens.json` — 108 domain tokens grouped by category
- `dataset/tokenizer/corpus.txt` — 3579 curated sentences (no templates)
- `dataset/tokenizer/corpus.jsonl` — same content as JSONL with metadata
- `dataset/tokenizer/<category>_terms.txt` — flat per-category lists

Sources of corpus content:

| Source | ~Lines | Purpose |
|---|---:|---|
| Per-tool curated | 285 | Varied phrasings, descriptions, CLI co-occurrence |
| Cross-domain contrast | 32 | Direct "X (memory) and Y (brightness) are unrelated" |
| Co-occurrence | 26 | "tool wraps CLI" patterns |
| Auxiliary-token coverage | 236 | KWin, Klipper, qdbus6, GGUF, etc. — 4-5 each |
| Mined dispatch_pairs | 3000 | Real user phrasings × 5-6 expansions each (capped) |

Skip if you haven't changed any tool YAMLs or token lists — the existing
`dataset/tokenizer/` is already up to date.

When to re-run:
- Added/removed a tool from `tools/catalog/`
- Edited `CORE_TOOLS` in `build_tokenizer_dataset.py`
- Added new dispatch pairs to `dataset/dispatch_pairs.jsonl` and want them
  in the corpus

---

## Stage 2 — Train the tokenizer extension

```bash
.fngemma-suryaos/bin/python training/train_tokenizer.py
```

Time: ~5 min on RTX 3080 Ti, ~40 min on CPU.

What happens:
1. **Phase 1 (Add tokens)** — 108 new tokens added to base Gemma's
   tokenizer. Fragmentation drops from avg 2.7 to 1.0 subwords/token.
2. **Phase 2 (Smart init)** — each new embedding initialized as the
   *average of its subword pieces* (e.g. `linux_memory_usage` starts
   near `mean('linux','memory','usage')`). Far better than random init.
3. **Phase 3 (Corpus warm-up)** — train ONLY the new embedding rows
   on `corpus.txt`. Base 262K vocab embeddings are frozen via gradient
   hook (cannot drift).
4. **Phase 4 (Save)** — write extended tokenizer + `embed_init.pt`.

Outputs in `training/tokenizer_extended/`:
- Standard HuggingFace tokenizer files (tokenizer_config.json,
  special_tokens_map.json, etc.)
- `embed_init.pt` — pre-trained embeddings for new tokens only
  (~648 KB; not the full model)
- `train_log.jsonl` — structured telemetry log

Watch progress live in Grafana at http://localhost:3000 (Functiongemma
Training Overview dashboard) or in the terminal — both `[LEARN]` /
`[PROGRESS]` / `[PLATEAU]` narration prints inline.

CLI flags:
- `--epochs N` (default 5; bump to 7-10 for higher quality)
- `--lr X` (default 5e-4; reduce to 1e-4 if loss oscillates)
- `--batch-size B` (default 16; reduce on smaller GPU/CPU)
- `--skip-corpus` (only smart init, no warm-up — 1 minute)
- `--neighbors K` (default 5; show top-K neighbours per new token)

---

## Stage 3 — Analyze the trained tokenizer

```bash
.fngemma-suryaos/bin/python training/analyze_embeddings.py
```

Six modules, each ending with `[LEARN]` / `[INSIGHT]` commentary:

1. **Nearest neighbours** — top-K base-vocab + new-token neighbours per
   trained token. Auto-detects whether neighbours are MEANINGFUL.
2. **Category cluster quality** — intra-category vs inter-category
   cosine. Goal: separation > 0.2.
3. **Embedding norm distribution** — base vs new norms with outlier
   detection.
4. **Drift from smart-init** — flags tokens that didn't move during
   training.
5. **Probe sentence completion** — feeds 8 probe sentences and reports
   top-10 hit rate for the expected tool token. Goal 4 generalization
   test.
6. **ASCII cluster map** — 2D PCA of new tokens rendered as text.
   Visual collapse-detection.

This is the *evaluation* step that tells you whether iteration N
improved over iteration N-1. Compare to [`../goals.md`](../goals.md)
targets.

CLI flags:
- `--top-k K` (default 5)
- `--max-tokens N` (default 30 — to keep neighbour output readable)
- `--no-model` — skip the probe sentence module (avoids loading 270M model)
- `--only MODULE` — run just one of: neighbours / clusters / norms /
  drift / probe / map

---

## Stage 4 — LoRA fine-tune + GGUF export

```bash
.fngemma-suryaos/bin/python training/finetune.py --mode all
```

Time: ~3-25 min (GPU vs CPU).

`--mode all` runs four sub-stages:
1. **check** — verify environment and required files
2. **convert** — Ollama GGUF blob → HF safetensors (only if `model_hf/`
   missing)
3. **train** — LoRA fine-tuning on `dataset/dispatch_pairs.jsonl`
4. **export** — merge LoRA into base, convert to GGUF, write `Modelfile`

Or run sub-stages individually:
- `--mode setup` — print env setup commands
- `--mode check` — verify dependencies and data files
- `--mode convert` — only the GGUF→HF conversion
- `--mode train` — only the LoRA training
- `--mode export` — only the merge + GGUF export

Outputs:
- `training/model_lora/` — LoRA adapter weights (~10 MB)
- `training/model_merged/` — merged HF model (~540 MB)
- `training/functiongemma-suryaos-270m.gguf` — final GGUF blob
- `training/Modelfile` — for `ollama create`

Telemetry written to `training/model_lora/training_log.jsonl` and
streamed to Grafana via Loki + Prometheus.

---

## Stage 5 — Register with Ollama

```bash
ollama create functiongemma:270m-suryaos -f training/Modelfile
ollama run functiongemma:270m-suryaos "is bluetooth active"
# Expected: calls linux_service_status({"name": "bluetooth"})
```

---

## Watching progress live

`http://localhost:3000` — Grafana, "functiongemma — training overview"
dashboard (under the *functiongemma* folder).

Key panels:
- Live loss curve, learning rate schedule, gradient norm, memory usage
- Per-tool loss heatmap (during LoRA phase) with mastered/learning/
  struggling status
- Cosine similarity probes per pair over epochs
- Embedding norm ratio + drift gauges
- Live event log tail (Loki)
- Run-id selector for multi-run comparison

Or watch the terminal — both training scripts print intuitive narration
inline.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Tokenizer load failed: gated repo` (401) | `export HF_TOKEN=hf_...`; accept Gemma license |
| `torch.cuda.is_available() == False` on a GPU box | `bash training/bootstrap.sh --reinstall` |
| Grafana shows "no data" | Verify training started writing to `training/*/train_log.jsonl` |
| `prometheus_client not installed` warning | `.fngemma-suryaos/bin/pip install prometheus_client` (optional) |
| OOM during corpus training | `--batch-size 8` on smaller GPU |
| Loss plateaus very high (> 6.5) | Iteration #3 corpus is not running; rebuild with `build_tokenizer_dataset.py` |

---

## Re-running cleanly

After making any corpus or token change:
```bash
rm -rf training/tokenizer_extended/
.fngemma-suryaos/bin/python training/build_tokenizer_dataset.py
.fngemma-suryaos/bin/python training/train_tokenizer.py
.fngemma-suryaos/bin/python training/analyze_embeddings.py
```

After making a LoRA / dispatch_pairs change:
```bash
rm -rf training/model_lora/ training/model_merged/
.fngemma-suryaos/bin/python training/finetune.py --mode all
```

---

## See also

- [`../RUN.md`](../RUN.md) — minimal-step run guide
- [`../goals.md`](../goals.md) — what success looks like
- [`tokenizer-explained.md`](tokenizer-explained.md) — intuitive walkthrough
- [`bug-fixes.md`](bug-fixes.md) — every bug we've caught
- [`learnings.md`](learnings.md) — decision log L1..L15
- [`../training/observability/README.md`](../training/observability/README.md) — Grafana stack docs
