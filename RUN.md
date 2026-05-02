# Run guide — minimal human steps

Two commands to start training. Everything else is automated.

---

## Prerequisites (one-time)

You need:
- Linux / WSL2 with Python 3.12+
- NVIDIA GPU (recommended) or CPU
- Docker (optional — for Grafana dashboards; can be skipped)
- A HuggingFace token with access to gated Gemma 3 270M
  ([accept license here](https://huggingface.co/google/gemma-3-270m-it))

---

## Step 1 — One-shot setup (creates venv, installs deps, starts dashboards)

```bash
bash training/bootstrap.sh
```

This script is **idempotent**. Re-running it is a fast no-op when everything
is already installed.

What it does:
1. Detects GPU and CUDA version (selects cu121/cu118/cpu PyTorch wheel)
2. Creates `.fngemma-suryaos/` virtual environment (skipped if exists)
3. Installs torch with the right wheel (skipped if already correct variant)
4. Installs all other requirements (skipped if all importable)
5. Installs bitsandbytes for QLoRA (GPU only; skipped if installed)
6. Starts the Grafana / Loki / Prometheus stack via docker compose
   (skipped if 5 containers are already running)
7. Waits for Grafana health endpoint and prints the dashboard URLs

**Output endpoints:**
- Grafana:    http://localhost:3000  (admin / admin)
- Prometheus: http://localhost:9090
- Loki API:   http://localhost:3100

**Flags if you need them:**
- `bash training/bootstrap.sh --no-dashboard` → skip docker compose step
- `bash training/bootstrap.sh --dashboard-only` → skip Python deps; just start stack
- `bash training/bootstrap.sh --reinstall` → force reinstall of everything

---

## Step 2 — Set HuggingFace token (Gemma 3 is gated)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Get your token at: https://huggingface.co/settings/tokens (any read-scope works).
Persist with `huggingface-cli login` once if you don't want to re-export.

---

## Step 3 — Train

### Option A — Tokenizer phase only (recommended first)

```bash
# Open Grafana in browser FIRST so you watch the run live:
xdg-open http://localhost:3000  # or just visit it

# Then start training:
.fngemma-suryaos/bin/python training/train_tokenizer.py
```

Time: ~40 min on CPU, ~5 min on RTX 3080 Ti or better.

Output: `training/tokenizer_extended/` (extended tokenizer + warmed embeddings).

### Option B — Full pipeline (tokenizer + LoRA fine-tune + GGUF export)

```bash
.fngemma-suryaos/bin/python training/train_tokenizer.py
.fngemma-suryaos/bin/python training/finetune.py --mode all
```

Time: tokenizer ~5-40 min, LoRA fine-tune ~3-25 min, GGUF export ~2 min.

### Option C — Regenerate corpus first (if you've changed tool YAMLs)

```bash
.fngemma-suryaos/bin/python training/build_tokenizer_dataset.py
.fngemma-suryaos/bin/python training/train_tokenizer.py
```

---

## Step 4 — Analyze the trained tokenizer (always do this)

```bash
.fngemma-suryaos/bin/python training/analyze_embeddings.py
```

Six analysis modules:
1. **Nearest neighbours** — what each new token "means" to the model
2. **Category cluster quality** — Goal 2 from goals.md (intra vs inter cosine)
3. **Embedding norm distribution** — Goal 3 (norm equivalence)
4. **Drift from smart-init** — Goal 4 (did training actually move things?)
5. **Probe sentence completion** — Goal 4 generalization (top-10 hit rate)
6. **ASCII cluster map** — visual sanity check for embedding collapse

Each module ends with `[LEARN]` / `[INSIGHT]` commentary referencing
[goals.md](goals.md) targets.

---

## Watching progress live

While training runs, open Grafana: http://localhost:3000

The pre-provisioned **"functiongemma — training overview"** dashboard
(under the *functiongemma* folder) shows:
- Live loss curve, learning rate schedule, gradient norm
- Per-tool loss heatmap (during LoRA phase)
- Cosine probe similarity per pair over epochs
- Embedding norm ratio + drift gauges
- Live event log tail (Loki)
- Multi-run comparison via the `run_id` selector

Or watch the terminal — both `train_tokenizer.py` and `finetune.py` print
intuitive `[LEARN]` / `[PROGRESS]` / `[PLATEAU]` / `[WARN]` narration as
training proceeds.

---

## Stopping the dashboard stack

```bash
cd training/observability && docker compose down       # stop, keep data
cd training/observability && docker compose down -v    # stop and wipe data
```

---

## Re-runs

Tokenizer training is deterministic per corpus. To re-run cleanly after a
corpus change:

```bash
rm -rf training/tokenizer_extended/
.fngemma-suryaos/bin/python training/train_tokenizer.py
```

LoRA training stamps a unique `run_id` per invocation so multiple runs
appear separately in Grafana for comparison.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Tokenizer load failed: gated repo` | Set HF_TOKEN (Step 2) and re-run |
| `CUDA: False` shown by bootstrap | Re-run with `--reinstall`; the script will swap CPU torch for GPU wheel |
| Grafana shows "no data" | Verify training ran: `ls training/tokenizer_extended/` |
| Containers won't start | `docker info` — ensure Docker daemon is running |
| Re-bootstrap takes minutes | This is correct on first run; subsequent runs print `[SKIP]` for everything |

---

## What success looks like

After a clean run, [`analyze_embeddings.py`](training/analyze_embeddings.py)
should report (per [goals.md](goals.md) targets):

| Metric | Target | What it means |
|---|---|---|
| Nearest 5 neighbours meaningful | > 70% | Goal 1 — embeddings live in sensible places |
| Same-tool cosine | 0.50–0.80 | Goal 2 — variants cluster but don't collapse |
| Cross-domain cosine | < 0.30 | Goal 2 — different tools are separated |
| Norm ratio (new ÷ base) | 0.70–1.20 | Goal 3 — new tokens not too quiet / loud |
| Drift from smart-init | > 0.10 | Goal 4 — training actually happened |
| Probe completion top-10 hit rate | > 50% | Goal 4 — generalization works |

If those are hit, the tokenizer phase has done its job. Next step: run LoRA
fine-tune via `finetune.py --mode all`.
