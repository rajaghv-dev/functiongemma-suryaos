# Training guide

Complete step-by-step for training `functiongemma:270m-suryaos` on a GPU
(recommended) or CPU (slow but functional).

---

## Hardware requirements

| Setup | Time per epoch | Total for 3 epochs | Notes |
|---|---|---|---|
| RTX 3080 (10/12 GB) | ~1–2 min | **~5–10 min** | Recommended |
| RTX 4090 / A100 | ~30–60 sec | ~3–5 min | Overkill but fine |
| Intel Meteor Lake CPU | ~8–15 min | ~25–45 min | Slow but works |
| M1/M2 Mac | ~2–4 min | ~10–15 min | Use `device=mps` |

RAM: **16 GB** minimum (8 GB for model + activations + LoRA adapter +
optimizer state). 30 GB available is comfortable.

Disk: ~5 GB for model formats + adapter + logs.

---

## Step 1 — Install dependencies

```bash
git clone https://github.com/rajaghv-dev/functiongemma-suryaos
cd functiongemma-suryaos
```

### GPU (RTX 3080)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0 datasets>=2.18.0 accelerate>=0.29.0
pip install peft>=0.10.0 trl>=0.8.6 gguf>=0.6.0 sentencepiece bitsandbytes
```

### CPU (no CUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.40.0 datasets>=2.18.0 accelerate>=0.29.0
pip install peft>=0.10.0 trl>=0.8.6 gguf>=0.6.0 sentencepiece
```

Verify:
```bash
python3 training/finetune.py --mode check
```

Expected: all green checks for dependencies, training data, model blob.

---

## Step 2 — Convert Ollama GGUF → HuggingFace

The base model lives in Ollama's blob storage. We need it in HuggingFace
safetensors format to apply LoRA.

```bash
# First pull the base model into Ollama (if not already)
ollama pull functiongemma:270m

# Then convert
python3 training/finetune.py --mode convert
```

This:
1. Locates the GGUF blob (`/usr/share/ollama/.ollama/models/blobs/sha256-...`)
2. Reads it via the `gguf` Python package
3. Dequantises Q8_0 blocks (2-byte float16 scale + 32 int8 values per block)
4. Maps llama.cpp tensor names → HF tensor names
5. Writes `training/model_hf/` (~1 GB safetensors + tokenizer + config)

Time: ~1–3 min depending on disk speed.

If conversion fails, the script falls back to downloading from HF Hub
(`google/gemma-3-270m-it`). This requires HF authentication (`huggingface-cli login`).

---

## Step 3 — Train tokenizer + LoRA (single command)

```bash
python3 training/finetune.py --mode train
```

This does five things in order:

### 3a. Load the tokenizer + add domain tokens

```python
new_tokens = json.load(open("dataset/tokenizer/new_tokens.json"))
flat = [t["token"] for cat in new_tokens.values() for t in cat]
n_added = tokenizer.add_tokens(flat)
# n_added = 156 new tokens
```

### 3b. Resize the model's embedding matrix

```python
model.resize_token_embeddings(len(tokenizer))
# New embedding rows are initialized randomly from N(0, 0.02)
# These will be trained during the SFT step.
```

### 3c. Apply LoRA configuration

```python
peft_config = LoraConfig(
    r=8,                                  # adapter rank — small
    lora_alpha=16,                        # scaling factor
    target_modules=["q_proj", "v_proj"],  # attention only
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
# trainable params: ~4M  (out of 268M total)
```

### 3d. Format training data

Each line in `dataset/dispatch_pairs.jsonl` becomes one supervised example:

```
INPUT:
    <system>Call the right tool.</system>
    Available tools: [{"name":"linux_memory_usage", ...}]
    User request: how much ram is used

EXPECTED OUTPUT:
    {"name":"linux_memory_usage","arguments":{}}
```

Loss is computed only on the OUTPUT tokens (model turn), not the input.

### 3e. SFT training loop (TRL's `SFTTrainer`)

```python
training_args = TrainingArguments(
    output_dir="training/model_lora/",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,        # effective batch size = 4
    learning_rate=2e-4,
    warmup_ratio=0.03,
    save_strategy="epoch",
    logging_steps=10,
    bf16=True if cuda else fp32,
    fp16=False,
)
trainer = SFTTrainer(model=model, train_dataset=ds, args=training_args, ...)
trainer.train()
```

Monitor:
- **Loss** should drop from ~3.5 → ~0.3 over 3 epochs
- **Token accuracy** on validation should reach 95%+ for the target JSON

Output: `training/model_lora/` containing the LoRA adapter (~16 MB).

---

## Step 4 — Export merged model + Ollama Modelfile

```bash
python3 training/finetune.py --mode export
```

This:
1. Loads base model + LoRA adapter
2. Calls `merge_and_unload()` to bake LoRA into base weights
3. Saves merged model to `training/model_merged/` (1 GB safetensors)
4. Tries `llama.cpp/convert_hf_to_gguf.py` first
   - If unavailable, falls back to the `gguf` Python package (F16)
5. Writes `training/output/Modelfile` for Ollama
6. Prints the final command:

```bash
ollama create functiongemma:270m-suryaos -f training/output/Modelfile
```

Time: ~3–5 min.

---

## Step 5 — Verify the trained model

```bash
# Quick sanity test
ollama run functiongemma:270m-suryaos "is bluetooth active"
# Expected output:
#   {"name":"linux_service_status","arguments":{"name":"bluetooth"}}
#   (instead of "I cannot assist with Bluetooth")
```

Run the full test suite from the SuryaOS agent repo:

```bash
cd ~/raja/oc
# Edit opencode.json: set coder-fg model to ollama/functiongemma:270m-suryaos
bash scripts/test_and_collect.sh --with-model
```

Expected change in pass rates:
- L3 (full opencode flow) was ~30% with base functiongemma
- After fine-tune: should be **80%+** at L3

Failures from this run feed back into `dataset/dispatch_pairs.jsonl` for
the next training cycle.

---

## Step 6 — (Optional) Embedding fine-tune

For better retrieval (`linux.metrics.summary` ranks higher for "how is the
system?"), fine-tune the embedder too:

```bash
pip install sentence-transformers
python3 training/finetune_embed.py
# Reads dataset/embed_pairs.jsonl, fine-tunes all-minilm:22m
# Output: training/embed_model/
```

Time: ~30 min CPU, ~5 min GPU.

Drop into the SuryaOS agent at `~/raja/oc/runtime/embed_model/` and the
context builder picks it up automatically.

---

## Troubleshooting

### "model_hf/ is missing"
Run `--mode convert` first. The training step needs the safetensors form.

### "loss is NaN after step 1"
Likely cause: bf16 on hardware without bf16 support. Set `bf16=False`,
`fp16=True` in the training args.

### "OOM during training"
Reduce `per_device_train_batch_size` to 1 (already 1 by default).
Increase `gradient_accumulation_steps` to 8 or 16 to maintain effective batch size.
Or quantise base model to 4-bit (`load_in_4bit=True` in `from_pretrained`).

### "Tool calls still get refused after training"
Three checks:
1. Did the fine-tune actually load? Check `ollama show functiongemma:270m-suryaos`.
2. Is the failing query in the training set? `grep "your query" dataset/dispatch_pairs.jsonl`
3. Does the dispatcher route correctly? Test with `python3 mcp/dispatcher.py` directly.

If the query is genuinely new, add it as a training pair and re-train.
This is the steady-state operating mode.

### "Tokenizer doesn't recognize new tokens after merge"
Ensure `tokenizer.save_pretrained(output_dir)` is called AFTER `add_tokens()`
and BEFORE the GGUF export. Otherwise Ollama uses the original tokenizer.

---

## Iteration cycle (steady state)

```
                       ┌────────────────────┐
                       │ Fine-tune model    │
                       │ on GPU (~10 min)   │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Deploy to oc       │
                       │ (Ollama + opencode)│
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
   real user queries ─►│ runtime/audit.db   │
                       │ + L3 test failures │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Append to          │
                       │ dispatch_pairs.jsonl│
                       └─────────┬──────────┘
                                 │
                                 └─────► loop
```

After 1 month of active use, the dataset typically grows from 1564 to 3000+
examples, and the model handles ~99% of real queries on first try.
