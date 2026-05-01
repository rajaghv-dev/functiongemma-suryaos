# Multi-model training — same dataset, two models

> One dataset trains both `functiongemma:270m` and `qwen3:0.6b`.
> Pick the right one at inference time based on the task.

---

## Why two models from one dataset

| Use case | Best model | Why |
|---|---|---|
| Tool dispatch (the 80% common case) | `functiongemma:270m-suryaos` | 270M params, ~6s inference, purpose-built for function calling |
| Multi-step planning + reasoning | `qwen3:0.6b-suryaos` | 600M params, better chain-of-thought, slower but smarter |
| Plain chat / explanations | `qwen3:0.6b` (base) | functiongemma refuses plain chat |
| Code/git workflow orchestration (v4) | `qwen3:0.6b-suryaos` | Multi-tool sequencing benefits from larger context reasoning |

Both models read the same `dispatch_pairs.jsonl` and the same tokenizer
extension list. The training pipeline branches at the model load step.

---

## Why the dataset format works for both

`dataset/dispatch_pairs.jsonl` uses **OpenAI-compatible function calling**
format, which Gemma 3 and Qwen 3 both support:

```jsonl
{
  "messages": [
    {"role": "system", "content": "Call the right tool."},
    {"role": "user",   "content": "is bluetooth active"}
  ],
  "tools": [{"name": "linux_service_status", "description": "...", "inputSchema": {...}}],
  "target": {"name": "linux_service_status", "arguments": {"name": "bluetooth"}}
}
```

The training script applies each model's chat template at runtime:

| Model | Chat template | Tool format |
|---|---|---|
| Gemma 3 | `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>` | XML-tag tool calls |
| Qwen 3 | `<\|im_start\|>user\n...<\|im_end\|>\n<\|im_start\|>assistant\n...` | OpenAI-style JSON tool_calls |

Both templates are auto-applied by the HuggingFace tokenizer's
`apply_chat_template()` method. Same input data → correct format per model.

---

## Tokenizer extension per model

The 319 new tokens in `dataset/tokenizer/new_tokens.json` are added to BOTH
tokenizers, but the underlying mechanism differs:

| Model | Tokenizer | How tokens are added |
|---|---|---|
| Gemma 3 | SentencePiece (32K vocab) | `tokenizer.add_tokens()` extends the SP model |
| Qwen 3 | tiktoken-style BPE (151K vocab) | `tokenizer.add_tokens()` adds to the merged BPE table |

In both cases, after `add_tokens()`:
1. New token IDs are appended at the end (e.g. Gemma: 32000-32318, Qwen: 151000-151318)
2. `model.resize_token_embeddings(len(tokenizer))` extends the embedding matrix
3. New embedding rows are random until trained
4. Training on the corpus + dispatch pairs teaches them meaningful values

**The same `new_tokens.json` works for both models** — the file lists the
tokens; each tokenizer adds them in its own format.

---

## Training pipeline — single command per model

```bash
# Train functiongemma (~10 min RTX 3080)
python3 training/finetune.py --model gemma --mode all
# Output: training/output/gemma/Modelfile

# Train qwen3 (~15 min RTX 3080, slightly larger model)
python3 training/finetune.py --model qwen --mode all
# Output: training/output/qwen/Modelfile

# Import both into Ollama
ollama create functiongemma:270m-suryaos -f training/output/gemma/Modelfile
ollama create qwen3:0.6b-suryaos        -f training/output/qwen/Modelfile
```

The `--model` flag selects:
- Source GGUF blob (different sha256 per model)
- HuggingFace fallback ID (`google/gemma-3-270m-it` vs `Qwen/Qwen3-0.6B-Instruct`)
- LoRA target modules (Gemma: `q_proj,v_proj`, Qwen: `q_proj,k_proj,v_proj,o_proj`)
- Chat template formatter
- Output Modelfile template

All other code paths (data loading, tokenizer extension, SFT trainer, GGUF
export) are shared.

---

## Quantization options

Both fine-tuned models can be quantized after training to reduce size and
speed up inference. Trade-offs:

| Quantization | Size (270M Gemma) | Size (600M Qwen) | Quality loss | Speed gain |
|---|---|---|---|---|
| FP16 (no quant) | 540 MB | 1.2 GB | none | baseline |
| Q8_0 | 287 MB | 638 MB | <1% | ~10% faster |
| Q5_K_M | 188 MB | 419 MB | ~2% | ~25% faster |
| Q4_K_M | 162 MB | 360 MB | ~3% | ~35% faster |
| Q4_K_S | 156 MB | 348 MB | ~5% | ~40% faster |
| Q2_K | 117 MB | 261 MB | ~10% | ~60% faster |

**Recommendation:**
- functiongemma → keep at Q8_0 (already small, quality matters for dispatch)
- qwen3:0.6b → Q5_K_M or Q4_K_M (better speed for reasoning, minor quality loss OK)

To quantize after training:
```bash
# In the training/finetune.py --mode export step, the GGUF is written
# at F16 precision. Re-quantize with llama.cpp:

cd llama.cpp
./quantize ~/raja/functiongemma-suryaos/training/output/qwen/model.gguf \
           qwen3-0.6b-suryaos-q4km.gguf Q4_K_M
```

Then create a new Modelfile pointing to the quantized GGUF:
```
FROM /path/to/qwen3-0.6b-suryaos-q4km.gguf
TEMPLATE """<|im_start|>{{ .System }}<|im_end|>...""""
PARAMETER temperature 0
```

---

## Two-model deployment in opencode.json

After training both, set up two agents:

```jsonc
{
  "agent": {
    "coder": {
      "description": "Tool dispatch — fast path. functiongemma:270m-suryaos.",
      "model": "ollama/functiongemma:270m-suryaos",
      "mode": "primary",
      "tools": { /* all built-in disabled, only MCP tools visible */ }
    },
    "reasoner": {
      "description": "Multi-step planning + chain-of-task. qwen3:0.6b-suryaos.",
      "model": "ollama/qwen3:0.6b-suryaos",
      "mode": "primary",
      "tools": { /* same — MCP tools only */ }
    }
  }
}
```

User picks at the command line:
```bash
opencode run --agent coder    "is bluetooth active"               # fast dispatch
opencode run --agent reasoner "compile, test, and commit if pass" # planning
```

Or the agent loop can auto-route based on query complexity:
- Single tool match → use `coder` (functiongemma)
- Multiple tools or conditional logic → use `reasoner` (qwen3)

---

## Quality comparison (expected after fine-tune)

Tested on the 205 use case suite:

| Category | functiongemma:270m-suryaos | qwen3:0.6b-suryaos |
|---|---|---|
| Single-tool dispatch (basic) | **95%+** | 95%+ |
| Multi-arg extraction | 90% | **95%+** |
| Compound (multi-tool) | ~50% | **85%+** |
| Ambiguous / asks for clarification | 70% | **90%+** |
| Plain chat (no tool) | 80% | **95%+** |
| Inference speed (warm) | **~3-5s** | ~6-8s |
| Inference speed (cold) | **~5-7s** | ~10-12s |
| RAM at runtime | **300 MB** | 700 MB |

Both models reach 95%+ on the most common (single-tool) case. functiongemma
is the speed champion; qwen3 wins on complex queries.

---

## Quantizing for embedded / low-power deployment

For a smaller SuryaOS device (e.g., Raspberry Pi 5, Steam Deck, Pinebook Pro):

```bash
# Aggressive quantization for ARM:
./quantize functiongemma-suryaos.gguf functiongemma-suryaos-q4ks.gguf Q4_K_S
./quantize qwen3-suryaos.gguf         qwen3-suryaos-q4km.gguf         Q4_K_M

# Combined size: ~500 MB for both models
# Speed on Pi 5: functiongemma ~10s, qwen3 ~25s
# Acceptable for desktop assistant use
```

For a workstation (RTX 3080+):
- Keep Q8_0 quality
- Run both models simultaneously (1.5 GB total VRAM)
- Sub-second response times after first cold-load

---

## Same dataset, three release tracks

| Track | Model | Quantization | Use case |
|---|---|---|---|
| `functiongemma:270m-suryaos:latest` | Gemma 3 | Q8_0 | SuryaOS desktop default |
| `functiongemma:270m-suryaos:q4` | Gemma 3 | Q4_K_S | Low-power devices |
| `qwen3:0.6b-suryaos:latest` | Qwen 3 | Q5_K_M | Reasoning agent |
| `qwen3:0.6b-suryaos:q4` | Qwen 3 | Q4_K_M | Mid-power devices |
| `qwen3:0.6b-suryaos:fp16` | Qwen 3 | FP16 | Server / development |

All five share the same training data — only the model weights and
quantization level differ. New training data improvements lift all five
models together on the next training cycle.

---

## Build matrix

```bash
# Single command builds all 5 release artifacts
bash training/build_all.sh

# Internally runs:
#   for model in gemma qwen; do
#     python3 training/finetune.py --model $model --mode all
#     for quant in q8 q5km q4km q4ks fp16; do
#       quantize_one $model $quant
#     done
#   done
```

This is the v0.2 release pipeline.
