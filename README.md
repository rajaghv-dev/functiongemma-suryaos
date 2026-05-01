# functiongemma-suryaos

Fine-tuned version of `functiongemma:270m` for SuryaOS desktop tool dispatch.

## The problem

`functiongemma:270m` (Google, 268M params, Gemma 3 architecture) is purpose-built for
function calling but has hardcoded safety refusals from its original training:

```
User: "how much ram is used"
Base model: "I cannot provide specific usage statistics for RAM."

User: "is bluetooth active"
Base model: "I cannot assist with tasks related to Bluetooth."

User: "disk space"
Base model: calls disk_usage(path="/disk")   ← wrong path
```

These are trained-in refusals — no prompt engineering can remove them.
Fine-tuning on domain-specific examples replaces them with correct behavior.

## The solution

Train a LoRA adapter on `(query, tool_schema, tool_call)` triples from the
SuryaOS tool catalog. The adapter teaches the model:

- "ram/memory" → call `linux_memory_usage()`
- "bluetooth" → call `linux_service_status(name="bluetooth")`
- "disk space" → call `linux_disk_usage(path="/")`
- "launch dolphin" → call `kde_krunner_launch(app="dolphin")`

After merging, `functiongemma:270m-suryaos` handles all 12 SuryaOS system
tools reliably at 270M params — no bigger model needed.

## Architecture

```
User query: "how much ram is used"
     │
     ▼
[Context builder] FTS + graph → narrows to memory_usage schema
     │
     ▼
[functiongemma:270m-suryaos] sees 1 schema → calls linux_memory_usage()
     │
     ▼
[MCP dispatch] free -h / Netdata API → "RAM: 5.1 GiB used / 30.6 GiB"
     │
     ▼
[qwen3:0.6b] formats → "5.1 GiB of your 30.6 GiB RAM is in use (17%)."
```

## Model details

| Property | Value |
|---|---|
| Base model | `functiongemma:270m` (Gemma 3, 268M params) |
| Architecture | gemma3 |
| Quantization | Q8_0 (base), merged to F16 |
| Fine-tune method | LoRA (r=8, alpha=16, q_proj + v_proj) |
| Training data | 77 base + ~530 augmented (query, tool) pairs |
| Training time | ~25 min CPU (Intel Meteor Lake, 30 GiB RAM) |
| Output | `functiongemma:270m-suryaos` Ollama model |

## Datasets

Three separate datasets, each training a different layer of the system:

| Dataset | Items | Trains | Doc |
|---|---|---|---|
| `dataset/dispatch_pairs.jsonl` | 462 | functiongemma weights (LoRA) | [dataset/README.md](dataset/README.md) |
| `dataset/embed_pairs.jsonl` | 461 | all-minilm:22m embedding model | [dataset/README.md](dataset/README.md) |
| `dataset/tokenizer/` | 156 tokens + 1559 sentences | SentencePiece vocabulary | [dataset/tokenizer/README.md](dataset/tokenizer/README.md) |

**Tokenizer dataset** adds 156 new atomic tokens (12 tool names × 3 forms +
KDE concepts + Linux daemons + arg values + v4 git/code terms). Without
extension, `system_metrics_summary` = 5 tokens; with extension = 1 token.
Saves ~50 prefill tokens per request and makes routing more reliable.

Sources:
- `yaml` (48): examples from SuryaOS tool YAML catalog
- `failure` (29): real failures recorded from user test sessions

## Tools covered (12)

| Tool (MCP name) | Handles |
|---|---|
| `linux_volume_set` | turn volume up/down, make it quieter |
| `linux_brightness_set` | dim screen, make display brighter |
| `linux_network_status` | wifi status, is wifi connected, am I online |
| `linux_battery_status` | battery level, is laptop charging |
| `linux_memory_usage` | how much RAM, memory usage |
| `linux_disk_usage` | disk space, how full is the drive |
| `linux_service_status` | is ollama running, is bluetooth active |
| `linux_metrics_summary` | system health, CPU+RAM+disk+battery overview |
| `kde_krunner_launch` | open kate, launch dolphin, start firefox |
| `kde_window_focus` | switch to firefox, focus terminal |
| `kde_notifications_send` | send a notification, desktop alert |
| `kde_dialog_confirm` | ask user to confirm, yes/no dialog |

## Quickstart

```bash
# 1. Install training deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.40.0 datasets>=2.18.0 accelerate>=0.29.0
pip install peft>=0.10.0 trl>=0.8.6 gguf>=0.6.0 sentencepiece

# 2. Generate more training data (optional, ~10 min)
python3 training/generate.py --mode augment --n-paraphrases 10

# 3. Convert Ollama GGUF → HF safetensors
python3 training/finetune.py --mode convert

# 4. Fine-tune (~25 min CPU)
python3 training/finetune.py --mode train

# 5. Export to Ollama
python3 training/finetune.py --mode export
ollama create functiongemma:270m-suryaos -f training/output/Modelfile

# 6. Test
ollama run functiongemma:270m-suryaos "is bluetooth active"
# Expected: calls linux_service_status(name="bluetooth")
```

## v4 scale target

Current scope: 12 tools, ~530 training examples, single desktop user.

v4 target (2000+ cases) covers chain-of-task workflows:
- Code compile → run tests → commit → push (git/IDE integration)
- Multi-agent coordination (orchestrator + sub-agents)
- Long-running tasks with status updates
- KDE Activity-based context switching

See [`docs/v4-roadmap.md`](docs/v4-roadmap.md) for the full plan.

## Related

- SuryaOS agent: `~/raja/oc` (the full agent stack)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Dataset spec: [`dataset/README.md`](dataset/README.md)
