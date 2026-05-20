# Architecture — functiongemma-suryaos

## The system view

The model is the **last step**, not the first.

```
User: "how much ram is used"
         │
         ▼  ~5ms
[Embedding: all-minilm:22m (fine-tuned)]
  query → 384-dim vector

         │
         ▼  ~2ms
[Vector + FTS search over 12 tool schemas]
  top-1: linux_memory_usage (score: 0.87)
  top-2: linux_metrics_summary (score: 0.71)

         │
         ▼  ~1ms
[Graph expand: add declared dependencies]
  linux_memory_usage has no deps → {linux_memory_usage}

         │
         ▼  ~2ms
[Context builder: assemble minimal prompt]
  system: "Call the right tool."
  tools:  [linux_memory_usage schema]   ← 1 tool, not 12
  user:   "how much ram is used"

         │
         ▼  ~300ms
[functiongemma:270m-suryaos (fine-tuned)]
  Sees 1 schema → no choice to get wrong
  → tool_calls: linux_memory_usage({})

         │
         ▼  ~100ms
[MCP dispatch: mcp/system.py]
  Netdata API → "RAM: 5.1 GiB used / 30.6 GiB total"

         │
         ▼  ~200ms
[qwen3:0.6b: format result → text]
  → "5.1 GiB of your 30.6 GiB RAM is in use (17%)."
```

Total: ~610ms

## Why fine-tuning fixes the base model

The base `functiongemma:270m` was trained by Google with safety constraints.
For SuryaOS system queries, it has two failure modes:

**1. Hardcoded refusals:**
```
"how much ram is used"   → "I cannot provide usage statistics"
"is bluetooth active"    → "I cannot assist with Bluetooth"
```
These are trained-in refusals. No prompt can remove them. LoRA fine-tuning
replaces the relevant attention patterns with correct tool-dispatch behavior.

**2. Wrong tool selection:**
```
"disk space"             → disk_usage(path="/disk")  ← wrong path
"launch dolphin"         → disk_usage(path="dolphin/")  ← completely wrong tool
"wifi status"            → system_wifi_status  ← tool doesn't exist
```
Fine-tuning on failure examples directly corrects these.

## The context builder (why 1 schema is enough)

With 12 schemas visible, a 270M model picks wrong ~40% of the time.
With 1 schema visible, it picks wrong ~0% of the time (nothing else to pick).

The context builder runs before the model:
```python
cb = ContextBuilder(fts_index, dep_graph, tool_registry)
ctx = cb.build("how much ram is used")
# ctx.tool_names = ["linux.memory.usage"]
# ctx.tool_schemas = [memory_usage_mcp_schema]
```

After fine-tuning, the model can also handle 3-5 schemas reliably (the
fine-tuning teaches it which schema matches which natural-language pattern).

## Training data design

**Key principle:** train on NARROW context (1 schema), not WIDE context (12 schemas).

At inference time, the context builder provides 1-3 schemas.
So training data has 1 schema per example — matches inference exactly.

```
Training:  query + [1 schema] → tool_call
Inference: query + [1 schema via context builder] → tool_call
```

If we trained on all 12 schemas (like the base model sees in opencode),
the model would still fail on the "which of 12?" selection problem.

## LoRA configuration

```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # attention layers only
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

- **r=8, alpha=16**: small rank, ~4M trainable params on top of 268M frozen
- **q_proj + v_proj**: targets attention query/value, sufficient for routing
- **No k_proj or o_proj**: keeps the adapter small, prevents overfitting on 77 examples

## Tokenizer extension

Tool names are added as atomic tokens before training:

```
Before: "metrics_summary" = ["metrics", "_", "summary"]  = 3 tokens
After:  "metrics_summary" = ["metrics_summary"]           = 1 token
```

This reduces tool-name fragmentation, speeds up inference (~33 tokens saved
per request with 12 tools), and makes the model's routing more reliable
(single token = clear category boundary).

New tokens added: `metrics_summary`, `volume_change`, `network_status`,
`battery_status`, `memory_usage`, `disk_usage`, `service_status`,
`krunner_launch`, `window_focus`, `brightness_set`, `notifications_send`,
`ebpf_summary` — 12 tokens total.
