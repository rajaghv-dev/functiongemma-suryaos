# Dataset — functiongemma-suryaos

Training data for fine-tuning `functiongemma:270m` on SuryaOS desktop tool dispatch.

## Files

| File | Lines | Purpose |
|---|---|---|
| `dispatch_pairs.jsonl` | 77 | Fine-tune functiongemma tool dispatch |
| `embed_pairs.jsonl` | 48 | Fine-tune all-minilm embedding recall |

Run `python3 ../training/generate.py --mode augment` to grow both to ~530+ lines.

---

## dispatch_pairs.jsonl

### What it is

Supervised examples that teach functiongemma to call the correct tool when given
a narrow context (1-3 schemas). Each line is one training example.

### Format

```jsonl
{
  "messages": [
    {"role": "system", "content": "Call the right tool."},
    {"role": "user",   "content": "how much ram is used"}
  ],
  "tools": [{
    "name": "linux_memory_usage",
    "description": "Check RAM/memory usage.",
    "inputSchema": {"type": "object", "properties": {}}
  }],
  "target": {"name": "linux_memory_usage", "arguments": {}},
  "source": "yaml"
}
```

Key design choices:
- **One tool per example** — the model sees only the correct schema, not all 12.
  This is what the context builder provides at inference time: the retriever
  narrows 12 tools → 1-3 before the model ever sees the prompt.
- **`source` field** — `yaml` = from tool YAML examples, `failure` = real
  failures recorded from user test sessions (highest value training signal).

### Sources breakdown

| Source | Count | Description |
|---|---|---|
| `yaml` | 48 | From `examples:` field in each tool's YAML |
| `failure` | 29 | Real failures from user test sessions (2026-05-01) |

### Failure examples (most valuable)

These are exact queries where the base functiongemma:270m either refused or
called the wrong tool. Recording real failures as training examples is the
highest-signal approach — the model learns exactly what it got wrong.

```
"how much ram is used"     → linux_memory_usage      (base: refused)
"disk space"               → linux_disk_usage /       (base: called /disk path)
"wifi status"              → linux_network_status     (base: called system_wifi_status)
"is bluetooth active"      → linux_service_status bt  (base: called system_bluetooth_status)
"launch dolphin"           → kde_krunner_launch       (base: called disk_usage dolphin/)
"send a notification"      → kde_notifications_send   (base: refused)
```

### Growing the dataset

**Option 1: Paraphrase via qwen3:0.6b (recommended)**
```bash
python3 ../training/generate.py --mode augment --n-paraphrases 10
# Generates ~10 variants per example → ~530 total
# Takes ~10 min (calls Ollama locally)
```

**Option 2: Add real failures manually**
```python
# Copy this template for each new failure you observe:
{
  "messages": [
    {"role": "system", "content": "Call the right tool."},
    {"role": "user",   "content": "<failing query here>"}
  ],
  "tools": [<correct_tool_schema>],
  "target": {"name": "<correct_tool_name>", "arguments": {<correct_args>}},
  "source": "failure"
}
```

**Option 3: Real usage from audit.db**
```bash
python3 ../training/generate.py --mode audit
# Extracts real queries from ~/raja/oc/runtime/audit.db
```

### Multi-arg examples (important gap)

Current examples mostly cover required args only. functiongemma needs examples
where optional args must be extracted from the query:

```jsonl
{"messages":[{"role":"user","content":"lower the volume by 20 percent"}],
 "tools":[volume_schema],
 "target":{"name":"linux_volume_set","arguments":{"direction":"down","step":20}}}

{"messages":[{"role":"user","content":"is sshd running"}],
 "tools":[service_schema],
 "target":{"name":"linux_service_status","arguments":{"name":"sshd"}}}
```

### v4 scale plan

| Phase | Examples | New tools | Description |
|---|---|---|---|
| Current | ~530 | 12 | SuryaOS system tools |
| v2 | ~1000 | +8 | KDE D-Bus tools (kmail, kontact, activities) |
| v3 | ~1500 | +10 | Code/git tools (compile, test, commit, push) |
| v4 | 2000+ | +20 | Chain-of-tasks, IDE, multi-agent workflows |

---

## embed_pairs.jsonl

### What it is

Contrastive pairs for fine-tuning the `all-minilm:22m` embedding model.
Teaches the embedder that "how is the system?" belongs with `linux.metrics.summary`
and NOT with `linux.volume.set`.

### Format

```jsonl
{
  "query":    "How is the system doing?",
  "positive": "linux.metrics.summary",
  "negative": "linux.volume.set"
}
```

### Why it matters

The context builder does FTS + vector search to narrow 12 tools → 1-3 before
the model sees anything. If the embedding model retrieves the wrong tool, the
model will call the wrong one even after fine-tuning.

`all-minilm:22m` fine-tuned on ~500 (query, tool) pairs achieves >95% recall@1
on the SuryaOS tool catalog. Training takes ~30 min on CPU.

### Graph-informed negatives

Random negatives are weak. Better negatives are tools that co-occur in workflows
(high semantic overlap but different functions):

```
metrics_summary ↔ battery_status    (both "system status" queries)
service_status  ↔ metrics_summary   (both "is X working" queries)
krunner_launch  ↔ window_focus      (both "desktop action" queries)
```

Use `DependencyGraph.from_registry(reg)._adj` to find these automatically.
See `../training/generate.py` for the implementation.

---

## tools/tool_schemas.json

Full MCP schema for all 12 tools, extracted from the SuryaOS tool YAML catalog.
Used by the training scripts to build `tools: [...]` in each training example.

```bash
# Regenerate from source:
cd ~/raja/oc
python3 -c "
import json, sys
sys.path.insert(0, 'src')
from surya_agent.tool_registry import ToolRegistry
from surya_agent.context_builder import _tool_to_mcp_schema
reg = ToolRegistry.load('tools')
schemas = {n: _tool_to_mcp_schema(t) for n,t in reg.tools.items()}
print(json.dumps(schemas, indent=2))
" > ~/raja/functiongemma-suryaos/tools/tool_schemas.json
```
