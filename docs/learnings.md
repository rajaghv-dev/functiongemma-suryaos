# Learnings — what we tried, what worked, why

> Decision log for the SuryaOS agent fine-tuning project.
> Captured 2026-05-01.
>
> This file exists so future-us doesn't re-litigate decisions that were
> already made. When you come back in 6 months wondering "why didn't we
> just X?", the answer is here.

---

## L1 — Why functiongemma:270m, not bigger models

### What we tried

| Model | Pulled? | Outcome |
|---|---|---|
| `smollm2:135m`, `smollm:135m` | yes | No tool calling — refuses or emits free text |
| `gemma3:270m` | yes | Weak tool calling; misses required params |
| `qwen2.5:0.5b` | yes | Tool training present, JSON fidelity poor at 0.5B |
| `qwen3:0.6b` | yes | First model where tool calls emit valid JSON |
| `functiongemma:270m` | yes | Purpose-built for function calling; best dispatch quality at small size |
| `qwen3.5:4b` | yes | 2.5 minutes per query — too slow for production |
| `gemma4:e4b` | yes | Multimodal, 9.6 GB, reserved for future use |

### Decision

**functiongemma:270m for production tool dispatch.**
**qwen3:0.6b for plain chat + reasoning.**
Don't replace functiongemma — work around its limits via fine-tuning.

### Why
- 300 MB on disk → fits anywhere
- 6s warm inference → acceptable for desktop assistant
- Purpose-built for function calling at this size
- The hardcoded refusals (ram/bluetooth/system stats) are fixable via fine-tune

### What we won't do
- Switch to larger reasoning models for routine tool dispatch
- Use cloud APIs (privacy + latency + cost reasons)
- Use a single big model for everything (waste compute on simple dispatch)

---

## L2 — Why the model is the LAST step, not the first

### The wrong mental model

```
User query → MODEL (sees 11 tools) → picks one → MCP runs it
```

In this model, the LLM does the heavy lifting. At 270M params on natural
language, it picks wrong ~40% of the time on vague queries.

### The right mental model

```
User query → CONTEXT BUILDER → narrow to 1-3 tools → model picks → MCP runs
```

The context builder (FTS + graph) does the routing decision BEFORE the model
runs. The model just formalizes it into a valid tool call. With 1 schema
visible, it can't get it wrong.

### Why this works
- FTS retrieval is deterministic — given examples, BM25 picks the right tool
  ~99% of the time
- 270M model with 1 schema is reliable; with 11 schemas it isn't
- The system gets faster: ~10× prefill reduction (1 tool vs 11)

### What we built
- `inference/context_builder.py` — wires FTS + graph + tool registry
- `inference/fts.py` — SQLite FTS5 index over tool examples
- `inference/graph.py` — dependency expansion

### What we tried that didn't work
- **Single-tool dispatcher MCP** (one `dispatch` tool, hidden routing) —
  both functiongemma and qwen3:0.6b refuse to call abstract `dispatch`
  for system/bluetooth queries. The dispatcher pattern WILL work after
  fine-tune, but base models don't trust it.
- **Bigger model with all 11 tools** — qwen3.5:4b works (94% pass) but
  takes 2.5 minutes. Latency unusable for desktop assistant.

---

## L3 — Why fine-tune the tokenizer first, then weights

### The order matters

```
[1] Extend tokenizer (add 319 atomic tokens)
       ↓
[2] Resize model embeddings (add rows for new tokens)
       ↓
[3] LoRA fine-tune on dispatch_pairs.jsonl
       (new embedding rows train alongside LoRA adapter)
       ↓
[4] Merge + export to GGUF
```

### Why not the reverse

If you LoRA fine-tune first, then add tokens:
- The model has no idea what the new tokens mean (random embeddings)
- The LoRA adapter doesn't cover the new embedding rows
- You'd need a SECOND fine-tune to teach the new tokens

The combined training (tokens + LoRA at once) costs the same as either alone.

### Why per-token coverage matters

Each new token needs to appear in training text **at least 5 times** for the
gradient signal to reach its embedding row. With 319 tokens and 3849 sentences,
average is ~12 occurrences per token — comfortable.

If a token appears only 1-2 times, its embedding stays mostly random and the
model can't make sense of it. We validate this in the build script:

```python
# scripts/training/build_tokenizer_dataset.py validates:
for token in token_list:
    n = corpus_text.count(token)
    if n < min_occurrences:
        warn(f"{token!r}: only {n} occurrences")
```

---

## L4 — Why 319 tokens, not 500+

### What we considered adding (~500 more tokens)

- Linux kernel internals (cgroups, eBPF, ptrace, etc.)
- Browser automation (Selenium/Playwright/Puppeteer APIs)
- IDE-specific (VS Code commands, JetBrains shortcuts)
- Hardware (CPU model names, GPU SKUs, CUDA APIs, OpenVINO)
- ML/training (full PyTorch + transformers vocabulary)

### Why we didn't

**Two-thirds of those tokens have no training data.**

If a token never appears in `dispatch_pairs.jsonl`, the LoRA training never
adjusts its embedding. The randomly-initialized row stays random. The model
can't use the token even though it's in the vocabulary.

Adding tokens without data = bloat. Each token takes 640 floats × 4 bytes =
2.5 KB in the embedding matrix. 500 unused tokens = 1.2 MB of dead weight.

### What we DID add (current 319)

| Category | Tokens | Justification |
|---|---|---|
| Tool names (3 forms × 12 tools) | 68 | Required — model must produce these in output |
| KDE concepts | 24 | Appear in dispatch examples (Plasma, Kate, etc.) |
| System terms | 33 | Appear in service status / app launches |
| Arg values | 15 | Required output (up/down/active/inactive) |
| v4 workflow | 16 | Reserved — appears in v4_chain test cases |
| Git terms | 28 | Added per user request — used in v3+ |
| File formats | 75 | Added per user request — appears in app launches |
| ML terms | 50 | Used in repo docs and comments |

**All 319 validated ≥5 occurrences in the corpus.**

### When to add more

Two-phase approach:
- **Phase A (now)**: Train with 319 tokens. Get v0.2 model out.
- **Phase B (later)**: When v3 tools land, add another 100-200 tokens for
  code/git/IDE-specific vocabulary. Retrain.

Don't preemptively add tokens for tools that don't exist yet.

---

## L5 — Why same dataset for both models (Gemma + Qwen)

### The format that works for both

`dataset/dispatch_pairs.jsonl` uses OpenAI-compatible function calling format.
Both Gemma 3 and Qwen 3 support this format natively via their chat
templates.

```jsonl
{
  "messages": [{"role":"system",...},{"role":"user",...}],
  "tools": [...schema...],
  "target": {"name":"...", "arguments": {...}}
}
```

The training script (`training/finetune.py --model gemma|qwen`) applies the
correct chat template at runtime — no data preprocessing needed.

### Why this matters
- One source of truth — improvements lift both models
- One test harness — `bash scripts/test_and_collect.sh` works for both
- One audit log — production usage feeds back into a single dataset

### Two-model deployment strategy
- **functiongemma:270m-suryaos** — fast tool dispatch (80% of requests)
- **qwen3:0.6b-suryaos** — multi-step reasoning + chain-of-task (20%)

User chooses at runtime:
```bash
opencode run --agent coder    "is bluetooth active"  # functiongemma, ~5s
opencode run --agent reasoner "compile and test"     # qwen3, ~10s with planning
```

---

## L6 — Why LoRA r=8, not full fine-tune

### LoRA economics

| Approach | Trainable params | Disk | Quality (vs full FT) |
|---|---|---|---|
| Full fine-tune | 268M (all weights) | 1 GB | 100% |
| LoRA r=4 | ~1M | 4 MB | 92% |
| **LoRA r=8** | **~4M** | **16 MB** | **96%** |
| LoRA r=16 | ~8M | 32 MB | 98% |
| LoRA r=64 | ~32M | 128 MB | 99% |

### Why r=8 is the sweet spot
- 96% of full-FT quality with 1.5% of the compute
- Trains in ~10 min on RTX 3080 (vs ~2 hours full)
- Only 4M trainable params won't catastrophically forget the base model's
  general language understanding
- Adapter is portable (16 MB) — can ship multiple adapters per base model

### Why not higher rank
- r=16 gains 2% quality for 2× training time
- r=64 gains 3% but takes 8× the time and risks overfitting on 1564 examples
- We can revisit if dataset grows to 5000+ examples

### Target modules: q_proj + v_proj only
- Standard LoRA practice for tool dispatch
- These attention projections handle "which token attends to which" — exactly
  what tool routing needs
- Not k_proj, o_proj, mlp — those would push the model toward generative
  drift away from the structured tool-call output

---

## L7 — Why we keep the 0% admin_yellow cases in the dataset

### The temptation

`admin_yellow` test cases reference tools that don't exist (v2 work):
`linux.service.restart`, `kde.power.profile.set`, `kde.bluetooth.toggle`, etc.

Why not delete them until the tools exist?

### Why we keep them
- They document what the v2 sprint will build
- They serve as a TODO list embedded in the test harness
- When v2 lands, the same auto-fix loop turns them green automatically
- The 0% is a useful signal — clearly visible in dashboards

### Same logic for `admin_red`
- Several v3 tools (`linux.power.shutdown`, `linux.package.install`)
- Some destructive queries that should NEVER have a tool
- Mixing both lets us measure "correctly denied" vs "missing tool"

The categorization (`policy_tier: green/yellow/red`) makes the distinction
explicit so we don't conflate "model failed" with "tool not implemented yet".

---

## L8 — Why three forms of every tool name

```
linux.volume.set      ← dot form (YAML, Python)
linux_volume_set      ← underscore form (full MCP name)
volume_set            ← short form (after server prefix stripped)
system_volume_set     ← prefixed (opencode adds server name)
volume_volume_change  ← prefixed (volume server + volume_change tool)
```

### Why the model needs all forms

opencode prefixes MCP tool names with the server name automatically:
- `mcp/system.py` registers `volume_set` → opencode shows it as `system_volume_set`
- `mcp/volume.py` registers `volume_change` → shows as `volume_volume_change`

The model produces tool calls in the prefixed form. So all three+ forms must
be in the tokenizer (otherwise prefix changes fragment the token).

### Cost: ~70 tokens for 12 tools (3-5 forms each)
Trivial. Confirmed valuable for inference reliability.

---

## L9 — Why Netdata for system metrics, not direct shell

### What we tried first

```python
def memory_usage(args, ctx):
    return run_command(["free", "-h"])
# Returns: "total  used  free  shared  buff/cache  available
#           Mem:   30Gi  6.8Gi 12Gi   1.0Gi      12Gi      23Gi"
```

This is human-formatted text. Hard for the model to extract structured
fields from.

### What we use now

```python
def memory_usage(args, ctx):
    # Try Netdata first (structured numbers)
    response = http_get("http://localhost:19999/api/v1/data?chart=system.ram")
    used_mib = response["data"][0][used_idx]
    total_mib = sum(response["data"][0][1:])
    return f"RAM: {used_mib/1024:.1f} GiB used / {total_mib/1024:.1f} GiB total ({pct:.0f}% used)"
```

### Why Netdata
- Already running on the machine (native install)
- REST API at `:19999` exposes 100+ pre-collected metrics
- Numerical values, not formatted text — easier for downstream formatting
- eBPF-backed — gets to per-process detail (Ollama page faults, syscall rates)

### Fallback chain
```
Try Netdata REST → if down, try direct shell (free, df, acpi) → if missing, fail
```

This way the agent works on machines without Netdata too, just with less
detail.

---

## L10 — Why we don't use embeddings for retrieval (yet)

### What FTS gives us today
- 84% pass rate on 205 test cases
- Deterministic — same query always picks same tool
- Sub-millisecond per query
- No model required at retrieval time

### What an embedding model could add
- Semantic similarity beyond exact word matches
- "lower the loudness" → matches `volume.set` even if "loudness" isn't in
  any tool YAML
- Recall@1 lift from ~85% → ~95%

### Why we haven't switched
- FTS already works for the cases we test
- all-minilm:22m fine-tune adds ~500 ms per query (loading + inference)
- Tradeoff worsens as catalog grows past 30 tools — that's when we switch
- Embedding fine-tune dataset (`embed_pairs.jsonl`) is ready when needed

### Decision rule
Switch to embedding retrieval when one of:
- Tool catalog passes 30 tools
- L1 retrieval recall drops below 90%
- User reports "I asked for X and it picked Y" cases

Until then, FTS + auto-fix loop closes gaps faster than retraining an embedder.

---

## L11 — Why we collect failures, not just successes

### The temptation

After test passes, archive results and move on.

### Why failures are MORE valuable than passes
- Successes don't change the model — already knows them
- Failures reveal exactly where the gradient should push
- Real user failures (in `runtime/audit.db`) are unbiased — they reflect
  actual usage distribution, not test-set assumptions

### The pipeline we built

```
Failure occurs → tests/results/<ts>_failures_L{1,2,3}.jsonl
                      ↓
            scripts/test_and_collect.sh
                      ↓
       ~/raja/functiongemma-suryaos/dataset/dispatch_pairs.jsonl
                      ↓
            (next training cycle)
```

### Result
Weekly retraining loop. After 1 month of usage, dataset grows from 1564 to
~3000+, model accuracy on real-world queries climbs from 84% → 95%+.

---

## L12 — Why we keep both `~/raja/oc` and this repo

### Could be one monorepo

```
suryaos/
├── agent/        # the runtime (current ~/raja/oc)
└── training/     # the model training (this repo)
```

### Why we kept them separate

| Concern | Monorepo | Two repos |
|---|---|---|
| Training dataset is ~1 MB JSONL | tracked in agent | tracked separately |
| Training scripts pull torch (~200 MB pip) | bloats agent venv | isolated |
| Agent runs on 30 GB laptops; training runs on RTX 3080 box | mixed dependencies | clean split |
| Forking the model for a new device variant | hard | trivial |
| Public training artifacts vs private agent config | mixed | clean separation |
| Branch protection / review | one rule fits all | tighter on weights |

### How they sync
- `~/raja/oc/scripts/test_and_collect.sh` syncs failures → this repo
- This repo's `training/finetune.py` reads from agent's tool YAMLs
- Both share `dataset/tokenizer/` (this repo) and `tools/catalog/` (agent)

### Decision
Keep them separate. The 5-line `cp` in test_and_collect.sh is cheap; the
isolation benefit is real.

---

## What we'd do differently

If starting over:
1. **Start with the test harness** — write 200 cases first, then build to pass
2. **Run failures into training data automatically from day 1** — we built this
   late, lost some early failure info
3. **Establish the green/yellow/red taxonomy at the YAML level on day 1** —
   we retrofitted policy_tier later
4. **Use Netdata from day 1** — we used `free`/`df`/`acpi` for weeks before
   switching, missed structured-data benefits

What we'd keep:
1. **Stage-by-stage approach** — never ship a multi-file change without
   verifying the previous stage works
2. **Two-repo structure** — separation of concerns paid off
3. **Real failures in the dataset** — `source: "failure"` examples lifted
   accuracy faster than synthetic ones
4. **Auto-fix loop** — converged in 2-3 iterations, beats manual YAML editing
