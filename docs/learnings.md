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

---

## L13 — Tokenizer training run #1 (postmortem)

### What happened

First real run of `train_tokenizer.py` — RTX 3080 Ti, 18 min, 2 epochs.
Loss dropped 8.51 → 6.73. Looked like progress. **Was actually broken.**

```
[OK]   Smart init complete: 0 via subword avg, 251 via global mean fallback
[OK]   New token embedding norms (post-init) — mean=0.4962  std=0.0000
```

`std=0.0000` is the smoking gun: every new token got the *same* embedding
(the global mean of 262K base vocab embeddings). 251 identical clones.

### Why every symptom made sense once we knew

| Symptom | Explanation |
|---|---|
| All cosine sims at +0.92 (incl. cross-domain) | Embeddings literally identical → cosine ≈ 1 by construction |
| Loss plateau at 6.7 | Model can't distinguish tools that occupy the same point |
| `<image_soft_token>` neighbour for every new token | Global mean of multilingual vocab happens to live near it |
| Sustained grad_norms 4-7 | Optimizer fighting the bad init for the entire run |

### The bug

```python
subword_ids = tokenizer.encode(token_str, add_special_tokens=False)
base_ids = [sid for sid in subword_ids if sid < base_vocab_size]
# encode() returns the NEW ID (because we already added the token);
# filter then drops everything; base_ids is always [];
# fallback uses global vocab mean → all 251 tokens identical.
```

We tokenized through the *already-extended* tokenizer, which had the new
tokens in its added-tokens trie. Every encode short-circuited to the new ID.

### Fix

Keep a separate base-tokenizer instance (loaded fresh, no `add_tokens()`)
solely for smart-init lookups. Encoding goes through the original vocab,
so subword decomposition actually works.

### Strategies + roadmap

Documented in [tokenizer-improvements.md](tokenizer-improvements.md).
Highest-leverage moves identified:
1. Fix smart-init bug (5 lines, unblocks everything)
2. Replace 70% templated corpus with co-occurrence-rich natural text
3. Defaults: 2 → 5 epochs (loss was still trending down at epoch 2)

### What this means for the project

The 18-minute training output is **discarded**. None of the new embeddings
are useful — they all point in the same direction. Re-running after the
fix is necessary; cached `tokenizer_extended/embed_init.pt` should be
deleted before the next run.

---

## L14 — Tokenizer training run #2 (the bottleneck shifts)

### What happened

Re-ran `train_tokenizer.py` with the BUG-001 smart-init fix. CPU, 41 minutes,
5 epochs (default bumped from 2). Loss 8.30 → 6.55.

### Three confirmations the fix worked

```
Smart init complete: 251 via subword avg, 0 via global mean fallback
New token embedding norms — mean=0.7160 std=0.0770   # std no longer 0!
init = mean(['kde','_','dialog','_','confirm'])      # actual subwords
```

Nearest neighbours are now meaningful instead of nonsense Kannada
characters:
```
linux_memory_usage  → 'memory'(0.50), 'usage'(0.46), '_'(0.61), '__'(0.49)
window_focus        → 'focus'(0.60), 'window'(0.60), 'Focus'(0.54)
notifications_send  → 'send'(0.56), 'notifications'(0.52)
```

`same-tool forms` cosine = +0.78 (good clustering).

### What's still wrong (the new bottleneck)

| Probe | Result | Expected | Diagnosis |
|---|---|---|---|
| cross-domain cosine | 0.66 | < 0.3 | Templates make all tool tokens look alike |
| `co-occurring ML libs` | 0.2937 → 0.2937 | should rise | **Probe measures frozen base-vocab tokens — can't move (BUG-005)** |
| `co-occurring git ops` | 0.3615 → 0.3615 | should rise | Same root cause as above |
| Loss plateau | 6.55 | 1-3 ideal | Corpus too repetitive |
| Sustained grad norms 3-7 | yes | < 1 | Same root cause: monotonous gradient signal |

### Root cause

The corpus is 70% rotated templates like `"Call {token} to handle this
request"`. Templates produce *monotonous* gradients — every new token
gets pushed in similar directions because every sentence looks similar.
Result: tool tokens cluster too tightly with each other instead of
spreading into the meaningful geometry the model needs for actual dispatch.

Also discovered BUG-005: 3 of our 9 probe pairs measure base-vocab
tokens (`torch`, `transformers`, `merge`, `commit`) which we *freeze*
via gradient hook. Those probes literally cannot move during training —
they were giving us false signals about lack of progress.

### Strategies → 20-lever roadmap

Documented in [dataset-strategies.md](dataset-strategies.md). The five
highest-priority moves identified:

1. **A4** — mine `dispatch_pairs.jsonl` failures into corpus (trivial cost)
2. **C1** — drop tokens already single-token in base vocab (-71 tokens)
3. **E2** — fix BUG-005 probe pairs (so we can measure progress)
4. **A3** — hard contrastive templates ("X is not Y") to fix cross-domain 0.66
5. **A1** — replace templates with natural language from man pages / docs

### What this means

Run #2 is **kept** but not used. Embeddings are usable but not great —
cross-domain similarity 0.66 means a query about disk could plausibly
get routed to a memory tool. We can ship LoRA fine-tuning on top of run
#2's embeddings as a baseline, but iteration #3 (with the dataset
overhaul) should significantly outperform.

Re-run procedure after iteration #3:
```bash
rm -rf training/tokenizer_extended/
.fngemma-suryaos/bin/python training/build_tokenizer_dataset.py
.fngemma-suryaos/bin/python training/train_tokenizer.py
# Target: cross-domain 0.66 → < 0.4, loss plateau 6.55 → < 5.0
```

---

## L15 — Iteration #3 dataset overhaul (commit 89e0f4d)

### What shipped

Three deliverables based on L13/L14 diagnoses:

**1. Token list pruned 319 → 108 (66% reduction)**

`build_tokenizer_dataset.py` rewritten with a fragmentation filter that
drops tokens already single-token in base Gemma. Also dropped 70+ generic
file-format extensions and 15 generic English words.

| Category | Before | After | Reason |
|---|---:|---:|---|
| tool_name | 68 | 44 | Kept all naming variants |
| kde | 24 | 11 | Dropped already-single tokens |
| system | 33 | 19 | Dropped CLI words Gemma already knows |
| file_format | 81 | 9 | Dropped redundant extensions; kept high-value |
| git | 41 | 11 | Dropped already-single git nouns |
| ml | 41 | 14 | Dropped framework words Gemma knows |
| arg_value | 15 | 0 | All dropped (corruption of generic English) |
| v4_workflow | 16 | 0 | All dropped (already-single in base vocab) |
| **Total** | **319** | **108** | |

**2. Templates removed; corpus is now curated content**

The old generator rotated 7 templates × 251 tokens = 3849 monotonous
sentences. Replaced with:

- **285 per-tool curated sentences** — varied phrasings, descriptions,
  CLI co-occurrence, naming-variant pairing, contrastive examples
- **32 cross-domain contrast** — direct "X (memory) and Y (brightness)
  are unrelated"
- **26 co-occurrence** — "tool wraps CLI" patterns
- **236 auxiliary token coverage** — KWin, Klipper, qdbus6, GGUF etc
  each get 4-5 sentences (was zero before)
- **3000 mined dispatch_pairs** (capped from 9360 to avoid drowning
  auxiliary tokens) — real user phrasings × 5-6 expansions each

Total: 3579 unique sentences (vs 3849 templated). Smaller corpus, vastly
richer signal. 15% multi-tool co-occurrence (was 0%).

**3. New analysis tool: `training/analyze_embeddings.py`**

Six modules to evaluate trained embeddings post-training:
1. Nearest neighbours (with auto meaningful-detection)
2. Category cluster quality (intra vs inter cosine)
3. Embedding norm distribution + outlier detection
4. Drift from smart-init (detect starved tokens)
5. Probe sentence completion (Goal 4 generalization)
6. ASCII PCA cluster map

Each module ends with `[LEARN]` / `[INSIGHT]` commentary referencing
goals.md targets.

**4. BUG-005 fixed**

Three frozen-token probe pairs in `PROBE_PAIRS` replaced with pairs that
include at least one trainable new token. All probes now give live
signal. See [bug-fixes.md BUG-005](bug-fixes.md).

### Why this matters

L13 fixed the smart-init bug → embeddings stopped collapsing.
L14 identified the corpus as the new bottleneck → cross-domain stuck at 0.62.
L15 ships the corpus overhaul that should drop cross-domain to < 0.40.

The tokenizer phase is now in its third generation:
- Run #1: completely broken (BUG-001)
- Run #2/#3: working but limited by templated corpus
- Run #4 (next): should hit the iter #3 intermediate targets

### Decision log: what's deferred

In scope of iter #3 (shipped):
- A4 mine dispatch_pairs (capped), C1 drop already-single tokens,
  E2 fix BUG-005 probes, A3 hard contrastive examples, A2 co-occurrence

Deferred to iter #4+:
- A1 LLM-generated natural-language corpus (vs current curated)
- D1 LLM paraphrase augmentation
- D2 production trace bootstrap
- B1-B4 diversity passes (length, structure, multi-sentence, signal vocab)
- E1, E3, E4 validation tooling (UMAP plots, holdout split, per-token loss)

### Re-run procedure

```bash
rm -rf training/tokenizer_extended/
bash training/bootstrap.sh                # idempotent setup
export HF_TOKEN=hf_...                    # auth for gated Gemma
.fngemma-suryaos/bin/python training/train_tokenizer.py
.fngemma-suryaos/bin/python training/analyze_embeddings.py
```

Targets vs Run #3:
- cross-domain cosine: 0.62 → < 0.40
- loss plateau:        6.55 → < 5.5
- BUG-005 probes:      live (move during training)

---

## L16 — Run #4 postmortem: iter #3 corpus REGRESSED on cross-domain

### What happened

After iter #3 dataset overhaul + GPU/bf16 fixes, ran train_tokenizer.py
on RTX 3080 Ti. Time: 2m 29s (16× faster than CPU). All BUG-005 probes
now alive. Loss reached 5.82 (better than Run #3's 6.55).

But the headline metric REGRESSED:

| Probe | Run #3 | Run #4 | Goal | Verdict |
|---|---:|---:|---:|---|
| **cross-domain** | **0.62** | **0.70** | **< 0.30** | **WORSE** |
| sibling linux | 0.77 | 0.78 | 0.30-0.50 | unchanged-bad |
| same-tool | 0.79 | 0.77 | 0.50-0.80 | both fine |

Cross-domain went UP by 0.08 — the opposite direction. The dataset
overhaul whose explicit goal was to fix cross-domain made it worse.

### The contradiction in the output

The narrator's old `_interpret_cosine_table` printed:
- "same-tool +0.77 — STRONG clustering (good)"
- "cross-domain +0.70 — DANGER (bad)"

These are 0.07 apart. If both are at ~0.7, the model is treating
unrelated tools as nearly the same as same-tool variants. There's no
"strong clustering" — there's just a single uniform blob. The narrator's
"good" and "bad" verdicts hinge on a numerically meaningless gap.

The summary block also claimed "tool-name tokens have meaningful starting
embeddings" and "LoRA can focus on routing logic from step 1." With
cross-domain at 0.70, both claims are false in the routing sense — the
embeddings are *meaningful* (smart-init worked) but not *differentiated*
(corpus failed to separate categories).

### Root cause

Looked at iter #3 corpus composition:

```
3000 mined dispatch_pairs  (84%)  ← same grammatical slot, all tools
 285 per-tool curated       (8%)
 236 auxiliary tokens       (7%)
  32 cross-domain contrast  (1%)  ← the only thing pushing tools apart
  26 co-occurrence          (1%)
```

Every mined sentence is shaped:
> `"<user query>" should dispatch to <TOOL>.`

Every tool name appears in the same grammatical position. The model
learns: *"all tool tokens occupy this slot."* That's an explicit signal
to **cluster them together** — the opposite of differentiation.

The 32 contrastive sentences (the only thing pushing tools APART) are
outnumbered **94 to 1** by mining sentences pulling them together. No
surprise contrast lost.

### Mental model

A useful corpus has three kinds of signal in roughly equal proportion:

1. **Coverage** — every token appears N times (we're good)
2. **Co-occurrence** — related tokens appear together (we have 26 — too few)
3. **Contrast** — unrelated tokens get explicitly separated (we have 32 — drowned)

Iter #3 over-invested in (1) via dispatch mining. Mining was supposed
to add real-user-phrasing diversity, but in expanding each pair to 6
sentences with the same syntactic shape, it actively *erased*
differentiation between tools.

### The deeper insight

> **Quantity isn't quality. The mining cap of 3000 was supposed to
> avoid drowning auxiliary tokens — it did. But it didn't avoid
> drowning *contrastive* signals, which is what we actually needed.**

I optimized for the wrong constraint when I capped mining at 3000.

### What iter #4 should do

Three concrete dial-flips, in order:

1. **Cut mining cap 3000 → 500.** Marginal value of mined sentence #501
   is near zero. The first 500 already cover the shape; more is just
   noise that strengthens the wrong signal.

2. **Auto-generate ~300 contrastive sentences.** For every cross-category
   pair (50+ such pairs across 12 tools × 2 categories), produce 6
   sentence templates: "X reports A; Y reports B — different concerns",
   "X is a memory tool, Y is a brightness tool", etc.
   Single biggest fix.

3. **Add ~100 varied-position sentences.** Tool names should appear
   not only as *dispatch targets* but as subjects, objects, modifiers:
   - "The output of linux_memory_usage is JSON-formatted" (subject)
   - "We added linux_memory_usage in v3.0" (object)
   - "linux_memory_usage and linux_disk_usage have similar response shapes
     but different domains" (contrastive subject + object)

Diversifying grammatical role forces the model to encode tool semantics,
not just "tokens that fill the dispatch slot."

### What I'm changing in the project's understanding

The dispatch_pair mining was too aggressive (BUG-008 — see bug-fixes.md
when added). The "more data is better" intuition was wrong here because
data with low syntactic variety reinforces the wrong invariant.

Going forward: corpus quality should be measured by **role variety per
token**, not just sentence count.

### Re-run after iter #4

```bash
rm -rf training/tokenizer_extended/
.fngemma-suryaos/bin/python training/build_tokenizer_dataset.py
.fngemma-suryaos/bin/python training/train_tokenizer.py
.fngemma-suryaos/bin/python training/analyze_embeddings.py
```

Targets vs Run #4:
- cross-domain cosine: 0.70 → < 0.40 (target — would put us *better*
  than Run #3's 0.62)
- sibling linux tools: 0.78 → < 0.55
- 5+ probes IN BAND (was 3)
