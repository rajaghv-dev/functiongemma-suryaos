# Contributing

How to add new tools, scenarios, training examples, and improve the model.

---

## Quick wins (no training required)

These updates take effect immediately, no GPU needed.

### 1. Add a missing query alias to a tool

If `"my new phrasing"` doesn't route to the right tool:

```bash
# In the companion ~/raja/oc repo:
cd ~/raja/oc
echo '  - my new phrasing' >> tools/linux/SOMETHING.yaml
python3 scripts/build_fts_index.py --mode dev
```

The query now matches via FTS retrieval.

### 2. Add a new test case

```bash
cd ~/raja/oc/tests/use_cases
echo '{"query":"my new query","expected_tool":"linux.foo","expected_args":{},"category":"my_tag"}' \
  >> 99_my_new_category.jsonl
python3 tests/run_use_cases.py --layer 1 --category my_tag
```

### 3. Run the auto-fix pipeline

```bash
cd ~/raja/oc
bash scripts/test_and_collect.sh
# Auto-fixes L1 retrieval gaps + collects model-level failures for training
```

---

## Adding a new tool

Pipeline: YAML → executor → MCP handler → tokenizer entry → training pair.

### Step 1: write the tool YAML

```yaml
# ~/raja/oc/tools/<domain>/<name>.yaml
name: linux.foo.bar
domain: linux
description: One-line description of what this tool does.
risk: low                      # low → green | medium → yellow | high → red
confirmation_required: false   # true if risk = medium or high
allowed_in_production: true
executor: surya_agent.executors.linux.foo_bar

parameters:
  type: object
  required: [arg1]
  properties:
    arg1:
      type: string
      description: What this arg means.

examples:
  - "natural query 1"
  - "natural query 2"
  - "natural query 3"

aliases:
  - linux.foo.alternative_name

tags:
  - tag1
  - tag2
```

### Step 2: implement the executor

```python
# ~/raja/oc/src/surya_agent/executors/linux.py

def foo_bar(args: dict[str, Any], ctx: ExecContext) -> ExecResult:
    arg1 = args.get("arg1")
    if not arg1:
        return ExecResult(ok=False, error="arg1 required")
    argv = ["the-cli-tool", arg1]
    return run_command(argv, timeout_s=ctx.timeout_s, dry_run=ctx.dry_run)
```

### Step 3: add MCP handler

```python
# ~/raja/oc/mcp/system.py — append to TOOLS list:
{
    "name": "foo_bar",
    "description": "Same as YAML description but trigger-rich.",
    "inputSchema": { ... copy from YAML parameters ... }
},

# And to HANDLERS dict:
"foo_bar": handle_foo_bar,

# And implement:
def handle_foo_bar(args: dict) -> tuple[bool, str]:
    return _run(["the-cli-tool", args.get("arg1", "")])
```

### Step 4: add token to tokenizer dataset

```bash
# In this repo, regenerate the tokenizer dataset:
cd ~/raja/functiongemma-suryaos
python3 training/build_tokenizer_dataset.py
# This auto-pulls the new tool name from the YAML catalog.
```

### Step 5: generate training pairs

```bash
cd ~/raja/oc
python3 scripts/training/generate_pairs.py --mode augment --n-paraphrases 10
# Generates ~10 paraphrases of each example via qwen3:0.6b
```

### Step 6: rebuild FTS + verify

```bash
cd ~/raja/oc
python3 scripts/build_fts_index.py --mode dev
python3 tests/run_use_cases.py --layer 1
```

### Step 7: sync to this repo

```bash
cd ~/raja/oc
cp training/dispatch_pairs.jsonl ~/raja/functiongemma-suryaos/dataset/
cp -r tools/ ~/raja/functiongemma-suryaos/tools/catalog/

cd ~/raja/functiongemma-suryaos
git add dataset/ tools/catalog/
git commit -m "data: add linux.foo.bar tool + N training pairs"
git push origin main
```

### Step 8: queue for next training cycle

The new tool is now in the dataset. The next time `python3 training/finetune.py
--mode train` runs (on the GPU box), the model will learn to dispatch to it.

---

## Adding new app aliases

Apps are listed in `~/raja/oc/scripts/training/build_apps_catalog.py`. To add
a new app:

```python
# In APPS list:
{"binary": "myapp",         "category": "media",
 "aliases": ["myapp", "natural name 1", "natural name 2"],
 "desc": "Short description for tokenizer corpus"},
```

Then regenerate:

```bash
cd ~/raja/oc
python3 scripts/training/build_apps_catalog.py
# Outputs to ~/raja/functiongemma-suryaos/dataset/apps/
```

The script generates 5 launch phrasings per alias automatically
(`open X`, `launch X`, `start X`, etc.).

---

## Adding new scenarios

```bash
cd ~/raja/oc/tests/use_cases

# Pick the right file or create a new one:
#   01_basic.jsonl        — read-only system queries
#   02_multi_arg.jsonl    — queries that extract args
#   03_service_aliases    — service name variations
#   04_app_aliases        — app name variations
#   05_negation           — should NOT call any tool
#   06_compound           — multi-tool sequences
#   07_ambiguous          — multiple correct answers
#   08_typos_casing       — robustness
#   09_chain_of_task      — v4 multi-step
#   10_user_green         — user persona, green tier
#   11_admin_yellow       — admin, yellow tier (v2 tools)
#   12_admin_red          — admin, red tier (security)
#   13_apps_browsers      — browser + dev/office/media

# Format per line:
{"query": "...",
 "expected_tool": "linux.foo" | null,
 "expected_args": {...} | null,
 "category": "tag",
 "policy_tier": "green" | "yellow" | "red",
 "persona": "user" | "admin"}
```

Then run:
```bash
bash scripts/test_and_collect.sh --no-sync
```

The pipeline auto-fixes L1 retrieval gaps and reports remaining failures.

---

## Reporting model failures

If you encounter a real failure during normal use:

```bash
# Method 1: Use the agent normally, audit log records it
opencode run --agent coder "your failing query"
# Check ~/raja/oc/runtime/audit.db for the entry

# Method 2: Add as a test case
echo '{"query":"your failing query","expected_tool":"linux.right_tool","expected_args":{}}' \
  >> ~/raja/oc/tests/use_cases/10_user_green.jsonl

# Method 3: Open an issue with the query
gh issue create -R rajaghv-dev/functiongemma-suryaos \
  --title "Model fails on: your query" \
  --body "Expected: linux.right_tool. Actual: refused / wrong tool."
```

All three eventually feed into the next training cycle.

---

## Code style

- Python: PEP 8 + type hints. Run `ruff check` before committing.
- YAML: 2-space indent, no trailing blank lines inside lists.
- JSON: 2-space indent for human-readable, single-line for JSONL.
- Markdown: 80-character line wrap. Use `#` headers, `|` tables.

---

## Commit message format

```
<type>: <scope> — <short description>

<longer body explaining what and why, not how>

Co-Authored-By: <name> <email>
```

Types: `feat` / `fix` / `data` / `docs` / `test` / `chore` / `refactor`.

---

## Pull request checklist

- [ ] All new YAMLs have at least 3 examples
- [ ] New tools have an MCP handler in `mcp/system.py` or new file
- [ ] `bash scripts/test_and_collect.sh --no-sync` passes at L1
- [ ] CHANGELOG.md updated with the change
- [ ] If tokenizer changes: ran `build_tokenizer_dataset.py`
- [ ] If apps changed: ran `build_apps_catalog.py`
- [ ] No Zoho or cloud-service references (scope is KDE desktop only)

---

## Roadmap & priorities

See [`docs/v4-roadmap.md`](docs/v4-roadmap.md) for the long-term plan.

Short-term priorities (in order):
1. **Run the fine-tune** on RTX 3080 box → verify L3 pass rate ≥ 80%
2. **Add v2 yellow-tier tools** — service.restart, power.suspend, screenlock.lock
3. **Add v3 red-tier tools** — power.shutdown, package.install (with policy gating)
4. **Embedding fine-tune** — better retrieval recall for ambiguous queries
5. **v4 chain-of-task** — code.compile + git.commit + push workflows
