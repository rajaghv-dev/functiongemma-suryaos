# v4 roadmap — scale to 2000+ tool calls

> Captured 2026-05-01. Current state: 12 tools, ~530 examples, SuryaOS system tools.
> v4 target: chain-of-task workflows for developers and power users.

## Why 2000+ examples

At 12 tools and 530 examples, the fine-tuned model handles all basic desktop
actions reliably. At v4 scale (2000+ cases), the agent handles complex
multi-step workflows where each step calls a different tool:

```
User: "compile my project, run the tests, and commit if they pass"

Step 1: code_compile(project="~/raja/oc")   → "Build successful"
Step 2: test_run(suite="unit")               → "47/47 passed"
Step 3: git_add(path=".")                    → staged
Step 4: git_commit(message="auto: tests pass") → committed
```

Each step is one tool call. The agent needs to plan the sequence AND call
each tool correctly. At 270M params this requires:
1. A reliable single-tool dispatcher (functiongemma fine-tuned — this repo)
2. A planner/orchestrator layer (qwen3:0.6b or larger) that decides sequence
3. A state tracker that passes results between steps

## Tool groups by v4 sprint

### Current: v0.2 — System tools (12 tools, 530 examples)

All in `mcp/system.py` and `mcp/volume.py`:
- Volume, brightness, battery, memory, disk, network, service status
- App launch, window focus, notifications, eBPF metrics
- Source: `dataset/dispatch_pairs.jsonl`

### v2 sprint — KDE D-Bus tools (+8 tools, target 1000 examples)

| Tool | D-Bus call | Example query |
|---|---|---|
| `kde.kmail.compose` | `org.kde.kmail2` | "draft email to Raja" |
| `kde.kmail.send_draft` | `org.kde.kmail2` | "send that draft" |
| `kde.kontact.event.add` | `org.kde.korgac` | "add meeting tomorrow 3pm" |
| `kde.dolphin.open` | `org.kde.dolphin` | "open ~/Downloads" |
| `kde.screenlock.lock` | `org.kde.screensaver` | "lock the screen" |
| `kde.activities.switch` | `org.kde.ActivityManager` | "switch to work mode" |
| `kde.krunner.query` | `org.kde.krunner` | "search for files named *.py" |
| `kde.clipboard.get` | `org.kde.klipper` | "what's in my clipboard" |

Training data: generate from D-Bus API docs + common user requests.

### v3 sprint — Code & git tools (+10 tools, target 1500 examples)

| Tool | Backend | Example query |
|---|---|---|
| `code.compile` | `make` / `cmake` / `cargo` | "build the project" |
| `code.test.run` | `pytest` / `cargo test` | "run the tests" |
| `code.lint` | `ruff` / `clippy` | "check for errors" |
| `git.status` | `git status` | "what's changed" |
| `git.add` | `git add` | "stage all changes" |
| `git.commit` | `git commit` | "commit with message X" |
| `git.push` | `git push` | "push to origin" |
| `git.diff` | `git diff` | "show what I changed" |
| `ide.open_file` | `kate` / `vscode` | "open main.py in editor" |
| `ide.goto_line` | editor D-Bus | "go to line 42 in kate" |

These require **confirmation** for destructive operations (commit, push):
```yaml
risk: medium
confirmation_required: true  # fires kdialog --yesno before executing
```

### v4 sprint — Chain-of-task orchestration (2000+ examples)

Compound queries that map to multi-step tool sequences:

| Query | Steps |
|---|---|
| "compile and test" | code.compile → code.test.run |
| "save and commit" | git.add → git.commit |
| "check system and notify if overloaded" | metrics_summary → if CPU>80% → notifications_send |
| "is the build passing? if yes push" | code.test.run → if ok → git.push |

Training format for multi-step:
```jsonl
{
  "messages": [{"role":"user","content":"compile and run tests"}],
  "tools": [compile_schema, test_schema],
  "target": [
    {"name":"code_compile","arguments":{}},
    {"name":"code_test_run","arguments":{"suite":"all"}}
  ],
  "type": "sequence"
}
```

## IDE integration

KDE has Kate and KDevelop. Both expose D-Bus APIs:
- Kate: `org.kde.kate` — open file, goto line, save, close
- KDevelop: `org.kde.kdevelop` — build, debug, run

The agent can act as an IDE co-pilot:
```
User: "open the file with the compilation error and go to line 42"
→ ide.open_file(path="src/main.rs") → ide.goto_line(42)
```

This requires adding Kate/KDevelop D-Bus tools and training examples
that combine file operations with code actions.

## Data generation strategy at scale

At 2000+ examples, manual labeling is too slow. Pipeline:

```
1. Tool YAML examples (ground truth)      →  12 × 5 = 60 base
2. qwen3:0.6b paraphrasing (×20)          →  60 × 20 = 1200
3. Real audit.db queries (grows daily)    →  +100/week at active use
4. Synthetic chain-of-task generation     →  100 multi-step examples
5. Adversarial: similar-but-wrong queries →  50 hard negatives
```

The audit.db loop is the key: every real user interaction where the agent
successfully calls a tool creates a new training example automatically.
After 1 month of active SuryaOS use, the dataset grows organically.
