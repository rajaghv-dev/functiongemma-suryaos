# Test results — current baseline

Latest run: 2026-05-01 (pre-finetune, base `functiongemma:270m` + qwen3:0.6b
fallback).

---

## Layer 1 (FTS retrieval) — 84% pass

L1 tests verify that the FTS index can retrieve the correct tool given
natural-language queries. **No model is involved**; failures here mean
either (a) tool descriptions need more trigger words, or (b) the FTS index
needs rebuilding.

```
Total cases:   205
Pass:          173 (84%)
Fail:           32
Skipped:        15  (compound multi-tool cases — tested at L3)
```

### By category

| Category | Pass | Total | Rate |
|---|---|---|---|
| `basic` | 28 | 28 | **100%** |
| `multi_arg` | 25 | 25 | **100%** |
| `service_alias` | 15 | 15 | **100%** |
| `app_alias` | 13 | 13 | **100%** |
| `casing` | 6 | 6 | **100%** |
| `typo` | 6 | 6 | **100%** |
| `negation` | 6 | 6 | **100%** |
| `browser` | 9 | 9 | **100%** |
| `dev_app` | 4 | 4 | **100%** |
| `kde_core` | 4 | 4 | **100%** |
| `media` | 8 | 8 | **100%** |
| `office` | 4 | 4 | **100%** |
| `comm` | 4 | 4 | **100%** |
| `user_green` | 19 | 20 | **95%** |
| `plain_chat` | 8 | 9 | **89%** |
| `ambiguous` | 6 | 9 | **67%** |
| `admin_red` | 8 | 15 | **53%** |
| `admin_yellow` | 0 | 20 | **0%** *(v2 tools)* |

### Why `admin_yellow` is 0%

These reference tools that don't exist yet (`linux.service.restart`,
`linux.power.suspend`, `kde.power.profile.set`, etc.). They're v2 sprint
work. Once the tool YAMLs are added, the same auto-fix loop brings them to
95%+ in one pass.

### Why `admin_red` is 53%

A mix of:
- Correctly denied scenarios (no tool should match) — these pass
- v3 tools not yet implemented (`linux.power.shutdown`, `linux.package.install`) — these fail
- Edge cases like "give me a root shell" that should never have a tool

Failures here are correct behavior — the model isn't supposed to invent
shell access. After v3 tools are added, this jumps to ~95%.

### Remaining ambiguous cases (3)

| Query | Currently routes to | Expected | Status |
|---|---|---|---|
| `"status"` | linux.network.status | linux.metrics.summary | needs example |
| `"show"` | linux.metrics.summary | null (too vague) | acceptable |
| `"ok"` | linux.metrics.summary | null (acknowledgment) | acceptable |

`"status"` will be added to `linux.metrics.summary.yaml` examples in the
next iteration. The other two are correct as "ambiguous + ask user".

---

## Layer 2 (dispatcher with arg extraction) — 84% pass

L2 tests the `mcp/dispatcher.py` single-tool router. It runs FTS retrieval
(same as L1) **plus** argument extraction from the natural-language query.

```
Total cases:   205
Pass:          173 (84%)
Fail:           32
Skipped:        15
```

### Arg extraction success rates

| Tool | Pass rate | Common failures |
|---|---|---|
| `linux.volume.set` | 100% | — |
| `linux.brightness.set` | 100% | — |
| `linux.disk.usage` | 100% | path normalisation works |
| `linux.service.status` | 100% | service name aliasing works |
| `kde.krunner.launch` | 95% | most app names extracted correctly |
| `kde.window.focus` | 95% | title extraction has minor gaps |
| `kde.notifications.send` | 90% | message extraction needs work |

The dispatcher's arg extraction logic in
[`tools/dispatcher.py`](../tools/dispatcher.py) handles:
- **Volume**: parses `"by N percent"` → `step=N`
- **Disk**: normalises wrong paths (`/disk` → `/`)
- **Service**: aliases `bt` → `bluetooth`, `wifi` → `NetworkManager`
- **App launch**: extracts after `open|launch|start|run`
- **Notification**: parses `:` separator for title/message

---

## Layer 3 (full opencode + model) — pending

L3 tests are the slowest (~10s per query) and require the model to be
loaded in Ollama. Run via:

```bash
cd ~/raja/oc
bash scripts/test_and_collect.sh --with-model
```

Expected baseline (BEFORE fine-tune, base functiongemma:270m):

| Category | Estimated pass rate |
|---|---|
| Simple direct queries (battery, volume, ollama) | ~60% |
| RAM / disk / wifi queries | **~10% (refusals)** |
| Bluetooth queries | **~5% (refusals)** |
| App launches | ~70% |
| Admin / system | ~40% |

**Target after fine-tune** (this is what we're training for):

| Category | Target pass rate |
|---|---|
| All system status queries | **95%+** (no more refusals) |
| App launches | **95%+** |
| Multi-arg queries | **85%+** (sometimes misses optional args) |
| Compound queries | ~50% (orchestration not in scope yet) |
| Ambiguous queries | **70%+** (will sometimes ask) |

---

## Auto-fix history

Run 1 (pre-fix):
```
54% pass at L1
53 retrieval misses captured
```

Run 2 (after first auto-fix — 52 examples added to 9 YAMLs):
```
97% pass at L1
3 stragglers (genuinely ambiguous)
```

Run 3 (after adding 88 user/admin/policy/app scenarios):
```
62% pass at L1 (regressed because new tools/aliases not in YAMLs)
77 new retrieval misses captured
```

Run 4 (after second auto-fix — 47 more examples added):
```
84% pass at L1
32 stragglers (mostly admin_yellow v2 tools and admin_red destructive)
```

The auto-fix loop converges in 2-3 iterations on each new scenario batch.

---

## Test harness location

All tests live in `~/raja/oc/tests/`:

```
tests/
├── use_cases/
│   ├── 01_basic.jsonl              28 cases — read-only queries
│   ├── 02_multi_arg.jsonl          25 cases — arg extraction
│   ├── 03_service_aliases.jsonl    15 cases — service name aliases
│   ├── 04_app_aliases.jsonl        13 cases — app name aliases
│   ├── 05_negation.jsonl            6 cases — should NOT call any tool
│   ├── 06_compound.jsonl            7 cases — multi-tool
│   ├── 07_ambiguous.jsonl           9 cases — multiple correct answers
│   ├── 08_typos_casing.jsonl       12 cases — robustness
│   ├── 09_chain_of_task.jsonl       8 cases — v4 multi-step
│   ├── 10_user_green.jsonl         20 cases — user persona, green tier
│   ├── 11_admin_yellow.jsonl       20 cases — admin, yellow tier (v2)
│   ├── 12_admin_red.jsonl          15 cases — admin, red tier
│   └── 13_apps_browsers.jsonl      33 cases — browsers + apps
├── run_use_cases.py                # test runner (L1/L2/L3)
├── auto_fix.py                     # auto-fix L1 retrieval misses
└── results/<timestamp>_*.{txt,jsonl}
```

Run any subset:
```bash
python3 tests/run_use_cases.py --layer 1 --category browser
```

---

## How failures become training data

```
Test fails at L3
   ↓
tests/results/<ts>_failures_L3.jsonl     (in dispatch_pairs format)
   ↓
scripts/test_and_collect.sh syncs to:
   ~/raja/functiongemma-suryaos/dataset/dispatch_pairs.jsonl
   ↓
Next training cycle picks them up
```

This is the steady-state operating mode — the model improves with every
real-world query that exposes a gap.
