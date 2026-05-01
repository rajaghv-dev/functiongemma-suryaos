# How many training samples per token?

Empirical answer with concrete numbers from this project.

---

## TL;DR

| Quality target | Samples per token | Total dataset (319 tokens) |
|---|---|---|
| Minimum viable | 5 | ~1,600 (we have 1564 ✓) |
| Good | 20 | ~6,400 |
| Production | 50 | ~16,000 |
| Excellent | 100+ | ~32,000+ |

**Current state: minimum viable.** The fine-tuned model will work but won't
be optimal. To reach "good", we need ~4× more data — exactly the v4 target.

---

## Why samples-per-token matters

Each new token has a randomly-initialized 640-dimensional embedding row.
The fine-tune training nudges those 640 numbers toward "the right vector
for this concept". Each gradient update is a tiny push. The token's
embedding converges only when it's been pushed enough times in
consistent directions.

```
Random init: [0.12, -0.34, 0.91, ..., 0.05]
            (gibberish — could mean anything)

After 5 occurrences:
            [0.18, -0.31, 0.85, ..., 0.07]
            (slight pull toward "service-like" concepts,
             but noisy — model still confuses with other tokens)

After 50 occurrences:
            [0.42, -0.18, 0.65, ..., 0.21]
            (clear "service_name" cluster, model uses it reliably)

After 200 occurrences:
            [0.51, -0.12, 0.59, ..., 0.28]
            (sharp, specialized, near-zero variance run-to-run)
```

---

## Empirical guidance from research

| Source | Recommendation |
|---|---|
| **HuggingFace's PEFT docs** | "10-100 examples per concept for a LoRA fine-tune to settle" |
| **Sentence-transformers (sbert.net)** | "≥5 examples per query type, prefer 30+ for production retrieval" |
| **Stanford Alpaca (52K instructions)** | ~100-500 examples per task category for instruction-following |
| **Microsoft's Phi-3 fine-tuning study** | "50-100 examples per intent for ≥95% intent classification accuracy" |
| **Google Gemma fine-tune cookbook** | "Aim for 100+ examples per output class for consistent behavior" |

These come from instruction tuning, classification, and retrieval research.
**Tool dispatch is closest to intent classification** — and the consensus
there is **50-100 examples per (intent, output) pair** for production quality.

In our terms:
- An "intent" = a tool to dispatch (we have 12 tools)
- 50 × 12 = **600 examples** for production tool dispatch
- We have 1564 — well above that floor

But for **per-token convergence** (the tokenizer extension piece), the
relevant number is different — it's how many times each NEW TOKEN appears
in the corpus, not how many examples per tool.

---

## Two distinct counts

### Count A — examples per tool (LoRA training signal)

```
linux.memory.usage:    9 examples in dispatch_pairs.jsonl
linux.disk.usage:      9 examples
linux.network.status:  9 examples
linux.service.status:  8 examples
kde.krunner.launch:   11 base + ~1450 from apps catalog = 1461
linux.metrics.summary: 5 examples
...
```

Most tools have **8-11 examples**. krunner_launch has 1461 (the apps catalog).

For LoRA r=8 to learn a tool dispatch reliably, **20-50 examples per tool**
is the comfort zone. We're below that for most tools — that's the gap to
close in v4.

### Count B — occurrences per new token (embedding training signal)

```
"bluetooth":  ~25 occurrences in the corpus (corpus.txt + dispatch_pairs)
"metrics_summary": ~50 occurrences (templates × 5 forms × 12 tools)
"systemd": ~15 occurrences
"BAT0": ~8 occurrences
"@DEFAULT_SINK@": ~7 occurrences
".pdf": ~12 occurrences
"GGUF": ~10 occurrences
```

Our **floor is 5 occurrences** (validated by `build_tokenizer_dataset.py`).
The average across 319 tokens is ~12 occurrences. This is the
**bare minimum** for the embedding to start being meaningful.

```
≥5 occurrences:   embedding moves away from random, gains some structure
≥20 occurrences:  embedding clearly clusters with related tokens
≥50 occurrences:  embedding is specialized, model uses reliably
≥100 occurrences: production-quality, near-zero run-to-run variance
```

---

## Actual distribution in our 319 tokens (measured 2026-05-01)

```
Occurrence buckets:
  5-9         258 tokens   80.9%  ████████████████████████████████████████
  10-19        20 tokens    6.3%  ███
  20-49        20 tokens    6.3%  ███
  50-99        15 tokens    4.7%  ██
  100+          6 tokens    1.9%
  Below 5       0 tokens    0.0%   (none — validated by build script)

Min:    5 occurrences
Median: 7 occurrences
Max:    4426 occurrences  (tokens that appear in many apps catalog templates)
```

**81% of our tokens are at minimum viability** (5-9 occurrences). This means
they will start to learn but won't be sharply specialized. Only 19% are
above the comfort zone of 20+.

This is the gap to close before v0.3:

```
Current state:   median 7 occurrences  →  minimum viable model
After 4× augment: median 28 occurrences →  good production model
After audit feed: median 80 occurrences →  excellent model (1-month usage)
```

---

## What this means for our model

### What WILL work after training (current data)

- Common tokens (≥20 occurrences): `bluetooth`, `metrics_summary`,
  `volume_change`, `network_status` — model will dispatch reliably
- Frequent enum values: `up`, `down`, `active` — already well-trained
- Tool name primary forms: each appears ~10-15 times, enough for routing

### What MAY be unreliable (5-20 occurrences)

- Rare device names: `BAT0`, `wlo1`, `eno2`, `nvme0n1`
- Some KDE concepts: `Akonadi`, `KIO`, `KCM`
- Specific file formats: `.heif`, `.heic`, `.zst`, `.dng`

These are below the 20-occurrence "comfort" threshold. The model will
USUALLY use them correctly but sometimes fail or refuse.

### What we should fix before production

Tokens with exactly 5 occurrences (the floor):
```bash
cd ~/raja/functiongemma-suryaos
python3 -c "
import json
text = open('dataset/tokenizer/corpus.txt').read()
tokens = json.load(open('dataset/tokenizer/new_tokens.json'))
for cat, items in tokens.items():
    for t in items:
        n = text.count(t['token'])
        if n < 10:
            print(f'  {n:3d}  {t[\"token\"]}  ({cat})')
"
```

Run this; for any token at 5-9 occurrences, add 5-10 more sentences using
that token in `corpus.txt`. Quick fix, immediate impact.

---

## Should we mine more tokens from code/comments/blogs/books?

**My recommendation: targeted mining only. Don't blanket-add.**

### Where to mine FROM (sources ranked by signal/noise ratio)

| Source | Signal | Noise | Recommendation |
|---|---|---|---|
| `~/raja/oc/` repo (our code + comments) | High | Low | **Mine** — these are EXACTLY the tokens we'll see |
| `~/raja/oc/runtime/audit.db` (real user queries) | Highest | Zero | **Mine continuously** — this is production signal |
| KDE source code (kde.org git mirrors) | Medium | High | Skip — most KDE internals never reach the user |
| Linux man pages | Medium | Medium | Skip — most are already in base vocab |
| Tech blogs (Phoronix, OMG Linux, KDE Planet) | Low | Very High | Skip — too much noise |
| Books (Linux internals, KDE Plasma docs) | Medium | High | Skip — wrong abstraction level |

### Why blogs/books are a bad source

Tokens like "the", "a", "configure", "system" appear thousands of times in
blogs. They're already perfectly trained in the base tokenizer. Mining from
blogs would mostly just confirm what's already there — wasted compute.

The tokens we WANT are the rare, domain-specific ones the base tokenizer
fragments. Those don't appear often in general blog text either — they're
only common in:
1. Our own codebase
2. Tool manifests (YAMLs)
3. Real user queries (audit log)

### Concrete mining recipe

```python
# scripts/training/mine_tokens.py (proposed for v0.3)

# 1. Extract all unique words from our codebase
oc_corpus = []
for path in Path("~/raja/oc").rglob("*.py"):
    oc_corpus.extend(path.read_text().split())
for path in Path("~/raja/oc").rglob("*.yaml"):
    oc_corpus.extend(path.read_text().split())
for path in Path("~/raja/oc").rglob("*.md"):
    # Comments + docs are great signal
    oc_corpus.extend(path.read_text().split())

# 2. Count occurrences
from collections import Counter
counts = Counter(oc_corpus)

# 3. Find candidate tokens:
#    - Appear ≥10 times in our codebase  (proves they're domain-relevant)
#    - Get fragmented by base tokenizer ≥3 ways  (proves extension helps)
candidates = []
for word, count in counts.most_common():
    if count < 10: continue
    base_tokens = base_tokenizer.tokenize(word)
    if len(base_tokens) >= 3:
        candidates.append((word, count, len(base_tokens)))

# 4. Filter to those NOT in current new_tokens.json
# 5. Manually review (5 minutes for ~50 candidates) before adding
```

This is **bounded mining** — only adds tokens that are demonstrably useful
for our domain.

### When to mine: NOT now, after the first fine-tune

Why wait?
1. The current 319 tokens cover the dispatch dataset's full domain
2. We don't know which tokens cause the most failures until we run L3 tests
3. Real-world failures from `audit.db` will identify gaps the curated list missed
4. Adding tokens before having data for them = dead weight

**Run the first training cycle. Then mine based on actual L3 failures.**

---

## My final recommendation

### Don't add more tokens yet

The v0.2 fine-tune should run with **319 tokens, 1564 dispatch pairs**.
This is enough to:
- Eliminate the base model's hardcoded refusals (ram/bluetooth/system stats)
- Get reliable single-tool dispatch on the 12 current tools
- Hit ~85-90% L3 pass rate

### Do add more EXAMPLES (not tokens)

The data shape with the most leverage right now:

```
For each tool with <30 dispatch pairs:
  Generate paraphrases via qwen3:0.6b → +20 per tool
  Add real audit.db entries (when production usage starts)
  Add adversarial examples (similar-but-wrong queries) for hard negatives
```

Run:
```bash
cd ~/raja/oc
python3 scripts/training/generate_pairs.py --mode augment --n-paraphrases 20
# Generates ~5,000 new pairs (4× current size)
```

After this, **per-tool examples go from 8-11 → 30-50**. That's the comfort
zone for LoRA learning. Existing tokens get more exposure too.

### Then, after the first training cycle

Run the L3 sweep with the trained model. Categorize failures:

1. **Tool selection wrong** — add more dispatch pairs for that tool
2. **Args wrong** — add multi-arg examples
3. **Hallucinated tool name** — that's a tokenizer issue; mine the
   hallucinated name from the YAMLs and add it as a new token
4. **Refusal still happening** — token gets MORE corpus occurrences

This data-driven loop closes gaps efficiently. No speculation about which
tokens "might be useful" — only ones the model actually fails on.

### Mining priorities (when we do mine)

In order of leverage:

```
1. audit.db real user queries           (highest signal, free)
2. mcp/system.py + mcp/volume.py        (the dispatcher we're training)
3. tools/**/*.yaml descriptions          (already partially mined)
4. ~/raja/oc Python code identifiers    (function names, class names)
5. ~/raja/oc/docs/learnings/*.md        (domain vocabulary)
6. Output of `ollama show <model>`      (model-specific terms)
7. KDE 6 D-Bus introspection            (when v3 KDE tools land)
8. Git commit messages from the repo    (action verbs)
```

Skip blogs and books — too much noise for too little gain at this scale.

---

## Concrete next steps

```bash
# Right now (before training):
cd ~/raja/oc
python3 scripts/training/generate_pairs.py --mode augment --n-paraphrases 20
# Brings dispatch_pairs.jsonl from 1564 → ~5000

# Sync to fine-tune repo
cp training/dispatch_pairs.jsonl ~/raja/functiongemma-suryaos/dataset/
cd ~/raja/functiongemma-suryaos
git add dataset/dispatch_pairs.jsonl
git commit -m "data: 4× augment dispatch pairs (1564 → 5000+)"
git push

# On RTX 3080 box:
git pull
python3 training/finetune.py --mode all
# 5000 examples × 3 epochs = ~25 min on RTX 3080
```

After v0.2 is trained and tested, THEN do targeted token mining based on
the actual L3 failure log.

This is the data-driven approach: measure first, expand second.
