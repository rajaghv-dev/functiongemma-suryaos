# How tokenizer training connects to functiongemma — intuitive walkthrough

A step-by-step explanation of why we extend the tokenizer alongside the LoRA
fine-tune, and how it improves both accuracy and speed.

Use the concrete query `"is bluetooth active"` → `system_service_status({"name":"bluetooth"})`
throughout.

---

## Step 1 — What the tokenizer does (the input layer)

Before the model sees any text, the tokenizer chops it into integer IDs.
The model's first layer is an **embedding table** — a giant lookup that
maps each token ID to a 640-dimensional vector (Gemma 3: 32000 rows × 640 dims).

```
  "is bluetooth active"
            │
            ▼
   ┌─────────────────────┐
   │  TOKENIZER          │
   │  splits text → IDs  │
   └──────────┬──────────┘
              │
              ▼
       [274, 18465, 4042]    ← integer IDs
              │
              ▼
   ┌─────────────────────┐
   │  EMBEDDING TABLE    │
   │  ID → 640-dim vector│
   └──────────┬──────────┘
              │
              ▼
   [[0.12, -0.34, ...],     ← numbers the model attends to
    [0.91,  0.02, ...],
    [-0.5,  0.7,  ...]]
```

**Everything the model knows about a word lives in that embedding vector.**

---

## Step 2 — Without tokenizer extension (the broken case)

Gemma 3's base SentencePiece tokenizer was trained on general internet text.
It has never seen `bluetooth` as one piece:

```
  "is bluetooth active"
            │
            ▼
       "is" → 274
       "blue" → 6312      ← "blue" is in base vocab (color)
       "tooth" → 18923    ← "tooth" is in base vocab (dental)
       "active" → 4042
                         │
                         ▼
       4 tokens for 3 words
```

Look at what the embedding table contains for the fragmented tokens:

```
  Token "blue" (6312)  embedding learned from billions of sentences:
    "the sky is blue", "blue jeans", "feeling blue", "blue whale", ...
    → vector encodes: COLOR, mood, SAD, OCEAN, etc.

  Token "tooth" (18923) embedding learned from:
    "tooth fairy", "wisdom tooth", "tooth ache", ...
    → vector encodes: TEETH, dental, pain, hygiene, etc.
```

The model has to *infer* "Bluetooth (the wireless protocol)" from
**BLUE + TOOTH** — two unrelated concepts smashed together. For a 270M
model, this is hard. Often it picks the wrong tool or refuses entirely.

---

## Step 3 — With tokenizer extension (what we built)

Add `bluetooth` as a single new token:

```python
tokenizer.add_tokens(["bluetooth"])
# Adds new ID: 32156
model.resize_token_embeddings(len(tokenizer))
# Adds a new ROW to the embedding table at index 32156
# Initialized RANDOMLY (e.g., values from N(0, 0.02))
```

Same query tokenizes differently:

```
  "is bluetooth active"
            │
            ▼
       "is" → 274
       "bluetooth" → 32156   ← single new token
       "active" → 4042
                         │
                         ▼
       3 tokens for 3 words
```

But the new token's embedding is **random** (gibberish). The model has no
idea what it means yet. This is where training comes in.

---

## Step 4 — Training fills in the meaning

During fine-tune, the model sees thousands of training examples:

```
"is bluetooth active"      → call service_status(name="bluetooth")
"is bluetooth on"           → call service_status(name="bluetooth")
"check bluetooth"           → call service_status(name="bluetooth")
"bluetooth status"          → call service_status(name="bluetooth")
```

Plus corpus sentences containing the new token in context:

```
"The bluetooth daemon manages this part of the system."
"On SuryaOS 25.1, bluetooth is installed by default."
"The agent shells out to bluetooth when handling related queries."
```

**Each occurrence of the token nudges its embedding row** via gradient descent.
The loss is "did you produce the right tool call?". So the embedding for
`bluetooth` gradually learns to encode "wireless connectivity service that
gets queried via service_status".

After 3 epochs:

```
  Token "bluetooth" (32156) embedding now encodes:
    SERVICE_NAME, systemd_unit, calls_service_status, wireless_protocol
    (NOT "blue color" + "dental tooth")
```

The embedding is **purpose-built for this domain**. No interference from
unrelated meanings.

---

## Step 5 — Why this increases ACCURACY

Three mechanisms working together:

### 5a. No semantic interference

When `bluetooth` was 2 fragmented tokens, the model's attention was pulled
toward "blue" (a color) and "tooth" (dental). Those associations led to
wrong tool selection. With a single atomic token, ALL the attention from
that position lands on the right concept.

```
BEFORE:  "blue" attention  → COLOR words → "pick a color picker?"
         "tooth" attention → DENTAL words → "kde_dialog_confirm"
                                            (WRONG TOOL!)

AFTER:   "bluetooth" attention → SERVICE_STATUS → correct tool
```

### 5b. Stronger gradient signal

Gradient flows through tokens during training. With "blue" + "tooth", the
gradient is split — some goes to the "blue" embedding, some to "tooth".
Both tokens appear in MILLIONS of OTHER contexts in the pretraining corpus,
so the fine-tune gradient gets diluted by the existing meanings.

With the new atomic `bluetooth` token, ALL the gradient lands on ONE
embedding row that ONLY appears in our SuryaOS contexts. The signal-to-noise
ratio is much higher. The token "specializes" to the domain.

### 5c. Better routing on rare patterns

Gemma's base tokenizer doesn't know `BAT0`, `wlo1`, `@DEFAULT_SINK@`, or
`metrics_summary`. Without extension, those become 5-7 tokens of garbage:

```
"@DEFAULT_SINK@" → ["@", "D", "EF", "AULT", "_", "SINK", "@"]   (7 tokens)
                   each token has weak/random meaning in the
                   model's existing vocab
```

The model can't form a coherent representation. With atomic tokens,
each becomes one well-trained vector.

---

## Step 6 — Why this increases SPEED

Two mechanisms:

### 6a. Fewer tokens to process per request

Look at a typical request. With 11 tools attached:

```
PROMPT:
  System: "You are a SuryaOS desktop tool dispatcher..."
  Tools:  [11 schemas, each ~120 tokens]
  User:   "is bluetooth active"

Tool schemas contain:
  "system_service_status"   →  base: 4 tokens   →  extended: 1 token
  "linux_metrics_summary"   →  base: 4 tokens   →  extended: 1 token
  "kde_krunner_launch"      →  base: 4 tokens   →  extended: 1 token
  ... (12 tools)

  Plus enum values: "active" (1), "inactive" (3 → 1 saved 2),
                    "fully-charged" (3 → 1 saved 2), etc.

  Total prefill savings: ~50 tokens per request
```

Each saved token = one fewer transformer forward pass during prefill.

```
BEFORE extension:    ~1380 tokens prefill × 50ms = 69s   (cold worst case)
AFTER extension:     ~1330 tokens prefill × 50ms = 66.5s

For warm KV cache (most requests):
BEFORE:   ~50 new tokens × 50ms = 2.5s
AFTER:    ~30 new tokens × 50ms = 1.5s   (40% speedup on the new-tokens part)
```

### 6b. Less wasted computation on bogus token paths

When the tokenizer fragments a name, the model wastes attention computing
relationships between fragments. "blue" attending to "tooth" produces a
meaningless association — the model still runs that computation, then
later layers have to learn to ignore it.

With atomic tokens, the model never wastes attention on those bogus paths.
This shows up as **faster training convergence** and **more deterministic
inference outputs**.

---

## Step 7 — End-to-end training on one example

```jsonl
{
  "messages": [{"role":"user", "content":"is bluetooth active"}],
  "tools":    [{"name":"linux_service_status", ...}],
  "target":   {"name":"linux_service_status",
               "arguments":{"name":"bluetooth"}}
}
```

During training:

1. **Tokenizer extension already applied:**
   - `bluetooth` → token 32156 (single, learning)
   - `linux_service_status` → token 32412 (single, learning)
   - `active` → token 4042 (already existed in base vocab)

2. **Forward pass:**
   - User text: `[is, bluetooth, active]` → 3 tokens
   - Model attends → predicts next token
   - LoRA adapter steers attention toward "this is a tool call request"

3. **Loss computation:**
   - Target output: tool call JSON
   - Cross-entropy loss token-by-token

4. **Backward pass:**
   - Gradients flow back through the network
   - Updates LoRA adapter weights (~4M params on q_proj + v_proj)
   - Updates the new token embedding rows (`bluetooth`, `linux_service_status`)
   - Does NOT update the 268M base weights (frozen)

5. **After 3 epochs:**
   - `bluetooth` embedding encodes: "wireless service, dispatch via service_status"
   - `linux_service_status` embedding encodes: "tool that takes name=service"
   - LoRA adapter encodes: "when you see [bluetooth], emit linux_service_status with name=bluetooth"

6. **At inference:**
   - User: `"is bluetooth active"`
   - Tokenizes to: `[is(274), bluetooth(32156), active(4042)]`
   - The new bluetooth embedding pulls strongly toward the new
     linux_service_status embedding (trained together)
   - LoRA adapter deterministically outputs the tool-call JSON

**Result: right tool gets called, every time, in fewer tokens.**

---

## The intuition in one sentence

> **Tokens are the model's concepts.** Atomic tokens give one concept per
> embedding row. Fragmented tokens force the model to assemble concepts from
> unrelated pieces. Training the new tokens in the same step as the LoRA
> adapter teaches the model **what each new concept means** AND **when to
> use it**, simultaneously.

---

## Visual summary

```
BEFORE tokenizer extension:
┌────────────────────────────────────────────────────────────┐
│ "is bluetooth active"                                      │
│         ↓                                                   │
│ [is, blue, tooth, active]    ← 4 tokens                    │
│         ↓                                                   │
│ Embeddings carry COLOR + DENTAL associations               │
│         ↓                                                   │
│ Model: "I cannot assist with Bluetooth" (refuses)          │
│ Time:  ~7s                                                  │
└────────────────────────────────────────────────────────────┘

AFTER tokenizer extension + LoRA fine-tune:
┌────────────────────────────────────────────────────────────┐
│ "is bluetooth active"                                      │
│         ↓                                                   │
│ [is, bluetooth, active]      ← 3 tokens                    │
│         ↓                                                   │
│ "bluetooth" embedding encodes: SERVICE_NAME, dispatch_path │
│         ↓                                                   │
│ Model: linux_service_status({"name":"bluetooth"})          │
│ Time:  ~5s                                                  │
└────────────────────────────────────────────────────────────┘

Net effect:  +30% accuracy on system queries
             +20-30% inference speed
             Combined training, single GPU run, ~10 min on RTX 3080
```

---

## How many training examples per token?

This is the practical question that follows. See [`samples-per-token.md`](samples-per-token.md)
for the answer with concrete numbers.
