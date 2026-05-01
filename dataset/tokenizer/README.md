# Tokenizer dataset

> Vocabulary extension for the SentencePiece tokenizer in `functiongemma:270m` (Gemma 3 architecture).

## Why a separate dataset?

The dispatch dataset (`../dispatch_pairs.jsonl`) trains the **model weights** —
it teaches the model when to call which tool. The tokenizer dataset is
different: it changes the **input layer** before the model ever sees text.

Without tokenizer extension:
```
"system_metrics_summary" → ["system", "_", "metrics", "_", "summary"]  = 5 tokens
"NetworkManager"         → ["Network", "Manager"]                        = 2 tokens
"@DEFAULT_SINK@"         → ["@", "DEFAULT", "_", "SINK", "@"]            = 5 tokens
```

With tokenizer extension:
```
"system_metrics_summary" → ["<|system_metrics_summary|>"]  = 1 token
"NetworkManager"         → ["<|NetworkManager|>"]           = 1 token
"@DEFAULT_SINK@"         → ["<|@DEFAULT_SINK@|>"]           = 1 token
```

Benefits per request (with 11 tools attached):
- **~50 tokens saved on prefill** (less computation per turn)
- **Tool names become atomic** (no fragmentation = clearer routing signal)
- **Faster inference** (~10% on Gemma 3 270M for typical prompts)

## Files

| File | Purpose |
|---|---|
| `new_tokens.json` | Categorized list of tokens to add to vocabulary |
| `corpus.txt` | Plain text corpus where each new token appears in context |
| `corpus.jsonl` | Same corpus in (text, token_count) format for training |
| `tool_terms.txt` | Just tool names + variants |
| `kde_terms.txt` | KDE/Plasma desktop concepts |
| `system_terms.txt` | Linux system terms (paths, daemons, syscalls) |

## Adding new tokens to the model

The training script (`../../training/finetune.py`) handles this in `--mode train`:

```python
# Pseudo-code (full implementation in finetune.py)
new_tokens = json.load(open("dataset/tokenizer/new_tokens.json"))
flat_tokens = [t["token"] for cat in new_tokens.values() for t in cat]

tokenizer.add_tokens(flat_tokens)            # extends vocabulary
model.resize_token_embeddings(len(tokenizer)) # adds rows to embedding matrix
# New embedding rows are initialized randomly; the corpus + LoRA training
# teaches them meaningful values.
```

The new embedding rows are trained alongside the LoRA adapter — both update
during the dispatch fine-tune. Without the corpus, the new token embeddings
would stay random and the model couldn't make sense of them.

## Token categories

### 1. Tool names (12 base × 3 forms = 36 tokens)

```
linux.volume.set     linux_volume_set     volume_set
linux.network.status linux_network_status network_status
...
```

The three forms cover:
- **Dot form** — used in YAML and Python code (`linux.volume.set`)
- **Underscore form** — used in MCP schemas after server prefix (`system_volume_set`)
- **Short form** — used in MCP schemas without server prefix (`volume_set`)

### 2. KDE/Plasma concepts (~30 tokens)

Application names: `Kate`, `Dolphin`, `Konsole`, `KMail`, `KRunner`, `KWin`,
`Akonadi`, `KOrganizer`, `KDevelop`, `Plasma`, `kdialog`, `kstart5`,
`kstart6`, `qdbus`, `qdbus6`.

These are KDE-specific and rarely appear in general-purpose pretraining corpora.

### 3. Linux/system terms (~30 tokens)

Daemon names: `pipewire`, `pulseaudio`, `NetworkManager`, `bluetoothd`,
`systemd`, `systemctl`, `journalctl`, `udev`, `cupsd`.

Device paths: `BAT0`, `wlo1`, `eno2`, `nvme0n1`, `@DEFAULT_SINK@`.

CLI tools: `pactl`, `nmcli`, `wmctrl`, `acpi`, `brightnessctl`.

### 4. Argument values (~10 tokens)

Common enum values that appear in tool calls: `up`, `down`, `active`,
`inactive`, `failed`, `connected`, `disconnected`.

## Corpus building

`corpus.txt` is built from three sources:

1. **Tool descriptions + examples** (~500 sentences)
   ```
   Use linux_volume_set with direction=down to lower the audio.
   The kde_krunner_launch tool opens applications like Kate or Dolphin.
   ```

2. **Synthetic context sentences** (~2000 sentences)
   Generated from templates that combine tool names with natural usage:
   ```
   Call system_battery_status to check the battery level.
   The NetworkManager service handles wifi connections.
   pactl set-sink-volume @DEFAULT_SINK@ +5% increases volume.
   ```

3. **Real audit log entries** (grows with usage)
   ```
   2026-05-01 14:23:11 ALLOW linux.volume.set {direction:down,step:5} ok
   ```

Each line in `corpus.txt` is one sentence. Tokens to learn appear at least 5
times each across the corpus — sufficient for the embedding to converge.

## Generation

```bash
cd ~/raja/functiongemma-suryaos
python3 training/generate.py --mode tokenizer
# → writes dataset/tokenizer/corpus.txt and corpus.jsonl
```

## Verification

After tokenizer extension during training, verify the new tokens:
```python
tokenizer = AutoTokenizer.from_pretrained("training/output/")
print(tokenizer.tokenize("system_metrics_summary"))
# Expected: ["<|system_metrics_summary|>"]   (1 token, not 5)
print(len(tokenizer))
# Expected: original_size + ~80 new tokens
```
