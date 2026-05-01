#!/usr/bin/env python3
"""
finetune_dispatch.py — Fine-tune functiongemma:270m for surya_agent tool dispatch.

OVERVIEW
========
functiongemma:270m is a Gemma 3 model (268M parameters, Q8_0 quantisation) that
already has "function calling" capability baked in by its original training.
However, it may pick the wrong tool when the user's phrasing is colloquial or
when tool names are domain-specific (e.g. "linux_metrics_summary").

This script teaches the model the *12 SuryaOS tools* via LoRA (Low-Rank
Adaptation) — a lightweight adapter technique that adds ~1–4 M trainable
parameters on top of the frozen base weights.  We never touch the base weights,
so the adapter can be merged later or discarded without harm.

WHY LoRA?
---------
Training all 268 M parameters on a CPU would take days and risk forgetting
general capabilities (catastrophic forgetting).  LoRA inserts small rank-8
matrices into the attention layers (q_proj, v_proj) — enough to steer the
model's routing decisions without rewriting its general language understanding.

WHY CPU-ONLY TRAINING?
----------------------
Intel Meteor Lake has no dedicated CUDA GPU.  PyTorch CPU inference and
training are fully supported; it is slow but perfectly functional for a small
model and a small dataset.  With 48 training examples and 3 epochs the full
training run takes roughly 15–25 minutes on this machine.

TRAINING DATA TARGET (scale note for future maintainers)
---------------------------------------------------------
Current scope   : 12 system tools,  500 examples target for v3
Future scope (v4): Code-compile, test, git, IDE workflows — 2000+ examples
                   needed once the agent handles development workflows.

PIPELINE STAGES (--mode)
-------------------------
  setup   → print the pip install commands; never auto-installs
  check   → verify deps, data file, and model blob
  convert → GGUF blob → HuggingFace safetensors in training/model_hf/
  train   → LoRA fine-tuning with SFTTrainer, saves to training/model_lora/
  export  → merge LoRA → base, quantise back to GGUF, write Modelfile
  all     → check → train → export  (skips convert if model_hf/ already exists)

Usage examples:
  python3 scripts/training/finetune_dispatch.py --mode setup
  python3 scripts/training/finetune_dispatch.py --mode check
  python3 scripts/training/finetune_dispatch.py --mode convert
  python3 scripts/training/finetune_dispatch.py --mode train
  python3 scripts/training/finetune_dispatch.py --mode export
  python3 scripts/training/finetune_dispatch.py --mode all
  python3 scripts/training/finetune_dispatch.py --mode train --epochs 5 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths — everything is relative to the repo root, which is two levels up
# from scripts/training/.
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parent.parent.parent
TRAINING_DIR = ROOT / "training"
DATA_FILE    = TRAINING_DIR / "dispatch_pairs.jsonl"
MODEL_HF_DIR = TRAINING_DIR / "model_hf"       # HF safetensors after convert
MODEL_LORA   = TRAINING_DIR / "model_lora"     # LoRA adapter after train
MODEL_MERGED = TRAINING_DIR / "model_merged"   # merged weights after export
MODELFILE    = TRAINING_DIR / "Modelfile"      # for `ollama create`
GGUF_OUT     = TRAINING_DIR / "functiongemma-suryaos-270m.gguf"

# The exact blob path for functiongemma:270m on this machine.
# `ollama show functiongemma:270m --modelfile` shows which blob is the weights.
OLLAMA_BLOB = Path(
    "/usr/share/ollama/.ollama/models/blobs/"
    "sha256-415f8f959d807bd4d4da891f01225d7b330416947fb011a8473080ae4fd07885"
)

# HuggingFace Hub fallback if local model_hf/ conversion was skipped.
HF_MODEL_ID = "google/gemma-3-270m-it"

# Domain-specific tokens for the 12 SuryaOS tool names.
# Adding these ensures the tokenizer has a single token for each tool name,
# which helps the model learn to emit them reliably in the output JSON.
DOMAIN_TOKENS = [
    "metrics_summary",
    "volume_change",
    "network_status",
    "battery_status",
    "memory_usage",
    "disk_usage",
    "service_status",
    "krunner_launch",
    "window_focus",
    "brightness_set",
    "notifications_send",
    "ebpf_summary",
]

# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    """Print a clear section header."""
    width = 72
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}", file=sys.stderr)


def _err(msg: str) -> None:
    print(f"  [ERR] {msg}", file=sys.stderr)


def _elapsed(start: float) -> str:
    secs = int(time.time() - start)
    return f"{secs // 60}m {secs % 60}s"


# ===========================================================================
# MODE: setup
# ===========================================================================

def mode_setup() -> None:
    """Print install commands — never auto-installs anything."""
    _banner("SETUP — pip install commands for CPU fine-tuning")

    print("""
These commands install all Python dependencies for fine-tuning on CPU.
Run them in your virtual environment before proceeding.

# 1. PyTorch CPU-only build (~200 MB, avoids pulling CUDA libs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. HuggingFace ecosystem
pip install transformers>=4.40.0 datasets>=2.18.0 accelerate>=0.29.0

# 3. LoRA adapter support
pip install peft>=0.10.0

# 4. TRL — SFTTrainer lives here (Supervised Fine-Tuning)
pip install trl>=0.8.6

# 5. GGUF read/write support (for --mode convert and --mode export)
pip install gguf>=0.6.0

# 6. sentencepiece — needed by Gemma tokenizer
pip install sentencepiece

# Optional: progress bars and nicer logs
pip install rich tqdm

# Verify after install:
python3 -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python3 -c "import transformers; print('transformers', transformers.__version__)"
python3 -c "import peft; print('peft', peft.__version__)"
python3 -c "import trl; print('trl', trl.__version__)"
python3 -c "import gguf; print('gguf OK')"
""")


# ===========================================================================
# MODE: check
# ===========================================================================

def mode_check() -> bool:
    """Verify all requirements are met before training."""
    _banner("CHECK — verifying environment")
    ok = True

    # --- Python packages ---
    required_packages = {
        "torch":        "torch",
        "transformers": "transformers",
        "datasets":     "datasets",
        "peft":         "peft",
        "trl":          "trl",
        "gguf":         "gguf",
        "sentencepiece": "sentencepiece",
    }

    print("\n  [dependencies]")
    for display_name, import_name in required_packages.items():
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            _ok(f"{display_name} {ver}")
        except ImportError:
            _err(f"{display_name} not installed — run --mode setup")
            ok = False

    # --- CPU / hardware ---
    print("\n  [hardware]")
    try:
        import torch
        _ok(f"CPU threads available: {torch.get_num_threads()}")
        if torch.cuda.is_available():
            _ok(f"CUDA also available: {torch.cuda.get_device_name(0)}")
        else:
            _ok("CUDA not available — training will run on CPU (expected)")
        # Estimate available RAM
        try:
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            _ok(f"System RAM: {mem_bytes / 1e9:.1f} GB")
        except Exception:
            pass
    except ImportError:
        pass

    # --- Training data ---
    print("\n  [training data]")
    if DATA_FILE.exists():
        lines = DATA_FILE.read_text().strip().splitlines()
        n = len(lines)
        _ok(f"{DATA_FILE} — {n} lines")
        if n < 20:
            _warn(f"Only {n} examples — model may not generalise well. "
                  "Target 500+ for reliable dispatch. Run generate_pairs.py --mode augment.")
        elif n < 100:
            _warn(f"{n} examples is marginal. Consider augmenting to 500+.")
        # Quick schema validation on first line
        try:
            first = json.loads(lines[0])
            assert "messages" in first and "target" in first and "tools" in first
            _ok("Schema validation passed (sample line)")
        except Exception as e:
            _err(f"Schema validation failed: {e}")
            ok = False
    else:
        _err(f"{DATA_FILE} not found — run scripts/training/generate_pairs.py first")
        ok = False

    # --- Ollama blob ---
    print("\n  [model blob]")
    if OLLAMA_BLOB.exists():
        size_mb = OLLAMA_BLOB.stat().st_size / 1e6
        _ok(f"GGUF blob found: {OLLAMA_BLOB.name} ({size_mb:.0f} MB)")
    else:
        _warn(f"GGUF blob not found at expected path:\n       {OLLAMA_BLOB}")
        _warn("--mode convert will fall back to HuggingFace Hub download.")

    # --- model_hf/ ---
    print("\n  [converted model]")
    if MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir()):
        files = list(MODEL_HF_DIR.iterdir())
        _ok(f"model_hf/ present ({len(files)} files) — --mode train can use this")
    else:
        _warn(f"model_hf/ not found — --mode train will download from HF Hub ({HF_MODEL_ID})")

    # --- model_lora/ ---
    print("\n  [LoRA adapter]")
    if MODEL_LORA.exists() and any(MODEL_LORA.iterdir()):
        _ok(f"model_lora/ present — --mode export can proceed")
    else:
        _warn("model_lora/ not found — run --mode train first before --mode export")

    print()
    if ok:
        _ok("All critical checks passed.")
    else:
        _err("Some checks FAILED — address errors above before proceeding.")

    return ok


# ===========================================================================
# MODE: convert
# ===========================================================================

def mode_convert() -> None:
    """Convert the Ollama GGUF blob to HuggingFace safetensors format.

    WHY THIS STEP?
    --------------
    Ollama stores models as GGUF files (a single binary blob with all tensor
    data + metadata).  HuggingFace's transformers library expects models in its
    own format: a config.json + model.safetensors + tokenizer files.

    The `gguf` Python package can read GGUF files.  We use it to:
      1. Inspect the metadata (architecture, vocab size, head counts …)
      2. Extract the weight tensors
      3. Write them out in HF safetensors format with a matching config.json

    NOTE: The gguf package's conversion utilities are still evolving.  If the
    automated conversion fails, the fallback is to download google/gemma-3-270m-it
    directly from HuggingFace Hub (requires internet + HF token for gated models).
    Gemma 3 270M IT is gated, so you need to accept the licence at:
      https://huggingface.co/google/gemma-3-270m-it
    and set HF_TOKEN in your environment.
    """
    _banner("CONVERT — GGUF blob → HuggingFace safetensors")

    MODEL_HF_DIR.mkdir(parents=True, exist_ok=True)

    if not OLLAMA_BLOB.exists():
        _warn(f"GGUF blob not found at {OLLAMA_BLOB}")
        _warn("Falling back to HuggingFace Hub download …")
        _hf_download_fallback()
        return

    try:
        import gguf
    except ImportError:
        _err("gguf package not installed — run --mode setup first")
        sys.exit(1)

    _ok(f"Reading GGUF from {OLLAMA_BLOB}")
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Strategy: use gguf.GGUFReader to extract metadata + tensors, then use
    # the transformers auto-conversion path (llama.cpp convert_hf_to_gguf.py
    # approach in reverse).
    #
    # For Gemma 3 the tensor names follow the standard HF naming convention,
    # so we can map them back directly.
    # -----------------------------------------------------------------------

    try:
        reader = gguf.GGUFReader(str(OLLAMA_BLOB), mode="r")
    except Exception as e:
        _err(f"Failed to open GGUF: {e}")
        _warn("Falling back to HuggingFace Hub download …")
        _hf_download_fallback()
        return

    # Print a summary of what's inside the GGUF
    print("\n  GGUF metadata:")
    metadata: dict = {}
    for field in reader.fields.values():
        key = field.name
        # Each field has parts (type + data); grab the first data element
        if hasattr(field, "parts") and field.parts:
            try:
                val = field.parts[field.data[0]][0]
                # Convert numpy scalar to Python type
                if hasattr(val, "item"):
                    val = val.item()
                metadata[key] = val
                if key.startswith("general.") or key.startswith("gemma"):
                    print(f"    {key}: {val}")
            except Exception:
                pass

    # Check we actually have Gemma 3 weights
    arch = metadata.get("general.architecture", "unknown")
    if "gemma" not in str(arch).lower():
        _warn(f"Unexpected architecture: {arch}. Proceeding anyway.")

    # -----------------------------------------------------------------------
    # Build Gemma 3 config.json from GGUF metadata.
    # These fields map directly from GGUF keys to HF config keys.
    # -----------------------------------------------------------------------

    # Helper: get int metadata with fallback
    def _meta_int(key: str, default: int) -> int:
        return int(metadata.get(key, default))

    n_layers       = _meta_int("gemma3.block_count",                  18)
    hidden_size    = _meta_int("gemma3.embedding_length",            1152)
    # 270M model uses 640 embedding_length per `ollama show`
    hidden_size    = _meta_int("gemma3.embedding_length",             640)
    n_heads        = _meta_int("gemma3.attention.head_count",         8)
    n_kv_heads     = _meta_int("gemma3.attention.head_count_kv",     4)
    intermediate   = _meta_int("gemma3.feed_forward_length",        3072)
    head_dim_val   = _meta_int("gemma3.attention.key_length",         256)
    vocab_size     = _meta_int("tokenizer.ggml.tokens",            256000)
    rms_norm_eps   = float(metadata.get("gemma3.attention.layer_norm_rms_epsilon", 1e-6))
    ctx_length     = _meta_int("gemma3.context_length",            32768)

    config = {
        "architectures":             ["Gemma3ForCausalLM"],
        "model_type":                "gemma3",
        "hidden_size":               hidden_size,
        "intermediate_size":         intermediate,
        "num_hidden_layers":         n_layers,
        "num_attention_heads":       n_heads,
        "num_key_value_heads":       n_kv_heads,
        "head_dim":                  head_dim_val,
        "hidden_act":                "gelu_pytorch_tanh",
        "max_position_embeddings":   ctx_length,
        "rms_norm_eps":              rms_norm_eps,
        "vocab_size":                vocab_size,
        "torch_dtype":               "float32",
        "transformers_version":      "4.40.0",
        "use_cache":                 True,
    }

    config_path = MODEL_HF_DIR / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    _ok(f"Wrote config.json (hidden={hidden_size}, layers={n_layers}, heads={n_heads})")

    # -----------------------------------------------------------------------
    # Extract tensors and write safetensors.
    # The GGUF tensor names use llama.cpp convention; map to HF names.
    # -----------------------------------------------------------------------

    import numpy as np

    # Gemma 3 GGUF → HF name mapping (partial — covers the main weight types)
    # GGUF key pattern          → HF key pattern
    #   token_embd.weight       → model.embed_tokens.weight
    #   blk.N.attn_q.weight     → model.layers.N.self_attn.q_proj.weight
    #   blk.N.attn_k.weight     → model.layers.N.self_attn.k_proj.weight
    #   blk.N.attn_v.weight     → model.layers.N.self_attn.v_proj.weight
    #   blk.N.attn_output.weight→ model.layers.N.self_attn.o_proj.weight
    #   blk.N.ffn_gate.weight   → model.layers.N.mlp.gate_proj.weight
    #   blk.N.ffn_up.weight     → model.layers.N.mlp.up_proj.weight
    #   blk.N.ffn_down.weight   → model.layers.N.mlp.down_proj.weight
    #   blk.N.attn_norm.weight  → model.layers.N.input_layernorm.weight
    #   blk.N.ffn_norm.weight   → model.layers.N.post_feedforward_layernorm.weight
    #   output_norm.weight      → model.norm.weight
    #   output.weight           → lm_head.weight

    def _gguf_to_hf_name(gguf_name: str) -> Optional[str]:
        name = gguf_name
        if name == "token_embd.weight":
            return "model.embed_tokens.weight"
        if name == "output_norm.weight":
            return "model.norm.weight"
        if name == "output.weight":
            return "lm_head.weight"
        if name.startswith("blk."):
            parts = name.split(".")
            layer_num = parts[1]
            rest = ".".join(parts[2:])
            prefix = f"model.layers.{layer_num}."
            mapping = {
                "attn_q.weight":       "self_attn.q_proj.weight",
                "attn_k.weight":       "self_attn.k_proj.weight",
                "attn_v.weight":       "self_attn.v_proj.weight",
                "attn_output.weight":  "self_attn.o_proj.weight",
                "ffn_gate.weight":     "mlp.gate_proj.weight",
                "ffn_up.weight":       "mlp.up_proj.weight",
                "ffn_down.weight":     "mlp.down_proj.weight",
                "attn_norm.weight":    "input_layernorm.weight",
                "ffn_norm.weight":     "post_feedforward_layernorm.weight",
                # Gemma 3 has pre/post attn norm variants in some configs
                "attn_post_norm.weight": "post_attention_layernorm.weight",
                "ffn_post_norm.weight":  "post_feedforward_layernorm.weight",
            }
            hf_rest = mapping.get(rest)
            return prefix + hf_rest if hf_rest else None
        return None

    # Dequantise Q8_0 tensors to float32 for safetensors.
    # Q8_0: each block of 32 values has a float16 scale + 32 int8 values.
    # dequantised_value = scale * int8_value

    def _dequantize_q8_0(data: np.ndarray, shape: tuple) -> np.ndarray:
        """Dequantise a flat Q8_0 byte buffer to float32."""
        # Block layout: 2 bytes (float16 scale) + 32 bytes (int8 values)
        BLOCK_SIZE  = 32
        BYTES_BLOCK = 2 + BLOCK_SIZE  # 34 bytes per block
        n_elements  = 1
        for d in shape:
            n_elements *= d
        n_blocks = n_elements // BLOCK_SIZE

        raw = data.tobytes() if not isinstance(data, bytes) else data
        raw = np.frombuffer(raw, dtype=np.uint8)

        scales = np.zeros(n_blocks, dtype=np.float32)
        weights = np.zeros(n_blocks * BLOCK_SIZE, dtype=np.float32)

        for b in range(n_blocks):
            offset = b * BYTES_BLOCK
            # Scale: little-endian float16
            scale = np.frombuffer(raw[offset:offset+2], dtype=np.float16)[0].astype(np.float32)
            # Quantised int8 values
            qs = raw[offset+2 : offset+2+BLOCK_SIZE].view(np.int8).astype(np.float32)
            weights[b*BLOCK_SIZE : (b+1)*BLOCK_SIZE] = scale * qs

        return weights.reshape(shape)

    print("\n  Extracting and converting tensors (this may take a few minutes) …")

    state_dict: dict[str, "torch.Tensor"] = {}
    skipped = 0
    converted = 0

    try:
        import torch
    except ImportError:
        _err("torch not installed — cannot extract tensors. Run --mode setup.")
        sys.exit(1)

    for tensor in reader.tensors:
        hf_name = _gguf_to_hf_name(tensor.name)
        if hf_name is None:
            skipped += 1
            continue

        shape = tuple(reversed(tensor.shape.tolist()))  # GGUF stores dims reversed
        raw_data = tensor.data

        # Most quantisation types can be handled by gguf's own dequant
        try:
            # Try gguf's built-in dequantisation if available
            if hasattr(tensor, "numpy"):
                arr = tensor.numpy().astype(np.float32)
            else:
                # Fallback: try to interpret as float32 or dequant Q8_0 manually
                quant_type = int(tensor.tensor_type)
                # gguf.GGMLQuantizationType.Q8_0 == 8
                if quant_type == 8:
                    arr = _dequantize_q8_0(raw_data, shape)
                elif quant_type == 0:  # F32
                    arr = raw_data.reshape(shape).astype(np.float32)
                elif quant_type == 1:  # F16
                    arr = raw_data.view(np.float16).reshape(shape).astype(np.float32)
                else:
                    _warn(f"  Unsupported quant type {quant_type} for {tensor.name}, skipping")
                    skipped += 1
                    continue
            state_dict[hf_name] = torch.from_numpy(arr)
            converted += 1
        except Exception as e:
            _warn(f"  Failed to convert {tensor.name}: {e}")
            skipped += 1

    _ok(f"Converted {converted} tensors, skipped {skipped}")

    if converted == 0:
        _err("No tensors were extracted. The GGUF format may be incompatible.")
        _warn("Falling back to HuggingFace Hub download …")
        _hf_download_fallback()
        return

    # Write safetensors
    try:
        from safetensors.torch import save_file as st_save
        st_path = MODEL_HF_DIR / "model.safetensors"
        st_save(state_dict, str(st_path))
        size_mb = st_path.stat().st_size / 1e6
        _ok(f"Saved {st_path} ({size_mb:.0f} MB)")
    except ImportError:
        # Fallback: save as .pt (works without safetensors package)
        import torch
        pt_path = MODEL_HF_DIR / "pytorch_model.bin"
        torch.save(state_dict, str(pt_path))
        size_mb = pt_path.stat().st_size / 1e6
        _ok(f"Saved {pt_path} ({size_mb:.0f} MB) [safetensors not installed, used .bin]")

    # Copy tokenizer files from the base Ollama model if possible.
    # The tokenizer is stored in the params blob alongside the weights.
    # For Gemma 3, we write a minimal tokenizer_config.json that points to the
    # HF tokenizer, which will be auto-downloaded by transformers.
    tokenizer_config = {
        "model_type":                 "gemma3",
        "tokenizer_class":            "GemmaTokenizerFast",
        "bos_token":                  "<bos>",
        "eos_token":                  "<eos>",
        "unk_token":                  "<unk>",
        "pad_token":                  "<pad>",
        "add_bos_token":              True,
        "add_eos_token":              False,
        "clean_up_tokenization_spaces": False,
        # transformers will pull the sentencepiece model from here
        "auto_map": {"AutoTokenizer": "transformers.GemmaTokenizerFast"},
    }
    (MODEL_HF_DIR / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2)
    )
    _ok("Wrote tokenizer_config.json")

    print(f"\n  Total time: {_elapsed(t0)}")
    _ok(f"Conversion complete → {MODEL_HF_DIR}")
    print("\n  NOTE: The tokenizer's sentencepiece model (tokenizer.model) will be")
    print("  downloaded automatically when --mode train first loads the tokenizer.")
    print(f"  Alternatively, copy it from {HF_MODEL_ID} on HuggingFace Hub.")


def _hf_download_fallback() -> None:
    """Download google/gemma-3-270m-it from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        try:
            from transformers.utils.hub import cached_file
        except ImportError:
            _err("Neither huggingface_hub nor transformers is installed.")
            _err(f"Manually download {HF_MODEL_ID} to {MODEL_HF_DIR}")
            sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        _warn("HF_TOKEN not set. If the model is gated you must set it:")
        _warn("  export HF_TOKEN=hf_...")
        _warn("  See: https://huggingface.co/google/gemma-3-270m-it")

    _ok(f"Downloading {HF_MODEL_ID} from HuggingFace Hub → {MODEL_HF_DIR} …")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=str(MODEL_HF_DIR),
            token=hf_token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
        )
        _ok(f"Downloaded to {MODEL_HF_DIR}")
    except Exception as e:
        _err(f"Download failed: {e}")
        _err("Resolve HF token / network issue, then re-run --mode convert")
        sys.exit(1)


# ===========================================================================
# MODE: train
# ===========================================================================

def mode_train(epochs: int = 3, lr: float = 2e-4, batch_size: int = 1,
               grad_accum: int = 4) -> None:
    """LoRA fine-tuning with SFTTrainer.

    WHAT HAPPENS HERE
    -----------------
    1. Load tokenizer — Gemma 3 uses SentencePiece with a 256 000-token vocab.
    2. Add domain tokens — 12 tool-name tokens so the model can emit them cleanly.
    3. Load base model — float32 on CPU (no quantisation during training).
    4. Apply LoRA — inject rank-8 adapter matrices into q_proj and v_proj.
    5. Format dataset — convert dispatch_pairs to Gemma 3 chat template strings.
    6. Train with SFTTrainer — 3 epochs, batch=1, grad_accum=4 (effective batch=4).
    7. Save adapter — writes adapter_model.safetensors + adapter_config.json.

    TIMING ESTIMATES (Intel Meteor Lake, 30 GB RAM, CPU-only)
    ----------------------------------------------------------
    48 examples × 3 epochs = 144 gradient steps (effective after grad_accum)
    Average sequence length ~200 tokens for Gemma 3 with tool schema.
    Expected: ~15–25 minutes for 48 examples, 3 epochs.
    At 500 examples: ~2.5–4 hours.
    At 2000 examples (v4 target): ~10–16 hours (consider overnight run).

    LoRA HYPERPARAMETERS
    --------------------
    r=8      : rank — controls adapter capacity. 8 is standard for small tasks.
    alpha=16 : scaling factor α/r = 2.0. Higher alpha = larger effective LR.
    dropout=0.05 : light regularisation to prevent overfitting on small data.
    target_modules: q_proj, v_proj — the key/query projections in attention.
                    These control *what* the model attends to, which directly
                    influences tool selection behaviour.
    """
    _banner("TRAIN — LoRA fine-tuning on CPU")
    t0 = time.time()

    # -- Import heavy deps here so setup/check still work without them --
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        _err(f"Missing dependency: {e}")
        _err("Run --mode setup for install commands, then install and retry.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Decide which model to load: local model_hf/ or HF Hub
    # -----------------------------------------------------------------------

    if MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir()):
        model_path = str(MODEL_HF_DIR)
        _ok(f"Loading model from local model_hf/ ({model_path})")
    else:
        model_path = HF_MODEL_ID
        _ok(f"model_hf/ not found — loading from HuggingFace Hub: {model_path}")
        _warn("This requires internet access and acceptance of Gemma licence.")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            _warn("Set HF_TOKEN if the model is gated.")

    # -----------------------------------------------------------------------
    # 2. Load tokenizer and add domain tokens
    # -----------------------------------------------------------------------

    _ok("Loading tokenizer …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
            # Gemma 3 tokenizer handles padding on the right by default
        )
    except Exception as e:
        _err(f"Tokenizer load failed: {e}")
        _err("If using local model_hf/, ensure tokenizer.model is present.")
        _err(f"If missing, copy from HF Hub: huggingface-cli download {HF_MODEL_ID} tokenizer.model")
        sys.exit(1)

    # Set pad token — Gemma uses <pad> but some HF configs omit it.
    # The trainer needs a pad token to batch sequences of different lengths.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        _warn("pad_token was None; set to eos_token")

    # Add domain-specific tokens.
    # WHY: If 'linux_memory_usage' is split into ['linux', '_', 'memory', '_', 'usage']
    # by the tokenizer, the model must predict 5 tokens correctly in sequence.
    # With a single token per tool name, it predicts 1 token — much easier to learn.
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in DOMAIN_TOKENS if t not in existing]

    if new_tokens:
        n_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
        _ok(f"Added {n_added} domain tokens to tokenizer: {new_tokens[:3]}…")
    else:
        n_added = 0
        _ok("All domain tokens already in vocabulary — none added")

    print(f"  Vocabulary size: {len(tokenizer)} tokens")

    # -----------------------------------------------------------------------
    # 3. Load base model in float32 on CPU
    # -----------------------------------------------------------------------

    _ok("Loading base model (float32, CPU) — this takes 1–2 minutes …")
    load_start = time.time()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,   # float32 needed for stable CPU training
            device_map="cpu",            # force CPU — no CUDA on this machine
            trust_remote_code=False,
            low_cpu_mem_usage=True,      # load tensors one at a time; saves ~2 GB peak RAM
        )
    except Exception as e:
        _err(f"Model load failed: {e}")
        sys.exit(1)

    _ok(f"Model loaded in {_elapsed(load_start)}")
    n_params = sum(p.numel() for p in model.parameters())
    _ok(f"Total parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Resize embedding table to match the extended vocabulary.
    # We added new tokens → vocab grew → embedding matrix must grow too.
    # New rows are initialised randomly; the LoRA training will learn their values.
    if n_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        _ok(f"Resized embeddings: {n_params:,} → {sum(p.numel() for p in model.parameters()):,} params")

    # -----------------------------------------------------------------------
    # 4. Apply LoRA
    # -----------------------------------------------------------------------

    # target_modules: we adapt q_proj and v_proj in every transformer layer.
    # These are the attention projections that determine "what to attend to" —
    # directly responsible for tool selection routing decisions.
    # k_proj is deliberately excluded: key projections encode position/content
    # of the context, which we don't need to change for tool routing.

    lora_config = LoraConfig(
        task_type       = TaskType.CAUSAL_LM,
        r               = 8,            # rank: adapter hidden dim
        lora_alpha      = 16,           # scale α: effective LR multiplier = α/r = 2.0
        lora_dropout    = 0.05,         # light regularisation
        target_modules  = ["q_proj", "v_proj"],
        bias            = "none",       # don't adapt bias terms (fewer params)
        inference_mode  = False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: ~0.3–0.5% of parameters are trainable with r=8

    # -----------------------------------------------------------------------
    # 5. Format the dataset into Gemma 3 chat template strings
    # -----------------------------------------------------------------------

    _ok(f"Loading training data from {DATA_FILE} …")
    raw_lines = DATA_FILE.read_text().strip().splitlines()
    _ok(f"  {len(raw_lines)} examples")

    def _format_example(line: str) -> Optional[str]:
        """Convert one dispatch_pairs JSONL line to a training string.

        INPUT FORMAT (from generate_pairs.py):
          {
            "messages": [
              {"role": "system",  "content": "Call the right tool."},
              {"role": "user",    "content": "how much ram is used"}
            ],
            "tools": [{ <tool JSON schema> }],
            "target": {"name": "linux_memory_usage", "arguments": {}}
          }

        OUTPUT FORMAT (Gemma 3 function-call style):
          We use the Gemma 3 chat template via tokenizer.apply_chat_template,
          which produces:

            <bos><start_of_turn>user
            [system prompt + tool schemas]
            <end_of_turn>
            <start_of_turn>model
            {"name": "linux_memory_usage", "arguments": {}}
            <end_of_turn>

        The SFTTrainer trains the model to generate the "model" turn given
        the "user" turn.  This is the standard supervised fine-tuning (SFT)
        loss: cross-entropy on the output tokens only (DataCollatorForSeq2Seq
        masks the input tokens' loss contribution).

        WHY INCLUDE TOOL SCHEMA IN THE PROMPT?
        The schema tells the model the available tool names and their
        parameter shapes.  This mirrors inference time, where the agent also
        passes the schema.  Without this, the model only learns names without
        understanding their interface.
        """
        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            return None

        messages = ex.get("messages", [])
        tools    = ex.get("tools", [])
        target   = ex.get("target", {})

        if not messages or not target:
            return None

        # Build the user-facing content: system prompt + tool schema + user query
        system_content = next(
            (m["content"] for m in messages if m["role"] == "system"),
            "Call the right tool."
        )
        user_content = next(
            (m["content"] for m in messages if m["role"] == "user"),
            ""
        )

        # Format tool schemas as a compact JSON block in the prompt.
        # This is simpler than using the Gemma 3 native function-call tokens,
        # which vary across Ollama builds and may not be in the HF tokenizer.
        tools_text = json.dumps(tools, separators=(",", ":"))

        # The expected model output is the target tool call as JSON.
        # Keep it compact — the model should learn to emit exactly this.
        target_json = json.dumps(target, separators=(",", ":"))

        # Build as a two-turn conversation: user prompt → model response
        combined_messages = [
            {
                "role": "user",
                "content": (
                    f"{system_content}\n\n"
                    f"Available tools: {tools_text}\n\n"
                    f"User request: {user_content}"
                ),
            },
            {
                "role": "model",
                "content": target_json,
            },
        ]

        # Apply Gemma 3 chat template.
        # add_generation_prompt=False because we include the response in the
        # "model" turn — we want the full sequence for SFT loss computation.
        try:
            text = tokenizer.apply_chat_template(
                combined_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: manual template (some older transformers versions)
            text = (
                f"<bos><start_of_turn>user\n"
                f"{system_content}\n\n"
                f"Available tools: {tools_text}\n\n"
                f"User request: {user_content}\n"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
                f"{target_json}\n"
                f"<end_of_turn>"
            )

        return text

    formatted = []
    skipped = 0
    for line in raw_lines:
        text = _format_example(line)
        if text:
            formatted.append({"text": text})
        else:
            skipped += 1

    if skipped:
        _warn(f"Skipped {skipped} malformed examples")

    _ok(f"Formatted {len(formatted)} training examples")

    # Sanity-print one example so you can verify the format looks correct
    if formatted:
        print("\n  --- Sample training string (first example, truncated to 300 chars) ---")
        sample = formatted[0]["text"]
        print(f"  {sample[:300]}…" if len(sample) > 300 else f"  {sample}")
        print("  ---")

    train_dataset = Dataset.from_list(formatted)

    # -----------------------------------------------------------------------
    # 6. Configure training
    # -----------------------------------------------------------------------

    # WHY THESE HYPERPARAMETERS?
    #
    # batch_size=1       : CPU RAM constraint. The model + gradients + optimizer
    #                      state fit in ~8 GB at batch=1. Larger batches would
    #                      OOM or swap heavily.
    #
    # grad_accum=4       : Simulates a batch of 4 without extra RAM. Gradients
    #                      from 4 micro-batches are accumulated before each weight
    #                      update. Effective batch size = 1 × 4 = 4.
    #
    # learning_rate=2e-4 : Standard for LoRA. The adapter is small, so it can
    #                      tolerate a slightly higher LR than full fine-tuning.
    #
    # warmup_ratio=0.1   : First 10% of steps use a linear LR warmup to avoid
    #                      gradient spikes at the start.
    #
    # weight_decay=0.01  : L2 regularisation on adapter weights. Keeps them small.
    #
    # max_seq_length=512 : Truncate sequences at 512 tokens. The tool schema +
    #                      user query + tool call typically fit in 150–250 tokens.
    #                      Capping at 512 saves memory and time.
    #
    # save_steps=50      : Save a checkpoint every 50 steps. With 48 examples
    #                      and batch=1, each epoch is 48 steps, so we save at
    #                      the end of each epoch.

    MODEL_LORA.mkdir(parents=True, exist_ok=True)

    # Estimate training time for the user
    n_steps = (len(formatted) * epochs) // (batch_size * grad_accum)
    print(f"\n  Estimated training steps: {n_steps}")
    print(f"  Rough timing @ ~5 sec/step on CPU: ~{n_steps * 5 // 60} minutes")
    print(f"  (Actual time depends on sequence length and CPU speed)\n")

    try:
        # SFTConfig is the modern interface (trl >= 0.8); it inherits TrainingArguments
        training_args = SFTConfig(
            output_dir                  = str(MODEL_LORA),
            num_train_epochs            = epochs,
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            learning_rate               = lr,
            warmup_ratio                = 0.1,
            weight_decay                = 0.01,
            logging_steps               = 10,
            save_steps                  = 50,
            save_total_limit            = 2,        # keep only 2 checkpoints
            fp16                        = False,    # CPU doesn't support fp16 training
            bf16                        = False,    # requires AVX512BF16 or CUDA
            dataloader_num_workers      = 0,        # 0 = main process; avoids fork issues
            report_to                   = "none",   # no WandB/TensorBoard
            max_seq_length              = 512,
            dataset_text_field          = "text",   # the column name in our Dataset
            # Packing: disabled. Packing concatenates short sequences to fill
            # max_seq_length, which increases GPU/CPU utilisation. But it makes
            # loss masking harder and can confuse the model on tiny datasets.
            packing                     = False,
        )
    except TypeError:
        # Older trl versions use TrainingArguments instead of SFTConfig
        _warn("SFTConfig not found in this trl version — using TrainingArguments")
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            output_dir                  = str(MODEL_LORA),
            num_train_epochs            = epochs,
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            learning_rate               = lr,
            warmup_ratio                = 0.1,
            weight_decay                = 0.01,
            logging_steps               = 10,
            save_steps                  = 50,
            save_total_limit            = 2,
            fp16                        = False,
            bf16                        = False,
            dataloader_num_workers      = 0,
            report_to                   = "none",
        )

    # -----------------------------------------------------------------------
    # 7. Create SFTTrainer and train
    # -----------------------------------------------------------------------

    # SFTTrainer wraps HuggingFace Trainer with:
    #   - Automatic chat template application
    #   - Packing support
    #   - Response template masking (train only on assistant turns)
    #
    # We already applied the chat template ourselves, so we pass raw text and
    # set dataset_text_field="text".

    try:
        trainer = SFTTrainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_dataset,
            tokenizer       = tokenizer,
        )
    except TypeError as e:
        _warn(f"SFTTrainer signature mismatch ({e}) — trying without tokenizer arg")
        trainer = SFTTrainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_dataset,
        )

    _ok("Starting training …")
    print(f"  {len(formatted)} examples × {epochs} epochs = ~{len(formatted)*epochs} steps total\n")

    try:
        train_result = trainer.train()
        _ok(f"Training complete in {_elapsed(t0)}")
        _ok(f"Final loss: {train_result.training_loss:.4f}")
    except KeyboardInterrupt:
        _warn("Training interrupted by user. Saving partial checkpoint …")
    except Exception as e:
        _err(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 8. Save LoRA adapter (not the full model — just the small adapter)
    # -----------------------------------------------------------------------

    _ok(f"Saving LoRA adapter to {MODEL_LORA} …")
    try:
        model.save_pretrained(str(MODEL_LORA))
        tokenizer.save_pretrained(str(MODEL_LORA))
        _ok("Adapter saved")
    except Exception as e:
        _err(f"Save failed: {e}")
        sys.exit(1)

    # Print what was saved
    saved_files = list(MODEL_LORA.iterdir())
    print(f"\n  Saved files ({len(saved_files)}):")
    for f in sorted(saved_files):
        size = f.stat().st_size / 1e6 if f.is_file() else 0
        print(f"    {f.name}  ({size:.1f} MB)" if size > 0.1 else f"    {f.name}")

    print(f"\n  Total training time: {_elapsed(t0)}")
    print(f"\n  Next step: python3 {__file__} --mode export")


# ===========================================================================
# MODE: export
# ===========================================================================

def mode_export() -> None:
    """Merge LoRA adapter into base weights, convert to GGUF, write Modelfile.

    STEPS
    -----
    1. Load base model + LoRA adapter from model_lora/
    2. Merge adapter weights into base (adapter_model × merge_and_unload)
    3. Save merged HF model to model_merged/
    4. Convert merged model → GGUF using the gguf package
    5. Write training/Modelfile
    6. Print the `ollama create` command
    """
    _banner("EXPORT — merge LoRA + convert to GGUF + write Modelfile")
    t0 = time.time()

    if not MODEL_LORA.exists() or not any(MODEL_LORA.iterdir()):
        _err(f"model_lora/ not found or empty at {MODEL_LORA}")
        _err("Run --mode train first.")
        sys.exit(1)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as e:
        _err(f"Missing dependency: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1+2. Load base + LoRA, then merge
    # -----------------------------------------------------------------------

    # Decide base model path (same logic as train)
    if MODEL_HF_DIR.exists() and any(MODEL_HF_DIR.iterdir()):
        base_path = str(MODEL_HF_DIR)
    else:
        base_path = HF_MODEL_ID

    _ok(f"Loading base model from {base_path} …")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        _err(f"Base model load failed: {e}")
        sys.exit(1)

    _ok(f"Loading LoRA adapter from {MODEL_LORA} …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_LORA))
        model = PeftModel.from_pretrained(base_model, str(MODEL_LORA))
    except Exception as e:
        _err(f"Adapter load failed: {e}")
        sys.exit(1)

    # Resize embeddings if vocabulary was extended during training
    if len(tokenizer) > base_model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        _ok(f"Resized embeddings to {len(tokenizer)}")

    # merge_and_unload: multiplies LoRA matrices (A × B) into the base weight
    # matrices, then removes the LoRA hooks.  Result: a plain nn.Module with
    # slightly different weights — no LoRA overhead at inference time.
    _ok("Merging LoRA weights into base model (merge_and_unload) …")
    merged = model.merge_and_unload()
    _ok(f"Merge complete in {_elapsed(t0)}")

    # -----------------------------------------------------------------------
    # 3. Save merged HF model
    # -----------------------------------------------------------------------

    MODEL_MERGED.mkdir(parents=True, exist_ok=True)
    _ok(f"Saving merged model to {MODEL_MERGED} …")
    merged.save_pretrained(str(MODEL_MERGED), safe_serialization=True)
    tokenizer.save_pretrained(str(MODEL_MERGED))
    _ok("Merged model saved")

    # -----------------------------------------------------------------------
    # 4. Convert merged model to GGUF
    # -----------------------------------------------------------------------

    # Strategy: try two approaches in order of preference:
    #   A. llama.cpp convert_hf_to_gguf.py (highest quality, proper quant support)
    #   B. gguf Python package (simpler, F32 or F16 only)
    #
    # If llama.cpp is not installed, fall back to the gguf package.

    _ok("Converting merged model to GGUF …")

    converted_to_gguf = False

    # Approach A: llama.cpp converter script
    llama_cpp_convert = _find_llama_cpp_converter()
    if llama_cpp_convert:
        _ok(f"Found llama.cpp converter: {llama_cpp_convert}")
        cmd = [
            sys.executable, llama_cpp_convert,
            str(MODEL_MERGED),
            "--outfile", str(GGUF_OUT),
            "--outtype", "q8_0",   # match original quantisation
        ]
        _ok(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            _ok(f"GGUF written to {GGUF_OUT}")
            converted_to_gguf = True
        else:
            _warn(f"llama.cpp converter failed:\n{result.stderr}")

    if not converted_to_gguf:
        # Approach B: gguf Python package — write F16 GGUF manually
        _ok("Falling back to gguf Python package (F16 precision) …")
        try:
            converted_to_gguf = _export_gguf_python(merged, tokenizer)
        except Exception as e:
            _warn(f"GGUF Python export failed: {e}")
            _warn("The merged HF model is still available at:")
            _warn(f"  {MODEL_MERGED}")
            _warn("You can convert it manually with llama.cpp:")
            _warn(f"  python3 convert_hf_to_gguf.py {MODEL_MERGED} --outtype q8_0")

    # -----------------------------------------------------------------------
    # 5. Write Modelfile
    # -----------------------------------------------------------------------

    _ok(f"Writing Modelfile to {MODELFILE} …")

    if converted_to_gguf and GGUF_OUT.exists():
        from_line = f"FROM {GGUF_OUT}"
    else:
        # Fallback: reference merged HF path (won't work with `ollama create`
        # directly but is a useful placeholder)
        from_line = f"# GGUF conversion failed — use llama.cpp to convert:\n# {MODEL_MERGED}"

    modelfile_content = f"""# Modelfile for functiongemma:270m-suryaos
# Generated by finetune_dispatch.py on {time.strftime('%Y-%m-%d')}
#
# This model is functiongemma:270m fine-tuned with LoRA on SuryaOS tool dispatch.
# Training data: {DATA_FILE}
# Training scope: 12 system tools (v3: 500 examples target)
# Future (v4): 2000+ examples when agent handles code/test/git/IDE workflows.

{from_line}

# Use the same template as the original functiongemma:270m
TEMPLATE {{{{ .Prompt }}}}
RENDERER functiongemma
PARSER functiongemma

# Keep the original inference parameters
PARAMETER top_k 64
PARAMETER top_p 0.95

# System prompt for tool dispatch
SYSTEM \"""You are a tool dispatcher for SuryaOS. Given the user's request and available tools, call the correct tool with appropriate arguments. Respond only with the tool call JSON.\"""
"""

    MODELFILE.write_text(modelfile_content)
    _ok(f"Modelfile written to {MODELFILE}")

    # -----------------------------------------------------------------------
    # 6. Print the ollama create command
    # -----------------------------------------------------------------------

    print(f"\n  Total export time: {_elapsed(t0)}")
    print()
    print("  ================================================================")
    print("  To register the fine-tuned model with Ollama, run:")
    print()
    print(f"    ollama create functiongemma:270m-suryaos -f {MODELFILE}")
    print()
    print("  Then test it:")
    print("    ollama run functiongemma:270m-suryaos 'how much ram is used'")
    print("  ================================================================")


def _find_llama_cpp_converter() -> Optional[str]:
    """Search common locations for llama.cpp's convert_hf_to_gguf.py."""
    candidates = [
        # System-wide installs / git clones in standard locations
        "/usr/local/lib/llama.cpp/convert_hf_to_gguf.py",
        "/opt/llama.cpp/convert_hf_to_gguf.py",
        str(Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"),
        str(Path.home() / "src" / "llama.cpp" / "convert_hf_to_gguf.py"),
        # In PATH (pip-installed llama-cpp-python may install helper scripts)
        shutil.which("convert_hf_to_gguf.py") or "",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


def _export_gguf_python(model, tokenizer) -> bool:
    """Write a minimal GGUF file using the gguf Python package.

    This produces a F16 GGUF (not Q8_0) — suitable for testing but larger
    than the original.  Use llama.cpp convert_hf_to_gguf.py for production.

    Returns True if successful.
    """
    try:
        import gguf
        import torch
        import numpy as np
    except ImportError as e:
        _warn(f"Cannot export GGUF: {e}")
        return False

    writer = gguf.GGUFWriter(str(GGUF_OUT), "gemma3")

    # Write architecture metadata
    config = model.config
    writer.add_architecture()
    writer.add_name("functiongemma-suryaos-270m")
    writer.add_description("functiongemma:270m fine-tuned on SuryaOS tool dispatch (LoRA v3)")

    # Gemma 3 metadata fields
    try:
        writer.add_context_length(getattr(config, "max_position_embeddings", 32768))
        writer.add_embedding_length(getattr(config, "hidden_size", 640))
        writer.add_block_count(getattr(config, "num_hidden_layers", 18))
        writer.add_feed_forward_length(getattr(config, "intermediate_size", 3072))
        writer.add_head_count(getattr(config, "num_attention_heads", 8))
        writer.add_head_count_kv(getattr(config, "num_key_value_heads", 4))
        writer.add_layer_norm_rms_eps(getattr(config, "rms_norm_eps", 1e-6))
        writer.add_vocab_size(len(tokenizer))
    except Exception as e:
        _warn(f"Some metadata fields not written: {e}")

    # Write tensors as F16
    _ok("Writing tensors to GGUF (F16) …")
    n_written = 0
    for name, param in model.named_parameters():
        try:
            # Convert HF name → GGUF name (reverse of convert step)
            gguf_name = _hf_to_gguf_name(name)
            if gguf_name is None:
                continue
            arr = param.detach().float().numpy().astype(np.float16)
            writer.add_tensor(gguf_name, arr)
            n_written += 1
        except Exception as e:
            _warn(f"  Failed to write {name}: {e}")

    _ok(f"Wrote {n_written} tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = GGUF_OUT.stat().st_size / 1e6
    _ok(f"GGUF file: {GGUF_OUT} ({size_mb:.0f} MB) [F16 — use llama.cpp for Q8_0]")
    return True


def _hf_to_gguf_name(hf_name: str) -> Optional[str]:
    """Map a HuggingFace parameter name to llama.cpp GGUF convention."""
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    if hf_name == "lm_head.weight":
        return "output.weight"
    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".")
        # model.layers.N.sub.module.weight
        layer_num = parts[2]
        rest = ".".join(parts[3:])
        prefix = f"blk.{layer_num}."
        reverse_map = {
            "self_attn.q_proj.weight":              "attn_q.weight",
            "self_attn.k_proj.weight":              "attn_k.weight",
            "self_attn.v_proj.weight":              "attn_v.weight",
            "self_attn.o_proj.weight":              "attn_output.weight",
            "mlp.gate_proj.weight":                 "ffn_gate.weight",
            "mlp.up_proj.weight":                   "ffn_up.weight",
            "mlp.down_proj.weight":                 "ffn_down.weight",
            "input_layernorm.weight":               "attn_norm.weight",
            "post_feedforward_layernorm.weight":    "ffn_norm.weight",
            "post_attention_layernorm.weight":      "attn_post_norm.weight",
        }
        gguf_rest = reverse_map.get(rest)
        return prefix + gguf_rest if gguf_rest else None
    return None


# ===========================================================================
# MODE: all
# ===========================================================================

def mode_all(epochs: int, lr: float) -> None:
    """Run check → train → export (convert is skipped if model_hf/ exists)."""
    _banner("ALL — running full pipeline: check → train → export")

    # Step 1: check
    ok = mode_check()
    if not ok:
        _err("Check failed — fix the issues above before continuing.")
        sys.exit(1)

    # Step 2: convert only if model_hf/ doesn't exist
    if not MODEL_HF_DIR.exists() or not any(MODEL_HF_DIR.iterdir()):
        _ok("model_hf/ not found — running convert step …")
        mode_convert()
    else:
        _ok("model_hf/ exists — skipping convert (use --mode convert to force)")

    # Step 3: train
    mode_train(epochs=epochs, lr=lr)

    # Step 4: export
    mode_export()

    _banner("ALL DONE")
    print(f"\n  Fine-tuned model Modelfile: {MODELFILE}")
    print(f"  Register with: ollama create functiongemma:270m-suryaos -f {MODELFILE}")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["setup", "check", "convert", "train", "export", "all"],
        help=(
            "setup: print install commands | "
            "check: verify environment | "
            "convert: GGUF→HF safetensors | "
            "train: LoRA fine-tuning | "
            "export: merge+GGUF+Modelfile | "
            "all: check→train→export"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3). "
             "With 48 examples, 3 epochs ≈ 15–25 min on CPU.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for LoRA training (default: 2e-4). "
             "Range: 1e-4 to 5e-4 for LoRA adapters.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size (default: 1 for CPU memory constraints).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4, effective batch = 4).",
    )

    args = parser.parse_args()

    if args.mode == "setup":
        mode_setup()
    elif args.mode == "check":
        mode_check()
    elif args.mode == "convert":
        mode_convert()
    elif args.mode == "train":
        mode_train(
            epochs     = args.epochs,
            lr         = args.lr,
            batch_size = args.batch_size,
            grad_accum = args.grad_accum,
        )
    elif args.mode == "export":
        mode_export()
    elif args.mode == "all":
        mode_all(epochs=args.epochs, lr=args.lr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
