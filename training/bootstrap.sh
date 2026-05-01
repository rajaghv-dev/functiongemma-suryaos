#!/usr/bin/env bash
# bootstrap.sh — create the Python venv and install all training dependencies.
#
# Supports:
#   RTX 30xx/40xx (Ampere/Ada)   — CUDA 12.x, fp16 training
#   NVIDIA DGX / A100 / H100     — CUDA 12.x, bf16 training
#   NVIDIA V100 / older          — CUDA 11.x, fp16 training
#   CPU-only (no GPU)            — float32 training (slow)
#
# Usage (run from repo root):
#   bash training/bootstrap.sh
#
# After this completes:
#   source .fngemma-suryaos/bin/activate
#   python training/train_tokenizer.py
#   python training/finetune.py --mode all

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.fngemma-suryaos"
REQ="$REPO_ROOT/training/requirements.txt"

# ── 1. Detect GPU / CUDA ───────────────────────────────────────────────────
echo ""
echo "[bootstrap] Detecting hardware ..."

TORCH_INDEX="https://download.pytorch.org/whl/cpu"
TORCH_LABEL="cpu-only"
HAS_GPU=false

if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name         --format=csv,noheader,nounits | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

    # Prefer nvcc for the toolkit version; fall back to nvidia-smi header
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
    else
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1 || echo "0.0")
    fi

    CUDA_MAJOR="${CUDA_VER%%.*}"

    echo "[bootstrap]   GPU:    $GPU_NAME"
    echo "[bootstrap]   VRAM:   ${GPU_VRAM} MiB"
    echo "[bootstrap]   Driver: $GPU_DRIVER"
    echo "[bootstrap]   CUDA:   $CUDA_VER (toolkit)"

    if   [ "${CUDA_MAJOR:-0}" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_LABEL="cu121 (CUDA 12.x)"
    elif [ "${CUDA_MAJOR:-0}" -ge 11 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_LABEL="cu118 (CUDA 11.x)"
    else
        echo "[bootstrap]   WARN: CUDA toolkit < 11 — falling back to CPU build"
        echo "[bootstrap]         Update CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
    fi

    HAS_GPU=true
else
    echo "[bootstrap]   No NVIDIA GPU detected — CPU-only build"
fi

echo "[bootstrap]   PyTorch wheel: $TORCH_LABEL"
echo "[bootstrap]   Index URL:     $TORCH_INDEX"
echo ""

# ── 2. Create venv if missing ──────────────────────────────────────────────
if [ ! -f "$VENV/bin/python3" ]; then
    echo "[bootstrap] Creating virtual environment at $VENV ..."
    python3 -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python3"

# ── 3. Upgrade pip ─────────────────────────────────────────────────────────
echo "[bootstrap] Upgrading pip ..."
"$PIP" install --upgrade pip --quiet

# ── 4. Install PyTorch (GPU or CPU wheel) ─────────────────────────────────
echo "[bootstrap] Installing PyTorch ($TORCH_LABEL) ..."
echo "            This downloads ~600 MB for GPU builds or ~200 MB for CPU."
"$PIP" install torch torchvision torchaudio --index-url "$TORCH_INDEX"

# ── 5. Install remaining requirements ─────────────────────────────────────
echo ""
echo "[bootstrap] Installing remaining requirements from $REQ ..."
"$PIP" install -r "$REQ"

# ── 6. Optional: bitsandbytes for QLoRA (GPU only) ────────────────────────
if [ "$HAS_GPU" = true ]; then
    echo ""
    echo "[bootstrap] Installing bitsandbytes (4-bit QLoRA support) ..."
    if "$PIP" install "bitsandbytes>=0.43.0" 2>/dev/null; then
        echo "[bootstrap]   bitsandbytes installed — QLoRA available"
    else
        echo "[bootstrap]   WARN: bitsandbytes failed to install (QLoRA disabled, regular LoRA still works)"
    fi
fi

# ── 7. Verify ──────────────────────────────────────────────────────────────
echo ""
echo "[bootstrap] Verifying installation ..."

_verify() {
    local label="$1"; local expr="$2"
    if result=$("$PYTHON" -c "$expr" 2>/dev/null); then
        printf "  [OK]   %s\n" "$result"
    else
        printf "  [FAIL] %s — check pip output above\n" "$label"
    fi
}

_verify "torch"         "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
_verify "GPU device"    "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
_verify "GPU VRAM"      "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM' if torch.cuda.is_available() else 'N/A')"
_verify "transformers"  "import transformers; print('transformers', transformers.__version__)"
_verify "peft"          "import peft;         print('peft', peft.__version__)"
_verify "trl"           "import trl;          print('trl', trl.__version__)"
_verify "datasets"      "import datasets;     print('datasets', datasets.__version__)"
_verify "accelerate"    "import accelerate;   print('accelerate', accelerate.__version__)"
_verify "sentencepiece" "import sentencepiece; print('sentencepiece OK')"
_verify "safetensors"   "import safetensors;  print('safetensors', safetensors.__version__)"
_verify "gguf"          "import gguf;         print('gguf OK')"
_verify "psutil"        "import psutil;       print('psutil', psutil.__version__)"
if [ "$HAS_GPU" = true ]; then
    _verify "bitsandbytes"  "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__, '(QLoRA ready)')"
fi

# ── 8. Print next steps ────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "  Bootstrap complete."
if [ "$HAS_GPU" = true ]; then
echo ""
echo "  GPU training enabled:"
echo "    $GPU_NAME / ${GPU_VRAM} MiB / $TORCH_LABEL"
fi
echo ""
echo "  Run in order:"
echo "    $PYTHON training/train_tokenizer.py         # extend + warm-up tokenizer"
echo "    $PYTHON training/finetune.py --mode check   # verify environment"
echo "    $PYTHON training/finetune.py --mode all     # full training pipeline"
echo ""
echo "  Or activate the venv first:"
echo "    source $VENV/bin/activate"
echo "    python training/finetune.py --mode all"
echo "========================================================================"
