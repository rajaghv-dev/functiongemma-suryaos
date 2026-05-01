#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — Idempotent environment + observability setup for training.
#
# IDEMPOTENT: every step checks current state and skips if already satisfied.
# Re-running this script after a successful run is a no-op (just verifies).
#
# What this script does, in order:
#   1. Detect GPU (nvidia-smi) and CUDA toolkit version (nvcc)
#   2. Choose the correct PyTorch wheel (cu121, cu118, or cpu-only)
#   3. Create the Python virtual environment (.fngemma-suryaos/) if missing
#   4. Install PyTorch IF not already installed with right CUDA support
#   5. Install requirements.txt IF any required package is missing
#   6. Install bitsandbytes for QLoRA (GPU only) IF not already installed
#   7. Verify every critical import and print versions
#   8. Bring up the observability stack (Grafana + Loki + Prometheus)
#      IF Docker is available and stack is not already running
#
# Supported hardware:
#   RTX 30xx / 40xx (Ampere / Ada)    CUDA 12.x → cu121 wheel, fp16 training
#   NVIDIA DGX / A100 / H100          CUDA 12.x → cu121 wheel, bf16 training
#   RTX 20xx / Tesla T4 / V100        CUDA 11.x → cu118 wheel, fp16 training
#   CPU-only (no GPU)                 cpu wheel, float32 training (slow)
#
# Usage (run from repo root):
#   bash training/bootstrap.sh                  # default: install + start dashboard
#   bash training/bootstrap.sh --no-dashboard   # skip docker compose step
#   bash training/bootstrap.sh --reinstall      # force reinstall all packages
#   bash training/bootstrap.sh --dashboard-only # skip Python deps; just start stack
#
# After bootstrap completes, run in order:
#   .fngemma-suryaos/bin/python training/train_tokenizer.py
#   .fngemma-suryaos/bin/python training/finetune.py --mode check
#   .fngemma-suryaos/bin/python training/finetune.py --mode all
# =============================================================================

set -euo pipefail  # -e: exit on error  -u: error on undefined vars  -o pipefail: pipe errors propagate

# ── Parse flags ─────────────────────────────────────────────────────────────
WITH_DASHBOARD=true   # default: also bring up the Grafana stack
FORCE_REINSTALL=false # default: skip install if package already present
SKIP_PY_DEPS=false    # default: install Python deps

for arg in "$@"; do
    case "$arg" in
        --no-dashboard)    WITH_DASHBOARD=false ;;
        --dashboard-only)  SKIP_PY_DEPS=true ;;
        --reinstall)       FORCE_REINSTALL=true ;;
        -h|--help)
            sed -n '2,/^# ===/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg" >&2
            echo "Use --help for usage" >&2
            exit 2
            ;;
    esac
done

# Resolve paths relative to this script, regardless of working directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.fngemma-suryaos"          # virtual environment directory
REQ="$REPO_ROOT/training/requirements.txt"  # non-torch dependencies
OBS_DIR="$REPO_ROOT/training/observability" # docker compose stack location

# =============================================================================
# STEP 1 — Detect GPU and choose the right PyTorch wheel
# =============================================================================
#
# Why we need to detect before installing:
#   PyTorch ships separate wheels for each CUDA version. If you install the
#   CPU wheel on a GPU machine, torch.cuda.is_available() returns False and
#   all training runs on CPU (10-100x slower). There is no warning.
#
# We use two sources to determine the CUDA version:
#   nvcc --version  → reports the CUDA Toolkit installed on the system
#                     (the version used to compile CUDA code)
#   nvidia-smi      → reports the driver's maximum supported CUDA version
#                     (always >= nvcc version because drivers are forward-compatible)
#
# We use the nvcc version because that's what the PyTorch wheel was compiled
# against. The driver version from nvidia-smi can be misleadingly high.
#
# PyTorch wheel index URLs:
#   cu121 → https://download.pytorch.org/whl/cu121  (CUDA 12.x)
#   cu118 → https://download.pytorch.org/whl/cu118  (CUDA 11.x)
#   cpu   → https://download.pytorch.org/whl/cpu    (no GPU)
# =============================================================================

echo ""
echo "[bootstrap] ============================================================"
echo "[bootstrap] functiongemma-suryaos environment setup"
echo "[bootstrap] ============================================================"
echo ""
echo "[bootstrap] STEP 1: Detecting hardware ..."

# Default: assume CPU-only until we find a GPU
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
TORCH_LABEL="cpu-only (no GPU detected)"
HAS_GPU=false

# nvidia-smi must be present AND able to query GPU info (fails gracefully in VMs)
if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null 2>&1; then

    # Read GPU properties from nvidia-smi structured query (CSV, no header, no units)
    GPU_NAME=$(nvidia-smi --query-gpu=name         --format=csv,noheader,nounits | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

    # Prefer nvcc for toolkit version — it's the authoritative build-time CUDA version.
    # Fall back to parsing nvidia-smi's header line if nvcc is not in PATH.
    if command -v nvcc &>/dev/null; then
        # nvcc --version output contains a line like: "Cuda compilation tools, release 12.0, V12.0.140"
        CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
    else
        # nvidia-smi header line contains: "CUDA Version: 12.0"
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1 || echo "0.0")
    fi

    CUDA_MAJOR="${CUDA_VER%%.*}"  # extract major version: "12.0" → "12"

    echo "[bootstrap]   GPU:     $GPU_NAME"
    echo "[bootstrap]   VRAM:    ${GPU_VRAM} MiB"
    echo "[bootstrap]   Driver:  $GPU_DRIVER"
    echo "[bootstrap]   CUDA:    $CUDA_VER (from nvcc toolkit)"

    # Map CUDA major version to PyTorch wheel index:
    #   CUDA 12.x → cu121 (PyTorch supports cu121 for all CUDA 12 minor versions)
    #   CUDA 11.x → cu118 (last CUDA 11 wheel is cu118)
    #   CUDA < 11 → warn and fall back to CPU (very old driver; upgrade recommended)
    if   [ "${CUDA_MAJOR:-0}" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_LABEL="cu121 — CUDA 12.x (RTX 30xx/40xx, A100, H100)"
    elif [ "${CUDA_MAJOR:-0}" -ge 11 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_LABEL="cu118 — CUDA 11.x (RTX 20xx, Tesla T4, V100)"
    else
        echo "[bootstrap]   WARN: CUDA toolkit version ${CUDA_VER} is below 11.0"
        echo "[bootstrap]         Falling back to CPU build. Update CUDA Toolkit:"
        echo "[bootstrap]         https://developer.nvidia.com/cuda-downloads"
    fi

    HAS_GPU=true
else
    echo "[bootstrap]   No NVIDIA GPU detected — using CPU-only PyTorch"
    echo "[bootstrap]   (Training will work but is 10-100x slower than GPU)"
fi

echo "[bootstrap]   PyTorch build: $TORCH_LABEL"
echo "[bootstrap]   Index URL:     $TORCH_INDEX"
echo ""

# =============================================================================
# STEP 2 — Create the Python virtual environment if it doesn't exist
# =============================================================================
#
# Why a virtual environment?
#   Keeps all training dependencies isolated from the system Python.
#   You can delete .fngemma-suryaos/ and re-run bootstrap.sh to get a
#   clean environment without touching any system packages.
#
# The venv is at the repo root (.fngemma-suryaos/) not inside training/
# so that both train_tokenizer.py and finetune.py can reference it with the
# same relative path: ../. fngemma-suryaos/bin/python3
# =============================================================================

echo "[bootstrap] STEP 2: Setting up virtual environment ..."

if [ ! -f "$VENV/bin/python3" ]; then
    echo "[bootstrap]   Creating new venv at $VENV ..."
    python3 -m venv "$VENV"
    echo "[bootstrap]   Virtual environment created."
else
    echo "[bootstrap]   Existing venv found at $VENV — reusing."
fi

PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python3"

echo "[bootstrap]   Python: $($PYTHON --version)"
echo ""

# =============================================================================
# STEP 3 — Upgrade pip
# =============================================================================
#
# Old pip versions (< 23) have a broken dependency resolver that can install
# incompatible package combinations. The pip bundled with the system Python
# (especially on Ubuntu 22.04) is often outdated.
# =============================================================================

echo "[bootstrap] STEP 3: Upgrading pip ..."
"$PIP" install --upgrade pip --quiet
echo "[bootstrap]   pip upgraded to $("$PIP" --version | awk '{print $2}')"
echo ""

# ── Helper: are all listed packages already importable? ───────────────────
# Returns 0 (success) when every package is importable, 1 otherwise.
# Used to decide whether to skip an install step.
_all_importable() {
    local missing=()
    for pkg in "$@"; do
        if ! "$PYTHON" -c "import $pkg" 2>/dev/null; then
            missing+=("$pkg")
        fi
    done
    if [ ${#missing[@]} -eq 0 ]; then
        return 0
    fi
    echo "[bootstrap]   Missing packages: ${missing[*]}"
    return 1
}

# ── Helper: short-circuit a step if --skip-py-deps was passed ─────────────
if [ "$SKIP_PY_DEPS" = true ]; then
    echo "[bootstrap] --dashboard-only — skipping Python dependency steps 4-7"
    echo ""
fi

# =============================================================================
# STEP 4 — Install PyTorch with the correct CUDA wheel
# =============================================================================
#
# We install PyTorch BEFORE requirements.txt because requirements.txt does not
# specify the index URL (to avoid hardcoding CPU vs GPU). If we installed
# requirements.txt first, pip would pull the default CPU torch from PyPI and
# a subsequent torch install would fight with it.
#
# --index-url overrides the default PyPI index for torch/torchvision/torchaudio.
# Other packages in this command still resolve from PyPI.
# =============================================================================

if [ "$SKIP_PY_DEPS" = false ]; then

    echo "[bootstrap] STEP 4: Checking PyTorch ..."

    # Decide if we can skip the install.
    # When GPU is expected, torch must report CUDA available; otherwise reinstall.
    # When CPU-only, any working torch is fine.
    NEED_TORCH_INSTALL=true
    if [ "$FORCE_REINSTALL" = true ]; then
        echo "[bootstrap]   --reinstall flag set — will reinstall PyTorch"
    elif [ "$HAS_GPU" = true ]; then
        # GPU machine — require torch with CUDA available
        if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)")
            CUDA_RT=$("$PYTHON"  -c "import torch; print(torch.version.cuda)")
            echo "[bootstrap]   [SKIP]  PyTorch $TORCH_VER with CUDA $CUDA_RT already installed"
            NEED_TORCH_INSTALL=false
        elif "$PYTHON" -c "import torch" 2>/dev/null; then
            echo "[bootstrap]   PyTorch installed but CUDA unavailable — reinstalling GPU build"
        fi
    else
        # CPU-only machine — any torch import is fine
        if "$PYTHON" -c "import torch" 2>/dev/null; then
            TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)")
            echo "[bootstrap]   [SKIP]  PyTorch $TORCH_VER already installed (CPU-only OK)"
            NEED_TORCH_INSTALL=false
        fi
    fi

    if [ "$NEED_TORCH_INSTALL" = true ]; then
        # If torch is already installed (e.g. CPU build when we need GPU), pip
        # will say "already satisfied" because the package name "torch" matches —
        # it doesn't know we want a different wheel variant. Uninstall first to
        # force a clean install from the cu121/cu118 index URL.
        if "$PYTHON" -c "import torch" 2>/dev/null; then
            EXISTING_TORCH=$("$PYTHON" -c "import torch; print(torch.__version__)")
            echo "[bootstrap]   Removing wrong-variant torch ($EXISTING_TORCH) before reinstall ..."
            "$PIP" uninstall -y torch torchvision torchaudio 2>/dev/null || true
        fi

        echo "[bootstrap]   Installing PyTorch ($TORCH_LABEL) ..."
        echo "[bootstrap]   Downloading ~600 MB for GPU builds, ~200 MB for CPU ..."
        echo "[bootstrap]   This can take 3-10 minutes depending on your connection."
        echo ""
        "$PIP" install torch torchvision torchaudio --index-url "$TORCH_INDEX"
        echo ""
        echo "[bootstrap]   PyTorch installed: $("$PYTHON" -c "import torch; print(torch.__version__)")"
        if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo "[bootstrap]   CUDA available: $("$PYTHON" -c "import torch; print(torch.version.cuda)")"
        elif [ "$HAS_GPU" = true ]; then
            echo "[bootstrap]   WARN: GPU detected but torch.cuda.is_available()=False"
            echo "[bootstrap]         Possible CUDA driver / wheel mismatch."
        fi
    fi
    echo ""

fi

# =============================================================================
# STEP 5 — Install remaining requirements (HuggingFace, LoRA, etc.)
# =============================================================================
#
# requirements.txt contains everything EXCEPT torch (torch is handled above).
# It includes:
#   transformers  — model loading, tokenizer, training infrastructure
#   peft          — LoRA adapter implementation (get_peft_model, LoraConfig)
#   trl           — SFTTrainer (supervised fine-tuning wrapper for HF Trainer)
#   datasets      — Dataset class for feeding training examples to the trainer
#   accelerate    — distributed training backend (used by HF Trainer)
#   safetensors   — fast, safe tensor file format for model weights
#   gguf          — read/write Ollama GGUF model blobs (for --mode convert/export)
#   sentencepiece — Gemma tokenizer backend (SentencePiece byte-pair encoding)
#   psutil        — process memory measurement for telemetry (optional)
# =============================================================================

if [ "$SKIP_PY_DEPS" = false ]; then

    echo "[bootstrap] STEP 5: Checking requirements.txt packages ..."

    # Critical packages from requirements.txt — note Python import names (not pip names).
    # Most pip names match import names, but a few differ:
    #   prometheus-client → import prometheus_client
    #   sentencepiece    → import sentencepiece (matches)
    REQUIRED_IMPORTS=(
        transformers datasets accelerate safetensors
        peft trl gguf sentencepiece
        psutil tqdm rich prometheus_client
    )

    if [ "$FORCE_REINSTALL" = true ]; then
        echo "[bootstrap]   --reinstall flag set — will reinstall all packages"
        "$PIP" install --force-reinstall -r "$REQ"
    elif _all_importable "${REQUIRED_IMPORTS[@]}"; then
        echo "[bootstrap]   [SKIP]  All ${#REQUIRED_IMPORTS[@]} required packages already importable"
    else
        echo "[bootstrap]   Installing from $REQ ..."
        "$PIP" install -r "$REQ"
    fi
    echo ""

fi

# =============================================================================
# STEP 6 — Install bitsandbytes for QLoRA (GPU only, optional)
# =============================================================================
#
# bitsandbytes enables 4-bit quantization (QLoRA) which loads the base model
# at 4-bit precision instead of fp16, cutting VRAM usage by ~4x. For the 270M
# model we don't need this (fp16 fits easily in 16 GB), but it becomes useful
# for larger base models (7B, 13B+).
#
# We try to install it but don't fail if it doesn't work. Some environments
# (older CUDA, Windows WSL1) can't compile bitsandbytes C extensions.
# Regular fp16/bf16 LoRA training works perfectly without it.
# =============================================================================

if [ "$SKIP_PY_DEPS" = false ] && [ "$HAS_GPU" = true ]; then

    echo "[bootstrap] STEP 6: Checking bitsandbytes (optional QLoRA support) ..."

    if [ "$FORCE_REINSTALL" = false ] && "$PYTHON" -c "import bitsandbytes" 2>/dev/null; then
        BNB_VER=$("$PYTHON" -c "import bitsandbytes; print(bitsandbytes.__version__)")
        echo "[bootstrap]   [SKIP]  bitsandbytes $BNB_VER already installed (QLoRA ready)"
    else
        echo "[bootstrap]   Installing bitsandbytes ..."
        if "$PIP" install "bitsandbytes>=0.43.0" 2>/dev/null; then
            BNB_VER=$("$PYTHON" -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null || echo "unknown")
            echo "[bootstrap]   bitsandbytes $BNB_VER installed — QLoRA (4-bit) available"
        else
            echo "[bootstrap]   WARN: bitsandbytes failed to install"
            echo "[bootstrap]         Regular fp16/bf16 LoRA training still works fine."
            echo "[bootstrap]         QLoRA is only needed for models > 7B parameters."
        fi
    fi
    echo ""

fi

# =============================================================================
# STEP 7 — Verify every critical import and print its version
# =============================================================================
#
# This catches silent failures like: torch installed as CPU when GPU was expected,
# or peft installed but incompatible with the transformers version.
#
# Each _verify call runs a one-liner in the venv Python. If it fails (exit != 0),
# we print FAIL with the package name so you know exactly what to fix.
# =============================================================================

if [ "$SKIP_PY_DEPS" = true ]; then
    echo "[bootstrap] Skipping Step 7 (verify) — --dashboard-only flag set"
    echo ""
else

echo "[bootstrap] STEP 7: Verifying installation ..."
echo ""

# Helper: run an expression in the venv Python and print OK/FAIL
_verify() {
    local label="$1"
    local expr="$2"
    if result=$("$PYTHON" -c "$expr" 2>/dev/null); then
        printf "  [OK]   %s\n" "$result"
    else
        printf "  [FAIL] %s — check pip output above for errors\n" "$label"
    fi
}

# Core: torch MUST report CUDA=True on a GPU machine. If it says False,
# the CPU wheel was installed — re-run bootstrap.sh to fix it.
_verify "torch"        "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# GPU details: only meaningful if CUDA=True above
_verify "GPU device"   "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only — no GPU')"
_verify "GPU VRAM"     "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB' if torch.cuda.is_available() else 'N/A')"
_verify "CUDA version" "import torch; print(torch.version.cuda or 'N/A')"

# HuggingFace stack: all four must be present and version-compatible
_verify "transformers" "import transformers; print('transformers', transformers.__version__)"
_verify "peft"         "import peft;         print('peft', peft.__version__)"
_verify "trl"          "import trl;          print('trl', trl.__version__)"
_verify "datasets"     "import datasets;     print('datasets', datasets.__version__)"
_verify "accelerate"   "import accelerate;   print('accelerate', accelerate.__version__)"

# Supporting packages
_verify "sentencepiece" "import sentencepiece; print('sentencepiece OK')"
_verify "safetensors"   "import safetensors;  print('safetensors', safetensors.__version__)"
_verify "gguf"          "import gguf;         print('gguf OK')"
_verify "psutil"        "import psutil;       print('psutil', psutil.__version__)"

# bitsandbytes is optional — don't fail if absent
if [ "$HAS_GPU" = true ]; then
    _verify "bitsandbytes" "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__, '(QLoRA ready)')"
fi

fi  # end of "if SKIP_PY_DEPS = false" block (Step 7)

# =============================================================================
# STEP 8 — Bring up the Grafana / Loki / Prometheus observability stack
# =============================================================================
#
# Idempotent: if all 5 containers are already running, this step is a no-op.
# Otherwise runs `docker compose up -d` which:
#   - creates containers that don't exist
#   - starts containers that are stopped
#   - leaves running containers untouched
#
# Skipped when:
#   - Docker is not installed
#   - Docker daemon is not running
#   - User passed --no-dashboard
#
# After this step, dashboards are at http://localhost:3000 (admin/admin).
# =============================================================================

if [ "$WITH_DASHBOARD" = false ]; then
    echo ""
    echo "[bootstrap] Skipping Step 8 (dashboard) — --no-dashboard flag set"
else
    echo ""
    echo "[bootstrap] STEP 8: Observability stack (Grafana + Loki + Prometheus) ..."

    # Verify Docker is available before attempting docker compose
    if ! command -v docker &>/dev/null; then
        echo "[bootstrap]   [SKIP]  Docker not installed — skipping dashboard."
        echo "[bootstrap]           Install: https://docs.docker.com/engine/install/"
        echo "[bootstrap]           Or run with --no-dashboard to silence this message."
    elif ! docker info &>/dev/null; then
        echo "[bootstrap]   [SKIP]  Docker daemon not running — skipping dashboard."
        echo "[bootstrap]           Start it: sudo systemctl start docker"
        echo "[bootstrap]           Or on WSL: open Docker Desktop"
    elif [ ! -f "$OBS_DIR/docker-compose.yml" ]; then
        echo "[bootstrap]   [SKIP]  $OBS_DIR/docker-compose.yml not found"
    else
        # Pick the right docker compose subcommand (newer "docker compose" vs older "docker-compose")
        if docker compose version &>/dev/null; then
            DC="docker compose"
        elif command -v docker-compose &>/dev/null; then
            DC="docker-compose"
        else
            DC=""
        fi

        if [ -z "$DC" ]; then
            echo "[bootstrap]   [SKIP]  Neither 'docker compose' nor 'docker-compose' found"
            echo "[bootstrap]           Install Compose: https://docs.docker.com/compose/install/"
        else
            cd "$OBS_DIR"

            # Count running services from this compose project
            RUNNING=$($DC ps --services --filter status=running 2>/dev/null | wc -l || echo 0)
            EXPECTED=5  # loki, promtail, pushgateway, prometheus, grafana

            if [ "$FORCE_REINSTALL" = true ]; then
                echo "[bootstrap]   --reinstall flag set — recreating containers"
                $DC up -d --force-recreate
            elif [ "$RUNNING" -ge "$EXPECTED" ]; then
                echo "[bootstrap]   [SKIP]  All $EXPECTED containers already running"
                echo "[bootstrap]           Restart with: cd $OBS_DIR && $DC restart"
            else
                echo "[bootstrap]   Starting $EXPECTED-container observability stack ..."
                echo "[bootstrap]   First run downloads ~500 MB of images (loki, prometheus, grafana, etc.)"
                echo ""
                $DC up -d
                echo ""

                # Wait briefly for Grafana to come online (max 30s)
                echo -n "[bootstrap]   Waiting for Grafana ..."
                for _ in $(seq 1 30); do
                    if curl -sf http://localhost:3000/api/health &>/dev/null; then
                        echo " ready"
                        break
                    fi
                    echo -n "."
                    sleep 1
                done
            fi

            cd "$REPO_ROOT"

            echo ""
            echo "[bootstrap]   Dashboard endpoints:"
            echo "[bootstrap]     Grafana:     http://localhost:3000  (admin / admin)"
            echo "[bootstrap]     Prometheus:  http://localhost:9090"
            echo "[bootstrap]     Pushgateway: http://localhost:9091"
            echo "[bootstrap]     Loki API:    http://localhost:3100"
        fi
    fi
fi

# =============================================================================
# Done — print the next steps
# =============================================================================

echo ""
echo "[bootstrap] ============================================================"
echo "[bootstrap]  Bootstrap complete!"
if [ "$HAS_GPU" = true ]; then
echo ""
echo "[bootstrap]  GPU training enabled:"
echo "[bootstrap]    $GPU_NAME  |  ${GPU_VRAM} MiB VRAM  |  $TORCH_LABEL"
fi
echo ""
echo "[bootstrap]  Run in this order:"
echo ""
echo "    # Step 1: extend tokenizer + warm up embeddings (~20-40 min GPU)"
echo "    $PYTHON training/train_tokenizer.py"
echo ""
echo "    # Step 2: verify the full environment is ready"
echo "    $PYTHON training/finetune.py --mode check"
echo ""
echo "    # Step 3: full training pipeline (convert → train → export)"
echo "    $PYTHON training/finetune.py --mode all"
echo ""
echo "[bootstrap]  Or activate the venv first (then just use 'python'):"
echo "    source $VENV/bin/activate"
echo "    python training/finetune.py --mode all"
echo ""
echo "[bootstrap]  Training telemetry will be written to:"
echo "    training/tokenizer_extended/train_log.jsonl   (tokenizer phase)"
echo "    training/model_lora/training_log.jsonl        (LoRA training phase)"
echo "[bootstrap] ============================================================"
