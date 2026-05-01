#!/usr/bin/env bash
# bootstrap.sh — create the Python venv and install all training dependencies.
#
# Run once from anywhere:
#   bash training/bootstrap.sh
#
# After this completes, two ways to use the env:
#   source .fngemma-suryaos/bin/activate
#   python training/train_tokenizer.py
#
#   # or without activating:
#   .fngemma-suryaos/bin/python training/train_tokenizer.py
#   .fngemma-suryaos/bin/python training/finetune.py --mode check

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.fngemma-suryaos"
REQ="$REPO_ROOT/training/requirements.txt"

# ── 1. Create venv if it doesn't have a Python binary ──────────────────────
if [ ! -f "$VENV/bin/python3" ]; then
    echo "[bootstrap] Creating virtual environment at $VENV ..."
    python3 -m venv "$VENV"
    echo "[bootstrap] Virtual environment created."
else
    echo "[bootstrap] Virtual environment exists: $VENV"
fi

PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python3"

# ── 2. Upgrade pip first (avoids resolver warnings) ────────────────────────
echo ""
echo "[bootstrap] Upgrading pip ..."
"$PIP" install --upgrade pip --quiet

# ── 3. Install all requirements ────────────────────────────────────────────
echo "[bootstrap] Installing requirements from $REQ ..."
echo "            (torch CPU build ~600 MB, first run takes several minutes)"
echo ""
"$PIP" install -r "$REQ"

# ── 4. Verify critical imports ─────────────────────────────────────────────
echo ""
echo "[bootstrap] Verifying installation ..."

verify() {
    local pkg="$1"
    local expr="$2"
    if "$PYTHON" -c "$expr" 2>/dev/null; then
        echo "  [OK]   $pkg"
    else
        echo "  [FAIL] $pkg — check pip output above"
    fi
}

verify "torch"         "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
verify "transformers"  "import transformers; print('transformers', transformers.__version__)"
verify "datasets"      "import datasets; print('datasets', datasets.__version__)"
verify "peft"          "import peft; print('peft', peft.__version__)"
verify "trl"           "import trl; print('trl', trl.__version__)"
verify "sentencepiece" "import sentencepiece; print('sentencepiece OK')"
verify "safetensors"   "import safetensors; print('safetensors', safetensors.__version__)"
verify "gguf"          "import gguf; print('gguf OK')"
verify "psutil"        "import psutil; print('psutil', psutil.__version__)"
verify "accelerate"    "import accelerate; print('accelerate', accelerate.__version__)"

# ── 5. Print next steps ─────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "  Bootstrap complete."
echo ""
echo "  Activate the environment:"
echo "    source $VENV/bin/activate"
echo ""
echo "  Or run scripts directly (no activation needed):"
echo "    $PYTHON training/train_tokenizer.py"
echo "    $PYTHON training/finetune.py --mode check"
echo "    $PYTHON training/finetune.py --mode all"
echo "========================================================================"
