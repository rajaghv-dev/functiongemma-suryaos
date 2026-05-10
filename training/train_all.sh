#!/usr/bin/env bash
# =============================================================================
# train_all.sh — end-to-end training pipeline orchestrator.
#
# Runs (idempotently):
#   1. bootstrap.sh                    — venv + torch + deps + (opt) dashboards
#   2. train_tokenizer.py              — extend tokenizer with 108 domain tokens
#   3. analyze_embeddings.py           — print 9-row probe table from goals.md
#   4. finetune.py --mode all          — LoRA SFT on dispatch_pairs_v4_train.jsonl
#                                        (per-epoch ProbeBank against
#                                         dispatch_pairs_v4_eval.jsonl, 162 pairs)
#   5. summarize accuracy from training/model_lora/probes.jsonl (last epoch)
#
# Hard prereq: HF_TOKEN must be exported (Gemma 3 270M is gated) OR a
# cached HuggingFace login at ~/.cache/huggingface/token must exist.
#
# Usage:
#   bash training/train_all.sh                       # full pipeline
#   bash training/train_all.sh --no-dashboard        # skip Grafana stack
#   bash training/train_all.sh --skip-tokenizer      # reuse existing tokenizer_extended/
#   bash training/train_all.sh --skip-lora           # tokenizer phase only
#   bash training/train_all.sh --epochs 5            # override LoRA epochs
#   bash training/train_all.sh --reinstall           # force-reinstall deps
#   bash training/train_all.sh --clean               # wipe tokenizer_extended/+model_lora/
#
# Exit codes:
#   0  success (all steps OK; accuracy printed)
#   1  HF_TOKEN / dataset / venv missing; or post-LoRA probes file missing
#   2  unknown CLI flag
#   anything else → propagated from the failing sub-step
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.fngemma-suryaos"
PY="$VENV/bin/python"
TOK_DIR="$REPO_ROOT/training/tokenizer_extended"
LORA_DIR="$REPO_ROOT/training/model_lora"
PROBES_JSONL="$LORA_DIR/probes.jsonl"

# ── Parse flags ─────────────────────────────────────────────────────────────
BOOTSTRAP_FLAGS=()
SKIP_TOKENIZER=false
SKIP_LORA=false
CLEAN=false
LORA_EPOCHS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --no-dashboard)   BOOTSTRAP_FLAGS+=("--no-dashboard"); shift ;;
        --reinstall)      BOOTSTRAP_FLAGS+=("--reinstall");    shift ;;
        --skip-tokenizer) SKIP_TOKENIZER=true; shift ;;
        --skip-lora)      SKIP_LORA=true;      shift ;;
        --clean)          CLEAN=true; shift ;;
        --epochs)         LORA_EPOCHS="$2"; shift 2 ;;
        --epochs=*)       LORA_EPOCHS="${1#*=}"; shift ;;
        -h|--help)
            sed -n '2,/^# ===/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "FATAL: unknown flag: $1" >&2
            echo "       use --help for usage" >&2
            exit 2
            ;;
    esac
done

# ── Pre-flight: HF token ────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
    cat >&2 <<EOF
FATAL: HF_TOKEN not set and no cached HuggingFace login found.

Gemma 3 270M is a gated repo. To unblock:
  1. Get a token (read scope is enough): https://huggingface.co/settings/tokens
  2. Accept the Gemma license:           https://huggingface.co/google/gemma-3-270m-it
  3. Export it in this shell:            export HF_TOKEN=hf_xxxxxxxxxxxx
  4. Re-run this script.
EOF
    exit 1
fi

# ── Pre-flight: dataset inputs (fail fast, before 600 MB of bootstrap) ─────
# Both phases have hard data dependencies. Without these the run will die
# minutes in with a Python KeyError or FileNotFoundError that is much
# harder to interpret than this single explicit check.
_REQ_PATHS=(
    "$REPO_ROOT/dataset/tokenizer/new_tokens.json"        # phase 1 input
    "$REPO_ROOT/dataset/tokenizer/corpus.txt"             # phase 1 input
    "$REPO_ROOT/dataset/dispatch_pairs_v4_train.jsonl"    # phase 2 train split
    "$REPO_ROOT/dataset/dispatch_pairs_v4_eval.jsonl"     # phase 2 eval split
)
_REQ_PURPOSES=(
    "tokenizer phase: 108 domain tokens to add to vocab"
    "tokenizer phase: warm-up corpus (~3.5k sentences)"
    "LoRA phase: training pairs (~1.4k examples)"
    "LoRA phase: held-out eval pairs (per-epoch ProbeBank)"
)
_DATA_MISSING=()
for i in "${!_REQ_PATHS[@]}"; do
    p="${_REQ_PATHS[$i]}"
    if [ ! -s "$p" ]; then
        _DATA_MISSING+=("$p — ${_REQ_PURPOSES[$i]}")
    fi
done
if [ "${#_DATA_MISSING[@]}" -gt 0 ]; then
    {
        echo "FATAL: required dataset files are missing or empty:"
        for m in "${_DATA_MISSING[@]}"; do echo "  - $m"; done
        echo
        echo "These are committed in the repo (iter #4). Recovery options:"
        echo "  - Restore from git:    git checkout HEAD -- dataset/"
        echo "  - Or re-clone:         (in a fresh dir) git clone <repo>"
        echo "  - Or rebuild manually:"
        echo "      python training/build_tokenizer_dataset.py    # tokenizer inputs"
        echo "      python training/build_real_dataset.py         # v4 dispatch pairs"
        echo "      python training/split_train_eval.py           # train/eval split"
    } >&2
    exit 1
fi

# ── Optional --clean: wipe stage-2/3 outputs for a fully fresh run ─────────
# Does NOT touch the venv (that's bootstrap's --reinstall) or the dataset.
if $CLEAN; then
    if [ -d "$REPO_ROOT/training/tokenizer_extended" ]; then
        echo "[train_all] --clean: removing training/tokenizer_extended/"
        rm -rf "$REPO_ROOT/training/tokenizer_extended"
    fi
    if [ -d "$LORA_DIR" ]; then
        echo "[train_all] --clean: removing $LORA_DIR"
        rm -rf "$LORA_DIR"
    fi
fi

# ── Helpers ─────────────────────────────────────────────────────────────────
banner() {
    echo
    echo "============================================================"
    echo "▶ $*"
    echo "============================================================"
}

now() { date +%s; }
elapsed() {
    local t=$(( $(now) - $1 ))
    printf "%dm%02ds" $((t/60)) $((t%60))
}

T0=$(now)

# ── 1. Bootstrap ────────────────────────────────────────────────────────────
banner "STEP 1/5  bootstrap (venv + torch + deps)"
t=$(now)
bash "$REPO_ROOT/training/bootstrap.sh" "${BOOTSTRAP_FLAGS[@]}"
echo "  ⏱ bootstrap: $(elapsed $t)"

# Bootstrap is only "done" when the venv actually has the imports we need.
# A previous bug let bootstrap exit 0 even when bin/pip was missing — then
# train_tokenizer.py died with a confusing ImportError half a minute later.
# This guard surfaces the failure here, with a copy-paste recovery command.
PIP_BIN="$REPO_ROOT/.fngemma-suryaos/bin/pip"
if [ ! -x "$PY" ] || [ ! -x "$PIP_BIN" ]; then
    echo "FATAL: venv at $REPO_ROOT/.fngemma-suryaos is incomplete after bootstrap." >&2
    echo "       Missing: $([ ! -x "$PY" ] && echo bin/python3) $([ ! -x "$PIP_BIN" ] && echo bin/pip)" >&2
    echo "       Recovery:  bash training/bootstrap.sh --reinstall" >&2
    exit 1
fi
if ! "$PY" -c "import torch, transformers, peft, trl, datasets, sentencepiece" 2>/dev/null; then
    echo "FATAL: venv exists but core packages do not import." >&2
    echo "       Diagnostics:" >&2
    "$PY" -c "
import importlib
for pkg in ('torch', 'transformers', 'peft', 'trl', 'datasets', 'sentencepiece'):
    try:
        importlib.import_module(pkg)
        print(f'         [OK]   {pkg}')
    except Exception as e:
        print(f'         [FAIL] {pkg}: {type(e).__name__}: {e}')
" >&2
    echo "       Recovery:  bash training/bootstrap.sh --reinstall" >&2
    exit 1
fi

# ── 2. Tokenizer extension + warm-up ────────────────────────────────────────
# A "complete" tokenizer phase produces both the saved tokenizer AND the
# warmed embedding tensor. Either alone is a partial run we should redo,
# otherwise finetune.py would seed LoRA with random rows.
_tokenizer_complete() {
    [ -d "$TOK_DIR" ]                          || return 1
    [ -s "$TOK_DIR/tokenizer.json" ]           || return 1
    [ -s "$TOK_DIR/embed_init.pt" ]            || return 1
    return 0
}

if $SKIP_TOKENIZER && _tokenizer_complete; then
    banner "STEP 2/5  train_tokenizer.py  [SKIPPED — --skip-tokenizer and $TOK_DIR is complete]"
else
    if $SKIP_TOKENIZER; then
        echo "[train_all] --skip-tokenizer requested but $TOK_DIR is missing/incomplete; running anyway."
    fi
    banner "STEP 2/5  train_tokenizer.py  (~5 min on RTX 3080 Ti)"
    t=$(now)
    "$PY" "$REPO_ROOT/training/train_tokenizer.py"
    echo "  ⏱ tokenizer: $(elapsed $t)"
fi

# ── 3. Embedding analysis (9-row probe table) ───────────────────────────────
if _tokenizer_complete; then
    banner "STEP 3/5  analyze_embeddings.py"
    # Non-fatal — analysis script may evolve; don't block training on it.
    "$PY" "$REPO_ROOT/training/analyze_embeddings.py" || \
        echo "  [WARN] analyze_embeddings failed; continuing"
else
    echo "  [WARN] tokenizer_extended/ missing or incomplete; skipping embedding analysis"
fi

# ── 4. LoRA fine-tune + GGUF export ─────────────────────────────────────────
if $SKIP_LORA; then
    banner "STEP 4/5  finetune.py --mode all  [SKIPPED — --skip-lora]"
else
    banner "STEP 4/5  finetune.py --mode all  (LoRA + per-epoch eval probes)"
    t=$(now)
    EXTRA=()
    [ -n "$LORA_EPOCHS" ] && EXTRA+=("--epochs" "$LORA_EPOCHS")
    "$PY" "$REPO_ROOT/training/finetune.py" --mode all "${EXTRA[@]}"
    echo "  ⏱ LoRA + export: $(elapsed $t)"
fi

# ── 5. Accuracy summary ─────────────────────────────────────────────────────
banner "STEP 5/5  accuracy summary  (last epoch from $PROBES_JSONL)"

if $SKIP_LORA; then
    echo "  [SKIP] LoRA was skipped — no accuracy to report."
elif [ ! -f "$PROBES_JSONL" ]; then
    echo "FATAL: expected $PROBES_JSONL after LoRA run, but it is missing." >&2
    echo "       Likely the eval split (dataset/dispatch_pairs_v4_eval.jsonl)" >&2
    echo "       was missing or empty — finetune.py would have warned." >&2
    exit 1
else
    "$PY" - "$PROBES_JSONL" <<'PYEOF'
import json, sys, pathlib
path = pathlib.Path(sys.argv[1])
lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
if not lines:
    print(f"  [WARN] {path} is empty")
    sys.exit(0)

last = lines[-1]
meta = last.get("_meta", {})
af   = last.get("arg_fidelity", {})
cm   = last.get("confusion_matrix", {})
sc   = last.get("schema_compliance", {})
so   = last.get("source_overfit", {})

epoch    = meta.get("epoch", "?")
n_pairs  = meta.get("n_pairs", "?")
elapsed  = meta.get("elapsed_s", "?")

print(f"  epoch={epoch}  n_eval_pairs={n_pairs}  probe_runtime={elapsed}s")
print()
print("  ── headline accuracies ──────────────────────────────────")
print(f"    tool_match            : {af.get('tool_match', 0):.1%}   (target ≥ 90%)")
print(f"    arg_keys_match        : {af.get('arg_keys_match', 0):.1%}   (target ≥ 85%)")
print(f"    arg_values_match      : {af.get('arg_values_match', 0):.1%}   (target ≥ 70%)")
print(f"    overall_accuracy(CM)  : {cm.get('overall_accuracy', 0):.1%}")
print(f"    cross_domain_rate     : {cm.get('cross_domain_rate', 0):.1%}   (lower is better)")
if sc:
    print(f"    schema_compliance     : json={sc.get('json_parse_rate', 0):.1%}  "
          f"tool_in_list={sc.get('tool_in_list_rate', 0):.1%}  "
          f"keys_in_schema={sc.get('keys_in_schema_rate', 0):.1%}")
print()
print("  ── per-tool accuracy (sorted, low→high) ────────────────")
per_tool = cm.get("per_tool_accuracy", {})
for tool, acc in sorted(per_tool.items(), key=lambda kv: kv[1]):
    bar = "█" * int(acc * 20)
    print(f"    {tool:<28} {acc:>5.1%}  {bar}")

confused = cm.get("top_confused", [])
if confused:
    print()
    print("  ── top 5 confusions (actual → predicted, count) ────────")
    for actual, pred, cnt in confused[:5]:
        print(f"    {actual:<28} → {pred:<28}  ×{cnt}")

if so and "by_source" in so:
    print()
    print("  ── accuracy by source (overfit check) ──────────────────")
    for src, acc in sorted(so["by_source"].items(), key=lambda kv: kv[1]):
        print(f"    {src:<28} {acc:>5.1%}")

interp_lines = [
    last.get(k, {}).get("interpretation", "") for k in
    ("confusion_matrix", "arg_fidelity", "schema_compliance",
     "first_token_entropy", "negative_margin", "source_overfit")
]
print()
print("  ── probe interpretations ───────────────────────────────")
for line in interp_lines:
    if line:
        print(f"    {line}")
PYEOF
fi

echo
echo "============================================================"
echo "✓ pipeline finished in $(elapsed $T0)"
echo "============================================================"
