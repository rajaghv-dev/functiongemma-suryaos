#!/usr/bin/env python3
"""
test_training_compat.py — pre-flight: are the v4 datasets ready to train?

Verifies the dataset → trainer wiring without needing torch installed:

  1. All four candidate dataset files exist and parse cleanly.
  2. Every line has a populated `target.arguments` for tools that require args
     (the iter #3 bug — empty args trained model to emit empty args).
  3. Every line's `tools[*].inputSchema` matches `tools/tool_schemas.json`.
  4. Per-tool count is sane (matches the v4 plan).
  5. (Optional) torch + CUDA: pick the dtype + batch size finetune.py would
     pick on this machine and print, so the user can confirm RTX 3080 path.

Run as a script:
    python3 tests/test_training_compat.py
    python3 tests/test_training_compat.py --gpu-only

Run as pytest:
    pytest tests/test_training_compat.py -v
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [
    REPO_ROOT / "dataset" / "dispatch_pairs.jsonl",
    REPO_ROOT / "dataset" / "dispatch_pairs_v4.jsonl",
    REPO_ROOT / "dataset" / "dispatch_pairs_v4_balanced30.jsonl",
    REPO_ROOT / "dataset" / "dispatch_pairs_v4_balanced.jsonl",
]
SCHEMAS_PATH = REPO_ROOT / "tools" / "tool_schemas.json"

REQUIRED_ARGS = {
    "linux_brightness_set":   ["direction"],
    "linux_volume_set":       ["direction"],
    "linux_volume_change":    ["direction"],
    "linux_service_status":   ["name"],
    "kde_krunner_launch":     ["app"],
    "kde_window_focus":       ["title"],
    "kde_notifications_send": ["title", "message"],
    "kde_dialog_confirm":     ["prompt"],
}


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def load_schemas() -> dict:
    if not SCHEMAS_PATH.exists():
        return {}
    raw = json.loads(SCHEMAS_PATH.read_text())
    return {entry["mcp_name"]: entry for entry in raw.values()}


def check_dataset(path: Path, schemas: dict) -> dict:
    pairs = load_jsonl(path)
    if not pairs:
        return {"path": path, "exists": path.exists(), "n": 0,
                "errors": [f"{'missing' if not path.exists() else 'empty'}"]}

    errs: list[str] = []
    arg_misses: Counter = Counter()
    schema_drift = 0

    for i, p in enumerate(pairs, start=1):
        target = p.get("target") or {}
        tool = target.get("name")
        if not tool:
            errs.append(f"line {i}: no target.name")
            continue
        # Required-arg check.
        args = target.get("arguments") or {}
        for req in REQUIRED_ARGS.get(tool, []):
            if args.get(req) in (None, "", []):
                arg_misses[(tool, req)] += 1
        # Schema-drift check.
        if schemas:
            tool_schema = schemas.get(tool)
            if tool_schema:
                want = tool_schema["mcp_schema"]["inputSchema"]
                for t in p.get("tools", []):
                    if t.get("name") == tool and t.get("inputSchema") != want:
                        schema_drift += 1

    return {
        "path":         path,
        "exists":       True,
        "n":            len(pairs),
        "by_tool":      Counter(p.get("target", {}).get("name") for p in pairs),
        "arg_misses":   dict(arg_misses),
        "schema_drift": schema_drift,
        "errors":       errs,
    }


def print_dataset_report(r: dict) -> None:
    print(f"\n  {r['path'].name}")
    if not r["exists"]:
        print(f"    [MISS] file not present")
        return
    if r["n"] == 0:
        print(f"    [MISS] file empty")
        return
    print(f"    pairs        : {r['n']}")
    by_tool = r["by_tool"]
    top, top_n = by_tool.most_common(1)[0]
    pct = 100.0 * top_n / r["n"]
    print(f"    top tool     : {top} = {top_n} ({pct:.1f}%)")
    print(f"    tools present: {len(by_tool)} / 12")
    if r["arg_misses"]:
        print(f"    arg misses   : {sum(r['arg_misses'].values())} pairs missing required args")
        for (tool, req), n in sorted(r["arg_misses"].items(), key=lambda x: -x[1])[:3]:
            print(f"                   {tool}.{req}: {n}")
    else:
        print(f"    arg misses   : 0 (all required args populated)")
    print(f"    schema drift : {r['schema_drift']}")


# ---------------------------------------------------------------------------
# GPU / dtype path — mirrors finetune.py's _detect_hardware() logic so the
# user sees the same answer without running the full trainer.
# ---------------------------------------------------------------------------

def detect_hardware() -> dict:
    """Reports what finetune.py will pick for this machine. Safe without torch."""
    info: dict = {
        "torch_available": False,
        "cuda_available":  False,
        "gpu_name":        None,
        "vram_gb":         0.0,
        "compute":         None,
        "dtype":           "float32",
        "use_bf16":        False,
        "use_fp16":        False,
        "batch_size":      1,
        "grad_accum":      4,
        "device":          "cpu",
        "verdict":         "CPU fallback (no torch installed)",
    }
    try:
        import torch  # noqa
        info["torch_available"] = True
    except ImportError:
        info["verdict"] = (
            "torch not installed — run `bash training/bootstrap.sh` to install. "
            "On Ampere+ GPUs (RTX 30xx/40xx) bootstrap selects the cu121 wheel."
        )
        return info

    import torch
    if not torch.cuda.is_available():
        info["verdict"] = "torch installed, no CUDA visible (CPU training)"
        return info

    props = torch.cuda.get_device_properties(0)
    info["cuda_available"] = True
    info["gpu_name"] = props.name
    info["vram_gb"]  = props.total_memory / 1e9
    info["compute"]  = (props.major, props.minor)
    is_ampere_plus = props.major >= 8
    use_bf16 = is_ampere_plus
    use_fp16 = (not use_bf16) and props.major >= 7
    info["use_bf16"] = use_bf16
    info["use_fp16"] = use_fp16
    info["dtype"]    = "bfloat16" if use_bf16 else ("float16" if use_fp16 else "float32")
    info["device"]   = "cuda"

    vram = info["vram_gb"]
    if vram >= 24:   info["batch_size"] = 12
    elif vram >= 16: info["batch_size"] = 8
    elif vram >= 10: info["batch_size"] = 4   # RTX 3080 10 GB lands here
    elif vram >= 8:  info["batch_size"] = 2
    else:            info["batch_size"] = 1
    info["grad_accum"] = max(1, 16 // info["batch_size"])

    info["verdict"] = (
        f"GPU OK: {info['gpu_name']} ({vram:.1f} GB, compute "
        f"{props.major}.{props.minor}) → dtype={info['dtype']}, "
        f"batch={info['batch_size']}, grad_accum={info['grad_accum']}"
    )
    return info


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-only", action="store_true",
                    help="skip dataset checks; just print the GPU/dtype path.")
    args = ap.parse_args()

    bar = "=" * 72
    print(f"\n{bar}\n  Training compat pre-flight\n{bar}")

    if not args.gpu_only:
        schemas = load_schemas()
        print(f"\n  Tool schemas loaded: {len(schemas)} tools")

        for ds in DATASETS:
            r = check_dataset(ds, schemas)
            print_dataset_report(r)

    print(f"\n  GPU / dtype path:")
    hw = detect_hardware()
    for k in ("torch_available", "cuda_available", "gpu_name", "vram_gb",
              "compute", "device", "dtype", "use_bf16", "use_fp16",
              "batch_size", "grad_accum"):
        if hw[k] is not None and hw[k] is not False:
            print(f"    {k:18s} {hw[k]}")
    print(f"\n  → {hw['verdict']}")

    print(f"\n{bar}\n  Suggested first run on RTX 3080 10 GB:")
    print(f"    bash training/bootstrap.sh")
    print(f"    export HF_TOKEN=hf_...")
    print(f"    .fngemma-suryaos/bin/python3 training/train_tokenizer.py \\")
    print(f"        --epochs 5 --batch-size 32")
    print(f"    .fngemma-suryaos/bin/python3 training/finetune.py --mode train \\")
    print(f"        --data dataset/dispatch_pairs_v4.jsonl --epochs 3 --batch-size 4")
    print(f"{bar}")
    return 0


# ---------------------------------------------------------------------------
# Pytest assertions (run with: pytest tests/test_training_compat.py -v)
# ---------------------------------------------------------------------------

def test_v4_dataset_exists():
    p = REPO_ROOT / "dataset" / "dispatch_pairs_v4.jsonl"
    assert p.exists(), f"missing iter-#4 dataset at {p}"


def test_v4_dataset_args_populated():
    schemas = load_schemas()
    p = REPO_ROOT / "dataset" / "dispatch_pairs_v4.jsonl"
    if not p.exists():
        return  # covered by previous test
    r = check_dataset(p, schemas)
    miss = sum(r["arg_misses"].values())
    assert miss == 0, f"v4 still has {miss} pairs with empty required args: {r['arg_misses']}"


def test_v4_no_schema_drift():
    schemas = load_schemas()
    p = REPO_ROOT / "dataset" / "dispatch_pairs_v4.jsonl"
    if not p.exists():
        return
    r = check_dataset(p, schemas)
    assert r["schema_drift"] == 0, f"{r['schema_drift']} schema-drift errors in v4"


def test_balanced_target_30_no_tool_drowns():
    schemas = load_schemas()
    p = REPO_ROOT / "dataset" / "dispatch_pairs_v4_balanced30.jsonl"
    if not p.exists():
        return
    r = check_dataset(p, schemas)
    top_n = max(r["by_tool"].values())
    assert top_n <= 30, f"top tool has {top_n} pairs (cap was 30)"


if __name__ == "__main__":
    sys.exit(main())
