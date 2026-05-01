#!/usr/bin/env python3
"""
Build the tokenizer extension dataset for functiongemma:270m.

Outputs:
    dataset/tokenizer/new_tokens.json   — categorized token list
    dataset/tokenizer/corpus.txt        — sentences with each token in context
    dataset/tokenizer/corpus.jsonl      — same with metadata
    dataset/tokenizer/tool_terms.txt    — tool names only
    dataset/tokenizer/kde_terms.txt     — KDE concepts only
    dataset/tokenizer/system_terms.txt  — Linux system terms only

Sources:
    1. tools/**/*.yaml — tool names, descriptions, examples
    2. mcp/system.py + mcp/volume.py — handler names, command lines
    3. configs/policy.yaml — service names referenced
    4. Hardcoded curated lists below — KDE concepts, device paths

Usage:
    python3 scripts/training/build_tokenizer_dataset.py
    python3 scripts/training/build_tokenizer_dataset.py --out ~/raja/functiongemma-suryaos/dataset/tokenizer/

Each token MUST appear at least N=5 times in corpus.txt for the embedding
to converge during fine-tune. The script enforces this via template repetition.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


# =============================================================================
# Curated token lists — these don't appear in tool YAMLs but matter for SuryaOS
# =============================================================================

KDE_TERMS = [
    # Plasma desktop
    "Plasma", "KWin", "KDE", "Wayland", "KRunner",
    # KDE applications
    "Kate", "Dolphin", "Konsole", "KMail", "KOrganizer",
    "KDevelop", "Akonadi", "Klipper", "kdialog",
    # KDE CLI tools
    "kstart5", "kstart6", "qdbus", "qdbus5", "qdbus6", "kioclient5", "kioclient6",
    # KDE concepts
    "KIO", "KCM", "kglobalaccel",
]

SYSTEM_TERMS = [
    # Daemons / services
    "systemd", "systemctl", "journalctl", "pipewire", "pulseaudio",
    "NetworkManager", "bluetoothd", "cupsd", "ollama", "udevd",
    "dbus-daemon", "logind", "rtkit",
    # CLI tools
    "pactl", "nmcli", "wmctrl", "acpi", "brightnessctl", "notify-send",
    "kdialog", "free", "df", "lsblk", "iotop",
    # Device names / paths (without leading slash for tokenization)
    "BAT0", "wlo1", "eno2", "nvme0n1", "DEFAULT_SINK",
    # System paths (will appear with / in corpus)
    "/proc", "/sys", "/etc/systemd", "/var/log", "/opt/surya-agent",
]

ARG_VALUES = [
    # Volume / brightness directions
    "up", "down",
    # Service states
    "active", "inactive", "failed", "deactivating", "activating",
    # Network states
    "connected", "disconnected", "limited", "portal",
    # Power
    "charging", "discharging", "fully-charged", "unknown",
]

# v4 tokens — code/git workflow (not yet active but worth indexing now)
V4_TERMS = [
    "compile", "build", "test", "commit", "push", "pull",
    "rebase", "merge", "checkout", "branch", "stash",
    "pytest", "cargo", "make", "cmake", "ninja",
]

# Git tokens — full set for v3 git workflow tools
GIT_TERMS = [
    # Core commands
    "clone", "fetch", "diff", "status", "log", "blame", "bisect", "reflog",
    "cherry-pick", "revert", "reset", "restore", "switch", "tag",
    # Concepts
    "HEAD", "origin", "upstream", "remote", "submodule", "worktree",
    "staged", "unstaged", "untracked", "conflict", "hunk",
    # Hosting + tools
    "github", "gitlab", "gitea", "codeberg", "gh-cli", "lazygit", "tig",
    # Workflow
    "PR", "MR", "merge-queue", "fast-forward", "squash-merge",
    "conventional-commits",
    # Files
    ".gitignore", ".gitattributes", ".gitmodules",
]

# File format + document tooling tokens
FILE_FORMAT_TERMS = [
    # PDF
    "PDF", ".pdf", "pdftotext", "pdftoppm", "qpdf", "poppler", "ghostscript",
    "pdfunite", "pdfseparate", "evince", "MuPDF",
    # LibreOffice / Office
    "LibreOffice", "Writer", "Calc", "Impress", "Draw", "Base", "Math",
    "soffice", "OnlyOffice", "Calligra",
    ".odt", ".ods", ".odp", ".odg", ".docx", ".xlsx", ".pptx",
    ".doc", ".xls", ".ppt", ".rtf", ".csv", ".tsv",
    # Image formats
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".tiff", ".bmp",
    ".avif", ".heif", ".heic", ".raw", ".dng",
    # Audio / video
    ".mp3", ".flac", ".ogg", ".wav", ".mp4", ".mkv", ".webm", ".mov",
    "ffmpeg", "ffprobe", "ImageMagick", "convert", "mogrify",
    # Archive
    ".zip", ".tar", ".gz", ".xz", ".7z", ".rar", ".zst",
    # Text / data
    ".md", "Markdown", ".rst", "AsciiDoc", "LaTeX", ".tex",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".xml",
    # Code-file markers (binary names commonly appearing in queries)
    "pandoc", "pdftk",
]

# ML / training tokens — used by docs and v4 chain-of-task workflows
ML_TERMS = [
    # PyTorch
    "torch", "pytorch", "cuda", "cudnn", "fp16", "bf16", "fp32", "int8", "int4",
    # HuggingFace
    "transformers", "datasets", "accelerate", "tokenizers", "peft", "trl",
    "AutoModel", "AutoTokenizer", "from_pretrained", "safetensors",
    # Models
    "Llama", "Gemma", "Qwen", "Mistral", "Phi",
    # Quantization formats
    "GGUF", "GGML", "Q8_0", "Q5_K_M", "Q4_K_S", "AWQ", "GPTQ",
    # Serving
    "vllm", "sglang", "ollama", "llama.cpp", "TGI",
    # Hardware
    "RTX", "A100", "H100", "Meteor-Lake", "OpenVINO", "ROCm",
]


# =============================================================================
# Step 1: Extract tool tokens from YAML catalog
# =============================================================================

def extract_tool_tokens() -> list[dict]:
    """Pull tool names in three forms from the YAML catalog."""
    from surya_agent.tool_registry import ToolRegistry
    reg = ToolRegistry.load(ROOT / "tools")

    tokens: list[dict] = []
    for name, tool in reg.tools.items():
        # Three forms — model sees all three across docs and MCP schemas.
        forms = [
            name,                              # linux.volume.set
            name.replace(".", "_"),            # linux_volume_set
            name.split(".", 1)[1].replace(".", "_") if "." in name else name,  # volume_set (mcp short)
        ]
        # opencode prefixes MCP tool names with the server name.
        # Add prefixed forms too (system_volume_set, volume_volume_change, etc.).
        prefixed = [f"system_{forms[2]}", f"volume_{forms[2]}", f"kde_{forms[2]}"]

        for f in dict.fromkeys(forms + prefixed):  # de-dup, keep order
            tokens.append({
                "token": f,
                "category": "tool_name",
                "source": "tool_registry",
                "for_tool": name,
            })
    return tokens


def extract_arg_value_tokens() -> list[dict]:
    """Pull enum values from tool parameter schemas."""
    from surya_agent.tool_registry import ToolRegistry
    reg = ToolRegistry.load(ROOT / "tools")

    seen: set[str] = set()
    tokens: list[dict] = []

    def walk(obj, path=""):
        if isinstance(obj, dict):
            if "enum" in obj and isinstance(obj["enum"], list):
                for v in obj["enum"]:
                    if isinstance(v, str) and v not in seen:
                        seen.add(v)
                        tokens.append({
                            "token": v,
                            "category": "arg_value",
                            "source": f"yaml:{path}",
                        })
            for k, v in obj.items():
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    for name, tool in reg.tools.items():
        walk(tool.parameters or {}, name)
    return tokens


# =============================================================================
# Step 2: Build corpus — sentences containing each new token
# =============================================================================

# Sentence templates — each template uses {token} as the slot and yields
# multiple natural-language variations per token. We rotate templates so that
# every token appears ≥5 times in the final corpus.

TOOL_TEMPLATES = [
    "Call {token} to handle this request.",
    "The {token} tool runs the appropriate system command.",
    "When the user asks for this, dispatch via {token}.",
    "{token} is the correct MCP tool for this query.",
    "Use {token} when the user wants to perform this action.",
    "{token} is registered in the SuryaOS tool catalog.",
    "After invoking {token}, format the result for the user.",
]

KDE_TEMPLATES = [
    "{token} is a core part of the KDE Plasma desktop.",
    "Launch {token} to open the user's preferred application.",
    "On SuryaOS, {token} integrates with the desktop session.",
    "The {token} component handles its specific responsibility.",
    "{token} runs in the user's KDE session.",
]

SYSTEM_TEMPLATES = [
    "The {token} daemon manages this part of the system.",
    "Run {token} to control the underlying service.",
    "{token} is part of the standard Linux toolchain.",
    "On SuryaOS 25.1, {token} is installed by default.",
    "The agent shells out to {token} when handling related queries.",
    "{token} is invoked via subprocess by the MCP server.",
]

ARG_TEMPLATES = [
    "The direction can be {token} for this command.",
    "Set the parameter to {token} when appropriate.",
    "A {token} state means the service is in this condition.",
    "{token} is one of the valid enum values for this argument.",
]


def build_corpus(token_list: list[dict], min_occurrences: int = 5) -> list[str]:
    """Generate sentences. Each token appears at least min_occurrences times."""
    sentences: list[str] = []

    for entry in token_list:
        token = entry["token"]
        category = entry["category"]
        if category == "tool_name":
            templates = TOOL_TEMPLATES
        elif category == "kde":
            templates = KDE_TEMPLATES
        elif category == "system":
            templates = SYSTEM_TEMPLATES
        elif category == "arg_value":
            templates = ARG_TEMPLATES
        else:
            templates = TOOL_TEMPLATES

        # Cycle through templates until we hit min_occurrences
        n = max(min_occurrences, len(templates))
        for i in range(n):
            sentences.append(templates[i % len(templates)].format(token=token))

    return sentences


def build_yaml_corpus() -> list[str]:
    """Extract every example + description sentence from the tool YAMLs."""
    from surya_agent.tool_registry import ToolRegistry
    reg = ToolRegistry.load(ROOT / "tools")

    sentences: list[str] = []
    for name, tool in reg.tools.items():
        if tool.description:
            sentences.append(tool.description.strip())
        for ex in (getattr(tool, "examples", []) or []):
            if isinstance(ex, str):
                sentences.append(ex.strip())
    return sentences


def build_dispatch_corpus(dispatch_path: Path) -> list[str]:
    """Pull user queries + tool name pairs from the dispatch dataset."""
    if not dispatch_path.exists():
        return []
    sentences: list[str] = []
    with open(dispatch_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                user = d["messages"][1]["content"]
                target = d["target"]["name"]
                # Both as a sentence
                sentences.append(f'When the user says "{user}", call {target}.')
            except Exception:
                pass
    return sentences


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path.home() / "raja/functiongemma-suryaos/dataset/tokenizer",
                        help="Output directory")
    parser.add_argument("--min-occur", type=int, default=5,
                        help="Minimum corpus occurrences per token")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. Build the token list -----
    print("[1/3] Building token list...")

    token_list: list[dict] = []
    token_list.extend(extract_tool_tokens())
    token_list.extend({"token": t, "category": "kde", "source": "curated"} for t in KDE_TERMS)
    token_list.extend({"token": t, "category": "system", "source": "curated"} for t in SYSTEM_TERMS)
    token_list.extend({"token": t, "category": "arg_value", "source": "curated"} for t in ARG_VALUES)
    token_list.extend(extract_arg_value_tokens())
    token_list.extend({"token": t, "category": "v4_workflow", "source": "v4_curated"} for t in V4_TERMS)
    token_list.extend({"token": t, "category": "git",          "source": "curated"} for t in GIT_TERMS)
    token_list.extend({"token": t, "category": "file_format",  "source": "curated"} for t in FILE_FORMAT_TERMS)
    token_list.extend({"token": t, "category": "ml",           "source": "curated"} for t in ML_TERMS)

    # Deduplicate by token (keep first occurrence/category)
    seen: dict[str, dict] = {}
    for t in token_list:
        if t["token"] not in seen:
            seen[t["token"]] = t
    token_list = list(seen.values())

    # Group by category for the JSON output
    by_cat: dict[str, list[dict]] = {}
    for t in token_list:
        by_cat.setdefault(t["category"], []).append(t)

    print(f"    {len(token_list)} unique tokens across {len(by_cat)} categories:")
    for cat, items in sorted(by_cat.items()):
        print(f"      {cat:20s} {len(items):3d}")

    # Write JSON
    new_tokens_path = out_dir / "new_tokens.json"
    new_tokens_path.write_text(json.dumps(by_cat, indent=2))
    print(f"    ✓ {new_tokens_path}")

    # Per-category text files
    for cat, items in by_cat.items():
        path = out_dir / f"{cat}_terms.txt"
        path.write_text("\n".join(t["token"] for t in items) + "\n")

    # ----- 2. Build the corpus -----
    print("[2/3] Building corpus...")

    sentences: list[str] = []
    # Synthetic context sentences (each token appears ≥ N times)
    sentences.extend(build_corpus(token_list, min_occurrences=args.min_occur))
    # Real text from tool YAMLs
    sentences.extend(build_yaml_corpus())
    # Real text from dispatch training data
    sentences.extend(build_dispatch_corpus(ROOT / "training" / "dispatch_pairs.jsonl"))

    print(f"    {len(sentences)} total sentences in corpus")

    corpus_path = out_dir / "corpus.txt"
    corpus_path.write_text("\n".join(sentences) + "\n")
    print(f"    ✓ {corpus_path}  ({sum(len(s) for s in sentences)} chars)")

    # JSONL form with per-line metadata
    corpus_jsonl_path = out_dir / "corpus.jsonl"
    with open(corpus_jsonl_path, "w") as f:
        for s in sentences:
            f.write(json.dumps({"text": s, "source": "tokenizer_corpus"}) + "\n")
    print(f"    ✓ {corpus_jsonl_path}")

    # ----- 3. Validation: confirm each token appears N+ times -----
    print("[3/3] Validating coverage...")
    corpus_text = "\n".join(sentences)
    weak: list[tuple[str, int]] = []
    for entry in token_list:
        n = corpus_text.count(entry["token"])
        if n < args.min_occur:
            weak.append((entry["token"], n))

    if weak:
        print(f"    ⚠ {len(weak)} tokens have < {args.min_occur} occurrences:")
        for t, n in weak[:10]:
            print(f"        {t!r}: {n}")
        if len(weak) > 10:
            print(f"        ... and {len(weak) - 10} more")
    else:
        print(f"    ✓ All {len(token_list)} tokens appear ≥{args.min_occur} times in corpus")

    print()
    print("Done.")
    print(f"Output: {out_dir}")
    print()
    print("Use in fine-tune:")
    print(f"  tokenizer.add_tokens([t['token'] for cat in tokens.values() for t in cat])")
    print(f"  Then train embeddings on corpus.txt during the SFT step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
