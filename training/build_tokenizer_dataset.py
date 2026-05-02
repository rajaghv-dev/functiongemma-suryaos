#!/usr/bin/env python3
"""
build_tokenizer_dataset.py — iteration #3 corpus generator.

REWRITTEN from the template-based generator that produced Run #1-#3 datasets.

What changed vs old version:

  1. TOKEN LIST PRUNING (319 → ~160)
     - Drop tokens already single-token in base Gemma (replacing pre-trained
       knowledge with random init = strict regression)
     - Drop redundant file-format extensions (.png/.jpg/.gif individually
       don't drive different routing — keep ~10 high-value ones)
     - Drop generic English words (active/inactive/up/down) — corrupts their
       pre-trained meaning when added as domain tokens

  2. CORPUS QUALITY (templates → curated)
     - REMOVED: 7 rotated templates × 251 tokens = monotonous gradient signal
     - ADDED:  per-tool curated natural sentences (15-25 per core tool)
     - ADDED:  hard contrastive examples ("X is RAM, Y is disk")
     - ADDED:  co-occurrence sentences (explicit "A uses B under the hood")
     - ADDED:  mining of dispatch_pairs.jsonl into 8 sentences each
     - ADDED:  cross-domain separation examples

  3. INSIGHT-RICH OUTPUT
     - Prints stats per phase: how many tokens kept/dropped and why
     - Reports per-tool sentence count and source mix
     - Final corpus stats: length distribution, co-occurrence count,
       diversity metrics

The intuition this whole script encodes:

  Embeddings learn from CONTEXT. Templates put every token in identical
  contexts → embeddings collapse together. Curated text puts each token
  in dozens of natural contexts that vary with the token's semantic role
  → embeddings spread into a useful geometry.

  Goal (from goals.md): cross-domain cosine 0.62 → < 0.30.
  Tactic: replace 70% of corpus with content that distinguishes tools.

Usage:
    python3 training/build_tokenizer_dataset.py
    python3 training/build_tokenizer_dataset.py --out path/to/output/dir
    python3 training/build_tokenizer_dataset.py --no-mine    # skip dispatch mining
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "dataset" / "tokenizer"
DISPATCH_PAIRS  = REPO_ROOT / "dataset" / "dispatch_pairs.jsonl"
HF_MODEL_ID     = "google/gemma-3-270m-it"


# =============================================================================
# Terminal output helpers (insight-rich, not just numbers)
# =============================================================================

def _banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def _ok(msg: str)     -> None: print(f"  [OK]      {msg}")
def _info(msg: str)   -> None: print(f"  [INFO]    {msg}")
def _drop(msg: str)   -> None: print(f"  [DROP]    {msg}")
def _keep(msg: str)   -> None: print(f"  [KEEP]    {msg}")
def _learn(msg: str)  -> None: print(f"  [LEARN]   {msg}")
def _insight(msg: str)-> None: print(f"  [INSIGHT] {msg}")
def _warn(msg: str)   -> None: print(f"  [WARN]    {msg}", file=sys.stderr)


# =============================================================================
# CANONICAL TOOL DEFINITIONS — single source of truth for the 12 core tools
# =============================================================================
# Each tool is defined with its routing-relevant metadata. The corpus
# generator uses these fields to produce contextually-rich sentences:
#
#   description    — for "what it does" sentences
#   cli_equiv      — for "X uses Y under the hood" co-occurrence
#   user_synonyms  — for "user query Q routes to T" patterns
#   contrasts_with — for hard contrastive "X is not Y" sentences
#   category       — for cross-category separation
#   key_concept    — for hierarchical "what it touches" sentences
# =============================================================================

CORE_TOOLS = {
    # ── Linux system tools ────────────────────────────────────────────────
    "linux_memory_usage": {
        "category": "linux",
        "description": "Check RAM/memory usage. Reads /proc/meminfo.",
        "cli_equiv":  "free -h",
        "user_synonyms": [
            "how much RAM is used", "memory usage", "show me free memory",
            "is memory low", "RAM consumption", "what's my memory looking like",
            "available RAM", "swap usage", "mem stats",
        ],
        "contrasts_with": [
            ("linux_disk_usage", "RAM, not disk"),
            ("linux_metrics_summary", "memory specifically, not all metrics"),
        ],
        "key_concept": "RAM",
        "naming_variants": ["linux_memory_usage", "memory_usage",
                            "system_memory_usage", "linux.memory.usage"],
    },
    "linux_disk_usage": {
        "category": "linux",
        "description": "Check filesystem disk usage. Reads /proc/diskstats and df.",
        "cli_equiv":  "df -h",
        "user_synonyms": [
            "how much disk space", "disk full?", "show free space",
            "is the SSD full", "storage left", "what's my disk looking like",
            "available space", "drive usage",
        ],
        "contrasts_with": [
            ("linux_memory_usage", "disk, not RAM"),
            ("linux_metrics_summary", "storage specifically, not all metrics"),
        ],
        "key_concept": "storage",
        "naming_variants": ["linux_disk_usage", "disk_usage",
                            "system_disk_usage", "linux.disk.usage"],
    },
    "linux_metrics_summary": {
        "category": "linux",
        "description": "Summary of CPU/RAM/disk/network metrics. Pulls from netdata.",
        "cli_equiv":  "top",
        "user_synonyms": [
            "system stats", "how is the system", "overall load",
            "performance summary", "everything at once", "system overview",
        ],
        "contrasts_with": [
            ("linux_memory_usage", "everything, not just RAM"),
            ("linux_disk_usage", "everything, not just disk"),
        ],
        "key_concept": "metrics",
        "naming_variants": ["linux_metrics_summary", "metrics_summary",
                            "system_metrics_summary", "linux.metrics.summary"],
    },
    "linux_battery_status": {
        "category": "linux",
        "description": "Battery percentage, charging state, and time remaining.",
        "cli_equiv":  "acpi -b",
        "user_synonyms": [
            "battery left", "is it charging", "how much charge",
            "battery percentage", "time on battery", "am I plugged in",
        ],
        "contrasts_with": [
            ("linux_brightness_set", "battery state, not screen brightness"),
        ],
        "key_concept": "battery",
        "naming_variants": ["linux_battery_status", "battery_status",
                            "system_battery_status", "linux.battery.status"],
    },
    "linux_brightness_set": {
        "category": "linux",
        "description": "Adjust screen brightness as a percentage.",
        "cli_equiv":  "brightnessctl set",
        "user_synonyms": [
            "screen too dark", "brighter please", "dim the screen",
            "set brightness to 50", "make the display brighter",
            "less bright please", "max brightness",
        ],
        "contrasts_with": [
            ("linux_battery_status", "screen, not battery"),
            ("linux_volume_change", "brightness, not volume"),
        ],
        "key_concept": "brightness",
        "naming_variants": ["linux_brightness_set", "brightness_set",
                            "system_brightness_set", "linux.brightness.set"],
    },
    "linux_volume_change": {
        "category": "linux",
        "description": "Adjust audio volume up/down or set to a level.",
        "cli_equiv":  "pactl set-sink-volume",
        "user_synonyms": [
            "volume up", "louder", "quieter", "mute", "unmute",
            "set volume to 50", "turn it down", "increase volume",
        ],
        "contrasts_with": [
            ("linux_brightness_set", "audio, not screen"),
        ],
        "key_concept": "audio",
        "naming_variants": ["linux_volume_change", "volume_change",
                            "system_volume_change", "linux.volume.change"],
    },
    "linux_network_status": {
        "category": "linux",
        "description": "WiFi/ethernet connection state, SSID, and IP.",
        "cli_equiv":  "nmcli",
        "user_synonyms": [
            "am I online", "wifi connected?", "what network",
            "internet status", "show me my IP", "connection state",
        ],
        "contrasts_with": [
            ("linux_service_status", "network, not arbitrary services"),
        ],
        "key_concept": "network",
        "naming_variants": ["linux_network_status", "network_status",
                            "system_network_status", "linux.network.status"],
    },
    "linux_service_status": {
        "category": "linux",
        "description": "Check if a systemd service is active/inactive/failed.",
        "cli_equiv":  "systemctl status",
        "user_synonyms": [
            "is bluetooth on", "check if cups is running",
            "is the service active", "service state",
            "is pipewire running", "check NetworkManager",
        ],
        "contrasts_with": [
            ("linux_network_status", "any service, not just network"),
            ("linux_metrics_summary", "service state, not metrics"),
        ],
        "key_concept": "systemd",
        "naming_variants": ["linux_service_status", "service_status",
                            "system_service_status", "linux.service.status"],
    },

    # ── KDE Plasma tools ──────────────────────────────────────────────────
    "kde_krunner_launch": {
        "category": "kde",
        "description": "Launch an application via KRunner.",
        "cli_equiv":  "kstart6",
        "user_synonyms": [
            "open Firefox", "launch Dolphin", "start Konsole",
            "run Kate", "open the browser", "launch the file manager",
        ],
        "contrasts_with": [
            ("kde_window_focus", "starts apps, doesn't focus existing ones"),
        ],
        "key_concept": "launcher",
        "naming_variants": ["kde_krunner_launch", "krunner_launch",
                            "kde.krunner.launch"],
    },
    "kde_window_focus": {
        "category": "kde",
        "description": "Focus an existing window by name or class.",
        "cli_equiv":  "wmctrl -a",
        "user_synonyms": [
            "switch to Firefox", "bring Dolphin to front",
            "focus the terminal", "show the editor",
            "go back to the browser",
        ],
        "contrasts_with": [
            ("kde_krunner_launch", "focuses existing, doesn't start new"),
        ],
        "key_concept": "window",
        "naming_variants": ["kde_window_focus", "window_focus",
                            "kde.window.focus"],
    },
    "kde_notifications_send": {
        "category": "kde",
        "description": "Display a desktop notification with title and body.",
        "cli_equiv":  "notify-send",
        "user_synonyms": [
            "tell me when done", "notify me", "show a popup",
            "remind me", "send a notification",
        ],
        "contrasts_with": [
            ("kde_dialog_confirm", "fire-and-forget, no user response"),
        ],
        "key_concept": "notification",
        "naming_variants": ["kde_notifications_send", "notifications_send",
                            "kde.notifications.send"],
    },
    "kde_dialog_confirm": {
        "category": "kde",
        "description": "Show a Yes/No confirmation dialog. Returns user's choice.",
        "cli_equiv":  "kdialog --yesno",
        "user_synonyms": [
            "ask before deleting", "confirm with the user",
            "are you sure?", "get user approval",
        ],
        "contrasts_with": [
            ("kde_notifications_send", "blocks for response, not fire-and-forget"),
        ],
        "key_concept": "dialog",
        "naming_variants": ["kde_dialog_confirm", "dialog_confirm",
                            "kde.dialog.confirm"],
    },
}


# =============================================================================
# CURATED auxiliary tokens — kept small and high-signal
# =============================================================================
# Replacing the bloated category lists. Each token here MUST:
#   - Fragment in base Gemma vocab (single-token already = no value)
#   - Have a clear semantic role distinct from base English
#   - Appear naturally in tool-dispatch contexts

# KDE concepts that are domain-specific (not generic English)
KDE_DOMAIN_TOKENS = [
    "Plasma", "KWin", "KRunner", "Akonadi", "Klipper",
    "Dolphin", "Kate", "Konsole",
    "kstart6", "qdbus6", "kioclient6", "kglobalaccel",
    "KIO",
]

# System tokens that are CLI commands / daemons
SYSTEM_DOMAIN_TOKENS = [
    "systemd", "systemctl", "journalctl", "pipewire", "pulseaudio",
    "NetworkManager", "bluetoothd", "ollama", "logind",
    "pactl", "nmcli", "wmctrl", "brightnessctl", "notify-send",
    "kdialog", "iotop",
    "BAT0", "wlo1", "DEFAULT_SINK",
]

# Curated FILE FORMAT tokens — only those that drive routing decisions.
# Dropped 70+ generic extensions like .png/.jpg/.gif/.tiff/.bmp/.avif/...
# because we never dispatch DIFFERENTLY for image extensions — they all
# go through ImageMagick. Same for most audio/video.
FILE_FORMAT_DOMAIN_TOKENS = [
    ".pdf",  ".docx", ".csv",   # high-value document formats
    ".json", ".yaml", ".md",    # structured data / docs (dispatch differently)
    "LibreOffice", "pandoc", "ffmpeg",  # major converter tools
    "ImageMagick",
]

# Git tokens — kept the conceptually distinct ones
GIT_DOMAIN_TOKENS = [
    "HEAD", "origin", "upstream", "submodule", "worktree",
    "lazygit", "gh-cli",
    "merge-queue", "fast-forward", "squash-merge", "conventional-commits",
    ".gitignore", ".gitattributes", ".gitmodules",
]

# ML tokens — kept the framework / quantisation / hardware specifics
ML_DOMAIN_TOKENS = [
    "transformers", "AutoTokenizer", "from_pretrained", "safetensors",
    "peft", "trl",
    "GGUF", "Q8_0", "Q5_K_M", "Q4_K_S", "AWQ", "GPTQ",
    "llama.cpp",
    "Meteor-Lake", "OpenVINO", "ROCm",
]

# DROPPED ENTIRELY (these were 15+ generic English words that corrupt their
# pre-trained meaning when re-added as new tokens):
#   "up", "down", "active", "inactive", "failed", "connected", "disconnected",
#   "charging", "discharging", "fully-charged", "limited", "portal",
#   "deactivating", "activating", "unknown"


# =============================================================================
# Token list builder with intuition-rich filtering
# =============================================================================

def build_token_list(verbose: bool = True) -> tuple[list[dict], dict]:
    """
    Returns (token_list, stats_dict).

    Filtering passes:
      1. Build candidate list from CORE_TOOLS naming variants + auxiliary tokens
      2. Load base tokenizer, check fragmentation per token
      3. DROP tokens that already tokenize as 1 piece (would just replace
         pre-trained embedding with random init = regression)
      4. KEEP everything else
    """
    if verbose:
        _banner("PHASE 1: Build + filter token list")

    # ─── Build the candidate list from canonical sources ─────────────────
    candidates: list[dict] = []

    # 1. Tool naming variants from CORE_TOOLS
    for tool_name, meta in CORE_TOOLS.items():
        for variant in meta["naming_variants"]:
            candidates.append({
                "token": variant,
                "category": "tool_name",
                "for_tool": tool_name,
            })

    # 2. Auxiliary domain tokens (KDE, system, file_format, git, ML)
    aux_categories = [
        ("kde",         KDE_DOMAIN_TOKENS),
        ("system",      SYSTEM_DOMAIN_TOKENS),
        ("file_format", FILE_FORMAT_DOMAIN_TOKENS),
        ("git",         GIT_DOMAIN_TOKENS),
        ("ml",          ML_DOMAIN_TOKENS),
    ]
    for cat, tokens in aux_categories:
        for tok in tokens:
            candidates.append({"token": tok, "category": cat})

    # Dedup while preserving order
    seen = set()
    deduped = []
    for c in candidates:
        if c["token"] not in seen:
            seen.add(c["token"])
            deduped.append(c)
    candidates = deduped

    if verbose:
        _ok(f"Built candidate list: {len(candidates)} unique tokens")

    # ─── Load tokenizer for fragmentation check ──────────────────────────
    if verbose:
        _info(f"Loading {HF_MODEL_ID} tokenizer for fragmentation analysis ...")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        _warn("transformers not installed — skipping fragmentation filter")
        _warn("Install: bash training/bootstrap.sh")
        return candidates, {"filtered": False}

    import os
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, token=hf_token)
    except Exception as e:
        _warn(f"Tokenizer load failed: {e}")
        _warn("Set HF_TOKEN env var. Continuing without fragmentation filter.")
        return candidates, {"filtered": False}

    # ─── Filter pass: drop tokens already tokenized as 1 piece ───────────
    if verbose:
        _info("Checking each candidate's fragmentation in base Gemma vocab ...")

    kept:    list[dict] = []
    dropped: list[dict] = []

    for c in candidates:
        ids = tok.encode(c["token"], add_special_tokens=False)
        c["base_fragmentation"] = len(ids)
        if len(ids) <= 1:
            dropped.append(c)
        else:
            kept.append(c)

    # Sort dropped by category for the report
    by_cat_dropped: dict[str, int] = {}
    for d in dropped:
        by_cat_dropped[d["category"]] = by_cat_dropped.get(d["category"], 0) + 1

    if verbose:
        _drop(f"Dropped {len(dropped)} tokens that already tokenize as 1 piece in Gemma:")
        for cat, n in sorted(by_cat_dropped.items()):
            print(f"             {cat:14s} {n:3d} dropped (already in base vocab)")
        _learn("Why drop these? Re-adding them as new tokens would REPLACE")
        _learn("their pre-trained embedding with a freshly-initialized one.")
        _learn("That's strictly worse than leaving them alone — we lose all")
        _learn("the contextual knowledge Gemma learned about them on web text.")
        print()

        # Show concrete examples of what was dropped
        sample_dropped = [d for d in dropped if d.get("category") in ("git", "system", "ml")][:8]
        if sample_dropped:
            _info("Examples of tokens dropped because they're already single-token:")
            for d in sample_dropped:
                print(f"             {d['token']!r:20s} ({d['category']}) — Gemma already knows this")
        print()

        _keep(f"Kept {len(kept)} tokens that genuinely fragment (≥ 2 pieces)")

    # Stats by category
    by_cat_kept: dict[str, int] = {}
    for k in kept:
        by_cat_kept[k["category"]] = by_cat_kept.get(k["category"], 0) + 1

    if verbose:
        for cat, n in sorted(by_cat_kept.items()):
            print(f"             {cat:14s} {n:3d} kept")

    stats = {
        "filtered":        True,
        "n_candidates":    len(candidates),
        "n_kept":          len(kept),
        "n_dropped":       len(dropped),
        "by_category":     by_cat_kept,
        "by_cat_dropped":  by_cat_dropped,
    }

    return kept, stats


# =============================================================================
# CORPUS GENERATOR — curated, no templates
# =============================================================================
#
# Where the corpus quality comes from:
#   1. Per-tool natural sentences (5 of each type × 12 tools = 60 per tool)
#      → 720 lines of high-variety natural language
#   2. Cross-tool contrastive sentences (~80 lines) — fixes cross-domain
#   3. Co-occurrence sentences (~60 lines)
#   4. Dispatch-pair mining (~600 lines from 77 pairs × 8 sentences)
#
# Total: ~1500 sentences, all natural, none templated.
# Smaller than the old 3849-line corpus but vastly more informative.
# =============================================================================

def gen_per_tool_sentences(tool_name: str, meta: dict) -> list[str]:
    """
    Generate 25-30 natural sentences per tool. Variety by design:
    - User query phrasings
    - Implementation/CLI details
    - Cross-tool comparisons (subset)
    - Casual phrasings
    - Expert-level discussions

    Each sentence references the canonical tool name AT LEAST ONCE.
    """
    sentences = []
    desc       = meta["description"]
    cli        = meta["cli_equiv"]
    synonyms   = meta["user_synonyms"]
    key        = meta["key_concept"]
    variants   = meta["naming_variants"]

    canon = variants[0]  # primary canonical name

    # ── User query → tool sentences (5) ───────────────────────────────────
    for syn in synonyms[:5]:
        sentences.append(f'When the user says "{syn}", call {canon}.')

    # ── Tool description in different framings (5) ────────────────────────
    sentences.append(f"{canon} is the SuryaOS tool that {desc.lower().rstrip('.')}.")
    sentences.append(f"To {desc.lower().rstrip('.').replace('check ', 'check ')}, use {canon}.")
    sentences.append(f"{canon} handles {key} queries on Linux/KDE systems.")
    sentences.append(f"The agent dispatches {canon} for {key}-related user requests.")
    sentences.append(f"Internally, {canon} wraps the underlying {key} subsystem.")

    # ── CLI equivalence (3) — co-occurrence pattern ───────────────────────
    sentences.append(f"{canon} is the structured equivalent of `{cli}`.")
    sentences.append(f"Where a shell user runs `{cli}`, the agent calls {canon}.")
    sentences.append(f"{canon} returns JSON; `{cli}` returns human-readable text.")

    # ── Naming-variant co-occurrence (3) — same-tool clustering ──────────
    if len(variants) >= 2:
        sentences.append(
            f"{variants[0]} and {variants[1]} refer to the same tool — "
            f"different naming conventions for the same {key} operation."
        )
    if len(variants) >= 3:
        sentences.append(
            f"The forms {variants[1]} and {variants[2]} both dispatch to "
            f"the same handler as {variants[0]}."
        )
    sentences.append(
        f"Whether written as {variants[0]} or {variants[-1]}, "
        f"the tool behaves identically."
    )

    # ── Hard contrastive sentences (3-6) — cross-domain separation ────────
    for other_tool, distinction in meta.get("contrasts_with", []):
        sentences.append(
            f"{canon} reports {distinction}; "
            f"{other_tool} is for a different domain — don't confuse them."
        )
        sentences.append(
            f"If the query is about {key}, dispatch {canon}, "
            f"NOT {other_tool}."
        )

    # ── Casual phrasings (3) — robustness to user style ──────────────────
    sentences.append(f"Casual phrasings like '{synonyms[0]}' should route to {canon}.")
    sentences.append(f"User vocabulary varies but {canon} handles {key} regardless of phrasing.")
    sentences.append(f"Even informal queries about {key} dispatch to {canon}.")

    # ── Expert-level / edge cases (2) ────────────────────────────────────
    sentences.append(
        f"{canon} runs in milliseconds and is safe to call frequently."
    )
    sentences.append(
        f"On systems where {cli} would fail, {canon} returns a structured error."
    )

    return sentences


def gen_cross_domain_contrast() -> list[str]:
    """
    Explicit cross-domain separation sentences. These are what fixes the
    cross-domain cosine 0.62 → < 0.30 problem from goals.md.
    """
    sentences = []

    # All pairs of tools from different categories
    tool_items = list(CORE_TOOLS.items())
    for i, (a_name, a_meta) in enumerate(tool_items):
        for j, (b_name, b_meta) in enumerate(tool_items[i+1:], start=i+1):
            if a_meta["category"] == b_meta["category"]:
                continue  # same category — handled by sibling contrasts in per-tool

            a_canon = a_meta["naming_variants"][0]
            b_canon = b_meta["naming_variants"][0]
            a_key   = a_meta["key_concept"]
            b_key   = b_meta["key_concept"]

            sentences.append(
                f"{a_canon} ({a_key}) and {b_canon} ({b_key}) "
                f"are in different categories and serve unrelated purposes."
            )

    # Cap to avoid overweighting any one tool — pick a balanced sample
    # 12 tools × 11 others = 132 pairs; we have ~50 cross-category pairs
    return sentences


def gen_co_occurrence() -> list[str]:
    """Sentences that intentionally pair related tokens."""
    sentences = []

    # Tool ↔ CLI equivalent pairings
    sentences.extend([
        "linux_memory_usage and free both report RAM stats — "
        "linux_memory_usage as JSON, free as a table.",
        "service_status uses systemctl under the hood; "
        "the agent prefers service_status for structured output.",
        "linux_disk_usage parses df output internally.",
        "linux_network_status calls nmcli to get the connection state.",
        "krunner_launch invokes kstart6 to spawn applications.",
        "notifications_send wraps notify-send for desktop alerts.",
        "kde_dialog_confirm uses kdialog --yesno under the hood.",
        "linux_brightness_set delegates to brightnessctl.",
        "linux_volume_change uses pactl for PipeWire/PulseAudio.",
    ])

    # Multi-tool flow sentences (real agent usage patterns)
    sentences.extend([
        "Performance investigation: first run linux_metrics_summary, "
        "then drill in with linux_memory_usage or linux_disk_usage.",
        "Connection issues: check linux_network_status; "
        "if a service is down, follow up with service_status.",
        "Battery alert flow: linux_battery_status detects low charge, "
        "then notifications_send shows the warning.",
        "App launch flow: krunner_launch starts Firefox, "
        "then window_focus brings it to front when needed.",
        "Confirmation flow: dialog_confirm asks the user, "
        "notifications_send tells them when done.",
    ])

    # KDE ecosystem co-occurrence
    sentences.extend([
        "KRunner is the KDE Plasma launcher; krunner_launch wraps it.",
        "Plasma desktop notifications go through notify-send, "
        "which kde_notifications_send invokes.",
        "Akonadi powers KDE PIM apps but isn't directly accessed by tools.",
        "Konsole, Dolphin, Kate are launched via krunner_launch.",
    ])

    # Linux ecosystem co-occurrence
    sentences.extend([
        "systemd manages services; service_status queries it via systemctl.",
        "pipewire is the modern audio daemon; "
        "linux_volume_change controls it via pactl.",
        "NetworkManager handles WiFi; "
        "linux_network_status reads its state via nmcli.",
        "logind tracks user sessions; the agent rarely touches it directly.",
    ])

    # ML ecosystem (for v4 tokens)
    sentences.extend([
        "transformers and peft work together: "
        "transformers loads the model, peft adds the LoRA adapter.",
        "GGUF is the quantized format ollama and llama.cpp both use.",
        "Q8_0 is moderate compression; Q4_K_S is more aggressive.",
        "AutoTokenizer.from_pretrained loads the tokenizer; "
        "safetensors stores the weights.",
    ])

    return sentences


def gen_auxiliary_token_sentences() -> list[str]:
    """
    Generate ~5-8 natural sentences per auxiliary token (KDE/system/git/ML/file).
    These tokens aren't tool names but appear in tool-dispatch contexts.

    Without this, mining dispatch_pairs floods the corpus with tool-name lines
    and starves the auxiliary tokens (zero corpus appearances → no learning).
    """
    sentences: list[str] = []

    # ── KDE concepts ──────────────────────────────────────────────────────
    kde_facts = {
        "Plasma": "KDE Plasma is the desktop environment SuryaOS ships with.",
        "KWin":   "KWin is the KDE window manager that handles Wayland sessions.",
        "KRunner":"KRunner is the KDE Plasma quick-launcher; krunner_launch wraps it.",
        "Akonadi":"Akonadi powers KDE PIM apps like KMail and KOrganizer.",
        "Klipper":"Klipper is the Plasma clipboard history daemon.",
        "Dolphin":"Dolphin is the default KDE file manager — kde_krunner_launch starts it.",
        "Kate":   "Kate is the KDE text editor with kparts, often invoked via krunner_launch.",
        "Konsole":"Konsole is the KDE terminal emulator launched via kde_krunner_launch.",
        "kstart6":"kstart6 is the Plasma 6 launcher used internally by krunner_launch.",
        "qdbus6":"qdbus6 is the D-Bus bridge used to talk to Plasma 6 services.",
        "kioclient6":"kioclient6 handles KIO URL operations for Plasma 6.",
        "kglobalaccel":"kglobalaccel manages global KDE keyboard shortcuts.",
        "KIO":    "KIO is the KDE input/output abstraction; kioclient6 is its CLI.",
    }
    for token, fact in kde_facts.items():
        sentences.extend([
            fact,
            f"{token} is part of the KDE Plasma ecosystem.",
            f"On SuryaOS, {token} runs as part of the user's Plasma session.",
            f"The agent doesn't call {token} directly — it goes through wrapper tools.",
        ])

    # ── System tokens ────────────────────────────────────────────────────
    system_facts = {
        "systemd":      "systemd is the Linux init system; service_status queries it.",
        "systemctl":    "systemctl is the systemd CLI; linux_service_status wraps it.",
        "journalctl":   "journalctl reads the systemd journal log buffer.",
        "pipewire":     "pipewire is the modern audio daemon controlled via pactl.",
        "pulseaudio":   "pulseaudio was the predecessor to pipewire; pactl works with both.",
        "NetworkManager":"NetworkManager handles WiFi; nmcli is its CLI.",
        "bluetoothd":   "bluetoothd is the Bluetooth service daemon — systemctl status bluetooth checks it.",
        "ollama":       "ollama serves local language models including functiongemma.",
        "logind":       "logind tracks user sessions via systemd.",
        "pactl":        "pactl is the PulseAudio/PipeWire control CLI; volume_change wraps it.",
        "nmcli":        "nmcli is the NetworkManager CLI; linux_network_status wraps it.",
        "wmctrl":       "wmctrl manipulates X11 windows; window_focus uses it on X11 sessions.",
        "brightnessctl":"brightnessctl reads/writes screen brightness; linux_brightness_set wraps it.",
        "notify-send":  "notify-send dispatches a desktop notification via D-Bus.",
        "kdialog":      "kdialog shows native KDE dialogs; kde_dialog_confirm uses --yesno.",
        "iotop":        "iotop is the per-process I/O monitor for Linux.",
        "BAT0":         "BAT0 is the standard Linux battery sysfs name; linux_battery_status reads it.",
        "wlo1":         "wlo1 is a typical predictable WiFi interface name.",
        "DEFAULT_SINK": "DEFAULT_SINK is the pactl placeholder for the active audio output.",
    }
    for token, fact in system_facts.items():
        sentences.extend([
            fact,
            f"{token} is a standard component of the Linux/SuryaOS stack.",
            f"On SuryaOS 25.1, {token} is installed and active by default.",
            f"The agent shells out to {token} via subprocess when handling related queries.",
        ])

    # ── Git tokens ────────────────────────────────────────────────────────
    git_facts = {
        "submodule":    "git submodule embeds one repo inside another.",
        "worktree":     "git worktree creates linked working trees from one repo.",
        "lazygit":      "lazygit is a TUI for git operations; v4 may dispatch to it.",
        "gh-cli":       "gh-cli is the GitHub CLI; v4 will use it for PR operations.",
        "merge-queue":  "merge-queue serialises merges to keep main green.",
        "fast-forward": "fast-forward is the simplest merge type — no merge commit.",
        "squash-merge": "squash-merge collapses a branch's commits into one.",
        "conventional-commits":"conventional-commits formats commit messages for changelog generation.",
        ".gitignore":   ".gitignore lists patterns git should not track.",
        ".gitattributes":".gitattributes sets per-path git settings like line endings.",
        ".gitmodules":  ".gitmodules tracks submodule configuration.",
    }
    for token, fact in git_facts.items():
        sentences.extend([
            fact,
            f"{token} is part of standard git workflow.",
            f"The agent's v4 git tools will reference {token} where relevant.",
        ])

    # ── ML tokens ────────────────────────────────────────────────────────
    ml_facts = {
        "AutoTokenizer": "AutoTokenizer.from_pretrained loads a tokenizer from a HuggingFace repo.",
        "from_pretrained":"from_pretrained downloads weights and configs from the HuggingFace Hub.",
        "safetensors":   "safetensors is the safe tensor file format that replaces pickle.",
        "peft":          "peft is the HuggingFace LoRA library — adds adapters to frozen models.",
        "trl":           "trl provides SFTTrainer for supervised fine-tuning of language models.",
        "GGUF":          "GGUF is the quantized model format used by ollama and llama.cpp.",
        "Q8_0":          "Q8_0 is moderate 8-bit quantisation in GGUF.",
        "Q5_K_M":        "Q5_K_M is a 5-bit GGUF quant variant balancing size and quality.",
        "Q4_K_S":        "Q4_K_S is a 4-bit GGUF quant for tight memory budgets.",
        "AWQ":           "AWQ is activation-aware weight quantisation, an alternative to GGUF.",
        "GPTQ":          "GPTQ is a post-training quantisation method.",
        "llama.cpp":     "llama.cpp serves GGUF models; ollama bundles it under the hood.",
        "Meteor-Lake":   "Meteor-Lake is the Intel CPU generation SuryaOS targets for inference.",
        "OpenVINO":      "OpenVINO accelerates ML inference on Intel hardware.",
        "ROCm":          "ROCm is AMD's GPU compute stack, an alternative to CUDA.",
    }
    for token, fact in ml_facts.items():
        sentences.extend([
            fact,
            f"{token} is part of the modern ML training ecosystem.",
            f"Documentation for {token} appears throughout the training scripts.",
        ])

    # ── File format tokens ───────────────────────────────────────────────
    file_facts = {
        ".pdf":          ".pdf documents are handled by poppler-based tools like evince.",
        ".docx":         ".docx is the Office Open XML word-processing format opened by LibreOffice.",
        ".csv":          ".csv stores tabular data, opened by LibreOffice Calc or pandas.",
        ".json":         ".json is the structured data format the agent uses internally.",
        ".yaml":         ".yaml is the configuration format for tool definitions.",
        ".md":           ".md is Markdown — used for docs and README files.",
        "LibreOffice":   "LibreOffice is the open-source office suite; soffice is its CLI.",
        "pandoc":        "pandoc converts between document formats including .md, .docx, .pdf.",
        "ffmpeg":        "ffmpeg is the universal audio/video conversion tool.",
        "ImageMagick":   "ImageMagick handles image format conversions and manipulations.",
    }
    for token, fact in file_facts.items():
        sentences.extend([
            fact,
            f"{token} is a tool the agent may invoke for file-handling tasks.",
            f"Document workflows on SuryaOS commonly involve {token}.",
        ])

    return sentences


def gen_from_dispatch_pairs(verbose: bool = True, cap: int = 3000) -> list[str]:
    """
    Mine dispatch_pairs.jsonl into corpus sentences.

    Each pair becomes ~8 corpus lines:
      - Raw user query (1)
      - Explicit pairing (1)
      - Alternative phrasing (1)
      - Cross-tool contrast (1)
      - Reversed direction (1)
      - Synonym hint (1)
      - Tool description in context (1)
      - Negative example (1)
    """
    if not DISPATCH_PAIRS.exists():
        if verbose:
            _warn(f"{DISPATCH_PAIRS} not found — skipping dispatch mining")
        return []

    sentences = []
    n_pairs = 0

    with open(DISPATCH_PAIRS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                user_msg = d["messages"][1]["content"].strip()
                tool_name = d["target"]["name"]
                if not user_msg or not tool_name:
                    continue
            except Exception:
                continue

            n_pairs += 1

            # Find category of this tool for cross-tool contrast
            meta = CORE_TOOLS.get(tool_name)
            sibling = None
            if meta:
                # Pick a contrast tool from same category
                same_cat = [t for t, m in CORE_TOOLS.items()
                            if m["category"] == meta["category"] and t != tool_name]
                if same_cat:
                    sibling = same_cat[0]

            # 1. Raw user query (the canonical phrasing)
            sentences.append(user_msg)
            # 2. Explicit pairing
            sentences.append(f'"{user_msg}" should dispatch to {tool_name}.')
            # 3. Alternative phrasing
            sentences.append(f'The query "{user_msg}" is best handled by {tool_name}.')
            # 4. Cross-tool contrast (if we have a sibling)
            if sibling:
                sentences.append(
                    f'For "{user_msg}", call {tool_name}; '
                    f'sibling tool {sibling} handles a different concern.'
                )
            # 5. Reversed direction
            sentences.append(
                f'{tool_name} handles natural-language queries like "{user_msg}".'
            )
            # 6. Tool description in context
            if meta:
                sentences.append(
                    f'When users phrase requests like "{user_msg}", '
                    f'{tool_name} ({meta["key_concept"]}) is the right answer.'
                )

    if verbose:
        _ok(f"Mined {n_pairs} dispatch pairs into {len(sentences)} corpus sentences "
            f"(~{len(sentences)//max(1,n_pairs)} per pair)")

    # ── Cap to avoid drowning auxiliary tokens ────────────────────────────
    # 1564 dispatch pairs × 6 sentences each = 9000+ lines, which would crowd
    # out the auxiliary-token coverage. Random sample to keep balance.
    if cap and len(sentences) > cap:
        import random
        random.seed(42)  # deterministic — re-runs produce the same corpus
        sentences = random.sample(sentences, cap)
        if verbose:
            _ok(f"Capped at {cap} mined sentences (random sample, seed=42) "
                f"to avoid drowning auxiliary tokens")

    return sentences


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--no-mine", action="store_true",
                        help="Skip dispatch_pairs mining (testing only)")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Build + filter token list ────────────────────────────────
    token_list, token_stats = build_token_list(verbose=True)

    # Group by category for the JSON output
    by_cat: dict[str, list[dict]] = {}
    for t in token_list:
        by_cat.setdefault(t["category"], []).append(t)

    # Write new_tokens.json
    new_tokens_path = out_dir / "new_tokens.json"
    new_tokens_path.write_text(json.dumps(by_cat, indent=2))
    _ok(f"Wrote {new_tokens_path}")

    # Per-category text files (one token per line)
    for cat, items in by_cat.items():
        path = out_dir / f"{cat}_terms.txt"
        path.write_text("\n".join(t["token"] for t in items) + "\n")

    # ── Phase 2: Generate corpus from curated content ─────────────────────
    _banner("PHASE 2: Generate corpus (no templates — curated content only)")

    all_sentences: list[str] = []

    # Per-tool sentences
    _info("Generating per-tool natural-language sentences ...")
    per_tool_count = {}
    for tool_name, meta in CORE_TOOLS.items():
        tool_sents = gen_per_tool_sentences(tool_name, meta)
        all_sentences.extend(tool_sents)
        per_tool_count[tool_name] = len(tool_sents)
    avg_per_tool = sum(per_tool_count.values()) / len(per_tool_count)
    _ok(f"Per-tool sentences: {sum(per_tool_count.values())} total "
        f"(avg {avg_per_tool:.1f}/tool across {len(CORE_TOOLS)} tools)")

    # Cross-domain contrastive
    _info("Generating cross-domain contrastive sentences ...")
    cross_domain = gen_cross_domain_contrast()
    all_sentences.extend(cross_domain)
    _ok(f"Cross-domain contrast sentences: {len(cross_domain)}")
    _learn("These directly target the cross-domain cosine 0.62 → < 0.30 goal.")
    _learn("Each sentence forces the model to learn that two tools are unrelated.")

    # Co-occurrence
    _info("Generating co-occurrence sentences ...")
    co_occur = gen_co_occurrence()
    all_sentences.extend(co_occur)
    _ok(f"Co-occurrence sentences: {len(co_occur)}")
    _learn("Co-occurrence is the only mechanism that makes related tokens cluster.")
    _learn("The corpus must EXPLICITLY pair related concepts in the same sentence.")

    # Auxiliary token coverage (KDE/system/git/ML/file_format)
    _info("Generating auxiliary-token sentences (KDE/system/git/ML/file) ...")
    aux_sents = gen_auxiliary_token_sentences()
    all_sentences.extend(aux_sents)
    _ok(f"Auxiliary token sentences: {len(aux_sents)}")
    _learn("Without these, dispatch_pairs mining would crowd out auxiliary tokens")
    _learn("(KWin, Klipper, qdbus6, GGUF, etc.) — each gets ~5-8 sentences here.")

    # Dispatch pair mining
    if not args.no_mine:
        _info("Mining dispatch_pairs.jsonl ...")
        mined = gen_from_dispatch_pairs(verbose=True)
        all_sentences.extend(mined)
        _learn(f"Real user phrasings from dispatch_pairs add {len(mined)} corpus lines")
        _learn("These are the GROUND TRUTH distribution — synthetic data approximates them")

    # Dedup while preserving order
    seen = set()
    deduped = []
    for s in all_sentences:
        if s and s not in seen:
            seen.add(s)
            deduped.append(s)

    # ── Phase 3: Write corpus + report stats ──────────────────────────────
    _banner("PHASE 3: Corpus output + diversity stats")

    corpus_path = out_dir / "corpus.txt"
    corpus_path.write_text("\n".join(deduped) + "\n")
    _ok(f"Wrote {corpus_path} ({len(deduped)} unique sentences)")

    corpus_jsonl_path = out_dir / "corpus.jsonl"
    with open(corpus_jsonl_path, "w") as f:
        for s in deduped:
            f.write(json.dumps({"text": s}) + "\n")
    _ok(f"Wrote {corpus_jsonl_path}")

    # Diversity statistics
    _info("Computing diversity stats ...")
    lengths = [len(s.split()) for s in deduped]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    # Co-occurrence count: how many sentences mention 2+ tool names?
    canon_names = [meta["naming_variants"][0] for meta in CORE_TOOLS.values()]
    multi_tool_count = sum(
        1 for s in deduped if sum(1 for c in canon_names if c in s) >= 2
    )

    # Per-token coverage
    coverage: dict[str, int] = {}
    corpus_text = "\n".join(deduped)
    for t in token_list:
        coverage[t["token"]] = corpus_text.count(t["token"])
    coverage_vals = list(coverage.values())
    weak = [(tok, n) for tok, n in coverage.items() if n < 5]
    avg_cov = sum(coverage_vals) / len(coverage_vals) if coverage_vals else 0
    min_cov = min(coverage_vals) if coverage_vals else 0
    max_cov = max(coverage_vals) if coverage_vals else 0

    print()
    _insight(f"Corpus length distribution: min={min_len}w, avg={avg_len:.1f}w, max={max_len}w")
    if avg_len < 8:
        _warn("Average length below 8 words — consider adding longer sentences")
    elif avg_len > 30:
        _warn("Average length above 30 words — consider shorter ones too")
    else:
        _learn(f"Length variation healthy ({avg_len:.0f}-word average — mix of short and long).")

    _insight(f"Multi-tool sentences (co-occurrence): {multi_tool_count} "
             f"({100*multi_tool_count/len(deduped):.0f}% of corpus)")
    if multi_tool_count < 50:
        _warn("Low co-occurrence count — embeddings may not cluster related tools")
    else:
        _learn(f"Strong co-occurrence signal ({multi_tool_count} sentences) — "
               f"related tools should cluster well.")

    _insight(f"Per-token corpus coverage: min={min_cov}, avg={avg_cov:.1f}, max={max_cov}")
    if weak:
        _warn(f"{len(weak)} tokens have < 5 corpus occurrences (may not converge):")
        for tok, n in sorted(weak, key=lambda x: x[1])[:5]:
            print(f"             {tok!r:30s} {n} occurrences")
    else:
        _learn(f"All {len(token_list)} tokens covered ≥ 5 times.")

    # Final summary
    _banner("SUMMARY: Iteration #3 corpus generation complete")
    print(f"  Token list:     {len(token_list)} tokens "
          f"({token_stats.get('n_dropped', 0)} dropped from {token_stats.get('n_candidates', 0)} candidates)")
    print(f"  Corpus size:    {len(deduped)} unique sentences "
          f"(was 3849 in old templated corpus)")
    print(f"  Sources:        {sum(per_tool_count.values())} per-tool curated")
    print(f"                  {len(cross_domain)} cross-domain contrast")
    print(f"                  {len(co_occur)} co-occurrence")
    print(f"                  {len(aux_sents)} auxiliary token coverage")
    if not args.no_mine:
        print(f"                  {len(mined)} mined from dispatch_pairs (capped)")
    print()
    print(f"  Output:         {out_dir}")
    print()
    _learn("Next: rm -rf training/tokenizer_extended/ then re-run train_tokenizer.py")
    _learn("Goal (from goals.md): cross-domain cosine 0.62 → < 0.40 (intermediate)")
    _learn("                      loss plateau         6.55 → < 5.5  (intermediate)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
