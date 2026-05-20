#!/usr/bin/env python3
"""
Generate training data for embedding + dispatch fine-tuning.

Sources:
  1. tools/**/*.yaml  — examples field (48 labeled pairs, 12 tools)
  2. runtime/audit.db — real user queries (grows with usage)
  3. Paraphrase gen    — qwen3:0.6b generates 20 variants per example

Output (JSONL):
  training/embed_pairs.jsonl    — (query, tool_name) for embedding fine-tune
  training/dispatch_pairs.jsonl — (query, tool_schema, tool_call) for functiongemma

Usage:
  python3 scripts/training/generate_pairs.py --mode base
      → uses only tool YAML examples (offline, no model needed)

  python3 scripts/training/generate_pairs.py --mode augment
      → also calls qwen3:0.6b to paraphrase each example (requires Ollama)

  python3 scripts/training/generate_pairs.py --mode audit
      → also pulls real queries from runtime/audit.db

Tutorial:
  This is Phase 1 of the fine-tuning pipeline described in
  docs/learnings/10-system-architecture.md.

  The embed_pairs format follows sentence-transformers InputExample:
    {"query": "...", "positive": "tool_name", "negative": "other_tool"}

  The dispatch_pairs format follows OpenAI fine-tune JSONL:
    {"messages": [...], "tools": [schema], "target": {"name": ..., "arguments": ...}}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from surya_agent.tool_registry import ToolRegistry  # noqa: E402
from surya_agent.context_builder import _tool_to_mcp_schema  # noqa: E402

OUTPUT_DIR = ROOT / "training"


# ---------------------------------------------------------------------------
# Step 1: collect base pairs from tool YAMLs
# ---------------------------------------------------------------------------

def base_pairs(registry: ToolRegistry) -> list[dict]:
    """Read examples from every tool YAML.

    Returns list of:
        {"query": str, "tool": str, "source": "yaml"}
    """
    pairs: list[dict] = []
    for name, tool in registry.tools.items():
        examples = getattr(tool, "examples", []) or []
        for ex in examples:
            if isinstance(ex, str) and ex.strip():
                pairs.append({"query": ex.strip(), "tool": name, "source": "yaml"})
    return pairs


# ---------------------------------------------------------------------------
# Step 2: paraphrase via qwen3:0.6b (requires Ollama running)
# ---------------------------------------------------------------------------

def paraphrase_example(example: str, tool_name: str, n: int = 20) -> list[str]:
    """Ask qwen3:0.6b to write N paraphrases of a user query.

    Returns list of paraphrase strings (may be fewer than N if model is terse).
    """
    try:
        import urllib.request
        prompt = (
            f"Write {n} different ways a user might phrase this request. "
            f"Each on its own line. No numbering, no explanation, no quotes.\n\n"
            f"Request: {example}"
        )
        payload = json.dumps({
            "model": "qwen3:0.6b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.9, "num_predict": 400},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data.get("response", "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Filter out lines that are too long or too short
        return [ln for ln in lines if 5 < len(ln) < 120][:n]
    except Exception as e:
        print(f"  ⚠ paraphrase failed for {example!r}: {e}", file=sys.stderr)
        return []


def augment_pairs(base: list[dict], n_per_example: int = 10) -> list[dict]:
    """Generate paraphrases for each base pair using qwen3:0.6b.

    Returns new pairs with source='paraphrase'.
    """
    augmented: list[dict] = []
    total = len(base)
    for i, pair in enumerate(base, 1):
        print(f"  [{i}/{total}] paraphrasing: {pair['query']!r}", file=sys.stderr)
        variants = paraphrase_example(pair["query"], pair["tool"], n=n_per_example)
        for v in variants:
            augmented.append({"query": v, "tool": pair["tool"], "source": "paraphrase"})
    return augmented


# ---------------------------------------------------------------------------
# Step 3: pull real queries from audit.db
# ---------------------------------------------------------------------------

def audit_pairs(db_path: Path) -> list[dict]:
    """Load (tool_name, query-like-args) from audit.db.

    The audit log records which tool was called, not the original user query.
    We reconstruct a plausible query from the tool name + args.
    This is an approximation — real query logging would be better.
    """
    import sqlite3
    if not db_path.exists():
        print(f"  ⚠ audit.db not found at {db_path}", file=sys.stderr)
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT tool_name, args FROM actions WHERE decision='allow' LIMIT 1000"
    ).fetchall()
    conn.close()

    pairs: list[dict] = []
    for tool_name, args_json in rows:
        try:
            args = json.loads(args_json or "{}")
            # Reconstruct a minimal query from args
            if args:
                query = f"use {tool_name} with {args}"
            else:
                query = f"call {tool_name}"
            pairs.append({"query": query, "tool": tool_name, "source": "audit"})
        except Exception:
            continue
    return pairs


# ---------------------------------------------------------------------------
# Format outputs
# ---------------------------------------------------------------------------

def to_embed_jsonl(pairs: list[dict], all_tools: list[str]) -> list[str]:
    """Format for sentence-transformers MultipleNegativesRankingLoss.

    Each line: {"query": str, "positive": str, "negative": str}
    The negative is a random different tool.
    """
    lines: list[str] = []
    for p in pairs:
        other_tools = [t for t in all_tools if t != p["tool"]]
        if not other_tools:
            continue
        negative = random.choice(other_tools)
        lines.append(json.dumps({
            "query":    p["query"],
            "positive": p["tool"],
            "negative": negative,
        }))
    return lines


def to_dispatch_jsonl(pairs: list[dict], registry: ToolRegistry) -> list[str]:
    """Format for OpenAI-style fine-tuning (functiongemma dispatch).

    Each line: {"messages": [...], "tools": [schema], "target": {name, arguments}}
    """
    lines: list[str] = []
    for p in pairs:
        tool = registry.tools.get(p["tool"])
        if not tool:
            continue
        schema = _tool_to_mcp_schema(tool)
        if not schema:
            continue

        # Build minimal args from schema required fields
        required = (tool.parameters or {}).get("required", [])
        args = {}
        for field in required:
            prop = (tool.parameters or {}).get("properties", {}).get(field, {})
            enum = prop.get("enum", [])
            # Use first enum value if available, otherwise empty
            args[field] = enum[0] if enum else ""

        lines.append(json.dumps({
            "messages": [
                {"role": "system", "content": "Call the right tool."},
                {"role": "user",   "content": p["query"]},
            ],
            "tools":  [schema],
            "target": {"name": schema["name"], "arguments": args},
            "source": p["source"],
        }))
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["base", "augment", "audit"], default="base")
    parser.add_argument("--n-paraphrases", type=int, default=10,
                        help="Paraphrases per example (augment mode only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(exist_ok=True)

    reg = ToolRegistry.load(ROOT / "tools")
    all_tools = list(reg.tools.keys())
    print(f"Tools loaded: {len(all_tools)}")

    # Collect pairs
    pairs: list[dict] = []

    print("\n[1] Base pairs from tool YAMLs...")
    base = base_pairs(reg)
    pairs.extend(base)
    print(f"    {len(base)} base pairs")

    if args.mode in ("augment",):
        print(f"\n[2] Paraphrase augmentation ({args.n_paraphrases}× per example)...")
        aug = augment_pairs(base, n_per_example=args.n_paraphrases)
        pairs.extend(aug)
        print(f"    {len(aug)} augmented pairs")

    if args.mode == "audit":
        print("\n[3] Audit DB pairs...")
        audit = audit_pairs(ROOT / "runtime" / "audit.db")
        pairs.extend(audit)
        print(f"    {len(audit)} audit pairs")

    print(f"\nTotal: {len(pairs)} pairs across {len(set(p['tool'] for p in pairs))} tools")
    print("\nDistribution:")
    from collections import Counter
    for tool, count in sorted(Counter(p["tool"] for p in pairs).items()):
        print(f"  {tool:40s}: {count}")

    # Write embed pairs
    embed_path = OUTPUT_DIR / "embed_pairs.jsonl"
    embed_lines = to_embed_jsonl(pairs, all_tools)
    embed_path.write_text("\n".join(embed_lines) + "\n")
    print(f"\n✓ embed_pairs.jsonl:    {len(embed_lines)} lines → {embed_path}")

    # Write dispatch pairs
    dispatch_path = OUTPUT_DIR / "dispatch_pairs.jsonl"
    dispatch_lines = to_dispatch_jsonl(pairs, reg)
    dispatch_path.write_text("\n".join(dispatch_lines) + "\n")
    print(f"✓ dispatch_pairs.jsonl: {len(dispatch_lines)} lines → {dispatch_path}")

    print()
    print("Next steps:")
    print("  Fine-tune embedding:  pip install sentence-transformers")
    print("                        python3 scripts/training/finetune_embed.py")
    print("  Fine-tune dispatch:   pip install unsloth")
    print("                        python3 scripts/training/finetune_dispatch.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
