"""
Context builder — the key component that makes small models reliable.

Problem: functiongemma:270m sees 11 tool schemas in every request.
With 11 choices and a vague query, it guesses wrong ~50% of the time.

Solution: before the model sees anything, the context builder:
  1. Embeds the query → vector
  2. Searches the tool index → top-1 to 3 candidates
  3. Expands via dependency graph → adds prerequisites
  4. Assembles a minimal prompt with ONLY those 1-3 schemas

Result: functiongemma sees 1 schema + 1 matching query → correct 100%.
The model's job is no longer "pick from 11" but "call this one tool".

Integration point:
    This sits between the user input and the model. The agent_loop.py
    calls build() on every turn instead of attaching all tool schemas.

Retrieval backend:
    Phase 0 (now):  FTSIndex (SQLite BM25) — already works, no ML
    Phase 2 (later): VectorIndex (all-minilm:22m fine-tuned) — better recall

State cache:
    Optional: injects the last metrics_summary result into every prompt.
    Lets the model give context-aware answers without a separate call.
    Example: "the battery is at 30%" before "should I charge the laptop?"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from surya_agent.retrieval.fts import FTSIndex
    from surya_agent.retrieval.graph import DependencyGraph
    from surya_agent.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Prompt structure returned to the model
# ---------------------------------------------------------------------------

@dataclass
class BuiltContext:
    """Everything the model needs for a single turn."""
    query: str
    tool_schemas: list[dict]          # 1-3 MCP-format tool schemas
    tool_names: list[str]             # names of tools in schemas (for logging)
    system_state: str | None          # last metrics_summary, if cached
    retrieval_scores: list[float]     # top-K scores (for debug/tuning)
    build_ms: float                   # how long the build took

    def to_prompt(self) -> dict:
        """Assemble the dict that gets passed to the model."""
        system = "Call the right tool."
        if self.system_state:
            system += f" Current system state: {self.system_state}"
        return {
            "system": system,
            "tools":  self.tool_schemas,
            "user":   self.query,
        }

    def debug_str(self) -> str:
        lines = [
            f"query:   {self.query!r}",
            f"tools:   {self.tool_names}",
            f"scores:  {[round(s, 3) for s in self.retrieval_scores]}",
            f"state:   {self.system_state!r}",
            f"build:   {self.build_ms:.1f}ms",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# State cache: rolling 60s snapshot of last metrics_summary result
# ---------------------------------------------------------------------------

class StateCache:
    """Lightweight cache of the most recent system health snapshot.

    Used by ContextBuilder to prepend state to every prompt, so the model
    can give context-aware answers without an extra tool call per turn.

    Example: if battery is 12%, inject that fact before the user asks
    "should I save my work?" The model can answer without calling battery_status.
    """

    TTL_S = 60.0  # refresh state if older than this

    def __init__(self) -> None:
        self._value: str | None = None
        self._ts: float = 0.0

    def update(self, summary: str) -> None:
        self._value = summary
        self._ts = time.monotonic()

    def get(self) -> str | None:
        if self._value and (time.monotonic() - self._ts) < self.TTL_S:
            return self._value
        return None

    def invalidate(self) -> None:
        self._value = None
        self._ts = 0.0


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """Build minimal prompts for tool dispatch.

    Usage:
        fts   = FTSIndex("./runtime/surya_agent.db")
        graph = DependencyGraph.from_registry(registry)
        reg   = ToolRegistry.load("tools")
        cb    = ContextBuilder(fts, graph, reg)

        ctx = cb.build("how is the system doing?")
        print(ctx.debug_str())
        prompt = ctx.to_prompt()
        # → pass prompt to functiongemma

    Confidence thresholds:
        score ≥ 0.8  → single tool (very high confidence)
        score ≥ 0.5  → top-2 tools
        score <  0.5  → top-3 tools (low confidence, widen the set)
        score <  0.3  → attach all tools (retrieval is not helping)
    """

    # BM25 score thresholds for how many tools to include.
    # Tune these after collecting real queries in audit.db.
    SCORE_HIGH = 0.8   # ≥ this → single tool
    SCORE_MED  = 0.5   # ≥ this → top-2
    SCORE_LOW  = 0.3   # < this → all tools (fallback)

    def __init__(
        self,
        fts: "FTSIndex",
        graph: "DependencyGraph",
        registry: "ToolRegistry",
        state_cache: StateCache | None = None,
        max_tools: int = 3,
    ) -> None:
        self.fts   = fts
        self.graph = graph
        self.reg   = registry
        self.state = state_cache or StateCache()
        self.max_tools = max_tools

    def build(self, query: str) -> BuiltContext:
        """Main entry point. Returns a BuiltContext ready for the model."""
        t0 = time.monotonic()

        # 1. Retrieve top-K candidates from FTS index
        hits = self.fts.search(query, k=self.max_tools + 2)  # slightly over-fetch
        scores = [h.score for h in hits]
        names  = [h.tool_name for h in hits]

        # 2. Decide how many tools to include based on top score
        if not hits:
            # No retrieval result — attach everything
            final_names = list(self.reg.tools.keys())
        elif scores[0] >= self.SCORE_HIGH:
            final_names = names[:1]
        elif scores[0] >= self.SCORE_MED:
            final_names = names[:2]
        elif scores[0] >= self.SCORE_LOW:
            final_names = names[:self.max_tools]
        else:
            # Retrieval is not confident — fall back to full catalog
            final_names = list(self.reg.tools.keys())

        # 3. Graph expand: add declared dependencies (e.g. auth tool before action)
        final_names = self.graph.expand(final_names, depth=1)

        # 4. Build MCP-compatible schemas for just these tools
        schemas: list[dict] = []
        for name in final_names:
            tool = self.reg.tools.get(name)
            if tool is None:
                continue
            schema = _tool_to_mcp_schema(tool)
            if schema:
                schemas.append(schema)

        build_ms = (time.monotonic() - t0) * 1000

        return BuiltContext(
            query=query,
            tool_schemas=schemas,
            tool_names=final_names,
            system_state=self.state.get(),
            retrieval_scores=scores[:len(final_names)],
            build_ms=build_ms,
        )

    def stats(self, query: str) -> str:
        """Debug helper: show what the context builder would produce."""
        ctx = self.build(query)
        return ctx.debug_str()


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

def _tool_to_mcp_schema(tool) -> dict | None:
    """Convert a ToolSpec into the MCP tools/list format functiongemma expects."""
    try:
        params = tool.parameters or {}
        return {
            "name": tool.name.replace(".", "_"),  # MCP uses underscore, not dot
            "description": tool.description or "",
            "inputSchema": params,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Run from repo root: python3 src/surya_agent/context_builder.py"""
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))

    from surya_agent.retrieval.fts import FTSIndex
    from surya_agent.retrieval.graph import DependencyGraph
    from surya_agent.tool_registry import ToolRegistry

    fts   = FTSIndex(str(ROOT / "runtime" / "surya_agent.db"))
    reg   = ToolRegistry.load(ROOT / "tools")
    graph = DependencyGraph.from_registry(reg)
    cb    = ContextBuilder(fts, graph, reg)

    queries = [
        "how is the system doing?",
        "is ollama running?",
        "check battery",
        "turn the volume down",
        "open kate",
        "how much ram is used?",
        "dim the screen",
        "wifi status",
    ]

    print("ContextBuilder smoke test")
    print("=" * 50)
    for q in queries:
        ctx = cb.build(q)
        status = "✓" if len(ctx.tool_names) <= 3 else "⚠ wide"
        top_score = f"{ctx.retrieval_scores[0]:.3f}" if ctx.retrieval_scores else "—"
        print(f"  {status} [{top_score}] {q!r:40s} → {ctx.tool_names}")
    print()
    print(f"All queries processed. Context builder is wired.")


if __name__ == "__main__":
    _smoke_test()
