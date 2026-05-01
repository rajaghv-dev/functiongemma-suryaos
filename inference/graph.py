"""
Optional graph-based dependency expansion.

Why graph?
    When tool A requires tool B to have run first (e.g. you must call
    `kde.auth.unlock` before `kde.kmail.send_draft`), the model
    needs to see BOTH in the prompt. Pure retrieval might find A but
    miss B.

    By declaring dependencies in the tool YAML and expanding them here,
    we guarantee prerequisite tools are included — without hard-coding
    knowledge in the agent loop.

Dependency format (in tool YAML):
    dependencies:
      - linux.auth.check      # must be in context when this tool is suggested
      - linux.audio.ensure_running

Implementation:
    A simple BFS up to `depth` hops in the dependency graph.
    The graph is derived lazily from the tool registry at load time.
    No external graph DB required — we use a plain dict of adj-lists.

Tutorial note:
    "Graph expansion" sounds complex. At the code level it's just:
        for each retrieved tool:
            add its declared deps to the candidate set
    Repeat up to `depth` times. That's it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from surya_agent.tool_registry import ToolRegistry


class DependencyGraph:
    """Adjacency list built from tool.yaml 'dependencies' fields."""

    def __init__(self) -> None:
        self._adj: dict[str, list[str]] = {}

    @classmethod
    def from_registry(cls, registry: "ToolRegistry") -> "DependencyGraph":
        g = cls()
        for tool in registry.tools.values():
            deps = getattr(tool, "dependencies", []) or []
            g._adj[tool.name] = list(deps)
        return g

    def expand(self, tool_names: list[str], depth: int = 1) -> list[str]:
        """Return tool_names plus all transitive dependencies up to `depth` hops.

        The original ordering is preserved; dependencies are appended.
        Names that don't exist in the registry are silently skipped.
        """
        if depth <= 0:
            return list(tool_names)
        seen: set[str] = set(tool_names)
        result: list[str] = list(tool_names)
        frontier = list(tool_names)
        for _ in range(depth):
            next_frontier: list[str] = []
            for name in frontier:
                for dep in self._adj.get(name, []):
                    if dep not in seen:
                        seen.add(dep)
                        result.append(dep)
                        next_frontier.append(dep)
            frontier = next_frontier
            if not frontier:
                break
        return result
