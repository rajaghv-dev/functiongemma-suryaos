#!/usr/bin/env python3
"""
MCP dispatcher — single-tool router for SuryaOS.

Why one tool instead of eleven:
    functiongemma:270m with 11 schemas gets confused and picks wrong ones
    or refuses entirely. With ONE schema, it has nothing to pick from —
    it just fills in `query` and the dispatcher does the routing.

    This is the context builder pattern in MCP form:
      model sees:   dispatch(query="how much ram is used")
      dispatcher:   FTS search → memory_usage → run → return result
      model gets:   "RAM: 5.1 GiB used / 30.6 GiB total (17%)"

    Once functiongemma:270m-suryaos is trained, swap back to mcp/system.py.
    The fine-tuned model will be reliable with all 11 schemas visible.

Wiring:
    In opencode.json set:
        "mcp": {
            "desktop": {"type":"local","command":["python3","mcp/dispatcher.py"],"enabled":true},
            "volume":  {"enabled":false},
            "system":  {"enabled":false}
        }
    Then opencode sees ONE tool: desktop_dispatch.
    functiongemma calls desktop_dispatch(query="...") every time.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "desktop", "version": "1.0.0"}

# The single tool the model sees. Simple, unambiguous.
TOOLS = [
    {
        "name": "dispatch",
        "description": (
            "Execute any SuryaOS desktop action. "
            "Pass the user's request as-is in the query field. "
            "Handles: volume, battery, memory, disk, wifi, brightness, "
            "service status, app launch, window focus, notifications, system health."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's request, e.g. 'check battery', 'turn volume down', 'open kate', 'is ollama running'.",
                },
            },
        },
    }
]

# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------

def write_msg(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()

def reply(req_id, result: dict) -> None:
    write_msg({"jsonrpc": "2.0", "id": req_id, "result": result})

def reply_error(req_id, code: int, message: str) -> None:
    write_msg({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})

def ok_result(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": False}

def err_result(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}

# ---------------------------------------------------------------------------
# Dispatcher — context builder → MCP call → result
# ---------------------------------------------------------------------------

def _mcp_call(server_script: str, tool_name: str, args: dict) -> tuple[bool, str]:
    """Call a tool in another MCP server via subprocess (one-shot)."""
    init   = json.dumps({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":PROTOCOL_VERSION,"clientInfo":{"name":"dispatcher"},"capabilities":{}}})
    notif  = json.dumps({"jsonrpc":"2.0","method":"notifications/initialized","params":{}})
    call   = json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":tool_name,"arguments":args}})
    stdin  = "\n".join([init, notif, call, ""])
    script = str(ROOT / "mcp" / server_script)
    try:
        proc = subprocess.run(
            [sys.executable, script],
            input=stdin, capture_output=True, text=True, timeout=15,
        )
        lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
        last  = json.loads(lines[-1]) if lines else {}
        result = last.get("result", {})
        content = result.get("content", [{}])
        text = content[0].get("text", "") if content else ""
        return not result.get("isError", False), text or "ok"
    except Exception as e:
        return False, f"dispatch error: {e}"


def handle_dispatch(args: dict) -> tuple[bool, str]:
    """Route the query to the correct tool using FTS context builder."""
    query = (args.get("query") or "").strip()
    if not query:
        return False, "query must be non-empty"

    # Use the context builder to find the best matching tool.
    try:
        from surya_agent.retrieval.fts import FTSIndex
        from surya_agent.retrieval.graph import DependencyGraph
        from surya_agent.tool_registry import ToolRegistry
        from surya_agent.context_builder import ContextBuilder

        db_path = str(ROOT / "runtime" / "surya_agent.db")
        reg     = ToolRegistry.load(ROOT / "tools")
        fts     = FTSIndex(db_path)
        graph   = DependencyGraph.from_registry(reg)
        cb      = ContextBuilder(fts, graph, reg)
        ctx     = cb.build(query)

        if not ctx.tool_names:
            return False, f"No tool found for: {query!r}"

        tool_name = ctx.tool_names[0]
    except Exception as e:
        return False, f"context builder error: {e}"

    # Map tool name → MCP server + handler name
    # Tools that live in system.py
    SYSTEM_TOOLS = {
        "linux.brightness.set":   ("system.py", "brightness_set"),
        "linux.network.status":   ("system.py", "network_status"),
        "linux.battery.status":   ("system.py", "battery_status"),
        "linux.memory.usage":     ("system.py", "memory_usage"),
        "linux.disk.usage":       ("system.py", "disk_usage"),
        "linux.service.status":   ("system.py", "service_status"),
        "kde.krunner.launch":     ("system.py", "krunner_launch"),
        "kde.window.focus":       ("system.py", "window_focus"),
        "linux.metrics.summary":  ("system.py", "metrics_summary"),
        "kde.notifications.send": ("system.py", "notifications_send"),
    }
    # Tools that live in volume.py
    VOLUME_TOOLS = {
        "linux.volume.set": ("volume.py", "volume_change"),
    }

    server_script, handler = (
        SYSTEM_TOOLS.get(tool_name) or
        VOLUME_TOOLS.get(tool_name) or
        (None, None)
    )

    if server_script is None:
        return False, f"No MCP handler for tool: {tool_name!r}"

    # Build minimal args for the target tool.
    # For service_status, infer the service name from the query.
    tool_args: dict = {}
    if tool_name == "linux.service.status":
        # Extract service name: "is bluetooth running" → bluetooth
        import re
        words = re.findall(r'\b\w+\b', query.lower())
        known = {"ollama","bluetooth","networkmanager","sshd","ssh","cups",
                 "pipewire","pulseaudio","docker","ufw","apache2","nginx"}
        for w in words:
            if w in known:
                tool_args["name"] = w
                break
        if not tool_args.get("name"):
            tool_args["name"] = "ollama"  # safe default

    elif tool_name == "kde.krunner.launch":
        import re, shutil
        # Extract app name: "open dolphin" → dolphin, "launch kate" → kate
        m = re.search(r'\b(?:open|launch|start|run)\s+(\w+)', query.lower())
        app = m.group(1) if m else query.split()[-1]
        tool_args["app"] = app
        # Fire-and-forget: kstart can block until window opens (slow apps).
        # Launch via nohup and return immediately instead of going through MCP.
        for launcher in ("kstart6", "kstart5", "kstart"):
            if shutil.which(launcher):
                import subprocess as _sp
                _sp.Popen([launcher, "--windowclass", app, app],
                          stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                return True, f"Launched {app!r}."
        if shutil.which(app):
            import subprocess as _sp
            _sp.Popen([app], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            return True, f"Launched {app!r}."
        return False, f"Cannot launch {app!r}: not found"

    elif tool_name == "linux.disk.usage":
        tool_args["path"] = "/"

    elif tool_name == "linux.volume.set":
        import re
        direction = "down"
        if re.search(r'\b(up|louder|raise|increase|higher|more)\b', query.lower()):
            direction = "up"
        tool_args = {"direction": direction}
        m = re.search(r'(\d+)\s*(?:percent|%)', query.lower())
        if m:
            tool_args["step"] = int(m.group(1))

    elif tool_name == "linux.brightness.set":
        import re
        direction = "down"
        if re.search(r'\b(up|brighter|raise|increase|higher|more)\b', query.lower()):
            direction = "up"
        tool_args = {"direction": direction}

    elif tool_name == "kde.notifications.send":
        # Extract message from query
        parts = query.split(":", 1)
        if len(parts) == 2:
            tool_args = {"title": "SuryaOS", "message": parts[1].strip()}
        else:
            tool_args = {"title": "SuryaOS", "message": query}

    return _mcp_call(server_script, handler, tool_args)


# ---------------------------------------------------------------------------
# Request router
# ---------------------------------------------------------------------------

def handle_request(req: dict) -> None:
    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        reply(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
        return

    if method == "notifications/initialized":
        return

    if method == "tools/list":
        reply(req_id, {"tools": TOOLS})
        return

    if method == "tools/call":
        params  = req.get("params", {})
        name    = (params.get("name") or "").strip()
        args    = params.get("arguments", {}) or {}

        if name != "dispatch":
            reply_error(req_id, -32602, f"unknown tool: {name!r}. Only 'dispatch' is available.")
            return

        try:
            ok, text = handle_dispatch(args)
        except Exception as exc:
            reply(req_id, err_result(f"dispatcher error: {exc}"))
            return

        reply(req_id, ok_result(text) if ok else err_result(text))
        return

    if req_id is not None:
        reply_error(req_id, -32601, f"method not found: {method!r}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        handle_request(req)


if __name__ == "__main__":
    main()
