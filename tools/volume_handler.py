#!/usr/bin/env python3
"""
MCP server: volume control for SuryaOS.

Speaks JSON-RPC 2.0 over stdio. Single tool: volume_change(direction, step).

Methods handled:
  - initialize                → server capabilities + protocolVersion   [stage 3]
  - notifications/initialized → noop (one-way notification)             [stage 3]
  - tools/list                → returns the volume_change tool schema   [stage 4]
  - tools/call                → executes the tool via pactl             [stage 5]

Run manually for stage-specific probes:
  ./scripts/manual/13-mcp-handshake.sh    # tests initialize
  ./scripts/manual/14-mcp-tools-list.sh   # tests tools/list
  ./scripts/manual/15-mcp-tools-call.sh   # tests tools/call (changes volume!)

Inspect the actual chat with the daemon:
  journalctl -u ollama -f                 # daemon-side
  cat /tmp/oc-probe-15-toolcall.log       # client-side
"""

import json
import shutil
import subprocess
import sys

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "volume", "version": "0.3.0"}

# Small models (including functiongemma:270m) sometimes emit direction
# values that convey the right intent but don't match the enum literally.
# We normalise here rather than reject, so the tool is robust to model phrasing.
_DIRECTION_UP = {"up", "u", "increase", "raise", "higher", "louder", "more"}
_DIRECTION_DOWN = {"down", "d", "decrease", "lower", "quieter", "softer", "less", "reduce"}


def _normalise_direction(raw: str) -> str | None:
    """Map a model-emitted direction string to 'up' or 'down', or None if unrecognised."""
    v = str(raw).strip().lower()
    if v in _DIRECTION_UP:
        return "up"
    if v in _DIRECTION_DOWN:
        return "down"
    return None

# Tool catalog declared by tools/list. Single tool today; structure
# leaves room for additions without restructuring the dispatcher.
TOOLS = [
    {
        "name": "volume_change",
        "description": "Adjust the audio volume up or down by a percentage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to change volume.",
                },
                "step": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Percentage to change volume by (default 5).",
                },
            },
            "required": ["direction"],
        },
    }
]


def write_msg(payload):
    """Emit one line of JSON to stdout and flush."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def reply(req_id, result):
    write_msg({"jsonrpc": "2.0", "id": req_id, "result": result})


def reply_error(req_id, code, message):
    write_msg({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def run_pactl(direction, step):
    """Execute the volume change. Returns (ok: bool, message: str)."""
    if not shutil.which("pactl"):
        return False, "pactl not found on PATH"
    sign = "+" if direction == "up" else "-"
    cmd = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{step}%"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, f"Volume {direction} by {step}%."
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or str(e)).strip()


def handle_request(req):
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
        # One-way notification per JSON-RPC; no response.
        return

    if method == "tools/list":
        reply(req_id, {"tools": TOOLS})
        return

    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {}) or {}

        if name != "volume_change":
            reply_error(req_id, -32602, f"unknown tool: {name!r}")
            return

        direction_raw = args.get("direction")
        direction = _normalise_direction(str(direction_raw)) if direction_raw is not None else None
        if direction is None:
            reply_error(req_id, -32602,
                        f"direction not recognised: {direction_raw!r}. "
                        f"Expected: up/down (or synonyms: raise/lower/increase/decrease)")
            return

        step = args.get("step", 5)
        if not isinstance(step, int) or not (1 <= step <= 100):
            reply_error(req_id, -32602,
                        f"step must be int 1..100, got: {step!r}")
            return

        ok, msg = run_pactl(direction, step)
        reply(req_id, {
            "content": [{"type": "text", "text": msg}],
            "isError": not ok,
        })
        return

    # Unknown method with an id → method-not-found error.
    # Notifications (no id) are silently ignored per JSON-RPC.
    if req_id is not None:
        reply_error(req_id, -32601, f"method not found: {method!r}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            # Skip garbled lines. Don't crash the server.
            continue
        handle_request(req)


if __name__ == "__main__":
    main()
