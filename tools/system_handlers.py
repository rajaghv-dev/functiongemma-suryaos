#!/usr/bin/env python3
"""
MCP server: system status + KDE desktop actions for SuryaOS.

Speaks JSON-RPC 2.0 over stdio. Multi-tool server — Group 1 (all green/low-risk).

Tools exposed:
  brightness_set   — screen brightness up/down (brightnessctl)
  network_status   — wifi/network connection state (nmcli)
  battery_status   — battery level + time remaining (acpi / upower)
  memory_usage     — RAM stats (free -h)
  disk_usage       — disk space on a path (df -h)
  service_status   — systemd service active/inactive (systemctl is-active)
  krunner_launch   — launch an app on the KDE desktop (kstart6/kstart5)
  window_focus     — raise a window to the front by title (wmctrl)
  metrics_summary  — compact health snapshot via Netdata REST API (:19999)

All tools are:
  - read-only or reversible
  - no confirmation required (green tier in policy)
  - safe to expose in the coder agent without file-management confusion

Methods handled:
  - initialize                → capabilities + protocolVersion
  - notifications/initialized → noop (one-way notification)
  - tools/list                → full TOOLS catalog
  - tools/call                → dispatch to handler by tool name

Run manually for probe/debug:
  echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"probe"},"capabilities":{}}}' | python3 mcp/system.py
  echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | python3 mcp/system.py
  echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"network_status","arguments":{}}}' | python3 mcp/system.py
  echo '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"service_status","arguments":{"name":"ollama"}}}' | python3 mcp/system.py
"""

import json
import shutil
import subprocess
import sys

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "system", "version": "0.2.0"}

# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------
# Each entry is a standard MCP tool schema. The model sees these via
# tools/list and picks one per turn. Keep descriptions short and action-
# oriented — functiongemma:270m is sensitive to description length.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "brightness_set",
        "description": "Change screen brightness. Use for: dim, bright, darker, lighter, brightness up/down.",
        "inputSchema": {
            "type": "object",
            "required": ["direction"],
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to change brightness.",
                },
                "step": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Percentage to change brightness by (default 10).",
                },
            },
        },
    },
    {
        "name": "network_status",
        "description": "Check wifi/internet/network status. Use for: wifi, online, connected, network, internet, ip address.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "battery_status",
        "description": "Check battery level and charging state. Use for: battery, charge, power, how long until dead.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "memory_usage",
        "description": "Check RAM/memory usage. Use for: ram, memory, how much memory, is memory full.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "disk_usage",
        "description": "Check disk/storage space. Use for: disk, storage, space, how full. Use path='/' for root disk.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filesystem path to check. Use '/' for root, '/home' for home. Default: '/'.",
                },
            },
        },
    },
    {
        "name": "service_status",
        "description": "Check if a service/daemon is running. Use for: 'is X running?', 'is X active?', 'is bluetooth on?'. Set name=service (e.g. ollama, bluetooth, NetworkManager, sshd).",
        "inputSchema": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Service name: ollama, bluetooth, NetworkManager, sshd, cups, apache2, nginx, etc.",
                },
            },
        },
    },
    {
        "name": "krunner_launch",
        "description": "Open/launch/start an application. Use for: open, launch, start, run app. Set app=name (e.g. kate, dolphin, konsole, firefox, spotify).",
        "inputSchema": {
            "type": "object",
            "required": ["app"],
            "properties": {
                "app": {
                    "type": "string",
                    "description": "App name or command to launch, e.g. kate, dolphin, konsole.",
                },
            },
        },
    },
    {
        "name": "window_focus",
        "description": "Bring a window to the foreground by matching its title.",
        "inputSchema": {
            "type": "object",
            "required": ["title"],
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Case-insensitive substring of the window title.",
                },
            },
        },
    },
    {
        "name": "metrics_summary",
        "description": "Get system health: CPU%, RAM, disk, battery, wifi, Ollama. Use for: 'how is the system?', 'system status', 'health check', 'is the system overloaded?'.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "notifications_send",
        "description": "Send a desktop notification/alert/popup. Use for: notify, alert, remind, popup, 'send notification', 'tell me when'.",
        "inputSchema": {
            "type": "object",
            "required": ["title", "message"],
            "properties": {
                "title":   {"type": "string", "description": "Notification title."},
                "message": {"type": "string", "description": "Notification body text."},
                "icon":    {"type": "string", "description": "Icon name (optional, e.g. dialog-information)."},
            },
        },
    },
    {
        "name": "ebpf_summary",
        "description": "Deep kernel-level stats via eBPF: Ollama page faults, open file descriptors, TCP sockets, syscall rates. Use for: 'what is ollama doing?', 'kernel stats', 'low-level system activity'.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def write_msg(payload: dict) -> None:
    """Emit one line of JSON to stdout and flush immediately."""
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
# Tool handlers
# Each handler receives args (dict) and returns (ok: bool, text: str).
# No subprocess exceptions escape — they are caught and surfaced as errors.
# ---------------------------------------------------------------------------

def _run(argv: list[str], *, timeout: float = 10.0) -> tuple[bool, str]:
    """Run a command safely. Returns (ok, output_or_error)."""
    binary = argv[0]
    if not shutil.which(binary):
        return False, f"{binary!r} not found on PATH"
    try:
        proc = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        text = (proc.stdout or proc.stderr or "").strip()
        return proc.returncode == 0, text or f"exit {proc.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except OSError as e:
        return False, f"OS error: {e}"


# Synonyms recognised for brightness direction (mirrors volume.py pattern).
_DIR_UP   = {"up", "u", "increase", "raise", "higher", "brighter", "more"}
_DIR_DOWN = {"down", "d", "decrease", "lower", "dimmer", "softer", "less", "dim", "reduce"}


def _normalise_dir(raw: str) -> str | None:
    v = str(raw).strip().lower()
    if v in _DIR_UP:
        return "up"
    if v in _DIR_DOWN:
        return "down"
    return None


def handle_brightness_set(args: dict) -> tuple[bool, str]:
    direction_raw = args.get("direction")
    direction = _normalise_dir(str(direction_raw)) if direction_raw else None
    if direction is None:
        return False, f"direction not recognised: {direction_raw!r}. Expected: up/down"

    step = args.get("step", 10)
    if not isinstance(step, int) or not (1 <= step <= 100):
        return False, f"step must be int 1..100, got: {step!r}"

    sign = "+" if direction == "up" else "-"
    ok, text = _run(["brightnessctl", "set", f"{sign}{step}%"])
    if ok:
        text = text or f"Brightness {direction} by {step}%."
    return ok, text


def handle_network_status(_args: dict) -> tuple[bool, str]:
    """Report active network/wifi connections via nmcli.

    Shows NAME, TYPE, STATE, DEVICE in terse colon-delimited format.
    """
    ok, text = _run(["nmcli", "-t", "-f", "NAME,TYPE,STATE,DEVICE", "connection", "show", "--active"])
    if ok and not text:
        text = "No active connections."
    return ok, text


def handle_battery_status(_args: dict) -> tuple[bool, str]:
    """Report battery level and charging state.

    Prefers acpi (concise output); falls back to upower when acpi is absent.
    """
    if shutil.which("acpi"):
        return _run(["acpi", "-b"])
    return _run(["upower", "-i", "/org/freedesktop/UPower/devices/battery_BAT0"])


def handle_memory_usage(_args: dict) -> tuple[bool, str]:
    """Report RAM usage. Tries Netdata first (structured MiB), falls back to free."""
    import urllib.request
    try:
        url = "http://localhost:19999/api/v1/data?chart=system.ram&after=-3&points=1&format=json"
        with urllib.request.urlopen(url, timeout=2) as r:
            d = json.loads(r.read())
        if d.get("data"):
            row = d["data"][0]
            labels = d.get("labels", [])
            idx = {l: i for i, l in enumerate(labels)}
            used    = row[idx.get("used",    0)]
            free_   = row[idx.get("free",    0)]
            cached  = row[idx.get("cached",  0)]
            buffers = row[idx.get("buffers", 0)]
            total   = used + free_ + cached + buffers
            avail   = free_ + cached
            pct     = (used / total * 100) if total else 0
            return True, (f"RAM: {used/1024:.1f} GiB used / {total/1024:.1f} GiB total "
                          f"({pct:.0f}% used, {avail/1024:.1f} GiB available)")
    except Exception:
        pass
    return _run(["free", "-h"])


def handle_disk_usage(args: dict) -> tuple[bool, str]:
    """Report disk usage. Tries Netdata first (structured GiB), falls back to df."""
    import urllib.request
    path = args.get("path", "/")
    if not isinstance(path, str) or not path:
        path = "/"

    # Try Netdata for root disk (structured data)
    if path in ("/", "/home"):
        chart = "disk_space./" if path == "/" else "disk_space./home"
        try:
            url = f"http://localhost:19999/api/v1/data?chart={chart}&after=-3&points=1&format=json"
            with urllib.request.urlopen(url, timeout=2) as r:
                d = json.loads(r.read())
            if d.get("data"):
                row = d["data"][0]
                labels = d.get("labels", [])
                idx = {l: i for i, l in enumerate(labels)}
                used  = row[idx.get("used", 0)]
                avail = row[idx.get("avail", 0)]
                total = used + avail
                pct   = (used / total * 100) if total else 0
                return True, f"Disk {path}: {used:.0f} GiB used / {total:.0f} GiB total ({pct:.0f}% full, {avail:.0f} GiB free)"
        except Exception:
            pass

    # Fallback: df
    return _run(["df", "-h", path])


def handle_service_status(args: dict) -> tuple[bool, str]:
    """Check whether a systemd service is active or inactive.

    Returns the service state string (active, inactive, failed, etc.).
    Exit code 3 from systemctl is-active (inactive) is treated as a valid
    answer, not an error.
    """
    name = (args.get("name") or "").strip()
    if not name:
        return False, "service name must be non-empty"

    # systemctl is-active exits 0 if active, 3 if inactive — both are valid answers.
    # We check exit code directly rather than relying on _run's ok flag.
    if not shutil.which("systemctl"):
        return False, "'systemctl' not found on PATH"
    try:
        proc = subprocess.run(
            ["systemctl", "is-active", name],
            capture_output=True, text=True, timeout=10.0,
        )
        state = (proc.stdout or "unknown").strip()
        return True, f"{name}: {state}"
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)


def handle_krunner_launch(args: dict) -> tuple[bool, str]:
    """Launch an application on the KDE desktop via kstart6/kstart5/kstart.

    Uses Popen (fire-and-forget) so the MCP call returns immediately even
    for slow-starting apps. Falls back to a bare exec if kstart is absent.
    """
    app = (args.get("app") or "").strip()
    if not app:
        return False, "app must be non-empty"

    # Fire-and-forget: kstart can block until window appears (slow apps).
    # Use Popen so the MCP call returns immediately.
    import subprocess as _sub
    for launcher in ("kstart6", "kstart5", "kstart"):
        if shutil.which(launcher):
            _sub.Popen([launcher, "--windowclass", app, app],
                       stdout=_sub.DEVNULL, stderr=_sub.DEVNULL)
            return True, f"Launched {app!r}."

    # Bare exec as last resort — no KDE session integration, but it works.
    if shutil.which(app):
        ok, text = _run([app])
        return ok, text or f"Launched {app!r}."

    return False, f"Cannot launch {app!r}: kstart not found and {app!r} not on PATH"


def handle_window_focus(args: dict) -> tuple[bool, str]:
    """Raise a window to the front by matching a substring of its title.

    Uses wmctrl -a for a case-insensitive partial title match.
    """
    title = (args.get("title") or "").strip()
    if not title:
        return False, "title must be non-empty"

    # wmctrl -a does a case-insensitive partial title match and raises the window.
    ok, text = _run(["wmctrl", "-a", title])
    return ok, text or (f"Focused window matching: {title!r}" if ok else f"No window matching {title!r}")


def handle_metrics_summary(_args: dict) -> tuple[bool, str]:
    """Compact health snapshot sourced from Netdata REST API + fallbacks.

    Queries Netdata at http://localhost:19999/api/v1/data for each chart.
    Falls back to direct system commands if Netdata is unreachable.

    Metric names confirmed on Netdata v2.10 (this machine):
      system.cpu               → percentage (user + system dimensions)
      system.ram               → MiB (free/used/cached/buffers dimensions)
      disk_space./             → GiB (used/avail dimensions)
      powersupply_capacity.BAT0→ percentage (capacity dimension)
      net.wlo1                 → kilobits/s (received/sent dimensions)
      app.ollama_cpu_utilization → percentage (user/system dimensions)
    """
    import re
    import urllib.request
    import urllib.error

    NETDATA = "http://localhost:19999/api/v1"

    def nd(chart: str) -> dict | None:
        """Fetch 1 latest data point for a Netdata chart. Returns None on any error."""
        url = f"{NETDATA}/data?chart={chart}&after=-3&points=1&format=json"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    def row_val(d: dict | None, dim: str) -> float | None:
        """Extract a single dimension value from a Netdata data response."""
        if not d or not d.get("data"):
            return None
        row = d["data"][0]
        labels = d.get("labels", [])
        try:
            return row[labels.index(dim)]
        except (ValueError, IndexError):
            return None

    parts: list[str] = []

    # --- CPU (user + system) ---
    d = nd("system.cpu")
    user  = row_val(d, "user")  or 0.0
    sys_  = row_val(d, "system") or 0.0
    if d:
        parts.append(f"CPU: {user + sys_:.0f}%")

    # --- RAM (used / total) ---
    d = nd("system.ram")
    if d:
        used    = row_val(d, "used")    or 0.0
        free_   = row_val(d, "free")    or 0.0
        cached  = row_val(d, "cached")  or 0.0
        buffers = row_val(d, "buffers") or 0.0
        total   = used + free_ + cached + buffers
        parts.append(f"RAM: {used/1024:.1f}/{total/1024:.1f} GiB")

    # --- Disk / ---
    d = nd("disk_space./")
    if d:
        used_d = row_val(d, "used")  or 0.0
        avail  = row_val(d, "avail") or 0.0
        total_d = used_d + avail
        pct = (used_d / total_d * 100) if total_d else 0
        parts.append(f"Disk: {used_d:.0f}/{total_d:.0f} GiB ({pct:.0f}%)")

    # --- Battery ---
    d = nd("powersupply_capacity.BAT0")
    cap = row_val(d, "capacity")
    if cap is not None:
        parts.append(f"Bat: {cap:.0f}%")
    else:
        ok, bat = handle_battery_status({})
        if ok:
            m = re.search(r"(\d+)%", bat)
            if m:
                parts.append(f"Bat: {m.group(1)}%")

    # --- WiFi (received kbps) ---
    d = nd("net.wlo1")
    recv = row_val(d, "received")
    if recv is not None:
        parts.append(f"WiFi: {abs(recv):.0f} kbps rx")
    else:
        ok, net = handle_network_status({})
        if ok and net:
            ssid = net.split(":")[0]
            parts.append(f"WiFi: {ssid}")

    # --- Ollama CPU (from Netdata per-app tracking) ---
    d = nd("app.ollama_cpu_utilization")
    o_user = row_val(d, "user")
    o_sys  = row_val(d, "system")
    if o_user is not None:
        parts.append(f"Ollama CPU: {(o_user or 0) + (o_sys or 0):.1f}%")
    else:
        ok, svc = handle_service_status({"name": "ollama"})
        if ok:
            state = svc.split(": ", 1)[1] if ": " in svc else svc
            parts.append(f"Ollama: {state}")

    if not parts:
        return False, "Netdata not reachable and fallbacks failed"

    return True, " | ".join(parts)


def handle_notifications_send(args: dict) -> tuple[bool, str]:
    """Send a KDE desktop notification via notify-send."""
    title   = (args.get("title")   or "SuryaOS").strip()
    message = (args.get("message") or "").strip()
    icon    = (args.get("icon")    or "dialog-information").strip()
    if not message:
        return False, "message must be non-empty"
    ok, text = _run(["notify-send", "-i", icon, title, message])
    return ok, text or f"Notification sent: {title} — {message}"


def handle_ebpf_summary(_args: dict) -> tuple[bool, str]:
    """Deep kernel-level stats via Netdata eBPF plugin.

    Charts queried (all confirmed on this machine via Netdata v2.10 + eBPF plugin):
      app.ollama_mem_page_faults  — Ollama minor/major page faults (inference spikes)
      system.file_nr_used         — total open file descriptors system-wide
      ipv4.sockstat_tcp_sockets   — TCP socket states (established, time_wait, etc.)
      ip.sockstat_sockets         — all socket types
    """
    import urllib.request

    NETDATA = "http://localhost:19999/api/v1"

    def nd(chart: str) -> dict | None:
        url = f"{NETDATA}/data?chart={chart}&after=-3&points=1&format=json"
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return json.loads(r.read())
        except Exception:
            return None

    def row_val(d, dim):
        if not d or not d.get("data"):
            return None
        row = d["data"][0]
        labels = d.get("labels", [])
        try:
            return row[labels.index(dim)]
        except (ValueError, IndexError):
            return None

    parts: list[str] = []

    # Ollama page faults — spikes during inference loading
    d = nd("app.ollama_mem_page_faults")
    minor = row_val(d, "minor") or 0.0
    major = row_val(d, "major") or 0.0
    if d:
        parts.append(f"Ollama page faults: {minor:.0f} minor / {major:.0f} major per sec")

    # Open file descriptors (system-wide)
    d = nd("system.file_nr_used")
    fds = row_val(d, "files") or row_val(d, "fd")
    if fds is not None:
        parts.append(f"Open FDs: {fds:.0f}")

    # TCP socket states
    d = nd("ipv4.sockstat_tcp_sockets")
    if d and d.get("data"):
        row = d["data"][0]
        labels = d.get("labels", [])
        tcp_parts = []
        for lbl in ("established", "time_wait", "close_wait"):
            if lbl in labels:
                val = row[labels.index(lbl)]
                if val and val > 0:
                    tcp_parts.append(f"{lbl}={val:.0f}")
        if tcp_parts:
            parts.append(f"TCP: {', '.join(tcp_parts)}")

    # Total sockets
    d = nd("ip.sockstat_sockets")
    used = row_val(d, "used")
    if used is not None:
        parts.append(f"Sockets: {used:.0f} total")

    if not parts:
        return False, "eBPF data not available — check if Netdata eBPF plugin is enabled at :19999"

    return True, " | ".join(parts)


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------
# functiongemma:270m sometimes hallucinate tool names close to but not exactly
# matching the registered names. This table maps guessed names → correct ones.
# All keys are lowercase; resolution is case-insensitive.
#
# Pattern: model sees "system_network_status" in tools/list but generates
# "system_wifi_status" — strip the server prefix and look up the remainder.
# ---------------------------------------------------------------------------

TOOL_ALIASES: dict[str, str] = {
    # Network / wifi aliases
    "wifi_status":        "network_status",
    "wifi_check":         "network_status",
    "internet_status":    "network_status",
    "network_check":      "network_status",
    "connection_status":  "network_status",
    "online_status":      "network_status",
    "network_info":       "network_status",

    # Memory / RAM aliases
    "ram_usage":          "memory_usage",
    "memory_check":       "memory_usage",
    "ram_check":          "memory_usage",
    "memory_info":        "memory_usage",
    "system_ram_usage":   "memory_usage",

    # Disk aliases
    "disk_check":         "disk_usage",
    "storage_usage":      "disk_usage",
    "disk_info":          "disk_usage",
    "storage_status":     "disk_usage",

    # Service / bluetooth aliases
    "bluetooth_status":   "service_status",
    "bluetooth_check":    "service_status",
    "service_check":      "service_status",

    # Launch aliases
    "app_launch":         "krunner_launch",
    "launch_app":         "krunner_launch",
    "open_app":           "krunner_launch",
    "app_open":           "krunner_launch",
    "start_app":          "krunner_launch",

    # Notification aliases
    "notification_send":  "notifications_send",
    "send_notification":  "notifications_send",
    "notify":             "notifications_send",
    "notification":       "notifications_send",
    "desktop_notification": "notifications_send",

    # Brightness aliases
    "screen_brightness":  "brightness_set",
    "display_brightness": "brightness_set",
    "brightness_change":  "brightness_set",
}

# Map common natural-language service names → actual systemd service names.
SERVICE_ALIASES: dict[str, str] = {
    "bt":              "bluetooth",
    "wifi":            "NetworkManager",
    "network":         "NetworkManager",
    "internet":        "NetworkManager",
    "sound":           "pipewire",
    "audio":           "pipewire",
    "pulse":           "pulseaudio",
    "printer":         "cups",
    "printing":        "cups",
    "firewall":        "ufw",
    "ssh":             "sshd",
    "docker":          "docker",
    "web":             "apache2",
    "http":            "apache2",
}

# Paths that functiongemma hallucinates for disk_usage — map to correct ones.
PATH_FIXES: dict[str, str] = {
    "/disk":   "/",
    "/disk/":  "/",
    "disk":    "/",
    "root":    "/",
    "/root":   "/",
    "/system": "/",
}


def _resolve_name(raw_name: str) -> str:
    """Resolve a tool name through aliases.

    opencode prefixes MCP tool names with the server name (e.g. system_network_status).
    Strip the prefix before lookup so alias keys stay server-agnostic.
    """
    name = (raw_name or "").strip().lower()
    # Strip server prefix (system_, volume_, etc.)
    for prefix in ("system_", "volume_", "kde_", "linux_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return TOOL_ALIASES.get(name, name)


def _resolve_service_args(args: dict, raw_name: str) -> dict:
    """Infer service name from context when model passes wrong or missing args.

    Handles two failure modes:
    1. Model calls service_status with no 'name' arg
    2. Model passes alias like "bt" or "wifi" instead of real service name
    """
    name = (args.get("name") or "").strip().lower()

    # If name is missing, try to infer from the original tool name the model called
    if not name:
        for keyword in ("bluetooth", "bt", "wifi", "ollama", "sshd", "ssh",
                        "network", "audio", "sound", "docker"):
            if keyword in raw_name.lower():
                name = keyword
                break

    # Normalise via alias table
    name = SERVICE_ALIASES.get(name, name)
    return {**args, "name": name}


def _resolve_disk_args(args: dict) -> dict:
    """Normalise disk_usage path — fix common hallucinated paths."""
    path = (args.get("path") or "/").strip()
    # Fix clearly wrong paths
    path = PATH_FIXES.get(path.lower(), path)
    # If path doesn't start with /, it's probably garbage — use root
    if not path.startswith("/"):
        path = "/"
    return {**args, "path": path}


# Map tool name → handler function.
HANDLERS = {
    "brightness_set":      handle_brightness_set,
    "network_status":      handle_network_status,
    "battery_status":      handle_battery_status,
    "memory_usage":        handle_memory_usage,
    "disk_usage":          handle_disk_usage,
    "service_status":      handle_service_status,
    "krunner_launch":      handle_krunner_launch,
    "window_focus":        handle_window_focus,
    "metrics_summary":     handle_metrics_summary,
    "notifications_send":  handle_notifications_send,
    "ebpf_summary":        handle_ebpf_summary,
}


# ---------------------------------------------------------------------------
# Request router
# ---------------------------------------------------------------------------

def handle_request(req: dict) -> None:
    """Route a single JSON-RPC request to the appropriate tool handler."""
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
        # One-way notification — no response per JSON-RPC spec.
        return

    if method == "tools/list":
        reply(req_id, {"tools": TOOLS})
        return

    if method == "tools/call":
        params = req.get("params", {})
        raw_name = params.get("name") or ""
        args = params.get("arguments", {}) or {}

        # Resolve aliases: model may guess system_wifi_status → network_status
        name = _resolve_name(raw_name)

        # Per-tool arg normalization
        if name == "service_status":
            args = _resolve_service_args(args, raw_name)
        if name == "disk_usage":
            args = _resolve_disk_args(args)
        if name == "krunner_launch" and not args.get("app"):
            # model sometimes passes path instead of app — use raw_name hint
            pass

        handler = HANDLERS.get(name)
        if handler is None:
            reply_error(req_id, -32602, f"unknown tool: {name!r}")
            return

        try:
            ok, text = handler(args)
        except Exception as exc:  # noqa: BLE001
            reply(req_id, err_result(f"handler error: {exc}"))
            return

        reply(req_id, ok_result(text) if ok else err_result(text))
        return

    # Unknown method with an id → method-not-found; notifications silently dropped.
    if req_id is not None:
        reply_error(req_id, -32601, f"method not found: {method!r}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Read JSON-RPC messages from stdin line-by-line and dispatch them."""
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
