# Policy tiers — green / yellow / red

The SuryaOS agent enforces a three-tier policy on every tool call. The
fine-tuned model dispatches actions; the policy engine decides whether
they execute.

---

## Tier definitions

| Tier | Color | Action | Examples |
|---|---|---|---|
| **Green** | 🟢 | `allow` | read-only or trivially reversible: status checks, app launches, volume/brightness/notifications |
| **Yellow** | 🟡 | `confirm` | state-changing but reversible: service restart, screenlock, suspend, theme change |
| **Red** | 🔴 | `deny` (default) or `confirm` (with admin overlay) | destructive or security-sensitive: shutdown, package install, kill process, governor change |

---

## How the policy engine works

```
User query → context builder → 1-3 tool schemas
                ↓
         functiongemma:270m-suryaos
                ↓
         Tool call: name + args
                ↓
         ┌──────────────────┐
         │ Policy engine    │
         │ configs/policy.  │
         │ yaml             │
         └────────┬─────────┘
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
    allow      confirm     deny
       │          │          │
       │     [kdialog]       │
       │     yes / no        │
       │          │          │
       ▼          ▼          ▼
    execute   execute or    log only
              cancel        return error
```

Source: [`policy.yaml`](policy.yaml) (mirror of `~/raja/oc/configs/policy.yaml`)

---

## Tier-by-tier breakdown

### 🟢 Green — auto-allow (10 tools)

Default action: execute immediately. No user prompt.

```yaml
"linux.battery.status":   allow
"linux.disk.usage":       allow
"linux.memory.usage":     allow
"linux.network.status":   allow
"linux.service.status":   allow
"linux.metrics.summary":  allow
"linux.volume.set":       allow
"linux.brightness.set":   allow
"kde.krunner.launch":     allow
"kde.notifications.send": allow
"kde.window.focus":       allow
```

Audit log records every call with timestamp, args, exit code. No interruption.

**Why volume + brightness are green:** trivially reversible. User can undo
in one keystroke (volume up / down). The risk-to-friction ratio favors auto-allow.

**Why app launch is green:** opening an app cannot delete data. User just
closes the window if not desired.

### 🟡 Yellow — confirm via kdialog (12 tools, mostly v2)

Default action: fire `kde.dialog.confirm` first, execute only on Yes.

```yaml
"linux.service.restart":      confirm
"linux.service.stop":         confirm
"linux.screenlock.lock":      confirm
"linux.power.suspend":        confirm
"kde.power.profile.set":      confirm
"kde.wifi.connect":           confirm
"kde.wifi.toggle":            confirm
"kde.bluetooth.toggle":       confirm
"kde.dnd.toggle":             confirm
"kde.nightcolor.toggle":      confirm
"kde.theme.set":              confirm
"kde.activity.switch":        confirm
```

User sees a kdialog popup:
```
┌────────────────────────────────────┐
│ SuryaOS Agent                      │
│                                    │
│ Restart NetworkManager?            │
│                                    │
│           [ Yes ]   [ No ]         │
└────────────────────────────────────┘
```

The agent waits for the user's choice. On No, the call is logged as
`decision=denied_by_user` but not executed.

**Why services restart is yellow, not green:** disconnects active wifi
sessions, pauses ongoing downloads. Reversible but not trivial.

**Why theme switch is yellow:** changing global appearance is jarring if
unintended. One-keystroke confirmation is a small price.

### 🔴 Red — deny by default (8 tools, mostly v3 + permanent)

Default action: `deny`. Model can dispatch the call but the engine refuses
to execute.

```yaml
# Permanently denied — no override accepted
"linux.shell.run":       deny  # never expose arbitrary shell
"linux.dbus.call":       deny  # never expose raw D-Bus
"linux.sudo.run":        deny  # never expose sudo

# Conditionally denied — admin overlay can grant confirm
"linux.power.shutdown":     deny
"linux.power.reboot":       deny
"linux.package.install":    deny
"linux.package.remove":     deny
"linux.process.kill":       deny
"linux.cpu.governor.set":   deny
```

When the model dispatches a denied tool, the agent responds:
```
This action is denied by SuryaOS security policy.
To enable: edit configs/policy.yaml and add an overlay.
```

The audit log entry: `decision=deny_by_policy`.

---

## Overlays — granting yellow access to red tools

Defined in [`policy.yaml`](policy.yaml) under `overlays:`. Activated via
environment variable.

Example overlay:

```yaml
overlays:
  trusted_admin_mode:
    description: "Lab/dev machine — relax some red-tier tools to confirm"
    rules:
      "linux.power.shutdown":  confirm
      "linux.power.reboot":    confirm
      "linux.package.install": confirm
      "linux.process.kill":    confirm
```

To activate:
```bash
SURYA_POLICY__OVERLAY=trusted_admin_mode opencode run --agent coder-fg "shutdown the computer"
```

The shutdown tool now triggers `kdialog` instead of being denied outright.
Without the overlay, it stays denied.

**No overlay can ever override `linux.shell.run`, `linux.dbus.call`, or
`linux.sudo.run`** — these are hardcoded as `permanent_deny` in the engine.

---

## Audit log — every decision recorded

```sql
SELECT ts, tool_name, decision, args FROM actions ORDER BY ts DESC LIMIT 5;

ts                  tool_name                decision    args
2026-05-01 18:30:11 linux.memory.usage       allow       {}
2026-05-01 18:29:45 kde.krunner.launch       allow       {"app":"firefox"}
2026-05-01 18:29:02 linux.service.restart    confirmed   {"name":"NetworkManager"}
2026-05-01 18:28:50 linux.power.shutdown     deny_policy {}
2026-05-01 18:28:30 linux.shell.run          deny_perm   {"cmd":"rm -rf"}
```

Stored at `~/raja/oc/runtime/audit.db`. Query via Prometheus exporter at
`:8765/metrics` or directly via SQLite.

---

## Training data implications

The fine-tuned model **does not** know about policy tiers — it just dispatches
the right tool call. The policy engine handles allow/confirm/deny.

But the training data includes scenarios that should NOT be dispatched:

```jsonl
{"query": "rm -rf my home",       "expected_tool": null}    ← model trained to refuse
{"query": "give me root shell",   "expected_tool": null}    ← model trained to refuse
{"query": "run arbitrary command","expected_tool": null}    ← model trained to refuse
```

These teach the model that some queries have no tool — it should say
"I cannot do that" rather than guessing the closest tool. This is the ONE
case where we keep the model's safety behavior; we just narrow it to the
exact destructive intent rather than blanket-refusing all system queries.

---

## Tier assignment in tool YAMLs

Each tool declares its tier in its YAML:

```yaml
# tools/linux/service.status.yaml
name: linux.service.status
risk: low                      # → green
confirmation_required: false   # → no kdialog

# tools/linux/service.restart.yaml (v2)
name: linux.service.restart
risk: medium                   # → yellow
confirmation_required: true    # → kdialog before execute

# tools/linux/power.shutdown.yaml (v3)
name: linux.power.shutdown
risk: high                     # → red
confirmation_required: true    # → kdialog AFTER overlay grants confirm tier
allowed_in_production: false   # → never compiled into production release
```

The policy engine reads these fields at startup and builds the rules table.

---

## Production vs development modes

Two policy modes shipped:

| Mode | strict | balanced | permissive |
|---|---|---|---|
| Green tools | allow | allow | allow |
| Yellow tools | confirm | confirm | allow |
| Red tools | deny | confirm | confirm |

Default in production: **strict**.
Default in dev: **balanced**.

Switch via:
```bash
SURYA_POLICY__MODE=permissive opencode run ...
```

Most users never change this. The default `strict` mode is what ships in
[`production.yaml`](production.yaml).
