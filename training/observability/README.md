# Training observability — Grafana + Loki + Prometheus

A self-contained observability stack for `functiongemma-suryaos` training.

## What you get

- **Grafana** dashboard at `http://localhost:3000` (admin / admin)
- **Loki** log aggregation — tails `training/*/train_log.jsonl` automatically
- **Prometheus** time-series metrics — receives pushes from the training scripts
- Pre-built dashboard "functiongemma — training overview" with:
  - Live loss curve, learning rate schedule, gradient norm, memory usage
  - Per-tool loss heatmap with mastered/learning/struggling status
  - Cosine similarity per probe pair over epochs
  - Embedding norm ratio and drift gauges
  - Live event log tail (Loki) with full JSONL detail
  - Multi-run comparison via `run_id` selector

## Quick start

```bash
# 1. Bring the stack up
cd training/observability
docker compose up -d

# 2. Run training as usual — metrics auto-push, logs auto-tail
.fngemma-suryaos/bin/python training/train_tokenizer.py
.fngemma-suryaos/bin/python training/finetune.py --mode all

# 3. Open Grafana
xdg-open http://localhost:3000   # or just visit it in a browser
```

The dashboard is in the **functiongemma** folder, named **"functiongemma — training overview"**.

## Architecture

```
JSONL files          Promtail tails files          Loki
    │                       │                        │
    ▼                       ▼                        ▼
training/                                      LogQL queries
  tokenizer_extended/   Promtail labels:           in Grafana
    train_log.jsonl ──► phase, event, epoch, ──►   (right side
  model_lora/                tool                   of dashboard)
    training_log.jsonl

MetricsPusher.push()
in finetune.py
+ train_tokenizer.py ──► Pushgateway:9091 ──► Prometheus:9090 ──► PromQL queries
                                                                  in Grafana
                                                                  (left side)
```

Two pipes:
- **Loki** for *event-shaped* JSONL — narration, per-tool tables, cosine probes
- **Prometheus** for *time-series* metrics — loss/lr/grad_norm pushed every 5 steps

## Container details

| Service | Port | Purpose |
|---|---|---|
| `loki` | 3100 | Log aggregation API |
| `promtail` | (none) | File tailer agent — ships JSONL to Loki |
| `pushgateway` | 9091 | Receives metric pushes from training scripts |
| `prometheus` | 9090 | Scrapes pushgateway every 5s, stores time-series |
| `grafana` | 3000 | Dashboards (UI) |

## Running training without the stack

The observability stack is **optional**. Training scripts gracefully degrade:

- If `prometheus_client` is not installed → MetricsPusher prints a warning and no-ops
- If Pushgateway is unreachable → first push prints a warning, subsequent pushes silently fail
- JSONL files are always written regardless

So you can develop without the stack running and bring it up when you want to visualise.

## Customising the dashboard

The dashboard auto-reloads from `grafana/dashboards/functiongemma-training.json`.
Edits in the Grafana UI are saved back to disk (because `allowUiUpdates: true`).
To restore the original, delete your edits and `docker compose restart grafana`.

## Querying without Grafana

**Prometheus directly** (`http://localhost:9090`):
```promql
training_loss{run_id="20260501-141522-mybox", phase="lora_train"}
training_per_tool_loss{tool="linux_memory_usage"}
rate(training_step{phase="lora_train"}[1m])     # steps/second
```

**Loki via curl**:
```bash
curl -G http://localhost:3100/loki/api/v1/query_range \
  --data-urlencode 'query={job="functiongemma", event="epoch_probe"}' \
  --data-urlencode 'start=1700000000000000000' \
  --data-urlencode 'end=1800000000000000000'
```

## Multi-run comparison

Each training run gets a unique `run_id` (timestamp + hostname). Select multiple
runs in the Grafana dropdown to overlay their loss curves, per-tool tables, etc.

To label a specific run for easier finding:
```bash
PUSHGATEWAY_JOB=experiment_alpha .fngemma-suryaos/bin/python training/finetune.py --mode train
```

## Troubleshooting

**Grafana says "no data" everywhere**
1. Is the training script actually running?
2. `docker compose ps` — are all 5 containers `Up`?
3. `curl http://localhost:9091/metrics | grep training_` — does Pushgateway have data?
4. Check Pushgateway UI directly: `http://localhost:9091`

**Prometheus has data but Grafana panel is empty**
- Time range filter — top-right of Grafana — set to "Last 1 hour" or wider.
- Check the `run_id` and `phase` selectors at the top of the dashboard.

**Promtail isn't shipping logs**
- `docker compose logs promtail` — should show it tailing the JSONL files
- Check the JSONL file actually exists and is being written to during training

**Disk usage growing**
- Loki keeps 30 days; Prometheus keeps 30 days. Adjust in `loki/config.yml`
  (`retention_period`) and `docker-compose.yml` (`--storage.tsdb.retention.time`).
- Wipe everything: `docker compose down -v`

## Stopping the stack

```bash
docker compose down       # stop containers, keep data
docker compose down -v    # stop containers AND wipe data
```
