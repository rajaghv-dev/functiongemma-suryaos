"""
metrics.py — push training metrics to Prometheus Pushgateway.

WHAT THIS MODULE DOES
=====================
Wraps prometheus_client to provide a simple API for the training scripts to
push time-series metrics to a Prometheus Pushgateway running locally (default:
http://localhost:9091).

WHY PUSHGATEWAY (and not direct scraping)?
==========================================
Prometheus normally PULLS metrics by scraping HTTP endpoints. That works for
long-running services. But our training scripts are SHORT-LIVED (15-25 min) —
by the time Prometheus would scrape them, they've already exited.

Pushgateway flips the model: training scripts PUSH their state to Pushgateway,
and Pushgateway is always up to be scraped by Prometheus. Metrics persist in
Pushgateway across short-lived job invocations.

WHAT METRICS ARE EXPOSED
========================
Time-series (push every step):
  training_loss{phase, run_id}                — current training loss
  training_grad_norm{phase, run_id}            — gradient L2 norm pre-clip
  training_learning_rate{phase, run_id}        — current lr on the schedule
  training_memory_mb{phase, run_id}            — process RSS memory in MB
  training_step{phase, run_id}                  — current optimizer step count
  training_epoch{phase, run_id}                 — fractional epoch number

Time-series (push per epoch):
  training_per_tool_loss{phase, run_id, tool}  — per-tool response loss
  tokenizer_cosine_sim{run_id, pair}            — per-probe-pair cosine similarity
  tokenizer_norm_ratio{run_id}                  — new vs base embedding norm ratio
  tokenizer_embedding_drift{run_id}             — drift from smart-init

GRACEFUL DEGRADATION
====================
If prometheus_client is missing OR pushgateway is unreachable, all push() calls
silently no-op so training continues normally. The training scripts work fully
without the observability stack running.

USAGE
=====
    from training.metrics import MetricsPusher
    pusher = MetricsPusher(phase="lora_train")  # auto-generates a run_id

    # Per training step
    pusher.push_step(step=100, loss=0.823, lr=2e-4, grad_norm=0.4, memory_mb=4521)

    # Per epoch end (LoRA)
    pusher.push_per_tool_loss(epoch=2, per_tool={
        "linux_memory_usage": 0.234,
        "kde_krunner_launch": 1.231,
    })

    # Tokenizer-phase only
    pusher.push_cosine_probes(epoch=2, sims={"same-tool forms": 0.71, ...})
    pusher.push_norm_ratio(ratio=0.92, drift=1.34)
"""

from __future__ import annotations

import math
import os
import socket
import sys
import time
from datetime import datetime
from typing import Optional  # kept for Python < 3.10 compat

# ---------------------------------------------------------------------------
# Try to import prometheus_client. If unavailable, we run in no-op mode so
# training never depends on the observability stack being available.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    from prometheus_client.exposition import default_handler
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False


# Default location of the Pushgateway. Overridable via env var, useful when
# running training inside docker (set to "pushgateway:9091" inside the network).
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "http://localhost:9091")
JOB_NAME        = os.environ.get("PUSHGATEWAY_JOB", "functiongemma_training")


def _new_run_id() -> str:
    """Generate a unique-per-run identifier so multiple runs can be compared."""
    # ISO-like timestamp + hostname for uniqueness across machines/runs
    ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname().replace(".", "-")[:16]
    return f"{ts}-{host}"


class MetricsPusher:
    """
    Push training metrics to Prometheus Pushgateway.

    All push_* methods are safe to call even when:
      - prometheus_client is not installed
      - Pushgateway is not running
      - Network is unreachable
    They silently fail rather than disrupt training.
    """

    def __init__(self, phase: str, run_id: Optional[str] = None,
                 url: Optional[str] = None, job: Optional[str] = None) -> None:
        """
        phase:  "tokenizer_corpus_train" or "lora_train" — labels all pushed metrics
        run_id: unique run identifier (auto-generated from timestamp+host if None)
        url:    Pushgateway URL; defaults to PUSHGATEWAY_URL env var or localhost:9091
        job:    Pushgateway job name; defaults to PUSHGATEWAY_JOB env var
        """
        self.phase   = phase
        self.run_id  = run_id or _new_run_id()
        self.url     = url or PUSHGATEWAY_URL
        self.job     = job or JOB_NAME
        self.enabled = _PROM_AVAILABLE
        self._silent_failures = 0   # to avoid spamming warnings

        # Each push uses a fresh registry so we control exactly which metrics
        # are present. Pushgateway treats absent metrics as "deleted" if we
        # used a single shared registry, so we re-create per call.
        # Pre-build gauge factories so we don't reimport prometheus_client every call.
        if self.enabled:
            self._build_gauge_factories()
            print(f"  [metrics]  MetricsPusher: phase={phase}, run_id={self.run_id}, "
                  f"url={self.url}")
        else:
            print(f"  [metrics]  prometheus_client not installed — Pushgateway disabled")
            print(f"  [metrics]  Install with: pip install prometheus_client")

    def _build_gauge_factories(self):
        """Hold the gauge classes — we instantiate per-push to control labels."""
        # Imported again here so subclasses / static analysis is happy
        from prometheus_client import CollectorRegistry, Gauge
        self._Registry = CollectorRegistry
        self._Gauge    = Gauge

    # ── Internal: do the actual push and swallow errors ────────────────────

    def _push(self, registry) -> None:
        """Send the registry to Pushgateway; silently no-op on failure."""
        if not self.enabled:
            return
        try:
            push_to_gateway(
                self.url,
                job=self.job,
                registry=registry,
                grouping_key={"run_id": self.run_id, "phase": self.phase},
                timeout=2,  # short timeout — don't block training on slow network
            )
        except Exception as e:
            self._silent_failures += 1
            # Print first failure so the user knows the stack isn't running,
            # then go silent to avoid flooding the terminal during training.
            if self._silent_failures == 1:
                print(f"  [metrics]  WARN: Pushgateway not reachable ({e}) — "
                      f"continuing without metrics")
                print(f"  [metrics]        Start the stack: "
                      f"cd training/observability && docker compose up -d")
            # silent for subsequent failures

    # ── Per-step metrics (LoRA + tokenizer corpus phase) ───────────────────

    def push_step(self, step: int, loss: float,
                  lr: Optional[float] = None,
                  grad_norm: Optional[float] = None,
                  memory_mb: Optional[float] = None,
                  epoch: Optional[float] = None) -> None:
        """
        Push the current training-step snapshot.

        Call this every logging_steps in DispatchCallback.on_log, OR every
        step in the tokenizer corpus training loop.

        Pushgateway holds the latest pushed values; Prometheus scrapes them
        every 5 seconds (per prometheus.yml).
        """
        if not self.enabled:
            return
        if loss is None or (isinstance(loss, float) and math.isnan(loss)):
            return  # don't push NaN — Prometheus rejects them anyway

        reg = self._Registry()

        # Each Gauge is created fresh per push, with the same labels.
        # The label set must be identical across pushes for clean Prometheus queries.
        labels = ["phase", "run_id"]

        loss_g = self._Gauge("training_loss",
                             "Cross-entropy training loss", labels, registry=reg)
        loss_g.labels(phase=self.phase, run_id=self.run_id).set(loss)

        step_g = self._Gauge("training_step",
                             "Current optimizer step", labels, registry=reg)
        step_g.labels(phase=self.phase, run_id=self.run_id).set(step)

        if lr is not None:
            lr_g = self._Gauge("training_learning_rate",
                               "Current learning rate", labels, registry=reg)
            lr_g.labels(phase=self.phase, run_id=self.run_id).set(lr)

        if grad_norm is not None and not math.isnan(grad_norm):
            gn_g = self._Gauge("training_grad_norm",
                               "Pre-clip gradient L2 norm", labels, registry=reg)
            gn_g.labels(phase=self.phase, run_id=self.run_id).set(grad_norm)

        if memory_mb is not None:
            mem_g = self._Gauge("training_memory_mb",
                                "Process RSS memory in MB", labels, registry=reg)
            mem_g.labels(phase=self.phase, run_id=self.run_id).set(memory_mb)

        if epoch is not None:
            ep_g = self._Gauge("training_epoch",
                               "Fractional epoch number", labels, registry=reg)
            ep_g.labels(phase=self.phase, run_id=self.run_id).set(epoch)

        self._push(reg)

    # ── Per-epoch metrics (LoRA only) ──────────────────────────────────────

    def push_per_tool_loss(self, epoch: int, per_tool: dict) -> None:
        """
        Push per-tool response loss after a probe.

        per_tool: {tool_name: mean_loss_for_that_tool}
        """
        if not self.enabled or not per_tool:
            return
        reg = self._Registry()
        g = self._Gauge("training_per_tool_loss",
                        "Per-tool response-only loss after probe",
                        ["phase", "run_id", "tool"], registry=reg)
        for tool, loss in per_tool.items():
            if loss is None or math.isnan(loss):
                continue
            g.labels(phase=self.phase, run_id=self.run_id, tool=tool).set(loss)
        self._push(reg)

    # ── Tokenizer-phase metrics ────────────────────────────────────────────

    def push_cosine_probes(self, epoch: int, sims: dict) -> None:
        """
        Push cosine similarity probe results.

        sims: {pair_description: similarity_value | None}
              from train_tokenizer.py PROBE_PAIRS.
        """
        if not self.enabled or not sims:
            return
        reg = self._Registry()
        g = self._Gauge("tokenizer_cosine_sim",
                        "Cosine similarity between probe-pair tokens",
                        ["run_id", "pair"], registry=reg)
        for desc, sim in sims.items():
            if sim is None:
                continue
            # Replace spaces in description with underscores for Prometheus labels
            label = desc.replace(" ", "_").replace("(", "").replace(")", "")
            g.labels(run_id=self.run_id, pair=label).set(sim)
        self._push(reg)

    def push_norm_ratio(self, ratio: float, drift: Optional[float] = None) -> None:
        """
        Push the new-vs-base embedding norm ratio and drift from smart-init.

        ratio: new_norm_mean / base_norm_mean    (1.0 = embeddings same magnitude)
        drift: average L2 distance moved from smart-init values (0 = no learning)
        """
        if not self.enabled:
            return
        reg = self._Registry()

        r = self._Gauge("tokenizer_norm_ratio",
                        "Ratio of new token norm vs base vocab norm",
                        ["run_id"], registry=reg)
        r.labels(run_id=self.run_id).set(ratio)

        if drift is not None:
            d = self._Gauge("tokenizer_embedding_drift",
                            "Mean L2 distance from smart-init embeddings",
                            ["run_id"], registry=reg)
            d.labels(run_id=self.run_id).set(drift)

        self._push(reg)

    # ── Generic event marker ───────────────────────────────────────────────

    def push_event(self, name: str, value: float = 1.0, **labels) -> None:
        """
        Push an arbitrary named metric. Useful for one-off events like
        'tokenizer_phase_complete' or 'training_started'.
        """
        if not self.enabled:
            return
        reg = self._Registry()
        label_keys = ["phase", "run_id"] + list(labels.keys())
        g = self._Gauge(name, f"Custom training event: {name}",
                        label_keys, registry=reg)
        all_labels = {"phase": self.phase, "run_id": self.run_id, **labels}
        g.labels(**all_labels).set(value)
        self._push(reg)
