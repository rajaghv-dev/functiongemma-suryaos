"""
training/probes.py — training-time probes library for function-calling LoRA SFT.

Why this exists
---------------
Total-loss is a single scalar; it tells us the model is "learning" but hides
which tool is starved, which arg slot is wrong, which source is being
memorized, and whether cross-domain (linux↔kde) confusion is collapsing or
growing.  This module provides a small bank of probes that run after each
epoch on the held-out eval split and emit per-tool / per-source / per-arg
diagnostics.

Public API
----------
    bank = ProbeBank(eval_pairs, tokenizer, config=ProbeConfig(...))
    metrics = bank.run(model, device, epoch=N)          # one greedy decode pass
    print(format_report(metrics))
    metrics_to_jsonl(metrics, out_path)                 # appends one line per epoch

Probes (each is a pure function on the shared decode batch where possible):

    1. per_tool_loss          — masked CE on the assistant segment, by target.name
    2. confusion_matrix       — actual→predicted tool-name table from greedy decode
    3. arg_fidelity           — tool / arg-keys / arg-values exact-match
    4. schema_compliance      — JSON-parse / tool-in-list / keys-in-schema
    5. first_token_entropy    — entropy of the next-token distribution restricted
                                to first-tokens of legal tool names
    6. negative_margin        — logp(positive) − max(logp(neg)) at the tool slot
    7. source_overfit         — accuracy grouped by source field
    8. arg_value_diversity    — distinct emitted arg values vs distinct truth

Constraints
-----------
Pure stdlib + torch + transformers.  Works on CPU and GPU.  Idempotent.
Probe runtime budget: <30 s per epoch on RTX 3080 for 162-pair eval.
All probes share ONE greedy-decode pass (cached in DecodeCache); per-token
forward passes (loss, entropy, margin) reuse a SECOND single forward over
the prompt+answer string so we never re-decode 8x.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ProbeConfig:
    """Knobs for the probe bank.  All thresholds are interpretation cutoffs
    that drive the [OK]/[WARN]/[FAIL] tag in format_report()."""

    # generation knobs
    max_new_tokens: int = 64
    do_sample: bool = False
    batch_size: int = 4
    run_every_n_epochs: int = 1
    max_eval_pairs: int = 0          # 0 = use all; positive = subsample

    # acceptance thresholds (tool-match rate)
    tool_match_ok: float = 0.90
    tool_match_warn: float = 0.75

    # arg fidelity thresholds (per-key exact match rate, excludes empty-arg tools)
    arg_keys_ok: float = 0.85
    arg_keys_warn: float = 0.65
    arg_values_ok: float = 0.70
    arg_values_warn: float = 0.45

    # schema compliance — parseable JSON / valid tool / clean keys
    schema_ok: float = 0.95
    schema_warn: float = 0.80

    # first-token entropy (nats, restricted to legal tool first-tokens; max=ln(12)=2.48)
    entropy_ok: float = 0.6           # confident
    entropy_warn: float = 1.4         # high but not random

    # negative margin (positive should beat best-negative by this many nats)
    margin_ok: float = 2.0
    margin_warn: float = 0.5

    # source-skew gap (max_source_acc − min_source_acc)
    source_skew_ok: float = 0.10
    source_skew_warn: float = 0.25

    # cross-domain confusion rate (linux↔kde wrong picks / total wrong)
    cross_domain_ok: float = 0.20
    cross_domain_warn: float = 0.50

    # arg-value diversity (distinct_emitted / distinct_truth)
    diversity_ok: float = 0.60
    diversity_warn: float = 0.30


# ---------------------------------------------------------------------------
# Domain inference (used for cross-domain confusion classification)
# ---------------------------------------------------------------------------


def _tool_domain(name: str) -> str:
    """Map a tool name to a coarse domain tag for cross-domain confusion stats.
    linux_*  → linux
    kde_*    → kde
    other    → other
    Returning the literal prefix keeps the bucket count low (3 in this repo)."""
    if not isinstance(name, str):
        return "other"
    if name.startswith("linux_"):
        return "linux"
    if name.startswith("kde_"):
        return "kde"
    return "other"


# ---------------------------------------------------------------------------
# Lightweight JSON-call extraction from generated text
# ---------------------------------------------------------------------------


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_call(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort: pull the first balanced {…} from a generated string and
    json.loads it.  Returns None if nothing parses."""
    if not isinstance(text, str):
        return None
    # Strip Gemma turn markers if present
    text = text.split("<end_of_turn>", 1)[0].strip()
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    # Try progressive shrink in case of trailing garbage / truncation
    for end in range(len(blob), 0, -1):
        try:
            obj = json.loads(blob[:end])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _tool_schema_keys(tool_schema: Dict[str, Any]) -> set:
    """Keys defined in the tool's inputSchema.properties (incl. optional)."""
    try:
        props = tool_schema["inputSchema"]["properties"]
        if isinstance(props, dict):
            return set(props.keys())
    except Exception:
        pass
    return set()


def _norm_value(v: Any) -> Any:
    """Normalise an arg value for equality checks: lowercase-strip strings,
    leave numbers/bools as-is, JSON-serialise other containers."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    except Exception:
        return repr(v)


# ---------------------------------------------------------------------------
# Eval pair record + cache
# ---------------------------------------------------------------------------


@dataclass
class _EvalRecord:
    """One eval pair, pre-formatted for the probe pass."""

    raw: Dict[str, Any]                # the original JSONL dict
    prompt_text: str                   # chat-template up to <start_of_turn>model
    full_text: str                     # prompt + assistant answer
    target_name: str
    target_args: Dict[str, Any]
    tool_names: List[str]              # candidate tool names from "tools" field
    tool_schemas: Dict[str, Dict[str, Any]]
    source: str
    negatives: List[str]


@dataclass
class _DecodeResult:
    """Output of a single greedy decode for one pair."""

    generated_text: str
    parsed: Optional[Dict[str, Any]]
    pred_name: Optional[str]
    pred_args: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Public ProbeBank
# ---------------------------------------------------------------------------


class ProbeBank:
    """Holds the eval set and the tokenizer; runs all probes in one pass.

    The bank is constructed once at training start.  Each call to .run(model)
    executes ONE greedy-decode batch over the eval set, then derives all 8
    probe metrics from the cached decodes plus a single teacher-forced
    forward pass for loss / entropy / margin.

    Eval pairs that fail to format are skipped silently — the probe returns
    a [WARN] tag if too few pairs survive.  The bank never raises into the
    Trainer loop.
    """

    def __init__(
        self,
        eval_pairs: Iterable[Dict[str, Any]],
        tokenizer,
        config: Optional[ProbeConfig] = None,
        formatter=None,
        prompt_formatter=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProbeConfig()
        self._fmt_full = formatter or _default_full_formatter
        self._fmt_prompt = prompt_formatter or _default_prompt_formatter
        self.records: List[_EvalRecord] = self._build_records(list(eval_pairs))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_records(self, pairs: List[Dict[str, Any]]) -> List[_EvalRecord]:
        out: List[_EvalRecord] = []
        cap = self.config.max_eval_pairs or len(pairs)
        for ex in pairs[:cap]:
            try:
                target = ex.get("target") or {}
                tools = ex.get("tools") or []
                tool_names = [t.get("name") for t in tools if isinstance(t, dict) and t.get("name")]
                tool_schemas = {t["name"]: t for t in tools if isinstance(t, dict) and t.get("name")}
                prompt_text = self._fmt_prompt(ex, self.tokenizer)
                full_text = self._fmt_full(ex, self.tokenizer)
                if not (prompt_text and full_text and target.get("name")):
                    continue
                out.append(
                    _EvalRecord(
                        raw=ex,
                        prompt_text=prompt_text,
                        full_text=full_text,
                        target_name=str(target.get("name")),
                        target_args=dict(target.get("arguments") or {}),
                        tool_names=tool_names,
                        tool_schemas=tool_schemas,
                        source=str(ex.get("source") or "unknown"),
                        negatives=[n for n in (ex.get("negatives") or []) if isinstance(n, str)],
                    )
                )
            except Exception:
                continue
        return out

    @property
    def n_pairs(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self, model, device: Optional[str] = None, epoch: int = 0) -> Dict[str, Any]:
        """Run all probes once.  Returns a dict of probe_name → metric block.
        Each block has raw numbers plus a 1-line 'interpretation' string."""
        import torch

        if not self.records:
            return {
                "_meta": {
                    "epoch": epoch,
                    "n_pairs": 0,
                    "interpretation": "[WARN] no eval pairs — probes skipped",
                }
            }

        model_was_training = bool(getattr(model, "training", False))
        model.eval()
        t0 = time.time()
        try:
            decodes = self._greedy_decode(model, device)
            losses, entropies, margins = self._teacher_forward(model, device)
        finally:
            if model_was_training:
                model.train()

        metrics: Dict[str, Any] = {
            "_meta": {
                "epoch": epoch,
                "n_pairs": len(self.records),
                "elapsed_s": round(time.time() - t0, 2),
            },
            "per_tool_loss":      self._probe_per_tool_loss(losses),
            "confusion_matrix":   self._probe_confusion(decodes),
            "arg_fidelity":       self._probe_arg_fidelity(decodes),
            "schema_compliance":  self._probe_schema(decodes),
            "first_token_entropy": self._probe_entropy(entropies),
            "negative_margin":    self._probe_margin(margins),
            "source_overfit":     self._probe_source(decodes),
            "arg_value_diversity": self._probe_diversity(decodes),
        }
        metrics["_meta"]["interpretation"] = (
            f"[INFO] probes: {len(self.records)} pairs in {metrics['_meta']['elapsed_s']}s"
        )
        return metrics

    # ------------------------------------------------------------------
    # Shared passes
    # ------------------------------------------------------------------

    def _greedy_decode(self, model, device) -> List[_DecodeResult]:
        """One greedy decode per pair, batched.  Cached for all probes."""
        import torch

        tok = self.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        bs = max(1, int(self.config.batch_size))
        out: List[_DecodeResult] = []
        prompts = [r.prompt_text for r in self.records]
        for i in range(0, len(prompts), bs):
            chunk = prompts[i : i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(device or model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    pad_token_id=tok.pad_token_id,
                )
            in_lens = enc["attention_mask"].sum(dim=1).tolist()
            for k, full_ids in enumerate(gen):
                gen_ids = full_ids[in_lens[k]:].tolist()
                text = tok.decode(gen_ids, skip_special_tokens=True)
                parsed = _extract_json_call(text)
                pname = (parsed or {}).get("name") if isinstance(parsed, dict) else None
                pargs = (parsed or {}).get("arguments") if isinstance(parsed, dict) else None
                if not isinstance(pargs, dict):
                    pargs = None
                out.append(_DecodeResult(
                    generated_text=text,
                    parsed=parsed,
                    pred_name=pname if isinstance(pname, str) else None,
                    pred_args=pargs,
                ))
        return out

    def _teacher_forward(
        self, model, device
    ) -> Tuple[List[float], List[Tuple[float, str]], List[Tuple[float, str]]]:
        """Single forward over prompt+answer.  Returns three parallel lists:
        - per-pair masked CE on the assistant segment
        - per-pair (entropy_at_first_tool_token, target_name)
        - per-pair (margin_logp(pos)−max_logp(neg), target_name) or (nan, name)
        """
        import torch
        import torch.nn.functional as F

        tok = self.tokenizer
        device = device or model.device
        losses: List[float] = []
        entropies: List[Tuple[float, str]] = []
        margins: List[Tuple[float, str]] = []

        # Pre-compute first-token-id of each tool name (used for entropy & margin)
        # We probe the first sub-token of `"name":"<tool>"` payload by looking
        # up the first id of the bare tool string.  This is an approximation;
        # if the tokenizer prefixes a space/quote, we still get a stable
        # ranking across tools so margins/entropies remain comparable.
        def _first_tok(s: str) -> Optional[int]:
            ids = tok(s, add_special_tokens=False)["input_ids"]
            return ids[0] if ids else None

        with torch.no_grad():
            for rec in self.records:
                try:
                    # Build labels: full sequence, prompt positions = -100
                    full_enc = tok(rec.full_text, return_tensors="pt",
                                   truncation=True, max_length=1024).to(device)
                    prompt_ids = tok(rec.prompt_text, add_special_tokens=False)["input_ids"]
                    plen = min(len(prompt_ids), full_enc.input_ids.shape[1] - 1)
                    labels = full_enc.input_ids.clone()
                    labels[0, :plen] = -100
                    if "attention_mask" in full_enc:
                        labels[full_enc.attention_mask == 0] = -100

                    out = model(**full_enc, labels=labels)
                    if out.loss is not None and torch.isfinite(out.loss):
                        losses.append(float(out.loss.item()))

                    # First-token logits at the start of the assistant segment.
                    # logits[t] predicts token[t+1], so logits at index plen-1
                    # predicts the first answer token.
                    logits = out.logits[0, max(0, plen - 1), :]    # [V]
                    # Restrict to first-token-ids of legal tool names
                    cand_ids: Dict[str, int] = {}
                    for name in rec.tool_names:
                        tid = _first_tok(name)
                        if tid is not None:
                            cand_ids[name] = tid
                    if cand_ids:
                        idx = torch.tensor(list(cand_ids.values()), device=logits.device)
                        sub_logits = logits[idx]
                        sub_logp = F.log_softmax(sub_logits, dim=-1)
                        sub_p = sub_logp.exp()
                        h = float(-(sub_p * sub_logp).sum().item())  # entropy in nats
                        entropies.append((h, rec.target_name))

                        names_in_order = list(cand_ids.keys())
                        name_to_logp = {n: float(sub_logp[i].item()) for i, n in enumerate(names_in_order)}
                        if rec.target_name in name_to_logp and rec.negatives:
                            neg_lps = [name_to_logp[n] for n in rec.negatives if n in name_to_logp]
                            if neg_lps:
                                margin = name_to_logp[rec.target_name] - max(neg_lps)
                                margins.append((margin, rec.target_name))
                except Exception:
                    continue

        return losses, entropies, margins

    # ------------------------------------------------------------------
    # Probe analyzers
    # ------------------------------------------------------------------

    def _probe_per_tool_loss(self, losses: List[float]) -> Dict[str, Any]:
        """1. per_tool_loss — masked CE on the assistant segment, grouped by
        target.name.  Reveals tools the adapter has barely touched.
        FAILS when one tool's mean loss is >2× the global mean (starvation)."""
        per: Dict[str, List[float]] = defaultdict(list)
        for rec, lo in zip(self.records, losses):
            per[rec.target_name].append(lo)
        per_mean = {t: round(sum(v) / len(v), 4) for t, v in per.items() if v}
        global_mean = round(sum(losses) / len(losses), 4) if losses else float("nan")
        worst = max(per_mean.items(), key=lambda kv: kv[1]) if per_mean else (None, float("nan"))
        starved = [t for t, m in per_mean.items() if global_mean and m > 2 * global_mean]
        tag = "[FAIL]" if starved else ("[WARN]" if global_mean > 1.5 else "[OK]")
        return {
            "per_tool_mean": per_mean,
            "global_mean": global_mean,
            "worst_tool": worst[0],
            "worst_loss": worst[1],
            "starved_tools": starved,
            "interpretation": (
                f"{tag} mean CE={global_mean:.3f}; worst={worst[0]} ({worst[1]:.3f}); "
                f"{len(starved)} starved tool(s)"
            ),
        }

    def _probe_confusion(self, decodes: List[_DecodeResult]) -> Dict[str, Any]:
        """2. confusion_matrix — actual→predicted tool-name from greedy decode.
        Reports per-tool top-1 accuracy and the cross-domain confusion rate
        (linux↔kde mistakes / total mistakes).  FAILS when overall accuracy
        is below tool_match_warn or cross-domain rate exceeds cross_domain_warn."""
        matrix: Dict[str, Counter] = defaultdict(Counter)
        per_tool_total: Counter = Counter()
        per_tool_correct: Counter = Counter()
        cross_domain = 0
        wrong_total = 0
        for rec, dec in zip(self.records, decodes):
            actual = rec.target_name
            predicted = dec.pred_name or "<no_parse>"
            matrix[actual][predicted] += 1
            per_tool_total[actual] += 1
            if predicted == actual:
                per_tool_correct[actual] += 1
            else:
                wrong_total += 1
                if _tool_domain(actual) != _tool_domain(predicted):
                    cross_domain += 1
        per_tool_acc = {
            t: round(per_tool_correct[t] / per_tool_total[t], 3)
            for t in per_tool_total
        }
        overall_acc = (
            sum(per_tool_correct.values()) / sum(per_tool_total.values())
            if per_tool_total else 0.0
        )
        cross_rate = (cross_domain / wrong_total) if wrong_total else 0.0
        # Top confused pairs
        confused: List[Tuple[str, str, int]] = []
        for actual, preds in matrix.items():
            for pred, cnt in preds.items():
                if pred != actual:
                    confused.append((actual, pred, cnt))
        confused.sort(key=lambda x: -x[2])
        cfg = self.config
        if overall_acc >= cfg.tool_match_ok and cross_rate <= cfg.cross_domain_ok:
            tag = "[OK]"
        elif overall_acc >= cfg.tool_match_warn and cross_rate <= cfg.cross_domain_warn:
            tag = "[WARN]"
        else:
            tag = "[FAIL]"
        return {
            "overall_accuracy": round(overall_acc, 3),
            "per_tool_accuracy": per_tool_acc,
            "top_confused": confused[:5],
            "cross_domain_rate": round(cross_rate, 3),
            "wrong_total": wrong_total,
            "interpretation": (
                f"{tag} tool acc={overall_acc:.1%}; cross-domain={cross_rate:.1%} "
                f"of {wrong_total} mistakes"
            ),
        }

    def _probe_arg_fidelity(self, decodes: List[_DecodeResult]) -> Dict[str, Any]:
        """3. arg_fidelity — three accuracies: tool_match, arg_keys_match,
        arg_values_match (case-insensitive equality, numeric equality for ints).
        FAILS if any of the three drops below its _warn threshold; tracked
        separately because iter #3 trained on empty arguments and lied
        about progress."""
        per_tool: Dict[str, Dict[str, List[bool]]] = defaultdict(
            lambda: {"tool": [], "keys": [], "values": []}
        )
        for rec, dec in zip(self.records, decodes):
            tool_ok = (dec.pred_name == rec.target_name)
            per_tool[rec.target_name]["tool"].append(tool_ok)
            truth_keys = set(rec.target_args.keys())
            pred_keys = set((dec.pred_args or {}).keys())
            keys_ok = (truth_keys == pred_keys) if (truth_keys or pred_keys) else True
            per_tool[rec.target_name]["keys"].append(keys_ok)
            if truth_keys and dec.pred_args is not None:
                vals_ok_count = sum(
                    1 for k in truth_keys
                    if k in dec.pred_args and _norm_value(dec.pred_args[k]) == _norm_value(rec.target_args[k])
                )
                values_ok = (vals_ok_count == len(truth_keys))
            elif not truth_keys:
                values_ok = True              # nothing to compare
            else:
                values_ok = False
            per_tool[rec.target_name]["values"].append(values_ok)

        def _agg(field: str) -> float:
            flat = [v for d in per_tool.values() for v in d[field]]
            return round(sum(flat) / len(flat), 3) if flat else 0.0

        tool_acc = _agg("tool")
        keys_acc = _agg("keys")
        vals_acc = _agg("values")
        per_tool_summary = {
            t: {f: round(sum(v) / len(v), 3) if v else 0.0 for f, v in d.items()}
            for t, d in per_tool.items()
        }
        cfg = self.config
        if (tool_acc >= cfg.tool_match_ok and
                keys_acc >= cfg.arg_keys_ok and
                vals_acc >= cfg.arg_values_ok):
            tag = "[OK]"
        elif (tool_acc >= cfg.tool_match_warn and
                keys_acc >= cfg.arg_keys_warn and
                vals_acc >= cfg.arg_values_warn):
            tag = "[WARN]"
        else:
            tag = "[FAIL]"
        return {
            "tool_match": tool_acc,
            "arg_keys_match": keys_acc,
            "arg_values_match": vals_acc,
            "per_tool": per_tool_summary,
            "interpretation": (
                f"{tag} tool={tool_acc:.1%}  keys={keys_acc:.1%}  values={vals_acc:.1%}"
            ),
        }

    def _probe_schema(self, decodes: List[_DecodeResult]) -> Dict[str, Any]:
        """4. schema_compliance — three pass/fail rates: parses-as-JSON,
        emits a known tool name from tools[], every arg key is in the
        tool's inputSchema.  FAILS when JSON-parse rate is below schema_warn:
        the model is producing free text instead of tool calls."""
        n = len(decodes)
        n_json = 0
        n_known = 0
        n_clean = 0
        for rec, dec in zip(self.records, decodes):
            if isinstance(dec.parsed, dict):
                n_json += 1
                pname = dec.pred_name
                if pname and pname in rec.tool_names:
                    n_known += 1
                    schema_keys = _tool_schema_keys(rec.tool_schemas.get(pname, {}))
                    pred_keys = set((dec.pred_args or {}).keys())
                    if not pred_keys or pred_keys.issubset(schema_keys):
                        n_clean += 1
        rates = {
            "json_parse_rate": round(n_json / n, 3) if n else 0.0,
            "known_tool_rate": round(n_known / n, 3) if n else 0.0,
            "clean_keys_rate": round(n_clean / n, 3) if n else 0.0,
        }
        cfg = self.config
        worst = min(rates.values())
        if worst >= cfg.schema_ok:
            tag = "[OK]"
        elif worst >= cfg.schema_warn:
            tag = "[WARN]"
        else:
            tag = "[FAIL]"
        return {
            **rates,
            "interpretation": (
                f"{tag} json={rates['json_parse_rate']:.1%} "
                f"known={rates['known_tool_rate']:.1%} "
                f"clean={rates['clean_keys_rate']:.1%}"
            ),
        }

    def _probe_entropy(self, entropies: List[Tuple[float, str]]) -> Dict[str, Any]:
        """5. first_token_entropy — entropy (nats) of next-token distribution
        restricted to legal tool first-tokens.  HIGH entropy = uncertain at
        the tool slot (cosine-collapse symptom).  FAILS when mean entropy
        exceeds entropy_warn (model effectively guessing)."""
        if not entropies:
            return {"mean": float("nan"), "p95": float("nan"), "per_tool": {},
                    "interpretation": "[WARN] no entropy data"}
        per_tool: Dict[str, List[float]] = defaultdict(list)
        for h, t in entropies:
            per_tool[t].append(h)
        per_tool_mean = {t: round(sum(v) / len(v), 3) for t, v in per_tool.items()}
        all_h = [h for h, _ in entropies]
        all_h_sorted = sorted(all_h)
        p95 = all_h_sorted[max(0, int(0.95 * len(all_h_sorted)) - 1)]
        mean_h = sum(all_h) / len(all_h)
        cfg = self.config
        tag = "[OK]" if mean_h <= cfg.entropy_ok else \
              "[WARN]" if mean_h <= cfg.entropy_warn else "[FAIL]"
        return {
            "mean": round(mean_h, 3),
            "p95": round(p95, 3),
            "per_tool_mean": per_tool_mean,
            "interpretation": (
                f"{tag} H̄={mean_h:.2f} nats (max=ln(12)≈2.48), p95={p95:.2f}"
            ),
        }

    def _probe_margin(self, margins: List[Tuple[float, str]]) -> Dict[str, Any]:
        """6. negative_margin — logp(positive) − max(logp(negatives)) at the
        tool-name slot.  Should grow each epoch.  FAILS when mean margin is
        below margin_warn — hard negatives are not being separated."""
        if not margins:
            return {"mean": float("nan"), "per_tool": {},
                    "interpretation": "[WARN] no negatives in eval pairs"}
        per_tool: Dict[str, List[float]] = defaultdict(list)
        for m, t in margins:
            per_tool[t].append(m)
        per_tool_mean = {t: round(sum(v) / len(v), 3) for t, v in per_tool.items()}
        mean_m = sum(m for m, _ in margins) / len(margins)
        cfg = self.config
        tag = "[OK]" if mean_m >= cfg.margin_ok else \
              "[WARN]" if mean_m >= cfg.margin_warn else "[FAIL]"
        return {
            "mean": round(mean_m, 3),
            "n_pairs": len(margins),
            "per_tool_mean": per_tool_mean,
            "interpretation": f"{tag} margin={mean_m:.2f} nats over {len(margins)} pairs",
        }

    def _probe_source(self, decodes: List[_DecodeResult]) -> Dict[str, Any]:
        """7. source_overfit — tool-match accuracy grouped by source field.
        Big gap = the model has memorized one source's surface phrasing.
        FAILS when (max_source_acc − min_source_acc) exceeds source_skew_warn."""
        per: Dict[str, List[bool]] = defaultdict(list)
        for rec, dec in zip(self.records, decodes):
            per[rec.source].append(dec.pred_name == rec.target_name)
        per_mean = {s: round(sum(v) / len(v), 3) for s, v in per.items() if v}
        if not per_mean:
            return {"per_source": {}, "skew": 0.0,
                    "interpretation": "[WARN] no source data"}
        skew = max(per_mean.values()) - min(per_mean.values())
        cfg = self.config
        tag = "[OK]" if skew <= cfg.source_skew_ok else \
              "[WARN]" if skew <= cfg.source_skew_warn else "[FAIL]"
        worst = min(per_mean.items(), key=lambda kv: kv[1])
        best = max(per_mean.items(), key=lambda kv: kv[1])
        return {
            "per_source_accuracy": per_mean,
            "skew": round(skew, 3),
            "best": best,
            "worst": worst,
            "interpretation": (
                f"{tag} skew={skew:.1%} (best={best[0]} {best[1]:.1%}, "
                f"worst={worst[0]} {worst[1]:.1%})"
            ),
        }

    def _probe_diversity(self, decodes: List[_DecodeResult]) -> Dict[str, Any]:
        """8. arg_value_diversity — distinct emitted arg values per tool vs
        distinct truth values.  Ratio ≪1 means the model collapsed to a
        single canned argument across queries.  FAILS when ratio is below
        diversity_warn (only relevant for tools with non-empty arguments)."""
        emitted_per_tool: Dict[str, set] = defaultdict(set)
        truth_per_tool: Dict[str, set] = defaultdict(set)
        for rec, dec in zip(self.records, decodes):
            t = rec.target_name
            if rec.target_args:
                truth_per_tool[t].add(json.dumps(
                    {k: _norm_value(v) for k, v in rec.target_args.items()},
                    sort_keys=True))
            if dec.pred_args:
                emitted_per_tool[t].add(json.dumps(
                    {k: _norm_value(v) for k, v in dec.pred_args.items()},
                    sort_keys=True))
        per_tool: Dict[str, Dict[str, int]] = {}
        ratios: List[float] = []
        for t in truth_per_tool:
            n_truth = len(truth_per_tool[t])
            n_emit = len(emitted_per_tool.get(t, set()))
            per_tool[t] = {"distinct_truth": n_truth, "distinct_emitted": n_emit}
            if n_truth > 0:
                ratios.append(n_emit / n_truth)
        ratio = sum(ratios) / len(ratios) if ratios else 0.0
        cfg = self.config
        tag = "[OK]" if ratio >= cfg.diversity_ok else \
              "[WARN]" if ratio >= cfg.diversity_warn else "[FAIL]"
        return {
            "per_tool": per_tool,
            "mean_ratio": round(ratio, 3),
            "interpretation": (
                f"{tag} emitted/truth={ratio:.1%} across {len(per_tool)} tool(s)"
            ),
        }


# ---------------------------------------------------------------------------
# Default chat-template formatters (fallbacks; the trainer can supply its own)
# ---------------------------------------------------------------------------


def _default_full_formatter(ex: Dict[str, Any], tokenizer) -> str:
    msgs = ex.get("messages", [])
    tools = ex.get("tools", [])
    target = ex.get("target", {})
    system = next((m["content"] for m in msgs if m.get("role") == "system"),
                  "Call the right tool.")
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    tools_text = json.dumps(tools, separators=(",", ":"))
    target_json = json.dumps(target, separators=(",", ":"))
    turns = [
        {"role": "user", "content": (f"{system}\n\nAvailable tools: {tools_text}\n\n"
                                      f"User request: {user}")},
        {"role": "model", "content": target_json},
    ]
    try:
        return tokenizer.apply_chat_template(turns, tokenize=False,
                                             add_generation_prompt=False)
    except Exception:
        return (f"<bos><start_of_turn>user\n{system}\n\nAvailable tools: {tools_text}\n\n"
                f"User request: {user}\n<end_of_turn>\n"
                f"<start_of_turn>model\n{target_json}\n<end_of_turn>")


def _default_prompt_formatter(ex: Dict[str, Any], tokenizer) -> str:
    msgs = ex.get("messages", [])
    tools = ex.get("tools", [])
    system = next((m["content"] for m in msgs if m.get("role") == "system"),
                  "Call the right tool.")
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    tools_text = json.dumps(tools, separators=(",", ":"))
    turns = [
        {"role": "user", "content": (f"{system}\n\nAvailable tools: {tools_text}\n\n"
                                      f"User request: {user}")},
    ]
    try:
        return tokenizer.apply_chat_template(turns, tokenize=False,
                                             add_generation_prompt=True)
    except Exception:
        return (f"<bos><start_of_turn>user\n{system}\n\nAvailable tools: {tools_text}\n\n"
                f"User request: {user}\n<end_of_turn>\n<start_of_turn>model\n")


# ---------------------------------------------------------------------------
# Reporting + persistence helpers
# ---------------------------------------------------------------------------


_PROBE_ORDER = [
    ("per_tool_loss",       "1. PER-TOOL LOSS"),
    ("confusion_matrix",    "2. CONFUSION MATRIX"),
    ("arg_fidelity",        "3. ARG FIDELITY"),
    ("schema_compliance",   "4. SCHEMA COMPLIANCE"),
    ("first_token_entropy", "5. FIRST-TOKEN ENTROPY"),
    ("negative_margin",     "6. NEGATIVE MARGIN"),
    ("source_overfit",      "7. SOURCE OVERFIT"),
    ("arg_value_diversity", "8. ARG VALUE DIVERSITY"),
]


def format_report(metrics: Dict[str, Any]) -> str:
    """Human-readable single-block report.  Each section starts with a
    [OK]/[WARN]/[FAIL] tag computed by the corresponding probe."""
    meta = metrics.get("_meta", {})
    out: List[str] = []
    out.append("─" * 72)
    out.append(f"  PROBE REPORT — epoch {meta.get('epoch','?')} "
               f"({meta.get('n_pairs', 0)} pairs, "
               f"{meta.get('elapsed_s', 0)}s)")
    out.append("─" * 72)

    for key, title in _PROBE_ORDER:
        block = metrics.get(key)
        if not isinstance(block, dict):
            continue
        out.append(f"  {title}")
        out.append(f"    {block.get('interpretation','(no data)')}")
        # Compact extra detail per probe
        if key == "confusion_matrix" and block.get("top_confused"):
            for a, p, c in block["top_confused"][:3]:
                out.append(f"    confused: {a:30s} → {p:30s} ×{c}")
        elif key == "per_tool_loss" and block.get("starved_tools"):
            out.append(f"    starved: {', '.join(block['starved_tools'])}")
        elif key == "source_overfit" and block.get("per_source_accuracy"):
            for s, a in sorted(block["per_source_accuracy"].items(),
                               key=lambda kv: kv[1]):
                out.append(f"    {s:30s} acc={a:.1%}")
        elif key == "arg_fidelity":
            out.append(f"    tool={block.get('tool_match',0):.1%}  "
                       f"keys={block.get('arg_keys_match',0):.1%}  "
                       f"values={block.get('arg_values_match',0):.1%}")
    out.append("─" * 72)
    return "\n".join(out)


def metrics_to_jsonl(metrics: Dict[str, Any], out_path) -> None:
    """Append one JSON line per epoch.  Parent dir is created if needed."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, sort_keys=True, default=str))
        f.write("\n")


# Bound method-style alias so the docstring usage `metrics.to_jsonl(path)`
# can be approximated via a tiny wrapper below.  Keeps the API loose: callers
# may use either form.

def attach_jsonl(metrics: Dict[str, Any]):
    """Optional: attach a .to_jsonl(path) method to the metrics dict by
    wrapping it in a thin subclass.  Pure convenience — the standalone
    metrics_to_jsonl(metrics, path) does the same job."""

    class _MetricsDict(dict):
        def to_jsonl(self, path):
            metrics_to_jsonl(self, path)

    return _MetricsDict(metrics)


def load_eval_pairs(path) -> List[Dict[str, Any]]:
    """Load eval JSONL into a list of dicts.  Returns [] if the file is
    missing — callers should surface a [WARN] in that case."""
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


__all__ = [
    "ProbeBank",
    "ProbeConfig",
    "format_report",
    "metrics_to_jsonl",
    "attach_jsonl",
    "load_eval_pairs",
]
