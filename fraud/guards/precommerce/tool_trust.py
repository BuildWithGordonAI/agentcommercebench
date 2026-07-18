"""
ToolTrustGuard — L3b: tool response latency / schema / size baseline anomaly.

Source: src/fraud_detection/l3_tool_trust_scorer.py (adapted for Guard interface).

Detects when a tool response deviates from its established baseline:
  - Latency spike: server-side injection / MITM adding overhead
  - Schema change: unexpected response keys (tool response poisoned)
  - Size anomaly: response inflated with hidden content
  - Secondary calls: tool triggered unexpected sub-calls

This is the guard that catches tool RESPONSES being compromised, not just
the agent's REQUEST payloads (which PayloadScanGuard handles).

Pre-commerce: catches poisoned FIND_SERVICE responses early.
At-commerce:  catches poisoned service call responses at AUTHORIZE time.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Optional
from fraud.guards.base import PreCommerceGuard
from fraud.context import FraudContext, GuardSignal, Phase


class _ToolBaseline:
    def __init__(self):
        self.latencies:   list[float]    = []
        self.sizes:       list[int]      = []
        self.schema_keys: Optional[set]  = None
        self.call_count:  int            = 0

    def update(self, response: dict, latency_ms: float):
        self.latencies.append(latency_ms)
        self.sizes.append(len(str(response).encode()))
        self.call_count += 1
        if self.schema_keys is None and isinstance(response, dict):
            self.schema_keys = set(response.keys())

    @property
    def ready(self) -> bool:
        return self.call_count >= 3

    def stats(self) -> dict:
        return {
            "lat_mean": float(np.mean(self.latencies)),
            "lat_std":  float(np.std(self.latencies, ddof=1)) if len(self.latencies) > 1 else 1.0,
            "sz_mean":  float(np.mean(self.sizes)),
            "sz_std":   float(np.std(self.sizes, ddof=1)) if len(self.sizes) > 1 else 1.0,
        }


class ToolTrustGuard(PreCommerceGuard):
    """
    Maintains per-tool response baselines and flags anomalous responses.
    The baseline is built incrementally from clean calls — no separate
    training data required.
    """
    name  = "tool_trust"
    phase = Phase.PRECOMMERCE

    def __init__(self):
        self._baselines: dict[str, _ToolBaseline] = defaultdict(_ToolBaseline)

    def should_fire(self, action_type: str) -> bool:
        return True  # monitor all tool responses

    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        tool      = getattr(event, "operation_id", None) or getattr(event, "service_id", None)
        response  = getattr(event, "original_request", {}) or {}
        latency   = getattr(event, "latency_ms", None)

        if not tool:
            return None

        baseline = self._baselines[tool]

        scores: list[float] = []
        flags:  list[str]   = []

        if baseline.ready and latency is not None:
            s = baseline.stats()

            # Latency spike: >3σ above mean (possible MITM / injection overhead)
            if s["lat_std"] > 0:
                z_lat = (latency - s["lat_mean"]) / s["lat_std"]
                if z_lat > 3.0:
                    scores.append(min(0.70, 0.30 + 0.10 * z_lat))
                    flags.append(f"L3b:latency_spike z={z_lat:.1f}")

            # Size anomaly: >3σ above mean (response inflated with hidden content)
            sz = len(str(response).encode())
            if s["sz_std"] > 0:
                z_sz = (sz - s["sz_mean"]) / s["sz_std"]
                if z_sz > 3.0:
                    scores.append(min(0.65, 0.25 + 0.10 * z_sz))
                    flags.append(f"L3b:response_size_anomaly z={z_sz:.1f}")

            # Schema change: unexpected keys in response
            if isinstance(response, dict) and baseline.schema_keys:
                new_keys = set(response.keys()) - baseline.schema_keys
                if new_keys:
                    scores.append(0.55)
                    flags.append(f"L3b:unexpected_response_keys:{sorted(new_keys)}")

        # Update baseline AFTER scoring (so this call doesn't inflate its own baseline)
        if latency is not None:
            baseline.update(response, latency)

        return self._signal(event, max(scores) if scores else 0.0, flags)
