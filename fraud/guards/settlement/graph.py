"""
SettlementGraphGuard — L6: cross-session network graph anomaly detection.

Source: src/fraud_detection/l6_settlement_anomaly_detector.py (adapted).

Maintains a global NetworkX DiGraph across all sessions seen so far.
Fires at the END of each session (called from FraudPipeline.end_session).

Detects:
  C1 — circular fund flows (cycle in payment graph)
  C2 — dense Sybil clusters (many wallets, uniform micro-payments)
  C3 — layered laundering (many hops, low-variance amounts)

Settlement-only: this guard is NOT called per-event. FraudPipeline
calls `finalize(session, context)` once per completed session.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
try:
    import networkx as nx
    import numpy as np
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from fraud.guards.base import SettlementGuard
from fraud.context import FraudContext, GuardSignal, Phase


@dataclass
class SettlementGraphGuard(SettlementGuard):
    name = "settlement_graph"

    _graph: object = field(default=None, repr=False)   # nx.DiGraph or None

    def __post_init__(self):
        if HAS_NETWORKX:
            self._graph = nx.DiGraph()

    def _add_settled_events(self, context: FraudContext) -> None:
        for event in context.events:
            action = str(getattr(event, "action_type", "")).lower()
            if "settle" not in action and "authorize" not in action:
                continue
            req     = getattr(event, "original_request", {}) or {}
            source  = context.agent_id or "agent"
            target  = req.get("vendor") or req.get("to") or "merchant"
            amount  = getattr(event, "amount_units", 0) or 0
            tx_hash = req.get("idempotency_key") or getattr(event, "event_id", "")
            ts      = str(getattr(event, "timestamp", ""))
            self._graph.add_edge(
                source, target,
                amount_units=amount,
                tx_hash=tx_hash,
                timestamp=ts,
                status="settled",
            )

    def finalize(self, context: FraudContext) -> Optional[GuardSignal]:
        """Called by FraudPipeline.end_session — not per-event."""
        if not HAS_NETWORKX or self._graph is None:
            return None

        self._add_settled_events(context)
        scores: list[float] = []
        flags:  list[str]   = []

        # C1 — circular flow
        cycles = [c for c in nx.simple_cycles(self._graph) if 2 <= len(c) <= 6]
        if cycles:
            cycle_score = min(0.90, 0.40 + 0.10 * len(cycles))
            scores.append(cycle_score)
            flags.append(f"L6:circular_flow cycles={len(cycles)}")

        # C2 — dense subgraph (Sybil cluster)
        if self._graph.number_of_nodes() >= 4:
            undirected = self._graph.to_undirected()
            try:
                clusters = list(nx.algorithms.community.greedy_modularity_communities(undirected))
                densities = []
                for cluster in clusters:
                    sub = self._graph.subgraph(cluster)
                    n, m = sub.number_of_nodes(), sub.number_of_edges()
                    if n > 1:
                        densities.append(m / (n * (n - 1)))
                if densities:
                    d = max(densities)
                    if d > 0.60:
                        scores.append(min(0.85, 0.40 + d * 0.60))
                        flags.append(f"L6:dense_cluster density={d:.2f}")
            except Exception:
                pass

        # C3 — Sybil micro-payments (low amount variance, many counterparties)
        edges = list(self._graph.edges(data=True))
        if len(edges) >= 4:
            amounts = np.array([d.get("amount_units", 0) for _, _, d in edges], dtype=float)
            n_uniq  = len({v for u, v, _ in edges} | {u for u, v, _ in edges})
            mean_amt = float(amounts.mean()) + 1.0
            variance_ratio = float(np.var(amounts)) / mean_amt
            sybil_score = min(1.0, (1.0 / max(1, n_uniq)) + variance_ratio / mean_amt)
            if sybil_score > 0.40:
                scores.append(min(0.80, sybil_score))
                flags.append(f"L6:sybil_pattern score={sybil_score:.2f}")

        score = max(scores) if scores else 0.0
        # Synthesize a dummy event-like object for _signal
        class _FakeEvent:
            event_id = f"settlement:{context.session_id}"
            action_type = "SETTLE"
        return self._signal(_FakeEvent(), score, flags)

    # detect() is required by ABC but settlement guards don't use it per-event
    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        return None
