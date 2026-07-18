"""
CrossSessionGuard — velocity and pattern anomalies across sessions.

Maintains a rolling window of recent sessions per agent_id.
Fires at end-of-session (finalize).

Detects:
  B5 — session velocity burst (same agent opens >N sessions/hour)
  A5 — agent identity reuse (same agent_id, wildly different personas)
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from fraud.guards.base import SettlementGuard
from fraud.context import FraudContext, GuardSignal

_SESSION_WINDOW_SECONDS = 3600     # 1 hour
_SESSION_BURST_THRESHOLD = 5       # sessions/hour before flagging
_PERSONA_DIVERSITY_THRESHOLD = 4   # distinct personas in window before flagging


@dataclass
class CrossSessionGuard(SettlementGuard):
    name = "cross_session"

    # agent_id → deque of (completed_at, persona)
    _history: dict = field(default_factory=lambda: defaultdict(deque), repr=False)

    def finalize(self, context: FraudContext) -> Optional[GuardSignal]:
        agent_id = context.agent_id or "unknown"
        now      = datetime.now(timezone.utc)
        persona  = context.persona or "unknown"

        q = self._history[agent_id]
        q.append((now, persona))

        # Drop entries older than window
        cutoff = now.timestamp() - _SESSION_WINDOW_SECONDS
        while q and q[0][0].timestamp() < cutoff:
            q.popleft()

        scores: list[float] = []
        flags:  list[str]   = []

        n_sessions = len(q)
        if n_sessions > _SESSION_BURST_THRESHOLD:
            burst_score = min(0.85, 0.40 + 0.05 * (n_sessions - _SESSION_BURST_THRESHOLD))
            scores.append(burst_score)
            flags.append(f"cross_session:velocity n={n_sessions}/hour")

        distinct_personas = len({p for _, p in q})
        if distinct_personas >= _PERSONA_DIVERSITY_THRESHOLD:
            scores.append(0.70)
            flags.append(f"cross_session:persona_diversity n_personas={distinct_personas}")

        score = max(scores) if scores else 0.0

        class _FakeEvent:
            event_id = f"cross_session:{context.session_id}"
            action_type = "SETTLE"
        return self._signal(_FakeEvent(), score, flags)

    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        return None
