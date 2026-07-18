"""
FraudContext — the inter-layer state object.

This is the core architectural concept. Every guard reads the full
accumulated context from prior guards and enriches it with its own signals.

Flow:
    event → PreCommerceGuards → AtCommerceGuards → SettlementGuards
                ↑                     ↑                   ↑
           (payload, intent,    (oracle, replay,    (graph, cross-session)
            sequence model)      velocity,           reads precommerce +
                                 trajectory)         atcommerce risk

Inter-layer amplification:
    At-commerce guards can see precommerce_risk and tighten thresholds.
    Settlement guards can see both and confirm / escalate.
    No guard is blind to what came before.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Phase(str, Enum):
    PRECOMMERCE  = "precommerce"   # before AUTHORIZE — agent harness side
    ATCOMMERCE   = "atcommerce"    # at AUTHORIZE/SETTLE — wallet / rail side
    SETTLEMENT   = "settlement"    # post-settle — card network / graph side


class Decision(str, Enum):
    ALLOW    = "allow"
    ESCALATE = "escalate"   # human-in-the-loop / hold
    BLOCK    = "block"


ALLOW_THRESHOLD    = 0.30
ESCALATE_THRESHOLD = 0.70


@dataclass
class GuardSignal:
    """One detection signal from one guard."""
    guard:    str           # e.g. "payload_scan", "price_oracle"
    phase:    Phase
    score:    float         # 0–1
    flags:    list[str]     # human-readable explanation flags
    event_id: str           # which event triggered this


@dataclass
class FraudContext:
    """
    Accumulated fraud state for one agent session.
    Grows as events are processed; each guard reads + enriches it.

    Guards communicate via this object — NOT via direct calls to each other.
    """
    session_id: str
    agent_id:   str
    persona:    Optional[str]

    # Full session event history (grows as events are processed)
    events: list = field(default_factory=list)

    # Signals emitted by every guard that fired
    signals: list[GuardSignal] = field(default_factory=list)

    # Caller-supplied metadata (consent_envelope, custom thresholds, etc.)
    metadata: dict = field(default_factory=dict)

    # ── computed risk per phase ───────────────────────────────────────────

    @property
    def precommerce_risk(self) -> float:
        """Max risk score from pre-commerce guards. Visible to at-commerce guards."""
        scores = [s.score for s in self.signals if s.phase == Phase.PRECOMMERCE]
        return max(scores, default=0.0)

    @property
    def atcommerce_risk(self) -> float:
        """Max risk score from at-commerce guards. Visible to settlement guards."""
        scores = [s.score for s in self.signals if s.phase == Phase.ATCOMMERCE]
        return max(scores, default=0.0)

    @property
    def settlement_risk(self) -> float:
        scores = [s.score for s in self.signals if s.phase == Phase.SETTLEMENT]
        return max(scores, default=0.0)

    @property
    def risk_score(self) -> float:
        """Overall max risk across all phases."""
        return max(self.precommerce_risk, self.atcommerce_risk, self.settlement_risk)

    @property
    def risk_flags(self) -> list[str]:
        return [f for s in self.signals for f in s.flags]

    @property
    def decision(self) -> Decision:
        if self.risk_score >= ESCALATE_THRESHOLD:
            return Decision.BLOCK
        if self.risk_score >= ALLOW_THRESHOLD:
            return Decision.ESCALATE
        return Decision.ALLOW

    # ── mutation (returns new context — immutable style) ──────────────────

    def with_signal(self, signal: GuardSignal) -> "FraudContext":
        return FraudContext(
            session_id=self.session_id,
            agent_id=self.agent_id,
            persona=self.persona,
            events=self.events,
            signals=self.signals + [signal],
            metadata=self.metadata,
        )

    def with_event(self, event) -> "FraudContext":
        return FraudContext(
            session_id=self.session_id,
            agent_id=self.agent_id,
            persona=self.persona,
            events=self.events + [event],
            signals=self.signals,
            metadata=self.metadata,
        )

    # ── factories ─────────────────────────────────────────────────────────

    @classmethod
    def for_session(
        cls,
        session_id: str,
        agent_id: str,
        persona: str = None,
        metadata: dict = None,
    ) -> "FraudContext":
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            persona=persona,
            metadata=metadata or {},
        )

    # ── serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "agent_id":         self.agent_id,
            "persona":          self.persona,
            "risk_score":       round(self.risk_score, 4),
            "precommerce_risk": round(self.precommerce_risk, 4),
            "atcommerce_risk":  round(self.atcommerce_risk, 4),
            "settlement_risk":  round(self.settlement_risk, 4),
            "decision":         self.decision.value,
            "flags":            self.risk_flags,
            "signals": [
                {
                    "guard":    s.guard,
                    "phase":    s.phase.value,
                    "score":    round(s.score, 4),
                    "flags":    s.flags,
                    "event_id": s.event_id,
                }
                for s in self.signals
            ],
            "n_events": len(self.events),
        }
