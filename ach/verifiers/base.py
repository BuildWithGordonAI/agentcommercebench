"""
Verifier protocol — the single extension point for ACP.

Any system that can assess a ConsequentialAction implements this interface:
  - Gordon fraud pipeline
  - Local rule engine
  - Price oracle
  - Compliance checker
  - Always-allow noop (for dev/testing)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ach.actions.base import ConsequentialAction
from ach.actions.wallet import AgentWallet


@dataclass
class Verification:
    verifier_id:    str
    decision:       str           # "allow" | "flag" | "block"
    score:          float         # 0.0–1.0  point estimate
    confidence:     float = 1.0   # how certain the verifier is about this score
    uncertainty:    float = 0.0   # epistemic: model knows it doesn't know
    score_interval: tuple[float, float] = (0.0, 1.0)  # 90% CI [low, high]
    flags:          list[str] = field(default_factory=list)
    metadata:       dict      = field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        return self.decision == "block"

    @property
    def is_flagged(self) -> bool:
        return self.decision == "flag"

    @property
    def should_escalate(self) -> bool:
        """True when uncertainty is too high to trust the point estimate."""
        ci_width = self.score_interval[1] - self.score_interval[0]
        return self.uncertainty > 0.30 or ci_width > 0.40

    def summary_str(self) -> str:
        ci = self.score_interval
        u  = f"u={self.uncertainty:.2f}" if self.uncertainty > 0 else ""
        ci_s = f"CI=[{ci[0]:.2f},{ci[1]:.2f}]" if ci != (0.0, 1.0) else ""
        parts = [p for p in [u, ci_s] if p]
        return f"score={self.score:.2f} " + (" ".join(parts) if parts else "")


@dataclass
class SessionContext:
    session_id: str
    agent_id:   str
    persona:    str  = "research"
    events:     list = field(default_factory=list)   # raw action history


@runtime_checkable
class Verifier(Protocol):
    verifier_id: str

    def verify(
        self,
        action:  ConsequentialAction,
        wallet:  AgentWallet,
        context: SessionContext,
    ) -> Verification: ...
