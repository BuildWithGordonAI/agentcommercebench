"""
Guard base class.

Every detector in the stack implements this interface:

    guard.should_fire(action_type) → bool
    guard.detect(event, context)   → GuardSignal | None

Guards are STATELESS per call — they read from the context object, never
from instance variables that accumulate across sessions.  Session state
lives in FraudContext.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from fraud.context import FraudContext, GuardSignal, Phase


class Guard(ABC):
    name:  str   # unique slug, used in GuardSignal.guard
    phase: Phase # which pipeline phase this belongs to

    @abstractmethod
    def should_fire(self, action_type: str) -> bool:
        """Return True if this guard applies to this action type."""

    @abstractmethod
    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        """
        Inspect event + accumulated context.
        Return a GuardSignal if suspicious, None if clean.

        Guards may read context.precommerce_risk / atcommerce_risk to
        amplify their own threshold based on what prior layers found.
        """

    def _signal(self, event, score: float, flags: list[str]) -> Optional[GuardSignal]:
        """Helper: return None if score==0, GuardSignal otherwise."""
        if score <= 0:
            return None
        return GuardSignal(
            guard=self.name,
            phase=self.phase,
            score=round(score, 4),
            flags=flags,
            event_id=getattr(event, "event_id", "?"),
        )


class PreCommerceGuard(Guard):
    """Fires on FIND_SERVICE and GET_SERVICE (before any payment)."""
    phase = Phase.PRECOMMERCE

    def should_fire(self, action_type: str) -> bool:
        return action_type in ("find_service", "get_service")


class AtCommerceGuard(Guard):
    """Fires on AUTHORIZE and SETTLE (at payment moment)."""
    phase = Phase.ATCOMMERCE

    def should_fire(self, action_type: str) -> bool:
        return action_type in ("authorize", "settle", "a2a_transfer")


class SettlementGuard(Guard):
    """
    Fires post-settlement.
    Unlike the other phases, settlement guards typically receive a BATCH
    of events / sessions, not a single event.  The should_fire() check
    always returns True — callers invoke these explicitly after a session
    is complete.
    """
    phase = Phase.SETTLEMENT

    def should_fire(self, action_type: str) -> bool:
        return True  # invoked explicitly at session end
