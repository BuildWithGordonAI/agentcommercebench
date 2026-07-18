"""
FraudPipeline — the main entry point for the fraud/ package.

Usage (per-event streaming):

    pipeline = FraudPipeline.default()
    context  = FraudContext.for_session(session_id, agent_id, persona)

    for event in session.events:
        context, signals = pipeline.process_event(event, context)
        if context.decision == Decision.BLOCK:
            raise Exception("Fraud detected")

    # End-of-session: run settlement guards
    context, settlement_signals = pipeline.end_session(context)

Usage (batch — whole session):

    context = pipeline.process_session(session)

Architecture (Invariant Labs-style layered guards):

    ┌─────────────────────────────────────────────────────────────┐
    │                     FraudPipeline                           │
    │                                                             │
    │  Pre-Commerce Layer (FIND_SERVICE / GET_SERVICE):           │
    │    PayloadScanGuard → IntentGuard → SequenceModelGuard      │
    │    ToolTrustGuard                                           │
    │         │                                                   │
    │         │ precommerce_risk flows into ↓                     │
    │         ▼                                                   │
    │  At-Commerce Layer (AUTHORIZE / SETTLE):                    │
    │    PriceOracleGuard → ReplayGuard → VelocityGuard           │
    │    TrajectoryGuard                                          │
    │         │                                                   │
    │         │ atcommerce_risk flows into ↓                      │
    │         ▼                                                   │
    │  Settlement Layer (end-of-session):                         │
    │    SettlementGraphGuard (L6) → CrossSessionGuard            │
    └─────────────────────────────────────────────────────────────┘

PayloadScanGuard is unusual: it fires on ALL events (not just pre-commerce)
because prompt injection can arrive at any moment.
"""
from __future__ import annotations
from typing import Optional

from fraud.context import (
    FraudContext, GuardSignal, Decision,
    ALLOW_THRESHOLD, ESCALATE_THRESHOLD,
)
from fraud.guards.precommerce import (
    PayloadScanGuard,
    IntentGuard,
    ToolTrustGuard,
    SequenceModelGuard,
)
from fraud.guards.atcommerce import (
    PriceOracleGuard,
    ReplayGuard,
    VelocityGuard,
    TrajectoryGuard,
)
from fraud.guards.settlement import (
    SettlementGraphGuard,
    CrossSessionGuard,
)


class FraudPipeline:
    """
    Routes each event to the correct guard layer based on action_type.
    Carries FraudContext between layers so every guard can read prior signals.

    Settlement guards are stateful (hold graphs/history across sessions);
    all other guards are stateless per-event.
    """

    def __init__(
        self,
        precommerce_guards=None,
        atcommerce_guards=None,
        settlement_guards=None,
        block_threshold: float  = ESCALATE_THRESHOLD,
        flag_threshold: float   = ALLOW_THRESHOLD,
    ):
        self.precommerce_guards = precommerce_guards or []
        self.atcommerce_guards  = atcommerce_guards  or []
        self.settlement_guards  = settlement_guards  or []
        self.block_threshold    = block_threshold
        self.flag_threshold     = flag_threshold

    @classmethod
    def default(cls) -> "FraudPipeline":
        """Factory: standard 3-layer pipeline."""
        return cls(
            precommerce_guards=[
                PayloadScanGuard(),     # fires on ALL event types
                IntentGuard(),
                SequenceModelGuard(),
                ToolTrustGuard(),
            ],
            atcommerce_guards=[
                PriceOracleGuard(),
                ReplayGuard(),
                VelocityGuard(),
                TrajectoryGuard(),
            ],
            settlement_guards=[
                SettlementGraphGuard(),
                CrossSessionGuard(),
            ],
        )

    @classmethod
    def precommerce_only(cls) -> "FraudPipeline":
        """Factory: pre-commerce harness layer only (no payment rail access)."""
        return cls(
            precommerce_guards=[
                PayloadScanGuard(),
                IntentGuard(),
                SequenceModelGuard(),
                ToolTrustGuard(),
            ],
        )

    @classmethod
    def atcommerce_only(cls) -> "FraudPipeline":
        """Factory: wallet / payment rail layer only (Stripe Radar equivalent)."""
        return cls(
            atcommerce_guards=[
                PriceOracleGuard(),
                ReplayGuard(),
                VelocityGuard(),
                TrajectoryGuard(),
            ],
        )

    # ── per-event processing ──────────────────────────────────────────────

    def process_event(
        self, event, context: FraudContext
    ) -> tuple[FraudContext, list[GuardSignal]]:
        """
        Process one event through the guard pipeline.
        Returns (updated_context, signals_emitted_this_event).
        """
        # Add event to history FIRST so guards can see it
        context = context.with_event(event)

        action = str(getattr(event, "action_type", "")).lower()
        signals: list[GuardSignal] = []

        # PayloadScanGuard fires on EVERY event
        payload_guard = next(
            (g for g in self.precommerce_guards if g.name == "payload_scan"),
            None,
        )
        if payload_guard:
            sig = payload_guard.detect(event, context)
            if sig and sig.score > 0:
                context = context.with_signal(sig)
                signals.append(sig)

        # Pre-commerce guards (non-payload) — FIND_SERVICE, GET_SERVICE
        is_precommerce = any(
            kw in action for kw in ("find_service", "get_service", "browse", "search")
        )
        if is_precommerce:
            for guard in self.precommerce_guards:
                if guard.name == "payload_scan":
                    continue  # already ran above
                if not guard.should_fire(action):
                    continue
                sig = guard.detect(event, context)
                if sig and sig.score > 0:
                    context = context.with_signal(sig)
                    signals.append(sig)

        # At-commerce guards — AUTHORIZE, SETTLE, A2A_TRANSFER
        is_atcommerce = any(
            kw in action for kw in ("authorize", "settle", "a2a", "transfer", "pay")
        )
        if is_atcommerce:
            for guard in self.atcommerce_guards:
                if not guard.should_fire(action):
                    continue
                sig = guard.detect(event, context)
                if sig and sig.score > 0:
                    context = context.with_signal(sig)
                    signals.append(sig)

        return context, signals

    # ── end of session ────────────────────────────────────────────────────

    def end_session(
        self, context: FraudContext
    ) -> tuple[FraudContext, list[GuardSignal]]:
        """
        Run settlement guards. Call once after all events are processed.
        Settlement guards are stateful — they accumulate across sessions.
        """
        signals: list[GuardSignal] = []
        for guard in self.settlement_guards:
            sig = guard.finalize(context)
            if sig and sig.score > 0:
                context = context.with_signal(sig)
                signals.append(sig)
        return context, signals

    # ── batch API ─────────────────────────────────────────────────────────

    def process_session(
        self,
        session,
        metadata: dict = None,
    ) -> FraudContext:
        """
        Process an entire Session object at once.
        Returns the final FraudContext with all signals accumulated.
        """
        context = FraudContext.for_session(
            session_id=getattr(session, "session_id", "unknown"),
            agent_id=getattr(session, "agent_id", "unknown"),
            persona=getattr(session, "persona", None),
            metadata=metadata or {},
        )

        for event in getattr(session, "events", []):
            context, _ = self.process_event(event, context)

        context, _ = self.end_session(context)
        return context

    # ── scoring shim (for benchmark adapter) ─────────────────────────────

    def score_session(self, session) -> tuple[float, list[str]]:
        """
        Returns (risk_score, flags) for a session.
        Benchmark-compatible interface — mirrors harness/detect/adapter.detect().
        """
        ctx = self.process_session(session)
        return ctx.risk_score, ctx.risk_flags
