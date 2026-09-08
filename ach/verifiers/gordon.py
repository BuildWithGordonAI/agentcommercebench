"""
GordonVerifier — bridges ACH into the existing fraud.pipeline.

Translates ConsequentialAction → harness Event, runs FraudPipeline,
translates FraudContext back to Verification. No network call required;
uses the local fraud/ package directly.
"""
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
import uuid

from ach.actions.base import ConsequentialAction, ActionType, to_harness_action
from ach.actions.wallet import AgentWallet
from ach.verifiers.base import Verification, SessionContext
from ach.verifiers.uncertainty import UncertaintyEstimator

from fraud.pipeline import FraudPipeline
from fraud.context import FraudContext, Decision


class GordonVerifier:
    """
    Wraps the Gordon fraud pipeline as an ACP Verifier.

    Maintains one FraudContext per session_id in memory.
    Call reset_session(session_id) to clear state between benchmark runs.
    """
    verifier_id = "did:gordon:verifier:fraud-pipeline"

    def __init__(self, pipeline: FraudPipeline = None):
        self._pipeline  = pipeline or FraudPipeline.default()
        self._contexts: dict[str, FraudContext] = {}
        self._unc = UncertaintyEstimator(
            cold_start_boost  = 0.30,
            ood_boost         = 0.22,
            boundary_width    = 0.18,
        )
        self._new_sessions: set[str] = set()

    def reset_session(self, session_id: str) -> None:
        self._contexts.pop(session_id, None)
        self._new_sessions.discard(session_id)

    def reset_all(self) -> None:
        self._contexts.clear()

    def verify(
        self,
        action:  ConsequentialAction,
        wallet:  AgentWallet,
        context: SessionContext,
    ) -> Verification:
        session_id = context.session_id
        agent_id   = wallet.agent_id

        # Retrieve or create FraudContext for this session
        is_new = session_id not in self._contexts
        fc = self._contexts.get(session_id)
        if fc is None:
            fc = FraudContext.for_session(session_id, agent_id, wallet.persona)
            self._new_sessions.add(session_id)

        # Build a harness-compatible Event from the ACH action
        event = _build_event(action, context)

        # Run through the full guard pipeline
        fc, signals = self._pipeline.process_event(event, fc)
        self._contexts[session_id] = fc

        # Translate FraudContext decision → ACP Verification
        fraud_decision = fc.decision
        if fraud_decision == Decision.BLOCK:
            decision = "block"
        elif fraud_decision == Decision.ESCALATE:
            decision = "flag"
        else:
            decision = "allow"

        ue = self._unc.estimate(
            score          = round(fc.risk_score, 4),
            agent_id       = agent_id,
            persona        = wallet.persona,
            category       = action.category,
            is_new_session = is_new,
        )

        return Verification(
            verifier_id    = self.verifier_id,
            decision       = decision,
            score          = round(fc.risk_score, 4),
            confidence     = ue.confidence,
            uncertainty    = ue.total,
            score_interval = (ue.ci_low, ue.ci_high),
            flags          = fc.risk_flags,
            metadata       = {
                "precommerce_risk":   fc.precommerce_risk,
                "atcommerce_risk":    fc.atcommerce_risk,
                "n_signals":          len(fc.signals),
                "uncertainty_sources": ue.sources,
            },
        )

    def end_session(self, session_id: str, agent_id: str, persona: str = None) -> Verification:
        """Run settlement guards. Call after all events in a session."""
        fc = self._contexts.get(session_id)
        if fc is None:
            fc = FraudContext.for_session(session_id, agent_id, persona)

        fc, signals = self._pipeline.end_session(fc)
        self._contexts[session_id] = fc

        fraud_decision = fc.decision
        decision = "block" if fraud_decision == Decision.BLOCK else \
                   "flag"  if fraud_decision == Decision.ESCALATE else "allow"

        return Verification(
            verifier_id = self.verifier_id,
            decision    = decision,
            score       = round(fc.risk_score, 4),
            flags       = fc.risk_flags,
        )


# ── event builder ─────────────────────────────────────────────────────────────

def _build_event(action: ConsequentialAction, context: SessionContext):
    """Build a harness-compatible Event from an ACH ConsequentialAction."""
    # Import here to avoid circular issues; harness schema is external
    from harness.simulate.schema import Event, ActionType as HarnessActionType

    harness_at = to_harness_action(action.action_type)

    # Build original_request for replay guard (needs idempotency_key)
    original_request: dict = {"idempotency_key": action.idempotency_key}
    if action.payload:
        original_request.update(action.payload)

    # Detect prompt injection hint in payload
    payload_str = str(action.payload)
    if any(marker in payload_str for marker in ("REPLAYED-", "ignore previous", "system:")):
        original_request["_suspicious"] = True

    return Event(
        event_id         = str(uuid.uuid4()),
        session_id       = context.session_id,
        agent_id         = context.agent_id,
        action_type      = HarnessActionType(harness_at),
        timestamp        = datetime.utcnow(),
        service_id       = action.merchant_id or None,
        operation_id     = action.action_type.value,
        # amount_units is USDC micro-units for on-chain rails; leave None for
        # off-chain commerce so PriceOracleGuard skips the on-chain threshold.
        amount_units     = None,
        vendor           = action.merchant_id or None,
        category         = action.category or "general",
        raw_endpoint     = action.merchant_name or None,
        original_request = original_request,
    )
