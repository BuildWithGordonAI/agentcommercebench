"""
TrajectoryGuard — L2: consent envelope + spend conformity.

Source: src/fraud_detection/l2_trajectory_monitor.py (adapted).

Enforces a consent envelope: the agent's declared mandate
(max spend, allowed categories, allowed vendors) against what it
actually does at AUTHORIZE time.

This is the "mandate compliance" guard. An agent can pass all
behavioral checks and still violate its declared consent scope.

Inter-layer: if precommerce_risk is elevated, the consent
envelope is enforced more strictly (reduce block_amount threshold).
"""
from __future__ import annotations
from typing import Optional
from fraud.guards.base import AtCommerceGuard
from fraud.context import FraudContext, GuardSignal


class TrajectoryGuard(AtCommerceGuard):
    """
    Validates AUTHORIZE events against a consent envelope if one is set.
    The consent envelope is stored in context.metadata by the caller.

    consent_envelope keys (all optional):
        max_amount      int   — maximum single-transaction spend in μUSDC
        block_amount    int   — hard block threshold
        allowed_cats    set   — allowed categories
        allowed_vendors set   — allowed vendor wallet addresses
    """
    name = "trajectory"

    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        envelope: dict = getattr(context, "metadata", {}).get("consent_envelope", {})
        if not envelope:
            return None     # no consent envelope configured — skip

        amount   = getattr(event, "amount_units", 0) or 0
        category = getattr(event, "category", None)
        vendor   = getattr(event, "vendor", None)

        scores: list[float] = []
        flags:  list[str]   = []

        # Inter-layer amplification: stricter if pre-commerce already suspicious
        strictness = 1.5 if context.precommerce_risk >= 0.30 else 1.0

        max_amt   = envelope.get("max_amount")
        block_amt = envelope.get("block_amount")

        if block_amt and amount > block_amt / strictness:
            scores.append(0.95)
            flags.append(f"L2:consent_block amount={amount}>{int(block_amt/strictness)}")
        elif max_amt and amount > max_amt / strictness:
            scores.append(0.65)
            flags.append(f"L2:consent_exceeded amount={amount}>{int(max_amt/strictness)}")

        allowed_cats = envelope.get("allowed_cats")
        if allowed_cats and category and category not in allowed_cats:
            scores.append(0.70)
            flags.append(f"L2:category_outside_mandate:{category}")

        allowed_vendors = envelope.get("allowed_vendors")
        if allowed_vendors and vendor and vendor not in allowed_vendors:
            scores.append(0.75)
            flags.append(f"L2:vendor_outside_mandate:{vendor[:20]}")

        return self._signal(event, max(scores) if scores else 0.0, flags)
