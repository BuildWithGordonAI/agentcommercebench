"""
At-Commerce Guards — fire AT the AUTHORIZE / SETTLE moment.

These are the wallet-side / payment-rail-side guards.
They can be deployed WITHOUT the pre-commerce layer (standalone, like Stripe Radar).
When the pre-commerce layer IS present, they read precommerce_risk from the
context and amplify their thresholds accordingly.

Guards:
    PriceOracleGuard   — amount vs category normal range (L4)
    ReplayGuard        — idempotency key replay detection (L4)
    VelocityGuard      — authorize call rate within session
    TrajectoryGuard    — consent envelope + spend pattern conformity (L2)
"""
from .oracle     import PriceOracleGuard
from .replay     import ReplayGuard
from .velocity   import VelocityGuard
from .trajectory import TrajectoryGuard

__all__ = [
    "PriceOracleGuard",
    "ReplayGuard",
    "VelocityGuard",
    "TrajectoryGuard",
]
