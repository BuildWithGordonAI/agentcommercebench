"""
PriceOracleGuard — L4: amount vs category normal range.

Source: harness/detect/l4_price.py (amount-check portion).

Inter-layer amplification:
    If precommerce_risk >= 0.50 (pre-commerce already suspicious),
    lower the 2× threshold to 1.5× — be 50% more aggressive.
    This is the core "inter-layer talk": the pre-commerce layer's
    signal makes the oracle tighter at payment time.
"""
from __future__ import annotations
import math
from typing import Optional
from fraud.guards.base import AtCommerceGuard
from fraud.context import FraudContext, GuardSignal

_NORMAL_MAX_UNITS: dict[str, int] = {
    "search":      15_000,
    "finance":     250_000,
    "ai":          100_000,
    "procurement": 500_000,
    "travel":      300_000,
    "a2a":         200_000,
    "default":     100_000,
}


class PriceOracleGuard(AtCommerceGuard):
    name = "price_oracle"

    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        amount   = getattr(event, "amount_units", None) or 0
        category = getattr(event, "category", None) or "default"
        if not amount:
            return None

        normal_max = _NORMAL_MAX_UNITS.get(category, _NORMAL_MAX_UNITS["default"])

        # Inter-layer amplification: tighten thresholds if pre-commerce alarmed
        multiplier_10x = 10
        multiplier_5x  = 5
        multiplier_2x  = 2.0 if context.precommerce_risk < 0.50 else 1.5

        scores: list[float] = []
        flags:  list[str]   = []

        if amount > normal_max * multiplier_10x:
            scores.append(0.92)
            flags.append(f"L4:price_10x:{amount}>{normal_max * multiplier_10x}")
        elif amount > normal_max * multiplier_5x:
            scores.append(0.78)
            flags.append(f"L4:price_5x:{amount}>{normal_max * multiplier_5x}")
        elif amount > normal_max * multiplier_2x:
            amp = "amplified" if context.precommerce_risk >= 0.50 else ""
            scores.append(0.55)
            flags.append(f"L4:price_{multiplier_2x}x:{amount}>{int(normal_max * multiplier_2x)} {amp}".strip())

        return self._signal(event, max(scores) if scores else 0.0, flags)
