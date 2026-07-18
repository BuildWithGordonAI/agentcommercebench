"""
IntentGuard — L3: category / amount vs persona behavioral profile.

Source: harness/detect/l3_behavioral.py (refactored for Guard interface).

Fires pre-commerce on FIND_SERVICE / GET_SERVICE and also at AUTHORIZE.
Checks:
  - Is this category blocked for this persona?
  - Is this category unexpected (but not blocked)?
  - Does the amount exceed the persona ceiling?

Inter-layer: if precommerce_risk is already elevated from PayloadScanGuard,
this guard tightens its soft-flag threshold.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Optional
from fraud.guards.base import PreCommerceGuard
from fraud.context import FraudContext, GuardSignal, Phase

# Persona profiles — calibrated from prod amounts (1 USDC = 1_000_000 μUSDC)
_PROFILE: dict[str, dict] = {
    "research": {
        "allowed":   {"finance", "search", "ai", "data"},
        "blocked":   {"travel", "infrastructure", "creative"},
        "max_units": 300_000,
    },
    "procurement": {
        "allowed":   {"procurement", "search", "data", "enrichment"},
        "blocked":   {"travel", "finance", "infrastructure"},
        "max_units": 550_000,
    },
    "travel": {
        "allowed":   {"travel", "search"},
        "blocked":   {"finance", "infrastructure"},
        "max_units": 350_000,
    },
}

_NORMAL_MAX_UNITS: dict[str, int] = {
    "search":      15_000,
    "finance":     250_000,
    "ai":          100_000,
    "procurement": 500_000,
    "travel":      300_000,
    "default":     100_000,
}


def _infer_persona(history: list) -> str:
    cats = [getattr(e, "category", None) for e in history if getattr(e, "category", None)]
    if not cats:
        return "research"
    freq: dict[str, int] = defaultdict(int)
    for c in cats:
        freq[c] += 1
    top = max(freq, key=freq.get)
    if top in ("travel", "hotel", "flight"):
        return "travel"
    if top in ("procurement",):
        return "procurement"
    return "research"


class IntentGuard(PreCommerceGuard):
    """
    Behavioral intent fingerprint — category + amount vs persona profile.
    Also fires on AUTHORIZE to catch drift at payment time.
    """
    name  = "intent_fingerprint"
    phase = Phase.PRECOMMERCE

    def should_fire(self, action_type: str) -> bool:
        return action_type in ("find_service", "get_service", "authorize", "settle")

    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        category = getattr(event, "category", None) or "unknown"
        amount   = getattr(event, "amount_units", None) or 0
        action   = getattr(event, "action_type", None)

        # Resolve persona from context, then event annotation, then history
        persona = (
            context.persona
            or getattr(event, "_persona", None)
            or _infer_persona(context.events)
        )
        profile = _PROFILE.get(persona, _PROFILE["research"])

        scores: list[float] = []
        flags:  list[str]   = []

        # ── Category check ────────────────────────────────────────────────
        if category in profile["blocked"]:
            scores.append(0.88)
            flags.append(f"L3:blocked_category:{category}_for_{persona}")

        elif category not in profile["allowed"]:
            # Inter-layer amplification: if payload scan already alarmed, be stricter
            soft_score = 0.65 if context.precommerce_risk >= 0.30 else 0.50
            scores.append(soft_score)
            flags.append(f"L3:unexpected_category:{category}")

        # ── Amount check (only meaningful on AUTHORIZE) ───────────────────
        if action and str(action) in ("authorize", "ActionType.AUTHORIZE") and amount:
            max_units = profile["max_units"]
            if amount > max_units:
                ratio = amount / max_units
                amt_score = min(0.85, 0.40 + 0.15 * math.log2(ratio))
                scores.append(amt_score)
                flags.append(f"L3:amount_exceeds_limit:{amount}>{int(max_units)}")

        return self._signal(event, max(scores) if scores else 0.0, flags)
