"""
VelocityGuard — AUTHORIZE call rate within session + timing anomaly.

Covers:
  B6 — velocity spike (>20 AUTHORIZE calls in session)
  B4 — off-hours timing (AUTHORIZE at 3am)
  B7 — policy probe (many AUTHORIZE calls probing spend limit)
"""
from __future__ import annotations
from typing import Optional
from fraud.guards.base import AtCommerceGuard
from fraud.context import FraudContext, GuardSignal

_ACTIVE_HOURS = (8, 22)     # 08:00–22:00 local
_VELOCITY_THRESHOLD = 20    # authorizations per session before flagging
_PROBE_THRESHOLD = 6        # small-amount AUTHORIZE calls indicating probing


class VelocityGuard(AtCommerceGuard):
    name = "velocity"

    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        scores: list[float] = []
        flags:  list[str]   = []

        # Count prior AUTHORIZE events in this session
        prior_auths = [
            e for e in context.events
            if str(getattr(e, "action_type", "")).lower().endswith("authorize")
        ]
        n_auth = len(prior_auths)

        # B6 — velocity burst
        if n_auth > _VELOCITY_THRESHOLD:
            burst_score = min(0.90, 0.50 + 0.02 * (n_auth - _VELOCITY_THRESHOLD))
            scores.append(burst_score)
            flags.append(f"velocity:burst auth_count={n_auth}")

        # B4 — off-hours (timing anomaly)
        ts = getattr(event, "timestamp", None)
        if ts is not None:
            h = ts.hour
            if not (_ACTIVE_HOURS[0] <= h < _ACTIVE_HOURS[1]):
                distance = min(abs(h - _ACTIVE_HOURS[0]), abs(h - _ACTIVE_HOURS[1]))
                off_score = min(0.60, 0.20 + 0.05 * distance)
                scores.append(off_score)
                flags.append(f"velocity:off_hours hour={h}")

        # B7 — policy probe: many small AUTHORIZE calls with varying amounts
        if n_auth >= _PROBE_THRESHOLD:
            amounts = [
                getattr(e, "amount_units", 0) or 0
                for e in prior_auths[-_PROBE_THRESHOLD:]
            ]
            if amounts and max(amounts) > 0:
                variance_ratio = (max(amounts) - min(amounts)) / max(amounts)
                if variance_ratio > 0.05 and all(a < 200_000 for a in amounts):
                    scores.append(0.65)
                    flags.append(f"velocity:policy_probe n={n_auth} range={min(amounts)}–{max(amounts)}")

        return self._signal(event, max(scores) if scores else 0.0, flags)
