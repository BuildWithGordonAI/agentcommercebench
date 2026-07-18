"""
ReplayGuard — L4: idempotency key replay detection.

Checks:
  1. Explicit REPLAYED- prefix in idempotency key (injected by attackers)
  2. Duplicate idempotency key within the current session history
"""
from __future__ import annotations
from typing import Optional
from fraud.guards.base import AtCommerceGuard
from fraud.context import FraudContext, GuardSignal


class ReplayGuard(AtCommerceGuard):
    name = "replay_guard"

    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        req       = getattr(event, "original_request", {}) or {}
        event_id  = getattr(event, "event_id", "")
        idem_key  = req.get("idempotency_key") or event_id

        scores: list[float] = []
        flags:  list[str]   = []

        # Explicit replay marker
        if "REPLAYED-" in idem_key:
            scores.append(0.95)
            flags.append(f"L4:explicit_replay_key:{idem_key[:40]}")
        else:
            # Duplicate within this session
            session_keys = {
                ((getattr(e, "original_request", {}) or {}).get("idempotency_key")
                 or getattr(e, "event_id", ""))
                for e in context.events
                if str(getattr(e, "action_type", "")).lower().endswith("authorize")
            }
            if idem_key in session_keys:
                scores.append(0.90)
                flags.append(f"L4:duplicate_key_in_session:{idem_key[:40]}")

        return self._signal(event, max(scores) if scores else 0.0, flags)
