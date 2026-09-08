"""
Registry layer — identity and settlement state.

This is the layer that needs memory the action itself does not carry: which
idempotency keys have already settled, which agent id owns this session, which
counterparties are new. Catching replay or impersonation without this state is
not a matter of a better model; the information is simply absent.

In-process here, so it holds only for the current run. In production this backs
onto durable stores, which is why these checks belong outside the sub-300ms
in-process decision path.
"""
from __future__ import annotations

from ..schema import ActionType, Action
from .base import Context

_PAY = (ActionType.AUTHORIZE, ActionType.A2A_TRANSFER)


class RegistryDetector:
    name = "registry"

    def score(self, action: Action, ctx: Context) -> tuple[float, list[str]]:
        risk = 0.0
        flags: list[str] = []

        # Replay: this key already settled in this run.
        if action.idempotency_key and action.idempotency_key in ctx.settled_keys:
            risk = max(risk, 0.95)
            flags.append("idempotency_replay")

        # Impersonation: the agent id moved mid-session.
        session_agent = ctx.session.agent_id if ctx.session else None
        if session_agent and action.agent_id != session_agent:
            risk = max(risk, 0.85)
            flags.append(f"agent_id_mismatch:{action.agent_id}")

        if action.action_type in _PAY and action.vendor:
            # Circular routing: value returning to the originating agent.
            if action.vendor == session_agent:
                risk = max(risk, 0.80)
                flags.append("circular_settlement")

            chain = action.payload.get("circular_chain")
            if isinstance(chain, (list, tuple)) and session_agent in chain:
                risk = max(risk, 0.85)
                flags.append("circular_chain_declared")

            # Sybil: a counterparty warmed up on dust, then paid at scale.
            to_vendor = [
                a for a in ctx.history if a.vendor == action.vendor and a.amount_units
            ]
            if len(to_vendor) >= 4 and action.amount_units:
                dust = [a for a in to_vendor if a.amount_units <= 50_000]
                if len(dust) >= 4 and action.amount_units >= 1_000_000:
                    risk = max(risk, 0.75)
                    flags.append("sybil_warmup_then_payout")

        return risk, flags
