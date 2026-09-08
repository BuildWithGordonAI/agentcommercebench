"""
LangGraph integration — CommerceCheckNode.

Drop this into any LangGraph StateGraph before the node that executes payment.
It verifies the pending action and writes the result to state.

Usage:
    graph.add_node("commerce_check", CommerceCheckNode(verifier, wallet))
    graph.add_conditional_edges("commerce_check", route_on_decision)
    graph.add_edge("execute_payment", ...) # only reached if allowed
"""
from __future__ import annotations
from typing import Any

from ach.actions.base import ConsequentialAction
from ach.actions.wallet import AgentWallet
from ach.verifiers.base import Verifier, SessionContext
from ach.receipts.car import CommerceActionReceipt


class CommerceCheckNode:
    """
    Synchronous LangGraph node that runs the verifier before payment.

    State keys consumed:
        pending_action: ConsequentialAction
        session_context: SessionContext

    State keys produced:
        last_car: CommerceActionReceipt
        decision: str  ("allow" | "flag" | "block")
        risk_score: float
        risk_flags: list[str]
    """
    def __init__(self, verifier: Verifier, wallet: AgentWallet):
        self.verifier = verifier
        self.wallet   = wallet

    def __call__(self, state: dict) -> dict:
        action  = state["pending_action"]
        context = state.get("session_context") or SessionContext(
            session_id = state.get("session_id", "unknown"),
            agent_id   = self.wallet.agent_id,
            persona    = self.wallet.persona,
        )

        verification = self.verifier.verify(action, self.wallet, context)
        car = CommerceActionReceipt.build(
            action        = action,
            wallet        = self.wallet,
            verifications = [verification],
        )
        car.session_id = context.session_id

        return {
            **state,
            "last_car":    car,
            "decision":    verification.decision,
            "risk_score":  verification.score,
            "risk_flags":  verification.flags,
        }


def route_on_decision(state: dict) -> str:
    """Conditional edge router for use with graph.add_conditional_edges."""
    d = state.get("decision", "allow")
    if d == "block":  return "handle_block"
    if d == "flag":   return "handle_flag"
    return "execute_payment"
