"""
Settlement Guards — fire at end-of-session, after all payments complete.

These are cross-session, network-graph-level guards.
They read `context.signals` from both precommerce and atcommerce
phases to decide whether to emit a settlement-level alert.

Guards:
    SettlementGraphGuard — circular flow, Sybil clusters, dense subgraphs (L6)
    CrossSessionGuard    — per-agent velocity burst, persona diversity
"""
from .graph         import SettlementGraphGuard
from .cross_session import CrossSessionGuard

__all__ = ["SettlementGraphGuard", "CrossSessionGuard"]
