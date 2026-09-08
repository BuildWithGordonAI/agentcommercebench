"""
ACH core action types.

Commerce primitives that map onto the existing harness ActionType,
but expressed in ACP terms: FIND / QUOTE / RESERVE / COMMIT / SETTLE / VOID / REFUND.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional
import uuid


class ActionType(str, Enum):
    FIND    = "find"     # read-only, zero financial consequence
    QUOTE   = "quote"    # read-only, price inquiry
    RESERVE = "reserve"  # holds funds; reversible via VOID
    COMMIT  = "commit"   # executes purchase; partially reversible
    SETTLE  = "settle"   # finalises settlement
    VOID    = "void"     # cancels a prior RESERVE (compensating)
    REFUND  = "refund"   # compensating action for a COMMIT


# Maps ACP ActionType → harness ActionType value for pipeline routing
_TO_HARNESS: dict[ActionType, str] = {
    ActionType.FIND:    "find_service",
    ActionType.QUOTE:   "get_service",
    ActionType.RESERVE: "authorize",
    ActionType.COMMIT:  "authorize",
    ActionType.SETTLE:  "settle",
    ActionType.VOID:    "authorize",   # void is still an auth-class event
    ActionType.REFUND:  "settle",
}


def to_harness_action(at: ActionType) -> str:
    return _TO_HARNESS[at]


class Reversibility(str, Enum):
    FULL    = "full"     # void available, no cost
    PARTIAL = "partial"  # refund possible; fees may apply
    NONE    = "none"     # wire, crypto — no recourse


@dataclass
class ConsequentialAction:
    action_type:      ActionType
    amount:           Decimal          = Decimal("0.00")
    currency:         str              = "USD"
    merchant_id:      str              = ""
    merchant_name:    str              = ""
    merchant_mcc:     str              = ""
    category:         str              = "general"
    reversibility:    Reversibility    = Reversibility.PARTIAL
    idempotency_key:  str              = field(default_factory=lambda: str(uuid.uuid4()))
    payload:          dict             = field(default_factory=dict)
    compensation:     Optional["ConsequentialAction"] = None  # saga rollback
