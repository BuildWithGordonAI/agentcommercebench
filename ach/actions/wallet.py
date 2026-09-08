from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class ConsentLevel(str, Enum):
    PRE_APPROVED = "pre_approved"  # execute within limits, no pause
    SOFT         = "soft"          # alert but execute
    HARD         = "hard"          # pause; wait for external confirmation


@dataclass
class SpendLimits:
    per_transaction: Decimal    = Decimal("1000.00")
    per_day:         Decimal    = Decimal("5000.00")
    per_month:       Decimal    = Decimal("20000.00")
    allowed_mcc:     list[str]  = field(default_factory=list)   # empty = all allowed
    blocked_mcc:     list[str]  = field(default_factory=list)

    def allows_mcc(self, mcc: str) -> bool:
        if mcc in self.blocked_mcc:
            return False
        if not self.allowed_mcc:
            return True
        return mcc in self.allowed_mcc

    def within_transaction_limit(self, amount: Decimal) -> bool:
        return amount <= self.per_transaction


@dataclass
class AgentWallet:
    wallet_id:     str
    agent_id:      str
    persona:       str              = "research"    # travel | procurement | research
    limits:        SpendLimits      = field(default_factory=SpendLimits)
    consent_level: ConsentLevel     = ConsentLevel.SOFT
    mit_token:     Optional[str]    = None   # Merchant-Initiated Transaction token
    metadata:      dict             = field(default_factory=dict)
