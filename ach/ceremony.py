"""
Ceremony Router — maps (action, verification) to ACP ceremony level and anchoring.

Ceremony levels (from ACP protocol spec):
  L0  FIND / QUOTE only — no consequential action, no verification required
  L1  COMMIT < $50 OR low uncertainty — LocalRules sufficient, local log
  L2  COMMIT $50-5000 OR medium uncertainty — verifier required, Merkle batch anchor
  L3  COMMIT > $5000 OR high uncertainty OR multi-verifier disagreement
       — multi-verifier required, real-time on-chain anchor

Escalation rules stack: a single L3 trigger is sufficient for L3 ceremony.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from ach.actions.base import ActionType, ConsequentialAction


class CeremonyLevel(str, Enum):
    L0 = "L0"   # information only — FIND / QUOTE
    L1 = "L1"   # local rules gate — small commit, low uncertainty
    L2 = "L2"   # verifier required — medium commit or medium uncertainty
    L3 = "L3"   # multi-verifier + anchor — large commit or high uncertainty


class AnchoringLevel(str, Enum):
    NONE   = "none"     # no log required (L0)
    LOCAL  = "local"    # hash-chained local log only (L1)
    MERKLE = "merkle"   # batched Merkle root (L2)
    CHAIN  = "chain"    # real-time on-chain anchor (L3)


# Amount thresholds (USD)
_L2_AMOUNT = Decimal("50")
_L3_AMOUNT = Decimal("5000")

# Uncertainty thresholds
_L2_UNCERTAINTY = 0.25
_L3_UNCERTAINTY = 0.45

# Score thresholds
_FLAG_SCORE = 0.45
_BLOCK_SCORE = 0.70


@dataclass(frozen=True)
class CeremonyDecision:
    level:       CeremonyLevel
    anchoring:   AnchoringLevel
    reasons:     tuple[str, ...]
    requires_multi_verifier: bool

    @property
    def requires_verifier(self) -> bool:
        return self.level in (CeremonyLevel.L2, CeremonyLevel.L3)


def route(
    action:      ConsequentialAction,
    score:       float       = 0.0,
    uncertainty: float       = 0.0,
    disagreement: float      = 0.0,
    n_verifiers: int         = 1,
) -> CeremonyDecision:
    """
    Determine the ceremony level for a (action, verification) pair.

    Parameters
    ----------
    action        : the ConsequentialAction being assessed
    score         : point-estimate risk score from the primary verifier
    uncertainty   : epistemic uncertainty from UncertaintyEstimator
    disagreement  : aleatory uncertainty from verifier ensemble disagreement
    n_verifiers   : number of verifiers that have already run
    """
    reasons: list[str] = []

    # --- L0: non-consequential action types ---
    if action.action_type in (ActionType.FIND, ActionType.QUOTE):
        return CeremonyDecision(
            level    = CeremonyLevel.L0,
            anchoring= AnchoringLevel.NONE,
            reasons  = ("non_consequential_action",),
            requires_multi_verifier=False,
        )

    amount = action.amount or Decimal("0")

    # Collect L3 triggers
    l3_triggers: list[str] = []
    if amount >= _L3_AMOUNT:
        l3_triggers.append(f"amount≥${_L3_AMOUNT}")
    if uncertainty >= _L3_UNCERTAINTY:
        l3_triggers.append(f"uncertainty={uncertainty:.2f}≥{_L3_UNCERTAINTY}")
    if disagreement >= 0.35:
        l3_triggers.append(f"verifier_disagreement={disagreement:.2f}")
    if score >= _BLOCK_SCORE and uncertainty >= 0.20:
        l3_triggers.append("uncertain_block")

    if l3_triggers:
        return CeremonyDecision(
            level    = CeremonyLevel.L3,
            anchoring= AnchoringLevel.CHAIN,
            reasons  = tuple(l3_triggers),
            requires_multi_verifier=True,
        )

    # Collect L2 triggers
    l2_triggers: list[str] = []
    if amount >= _L2_AMOUNT:
        l2_triggers.append(f"amount≥${_L2_AMOUNT}")
    if uncertainty >= _L2_UNCERTAINTY:
        l2_triggers.append(f"uncertainty={uncertainty:.2f}≥{_L2_UNCERTAINTY}")
    if score >= _FLAG_SCORE:
        l2_triggers.append(f"score={score:.2f}≥{_FLAG_SCORE}")

    if l2_triggers:
        return CeremonyDecision(
            level    = CeremonyLevel.L2,
            anchoring= AnchoringLevel.MERKLE,
            reasons  = tuple(l2_triggers),
            requires_multi_verifier=False,
        )

    # L1: everything else that is consequential
    return CeremonyDecision(
        level    = CeremonyLevel.L1,
        anchoring= AnchoringLevel.LOCAL,
        reasons  = ("low_risk_commit",),
        requires_multi_verifier=False,
    )
