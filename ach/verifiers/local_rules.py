"""
LocalRulesVerifier — deterministic, no-API-key verifier.

Checks spend limits, MCC allowlist, explicit replay prefix, and basic velocity.
Ships in the open-source repo. Good for dev; not for production.

Uncertainty model:
  Rules are deterministic → confidence=1.0, uncertainty=0.0 for triggered rules.
  When no rule fires (score=0), there is mild epistemic uncertainty because the
  rules don't cover behavioural patterns — reflected as a small cold-start boost.
"""
from __future__ import annotations
from collections import defaultdict
from decimal import Decimal

from ach.actions.base import ConsequentialAction, ActionType
from ach.actions.wallet import AgentWallet
from ach.verifiers.base import Verification, SessionContext
from ach.verifiers.uncertainty import UncertaintyEstimator

_VELOCITY_WARN  = 3
_VELOCITY_BLOCK = 8


class LocalRulesVerifier:
    verifier_id = "did:ach:verifier:local-rules"

    _daily_spend: dict[str, Decimal] = defaultdict(Decimal)

    def __init__(self):
        self._unc = UncertaintyEstimator(
            cold_start_boost  = 0.10,   # rules are structural; cold-start matters less
            ood_boost         = 0.08,
            boundary_width    = 0.12,
        )

    def verify(
        self,
        action:  ConsequentialAction,
        wallet:  AgentWallet,
        context: SessionContext,
    ) -> Verification:
        flags: list[str] = []
        score = 0.0
        at = action.action_type

        if at in (ActionType.FIND, ActionType.QUOTE):
            return Verification(
                self.verifier_id, "allow", 0.0,
                confidence=1.0, uncertainty=0.0, score_interval=(0.0, 0.0),
            )

        # ── 1. Explicit replay ────────────────────────────────────────────────
        if "REPLAYED-" in action.idempotency_key:
            flags.append(f"rules:explicit_replay_key:{action.idempotency_key[:30]}")
            score = max(score, 0.95)

        # ── 2. Per-transaction limit ──────────────────────────────────────────
        if not wallet.limits.within_transaction_limit(action.amount):
            flags.append(
                f"rules:over_per_txn_limit "
                f"amount={action.amount} limit={wallet.limits.per_transaction}"
            )
            score = max(score, 0.90)

        # ── 3. MCC allowlist ──────────────────────────────────────────────────
        if action.merchant_mcc and not wallet.limits.allows_mcc(action.merchant_mcc):
            flags.append(f"rules:mcc_not_allowed:{action.merchant_mcc}")
            score = max(score, 0.85)

        # ── 4. Session velocity ───────────────────────────────────────────────
        n_commits = sum(
            1 for e in context.events
            if getattr(e, "action_type", None) in (ActionType.COMMIT, ActionType.RESERVE)
        )
        if n_commits >= _VELOCITY_BLOCK:
            flags.append(f"rules:velocity_block n={n_commits}")
            score = max(score, 0.92)
        elif n_commits >= _VELOCITY_WARN:
            flags.append(f"rules:velocity_warn n={n_commits}")
            score = max(score, 0.45)

        # ── 5. Daily spend limit ──────────────────────────────────────────────
        spent = self._daily_spend[wallet.agent_id]
        if spent + action.amount > wallet.limits.per_day:
            flags.append(
                f"rules:over_daily_limit "
                f"spent={spent} adding={action.amount} limit={wallet.limits.per_day}"
            )
            score = max(score, 0.88)

        if score < 0.70:
            self._daily_spend[wallet.agent_id] += action.amount

        decision = "block" if score >= 0.70 else "flag" if score >= 0.30 else "allow"

        # Rules are deterministic — no epistemic uncertainty when a rule fires.
        # When clean (score=0), add small uncertainty because rules can't see behaviour.
        if score > 0.0:
            confidence, uncertainty, ci = 1.0, 0.0, (max(0.0, score - 0.05), min(1.0, score + 0.05))
        else:
            ue = self._unc.estimate(
                score=0.0,
                agent_id=context.agent_id,
                persona=context.persona,
                category=action.category,
            )
            confidence   = ue.confidence
            uncertainty  = ue.total
            ci           = (ue.ci_low, ue.ci_high)

        return Verification(
            verifier_id    = self.verifier_id,
            decision       = decision,
            score          = score,
            confidence     = round(confidence, 4),
            uncertainty    = round(uncertainty, 4),
            score_interval = ci,
            flags          = flags,
        )
