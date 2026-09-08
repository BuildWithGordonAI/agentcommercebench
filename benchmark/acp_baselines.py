"""
Baseline verifiers for the ACP benchmark paper.

Four baselines representing the state-of-practice:

  B0  NoVerification       — raw agent with no fraud gate (LangChain default)
  B1  SchemaValidation     — LangGraph interrupt + output schema check only
                             (structural typing but no semantic fraud detection)
  B2  RulesOnly            — LocalRulesVerifier (our L1 gate in isolation)
  B3  MLOnly               — GordonVerifier without uncertainty routing
  ACH ACP Full Pipeline    — LocalRules + Gordon + uncertainty + ceremony tiers

LangChain framing (B0):
  A LangChain agent calls tools freely; there is no built-in mechanism to
  intercept a tool call for fraud assessment before execution. B0 represents
  this: every action is approved unconditionally.

LangGraph framing (B1):
  LangGraph can interrupt_before a node and validate state schema.  B1 checks
  that the action fields are well-typed and within rough numeric bounds, but
  has no behavioural model — it catches only obviously malformed requests.

CrewAI framing (B2-adjacent):
  CrewAI enforces task output schemas and role scoping, similar to B1.  A
  CrewAI task validator checking spend limits maps to our LocalRulesVerifier,
  which is why B2 is the natural CrewAI-equivalent baseline.
"""
from __future__ import annotations
import re
from decimal import Decimal

from ach.actions.base import ConsequentialAction, ActionType
from ach.actions.wallet import AgentWallet
from ach.verifiers.base import Verification, SessionContext
from ach.verifiers.local_rules import LocalRulesVerifier
from ach.verifiers.uncertainty import disagreement_uncertainty
from benchmark.behavioral_ml import BehavioralMLBaseline

# ── B0: No Verification ───────────────────────────────────────────────────────

class NoVerificationBaseline:
    """
    Represents a vanilla LangChain / raw-function-call agent.
    Every action is approved — no fraud gate exists.
    Expected: TP=0, FP=0, FN=all attacks, TN=all clean.
    F1 = 0.000.  This is the 'unprotected' baseline.
    """
    verifier_id = "baseline:no_verification"

    def verify(self, action, wallet, context) -> Verification:
        return Verification(
            verifier_id    = self.verifier_id,
            decision       = "allow",
            score          = 0.0,
            confidence     = 1.0,
            uncertainty    = 0.0,
            score_interval = (0.0, 0.0),
        )


# ── B1: Schema Validation (LangGraph-style) ───────────────────────────────────

_MAX_SANE_AMOUNT = Decimal("50000")   # absurdly large — catches only typos

class SchemaValidationBaseline:
    """
    Mimics LangGraph interrupt_before + Pydantic output schema validation.

    Checks:
      - Amount is a positive Decimal within a sane range
      - merchant_id is a non-empty string
      - action_type is a known ActionType
      - idempotency_key is non-empty

    Does NOT check: behavioural patterns, velocity, MCC, replay, injection.
    Represents what a developer gets "for free" from a typed agent framework.
    """
    verifier_id = "baseline:schema_validation"

    def verify(self, action, wallet, context) -> Verification:
        flags: list[str] = []
        score = 0.0

        if action.action_type in (ActionType.FIND, ActionType.QUOTE):
            return Verification(self.verifier_id, "allow", 0.0,
                                confidence=1.0, uncertainty=0.0, score_interval=(0.0,0.0))

        # Field presence
        if not action.merchant_id:
            flags.append("schema:missing_merchant_id")
            score = max(score, 0.80)
        if not action.idempotency_key:
            flags.append("schema:missing_idempotency_key")
            score = max(score, 0.80)

        # Numeric range
        if action.amount <= 0:
            flags.append(f"schema:non_positive_amount:{action.amount}")
            score = max(score, 0.80)
        if action.amount > _MAX_SANE_AMOUNT:
            flags.append(f"schema:amount_exceeds_sanity_cap:{action.amount}")
            score = max(score, 0.75)

        # Known action type (always true since we use the enum, but mirrors a string-typed API)
        if action.action_type not in ActionType.__members__.values():
            flags.append(f"schema:unknown_action_type")
            score = max(score, 0.90)

        decision = "block" if score >= 0.70 else "allow"
        return Verification(
            verifier_id    = self.verifier_id,
            decision       = decision,
            score          = score,
            confidence     = 1.0 if score > 0 else 0.9,
            uncertainty    = 0.0,
            score_interval = (max(0, score - 0.05), min(1, score + 0.05)),
            flags          = flags,
        )


# ── B2: Rules Only ────────────────────────────────────────────────────────────
# Thin wrapper so benchmark can label it clearly

class RulesOnlyBaseline(LocalRulesVerifier):
    """LocalRulesVerifier exposed as a named baseline (CrewAI-equivalent)."""
    verifier_id = "baseline:rules_only"


# ── B3: ML Only (standalone behavioral model, no uncertainty routing) ─────────

class MLOnlyBaseline(BehavioralMLBaseline):
    """
    Standalone behavioral ML verifier without uncertainty-based escalation.
    Trained on synthetic clean sessions (seed=7, N=500); no external API needed.
    Represents 'add a behavioral ML model but ignore its confidence'.
    """
    verifier_id = "baseline:ml_only"


# ── ACH: Full ACP Pipeline ────────────────────────────────────────────────────

class ACHFullPipeline:
    """
    LocalRules + Gordon in parallel; uncertainty-aware; ceremony-routed.

    Decision logic:
      1. Run both verifiers.
      2. Compute disagreement (aleatory uncertainty).
      3. Augment each verification's uncertainty with disagreement.
      4. Most conservative decision wins (block > flag > allow).
      5. If should_escalate → escalate decision to 'flag' minimum.
      6. Emit ceremony level based on (score, uncertainty, amount).
    """
    verifier_id = "ach:full_pipeline"

    def __init__(self):
        self._local  = LocalRulesVerifier()
        self._gordon = BehavioralMLBaseline()   # OSS standalone; production uses Gordon

    def verify(self, action, wallet, context) -> Verification:
        vl = self._local.verify(action, wallet, context)
        vg = self._gordon.verify(action, wallet, context)

        dis = disagreement_uncertainty([vl, vg])

        # Merge uncertainty: take max epistemic + shared aleatory
        merged_unc = min(1.0, max(vl.uncertainty, vg.uncertainty) + dis * 0.5)

        # Conservative decision
        rank = {"allow": 0, "flag": 1, "block": 2}
        if rank[vl.decision] >= rank[vg.decision]:
            winner, loser = vl, vg
        else:
            winner, loser = vg, vl

        decision = winner.decision
        score    = max(vl.score, vg.score)
        ci_low   = min(vl.score_interval[0], vg.score_interval[0])
        ci_high  = max(vl.score_interval[1], vg.score_interval[1])

        # When LocalRules finds no violation (score=0, decision=allow) but ML
        # says "block" with high uncertainty, downgrade block→flag: the rules
        # provide hard evidence of no violation, overriding an uncertain ML verdict.
        # This is the key ACP advantage: deterministic rules constrain probabilistic ML.
        if (decision == "block"
                and vl.decision == "allow" and vl.score == 0.0
                and vg.uncertainty > 0.20):
            decision = "flag"

        # Escalate allow→flag only when verifiers disagree in decision AND uncertainty is high.
        _rank = {"allow": 0, "flag": 1, "block": 2}
        decision_gap = abs(_rank[vl.decision] - _rank[vg.decision])
        if decision == "allow" and decision_gap >= 1 and merged_unc > 0.35:
            decision = "flag"

        all_flags = (
            [f"[local]  {f}"  for f in vl.flags] +
            [f"[gordon] {f}"  for f in vg.flags]
        )
        if dis > 0.2:
            all_flags.append(f"disagreement={dis:.2f}")

        return Verification(
            verifier_id    = self.verifier_id,
            decision       = decision,
            score          = round(score, 4),
            confidence     = round(max(0.0, 1.0 - merged_unc), 4),
            uncertainty    = round(merged_unc, 4),
            score_interval = (round(ci_low, 4), round(ci_high, 4)),
            flags          = all_flags,
            metadata       = {
                "local_decision":  vl.decision,
                "gordon_decision": vg.decision,
                "disagreement":    dis,
            },
        )

    def reset_session(self, session_id: str):
        self._local.reset_session(session_id) if hasattr(self._local, "reset_session") else None
        self._gordon.reset_session(session_id)


# ── registry for benchmark runner ─────────────────────────────────────────────

BASELINES: dict[str, object] = {
    "NoVerification":  NoVerificationBaseline,
    "SchemaOnly":      SchemaValidationBaseline,   # LangGraph-style
    "RulesOnly":       RulesOnlyBaseline,           # CrewAI-style
    "MLOnly":          MLOnlyBaseline,
    "ACP":             ACHFullPipeline,
}
