"""
Commerce Action Receipt (CAR) — the atomic unit of ACP.

Every consequential action produces one CAR.
The CAR is the thing that gets signed, stored, and verified.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
import uuid, json

from ach.actions.base import ConsequentialAction
from ach.actions.wallet import AgentWallet
from ach.verifiers.base import Verification
from ach.receipts.signer import sign


@dataclass
class ProvenanceStep:
    step:       int
    type:       str    # FIND | QUOTE | COMMIT | SESSION_START | SESSION_END
    tool:       str    = ""
    args_hash:  str    = ""
    car_id:     str    = ""
    timestamp:  str    = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CommerceActionReceipt:
    car_id:          str
    session_id:      str
    agent_id:        str
    wallet_id:       str
    action:          ConsequentialAction
    verifications:   list[Verification]
    final_decision:  str            # "allow" | "flag" | "block"
    provenance:      list[ProvenanceStep] = field(default_factory=list)
    timestamp:       str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    outcome:         dict           = field(default_factory=dict)
    signature:       str            = ""

    @classmethod
    def build(
        cls,
        action:        ConsequentialAction,
        wallet:        AgentWallet,
        verifications: list[Verification],
        session_id:    str = None,
        provenance:    list[ProvenanceStep] = None,
        outcome:       dict = None,
    ) -> "CommerceActionReceipt":
        # Aggregate decision: any BLOCK wins; any FLAG beats ALLOW
        decision = "allow"
        for v in verifications:
            if v.decision == "block":
                decision = "block"
                break
            if v.decision == "flag":
                decision = "flag"

        car = cls(
            car_id         = "car_" + str(uuid.uuid4())[:8],
            session_id     = session_id or wallet.agent_id,
            agent_id       = wallet.agent_id,
            wallet_id      = wallet.wallet_id,
            action         = action,
            verifications  = verifications,
            final_decision = decision,
            provenance     = provenance or [],
            outcome        = outcome or {},
        )
        car.signature = car._sign()
        return car

    def _sign(self) -> str:
        payload = {
            "car_id":         self.car_id,
            "session_id":     self.session_id,
            "agent_id":       self.agent_id,
            "action_type":    self.action.action_type.value,
            "amount":         str(self.action.amount),
            "merchant_id":    self.action.merchant_id,
            "idempotency_key": self.action.idempotency_key,
            "final_decision": self.final_decision,
            "timestamp":      self.timestamp,
        }
        return sign(payload)

    def is_valid(self) -> bool:
        from ach.receipts.signer import verify_signature
        payload = {
            "car_id":         self.car_id,
            "session_id":     self.session_id,
            "agent_id":       self.agent_id,
            "action_type":    self.action.action_type.value,
            "amount":         str(self.action.amount),
            "merchant_id":    self.action.merchant_id,
            "idempotency_key": self.action.idempotency_key,
            "final_decision": self.final_decision,
            "timestamp":      self.timestamp,
        }
        return verify_signature(payload, self.signature)

    def to_dict(self) -> dict:
        return {
            "car_id":         self.car_id,
            "session_id":     self.session_id,
            "agent_id":       self.agent_id,
            "wallet_id":      self.wallet_id,
            "action": {
                "type":        self.action.action_type.value,
                "amount":      str(self.action.amount),
                "currency":    self.action.currency,
                "merchant_id": self.action.merchant_id,
                "reversibility": self.action.reversibility.value,
                "idempotency_key": self.action.idempotency_key,
            },
            "verifications": [
                {
                    "verifier_id": v.verifier_id,
                    "decision":    v.decision,
                    "score":       v.score,
                    "flags":       v.flags,
                }
                for v in self.verifications
            ],
            "final_decision": self.final_decision,
            "timestamp":      self.timestamp,
            "outcome":        self.outcome,
            "signature":      self.signature,
            "signature_valid": self.is_valid(),
        }

    def summary(self) -> str:
        flags_str = ", ".join(self.verifications[0].flags[:2]) if self.verifications else ""
        return (
            f"CAR {self.car_id} | {self.action.action_type.value.upper():8s} "
            f"${self.action.amount:>8.2f} @ {self.action.merchant_name or self.action.merchant_id} | "
            f"{self.final_decision.upper():5s} score={self.verifications[0].score:.2f} "
            + (f"flags=[{flags_str}]" if flags_str else "")
        )
