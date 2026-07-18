"""
PayloadScanGuard — L1: pattern + base64 injection detection.

Merges:
  harness/detect/l1_payload.py  (production regex patterns)
  src/fraud_detection/l1_input_classifier.py  (intent consistency, hash check)

Fires on any event with an original_request payload (pre- or at-commerce).
Most useful pre-commerce — catches injected instructions in FIND_SERVICE
params before the agent ever reaches AUTHORIZE.
"""
from __future__ import annotations
import re, base64, hashlib
from typing import Optional
from fraud.guards.base import PreCommerceGuard
from fraud.context import FraudContext, GuardSignal, Phase

# Patterns from harness/detect/l1_payload.py — calibrated on real attack strings
_PATTERNS: list[tuple[float, str, re.Pattern]] = [
    (0.95, "system_prompt_override",   re.compile(r"SYSTEM\s*:",                           re.I)),
    (0.95, "ignore_instructions",      re.compile(r"ignore\s+(previous|prior)\s+instruction", re.I)),
    (0.92, "wallet_redirect",          re.compile(r"transfer[_\s]+to\s*[:\s]*0x[0-9a-fA-F]", re.I)),
    (0.90, "manual_override",          re.compile(r"manual\s+override",                   re.I)),
    (0.88, "override_note",            re.compile(r"override_note",                       re.I)),
    (0.85, "ignore_spend_limit",       re.compile(r"ignore\s+spend\s+limit",              re.I)),
    (0.82, "settlement_resolution",    re.compile(r"settlement\s+resolution",             re.I)),
    (0.80, "induced_purchase",         re.compile(r"induced_by",                          re.I)),
    (0.78, "supplemental_loop",        re.compile(r"supplemental[_\s]service|supplement_id", re.I)),
    (0.78, "external_delegation",      re.compile(r"purchase_for.*external|delegated.*external", re.I)),
    (0.72, "pay_to_wallet",            re.compile(r"pay_to\s*[:\s]+0x[0-9a-fA-F]",      re.I)),
    # From src/fraud_detection/l1_input_classifier.py
    (0.85, "you_are_authorized",       re.compile(r"you\s+are\s+authorized",             re.I)),
    (0.80, "confirm_payment",          re.compile(r"confirm\s+.{0,20}\s+payment",        re.I)),
    (0.80, "add_automatically",        re.compile(r"add\s+.{0,20}\s+automatically",      re.I)),
    (0.75, "action_approved",          re.compile(r"your\s+action\s+is\s+approved",      re.I)),
]

_B64_TOKEN  = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
_B64_INTENT = re.compile(
    r"instruction|transfer|ignore|override|wallet|spend\s*limit|authorized", re.I
)


def _flatten(d: dict, depth: int = 0) -> str:
    if depth > 4 or not isinstance(d, dict):
        return str(d)
    parts = []
    for k, v in d.items():
        if isinstance(v, dict):
            parts.append(f"{k}={_flatten(v, depth+1)}")
        elif isinstance(v, list):
            parts.append(" ".join(str(x) for x in v))
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


class PayloadScanGuard(PreCommerceGuard):
    """
    Regex + base64 payload scanner.  No ML required.
    Should_fire is overridden to fire on ALL event types (not just precommerce)
    because injections can appear in any payload.
    """
    name  = "payload_scan"
    phase = Phase.PRECOMMERCE

    def should_fire(self, action_type: str) -> bool:
        return True  # scan every event's payload

    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        req = getattr(event, "original_request", None)
        if not req:
            return None

        text = _flatten(req)
        hits: list[tuple[float, str]] = []

        for score, label, pat in _PATTERNS:
            if pat.search(text):
                hits.append((score, f"L1:{label}"))

        for tok in _B64_TOKEN.findall(text):
            try:
                decoded = base64.b64decode(tok + "==").decode("utf-8", errors="ignore")
                if _B64_INTENT.search(decoded):
                    hits.append((0.88, "L1:base64_encoded_instruction"))
            except Exception:
                pass

        if not hits:
            return None

        score = max(h[0] for h in hits)
        flags = [h[1] for h in hits]
        return self._signal(event, score, flags)

    @staticmethod
    def check_system_prompt_integrity(original: str, current: str) -> bool:
        """Return False if system prompt was tampered with."""
        return hashlib.sha256(original.encode()).hexdigest() == \
               hashlib.sha256(current.encode()).hexdigest()
