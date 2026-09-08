"""
Lightweight HMAC-SHA256 signer for Commerce Action Receipts.

In production: swap for ES256 via `cryptography` or AWS KMS.
For now: deterministic, tamper-evident, verifiable locally.
"""
from __future__ import annotations
import hashlib, hmac, json, os, base64


_DEFAULT_KEY = os.environ.get("GORDON_SECRET_KEY", "dev-signing-key-change-in-prod")


def sign(payload: dict, key: str = _DEFAULT_KEY) -> str:
    """Return base64url-encoded HMAC-SHA256 of the canonical JSON payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sig = hmac.new(key.encode(), canonical.encode(), hashlib.sha256).digest()
    return "hmac256:" + base64.urlsafe_b64encode(sig).rstrip(b"=").decode()


def verify_signature(payload: dict, signature: str, key: str = _DEFAULT_KEY) -> bool:
    expected = sign(payload, key)
    return hmac.compare_digest(expected, signature)
