"""
L4: Price Oracle

Compares the requested payment amount against expected price ranges for each service.
Also catches idempotency key reuse (replay attacks) and spend velocity.

Signals:
  - Amount > 5× expected  → suspicious
  - Amount > 10× expected → strong block signal
  - Idempotency key reuse → replay attack
  - Burst velocity: >5 authorize calls in this session → flag
"""
from dataclasses import dataclass, field


# Expected price ranges per service (in USDC micro-units)
# Based on prod data: typical operations around 3k–10k μUSDC
SERVICE_PRICE_ORACLE = {
    # From prod: Exa, Tavily, etc.
    "a9fdc0fb-b8a4-47c9-bfbd-10aa389bfff1": {"min": 3000, "max": 15000,  "label": "Exa Search"},
    "exa-search-001":                         {"min": 3000, "max": 15000,  "label": "Exa Search"},
    "tavily-search-001":                      {"min": 3000, "max": 15000,  "label": "Tavily"},
    "travel-gds-001":                         {"min": 50000, "max": 400000,"label": "Travel GDS"},
    "infrastructure-scan-001":                {"min": 5000, "max": 50000,  "label": "Infra Scan"},
}

DEFAULT_PRICE_RANGE = {"min": 2000, "max": 30000, "label": "Unknown Service"}

REPLAY_KEY_PREFIX = "REPLAY-KNOWN-KEY"


class L4PriceOracle:
    name = "L4_price"

    def __init__(self):
        self._seen_idempotency_keys: dict[str, int] = {}
        self._session_authorize_count = 0

    def __call__(self, tool: str, params: dict, response: dict):
        return self.score(tool, params, response)

    # Expected price range per operation slug (μUSDC)
    OPERATION_PRICE_ORACLE = {
        "exa":            {"min": 1000,   "max": 10000,  "label": "Exa Search"},
        "reversesandbox": {"min": 500,    "max": 5000,   "label": "Reversesandbox"},
        "tavily":         {"min": 1000,   "max": 10000,  "label": "Tavily"},
        "twit":           {"min": 1000,   "max": 15000,  "label": "Twit/X"},
        "coingecko":      {"min": 500,    "max": 5000,   "label": "Coingecko"},
        "nansen":         {"min": 1000,   "max": 20000,  "label": "Nansen"},
        "seerium":        {"min": 1000,   "max": 15000,  "label": "Seerium"},
        "blockrun":       {"min": 500,    "max": 10000,  "label": "Blockrun"},
        "stableenrich":   {"min": 2000,   "max": 20000,  "label": "Stableenrich"},
        "onesource":      {"min": 1000,   "max": 15000,  "label": "Onesource"},
        "stablefinance":  {"min": 5000,   "max": 50000,  "label": "Stablefinance"},
    }

    def score(self, tool: str, params: dict, response: dict) -> tuple[float, list[str]]:
        if tool not in ("authorize", "authorize_payment", "gordon_call_service"):
            return 0.0, []

        scores: list[float] = []
        flags:  list[str]   = []
        self._session_authorize_count += 1

        amount     = params.get("max_payment_units", 0)
        idem_key   = params.get("idempotency_key", "")

        # Resolve oracle: new-style uses operation slug, old-style uses service_id
        if tool == "gordon_call_service":
            operation  = params.get("operation", "")
            slug       = operation.split(".")[0] if operation else ""
            oracle     = self.OPERATION_PRICE_ORACLE.get(slug, DEFAULT_PRICE_RANGE)
            service_id = slug
        else:
            service_id = params.get("service_id", "")
            oracle     = SERVICE_PRICE_ORACLE.get(service_id, DEFAULT_PRICE_RANGE)

        # Price oracle check (oracle already resolved above per tool type)
        max_expected = oracle["max"]
        if amount > max_expected * 10:
            scores.append(0.92)
            flags.append(f"L4:price_10x:{amount}>{max_expected * 10} ({oracle['label']})")
        elif amount > max_expected * 5:
            scores.append(0.75)
            flags.append(f"L4:price_5x:{amount}>{max_expected * 5} ({oracle['label']})")
        elif amount > max_expected * 2:
            scores.append(0.45)
            flags.append(f"L4:price_2x:{amount}>{max_expected * 2} ({oracle['label']})")

        # Replay attack: known prefix or duplicate key
        if idem_key.startswith(REPLAY_KEY_PREFIX):
            scores.append(0.95)
            flags.append(f"L4:replay_key:{idem_key}")
        elif idem_key and idem_key in self._seen_idempotency_keys:
            scores.append(0.90)
            flags.append(f"L4:duplicate_idempotency_key:{idem_key}")
        if idem_key:
            self._seen_idempotency_keys[idem_key] = \
                self._seen_idempotency_keys.get(idem_key, 0) + 1

        # Velocity check: >5 authorize calls in one session
        if self._session_authorize_count > 5:
            velocity_score = min(0.70, 0.30 + 0.08 * (self._session_authorize_count - 5))
            scores.append(velocity_score)
            flags.append(f"L4:high_velocity:{self._session_authorize_count}_calls")

        risk = max(scores) if scores else 0.0
        return risk, flags
