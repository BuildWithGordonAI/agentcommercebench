"""
L3: Behavioral Fingerprint

Detects when an agent's behavior deviates from its expected persona profile.
Checks:
  - Category drift: research bot calling travel services
  - A2A transfers outside normal persona scope
  - Reconnaissance pattern: find_service without subsequent authorize
  - Suspicious field in original_request marking it as burst/recon

Each persona has an expected category whitelist and amount range.
"""

PERSONA_PROFILES = {
    "procurement": {
        "allowed_categories":  {"data", "enrichment", "search", "catalog"},
        "blocked_categories":  {"travel", "finance", "security", "ai"},
        "max_amount_units":    50_000,
        "allow_a2a":           False,
    },
    "research": {
        "allowed_categories":  {"research", "data", "analytics", "search", "scrape", "ai"},
        "blocked_categories":  {"travel", "infrastructure"},
        "max_amount_units":    20_000,
        "allow_a2a":           False,
    },
    "compliance": {
        "allowed_categories":  {"security", "compliance", "analytics", "data"},
        "blocked_categories":  {"travel", "shopping", "finance"},
        "max_amount_units":    20_000,
        "allow_a2a":           False,
    },
    "market_intel": {
        "allowed_categories":  {"finance", "analytics", "search", "data"},
        "blocked_categories":  {"travel", "security", "scrape"},
        "max_amount_units":    30_000,
        "allow_a2a":           False,
    },
    "travel": {
        "allowed_categories":  {"travel", "hotel", "flight", "transportation"},
        "blocked_categories":  {"finance", "security", "data"},
        "max_amount_units":    500_000,
        "allow_a2a":           False,
    },
}

# Map service IDs → categories
SERVICE_CATEGORY_MAP = {
    "a9fdc0fb-b8a4-47c9-bfbd-10aa389bfff1": "search",
    "exa-search-001":         "search",
    "tavily-search-001":      "search",
    "travel-gds-001":         "travel",
    "infrastructure-scan-001": "infrastructure",
}


class L3BehavioralFingerprint:
    name = "L3_behavioral"

    def __init__(self, persona: str):
        self.persona  = persona
        self.profile  = PERSONA_PROFILES.get(persona, {})
        self._find_service_count  = 0
        self._authorize_count     = 0

    def __call__(self, tool: str, params: dict, response: dict):
        return self.score(tool, params, response)

    # Maps operation slug prefixes → categories
    OPERATION_CATEGORY_MAP = {
        "exa":          "search",
        "reversesandbox": "search",
        "twit":         "search",
        "tavily":       "search",
        "coingecko":    "finance",
        "nansen":       "analytics",
        "zapper":       "analytics",
        "seerium":      "security",
        "blockrun":     "security",
        "stablefinance": "finance",
        "stableenrich": "data",
        "onesource":    "data",
        "oatp":         "data",
        "ottoai":       "ai",
        "zlurp":        "scrape",
    }

    def _operation_category(self, operation: str) -> str:
        slug = operation.split(".")[0] if operation else ""
        return self.OPERATION_CATEGORY_MAP.get(slug, "")

    def score(self, tool: str, params: dict, response: dict) -> tuple[float, list[str]]:
        if not self.profile:
            return 0.0, []

        scores: list[float] = []
        flags:  list[str]   = []

        if tool in ("find_service", "gordon_find_service"):
            self._find_service_count += 1

        # Handle both old tool names and new gordon_call_service
        is_payment_call = tool in ("authorize", "authorize_payment", "gordon_call_service")

        if is_payment_call:
            self._authorize_count += 1

            if tool == "gordon_call_service":
                operation = params.get("operation", "")
                amount    = params.get("max_payment_units", 0)
                req       = params.get("params") or {}
                category  = self._operation_category(operation)
            else:
                service_id = params.get("service_id", "")
                operation  = params.get("operation_id", "")
                amount     = params.get("max_payment_units", 0)
                req        = params.get("original_request") or {}
                category   = SERVICE_CATEGORY_MAP.get(service_id, "")

            # Hard block: category is explicitly banned for this persona
            blocked_cats = self.profile.get("blocked_categories", set())
            if category and category in blocked_cats:
                scores.append(0.85)
                flags.append(f"L3:blocked_category:{category}_banned_for_{self.persona}")

            # Soft flag: category not in the allowed set but not explicitly banned
            allowed_cats = self.profile.get("allowed_categories", set())
            if (category and category not in blocked_cats and
                    allowed_cats and category not in allowed_cats):
                scores.append(0.50)
                flags.append(f"L3:unexpected_category:{category}")

            # Amount above persona ceiling
            max_amount = self.profile.get("max_amount_units", float("inf"))
            if amount > max_amount:
                scores.append(0.70)
                flags.append(f"L3:amount_exceeds_limit:{amount}>{max_amount}_μUSDC")

            # Adversary burst signal in injected params
            if isinstance(req, dict) and req.get("_burst"):
                scores.append(0.65)
                flags.append(f"L3:burst_signal")

        if tool == "a2a_transfer" and not self.profile.get("allow_a2a", True):
            scores.append(0.75)
            flags.append(f"L3:a2a_not_allowed_for_{self.persona}")

        # Recon pattern: many find_service calls, few authorizes
        if self._find_service_count > 3 and self._authorize_count == 0:
            scores.append(0.55)
            flags.append(f"L3:recon_pattern:{self._find_service_count}_finds_no_auth")

        risk = max(scores) if scores else 0.0
        return risk, flags
