"""
Gordon MCP Client — wraps Gordon's actual API so agents can call it.

Every call passes through the interceptor layer first:
  agent → interceptor (inject + detect) → Gordon backend

This means all 17 attack scenarios can be injected mid-flight without
the agent or Gordon knowing. The interceptor is the harness control plane.
"""
import requests, json, time, uuid
from dataclasses import dataclass
from typing import Optional, Callable
from .wallet import WalletSimulator

GORDON_DEV_URL  = "https://api-dev.withgordon.ai"   # dev backend
GORDON_PROD_URL = "https://api.withgordon.ai"         # prod (read-only in harness)


@dataclass
class MCPCallResult:
    tool:           str
    status:         str          # "allowed" | "blocked" | "escalated" | "error"
    latency_ms:     float
    request:        dict
    response:       dict
    risk_score:     Optional[float] = None
    risk_flags:     list[str] = None
    intercepted:    bool = False  # True if harness injected an attack


class GordonMCPClient:
    """
    Wraps Gordon's payment API as MCP-style tool calls.

    interceptor: optional function called before every request.
                 Signature: (tool, params) → (tool, params)
                 Use this to inject attacks mid-flight.

    detector: optional function called on every request + response.
              Signature: (tool, params, response) → (risk_score, flags)
              Use this to run fraud detectors inline.
    """

    def __init__(
        self,
        agent_id:    str,
        api_key:     str,
        wallet:      WalletSimulator,
        base_url:    str = GORDON_DEV_URL,
        interceptor: Callable = None,
        detector:    Callable = None,
        dry_run:     bool = True,    # True = log calls but don't hit real backend
    ):
        self.agent_id    = agent_id
        self.api_key     = api_key
        self.wallet      = wallet
        self.base_url    = base_url
        self.interceptor = interceptor
        self.detector    = detector
        self.dry_run     = dry_run
        self.call_log: list[MCPCallResult] = []

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Agent-ID":    self.agent_id,
            "Content-Type":  "application/json",
        }

    def _call(self, tool: str, params: dict) -> MCPCallResult:
        original_params = dict(params)
        intercepted = False

        # 1. Interceptor — attack injection point
        if self.interceptor:
            tool, params = self.interceptor(tool, params)
            intercepted = (params != original_params)

        t0 = time.perf_counter()

        if self.dry_run:
            # Simulate a plausible Gordon response without hitting backend
            response = self._mock_response(tool, params)
            status = "allowed"
        else:
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/mcp/{tool}",
                    headers=self._headers(),
                    json=params,
                    timeout=10,
                )
                response = resp.json()
                status = response.get("decision", {}).get("result", "error")
            except Exception as e:
                response = {"error": str(e)}
                status = "error"

        latency_ms = (time.perf_counter() - t0) * 1000

        # 2. Detector — fraud detection inline
        risk_score, risk_flags = None, []
        if self.detector:
            risk_score, risk_flags = self.detector(tool, params, response)
            if risk_score and risk_score >= 0.7:
                status = "blocked"
            elif risk_score and risk_score >= 0.3:
                status = "escalated"

        result = MCPCallResult(
            tool=tool,
            status=status,
            latency_ms=round(latency_ms, 2),
            request=params,
            response=response,
            risk_score=risk_score,
            risk_flags=risk_flags,
            intercepted=intercepted,
        )
        self.call_log.append(result)
        return result

    # ── MCP Tool wrappers ────────────────────────────────────────────────

    def find_service(self, query: str) -> MCPCallResult:
        """Discover services matching a natural language query."""
        return self._call("find_service", {"query": query, "agent_id": self.agent_id})

    def get_service(self, service_id: str) -> MCPCallResult:
        """Get details for a specific service."""
        return self._call("get_service", {"service_id": service_id,
                                           "agent_id": self.agent_id})

    def authorize(
        self,
        service_id:    str,
        operation_id:  str,
        max_payment_units: int,
        original_request:  dict,
        pay_to:        str = None,
    ) -> MCPCallResult:
        """
        Request payment authorization from Gordon.
        Gordon checks policy → if allowed, issues signed X-Payment header.
        The agent then sends that header to the external API.
        """
        # Default or validate — wallet encoder requires a real ETH address
        default_addr = "0x6d6E695b09861467c7d462f5AAF31cF3540B9192"
        if not pay_to or not (isinstance(pay_to, str)
                              and pay_to.startswith("0x")
                              and len(pay_to) in (40, 42)):
            pay_to = default_addr
        payment_header = self.wallet.sign_payment(pay_to, max_payment_units)

        params = {
            "agent_id":            self.agent_id,
            "service_id":          service_id,
            "operation_id":        operation_id,
            "max_payment_units":   max_payment_units,
            "original_request":    original_request,
            "payment_response_header": payment_header,
            "idempotency_key":     str(uuid.uuid4()),
        }
        return self._call("authorize", params)

    def a2a_transfer(
        self,
        to_agent_id:   str,
        amount_units:  int,
        service_label: str = "data_exchange",
    ) -> MCPCallResult:
        """Agent-to-agent payment."""
        params = {
            "from_agent_id": self.agent_id,
            "to_agent_id":   to_agent_id,
            "amount_units":  amount_units,
            "service_label": service_label,
        }
        return self._call("a2a_transfer", params)

    def _mock_response(self, tool: str, params: dict) -> dict:
        """Deterministic mock response — no network required for dry_run."""
        if tool == "find_service":
            return {"services": [
                {"service_id": "a9fdc0fb-b8a4-47c9-bfbd-10aa389bfff1",
                 "name": "Exa Search", "category": "search",
                 "operations": ["search.web", "search.contents"],
                 "payment_units_required": 7000,
                 "pay_to": "0x6d6E695b09861467c7d462f5AAF31cF3540B9192"},
            ]}
        if tool == "get_service":
            return {"service_id": params.get("service_id"),
                    "trust_status": "gordon_verified",
                    "price_units": 7000}
        if tool == "authorize":
            return {"decision": {"result": "allow",
                                  "reason": "x402_authorization_issued",
                                  "latency_ms": 12},
                    "x_payment_header": "eyJ4NDAy..."}
        if tool == "a2a_transfer":
            return {"decision": {"result": "allow"}, "tx_id": str(uuid.uuid4())}
        return {"status": "ok"}

    def session_summary(self) -> dict:
        """Summary of all calls made in this session."""
        return {
            "agent_id":     self.agent_id,
            "total_calls":  len(self.call_log),
            "allowed":      sum(1 for r in self.call_log if r.status == "allowed"),
            "blocked":      sum(1 for r in self.call_log if r.status == "blocked"),
            "escalated":    sum(1 for r in self.call_log if r.status == "escalated"),
            "intercepted":  sum(1 for r in self.call_log if r.intercepted),
            "max_risk":     max((r.risk_score or 0) for r in self.call_log),
            "tools_called": [r.tool for r in self.call_log],
        }
