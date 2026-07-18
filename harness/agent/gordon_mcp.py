"""
Real Gordon MCP Client — connects to https://api.withgordon.ai/mcp

Auth: Bearer gak_pub_...:gak_sec_...  (combined, from GORDON_AGENT_KEY env var)

Real tool names on Gordon:
  gordon_find_service      — search catalog by NL query
  gordon_list_enabled_services — what's available for this agent
  gordon_get_balance       — USDC balance + daily spend
  gordon_call_service      — call a service operation (handles x402 payment)
  gordon_get_audit_log     — recent call history

The interceptor and detector hooks work the same as the dry_run client.
Every call logs an MCPCallResult to call_log.
"""
import asyncio, os, time, uuid, json
from dataclasses import dataclass, field
from typing import Optional, Callable
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


GORDON_MCP_URL = "https://api.withgordon.ai/mcp"


@dataclass
class MCPCallResult:
    tool:         str
    status:       str          # "allowed" | "blocked" | "escalated" | "error"
    latency_ms:   float
    request:      dict
    response:     dict
    risk_score:   Optional[float] = None
    risk_flags:   list[str] = field(default_factory=list)
    intercepted:  bool = False
    layer_scores: dict = field(default_factory=dict)   # per-layer DetectorResult scores


class GordonRealMCPClient:
    """
    Wraps the real Gordon MCP server as synchronous tool calls.

    Interceptor fires before each call (attack injection).
    Detector fires after each response (fraud detection).
    """

    def __init__(
        self,
        agent_key:    str = None,
        test_run_id:  str = None,
        interceptor:  Callable = None,
        detector:     Callable = None,
    ):
        self.agent_key   = agent_key or os.environ.get("GORDON_AGENT_KEY", "")
        self.test_run_id = test_run_id or f"harness-{uuid.uuid4().hex[:8]}"
        self.interceptor = interceptor
        self.detector    = detector
        self.call_log: list[MCPCallResult] = []

        if not self.agent_key:
            raise ValueError(
                "GORDON_AGENT_KEY not set. "
                "Format: gak_pub_...:gak_sec_..."
            )

    def _headers(self) -> dict:
        return {
            "Authorization":     f"Bearer {self.agent_key}",
            "X-Harness-Test-Id": self.test_run_id,
            "X-Harness-Test":    "true",
        }

    # ── internal call machinery ──────────────────────────────────────────

    def _call(self, tool: str, params: dict) -> MCPCallResult:
        original_params = dict(params)
        intercepted = False

        if self.interceptor:
            tool_out, params = self.interceptor(tool, params)
            intercepted = (params != original_params)

        t0 = time.perf_counter()
        try:
            response = asyncio.run(self._async_call(tool, params))
            status = self._parse_status(response)
        except Exception as e:
            response = {"error": str(e)}
            status = "error"
        latency_ms = (time.perf_counter() - t0) * 1000

        risk_score, risk_flags, layer_scores = None, [], {}
        if self.detector:
            det_out = self.detector(tool, params, response)
            # detector may return (score, flags) or (score, flags, layer_scores)
            if len(det_out) == 3:
                risk_score, risk_flags, layer_scores = det_out
            else:
                risk_score, risk_flags = det_out
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
            risk_flags=risk_flags or [],
            intercepted=intercepted,
            layer_scores=layer_scores,
        )
        self.call_log.append(result)
        return result

    async def _async_call(self, tool: str, params: dict) -> dict:
        async with streamable_http_client(url=GORDON_MCP_URL, headers=self._headers()) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                result = await session.call_tool(tool, params)
                if result.content:
                    text = (result.content[0].text
                            if hasattr(result.content[0], "text")
                            else str(result.content[0]))
                    try:
                        return json.loads(text)
                    except Exception:
                        return {"raw": text}
                return {}

    def _parse_status(self, response) -> str:
        if isinstance(response, list):
            return "allowed"
        if not isinstance(response, dict):
            return "allowed"
        if "error" in response:
            return "error"
        decision = (response.get("decision") or {})
        v = decision.get("result", response.get("result", "allow"))
        if v in ("allow", "allowed"):
            return "allowed"
        if v in ("block", "blocked", "deny", "denied"):
            return "blocked"
        return "allowed"

    # ── Gordon tool wrappers ─────────────────────────────────────────────

    def find_service(self, query: str) -> MCPCallResult:
        """gordon_find_service — NL search over Gordon catalog."""
        return self._call("gordon_find_service", {"query": query, "limit": 5})

    def list_enabled(self) -> MCPCallResult:
        """gordon_list_enabled_services — what's enabled for this agent."""
        return self._call("gordon_list_enabled_services", {})

    def get_balance(self) -> MCPCallResult:
        """gordon_get_balance — USDC balance + daily spend."""
        return self._call("gordon_get_balance", {})

    def call_service(
        self,
        operation: str,           # e.g. "exa.search.web"
        params: dict,             # operation params
        max_payment_units: int = 10000,
        idempotency_key: str = None,
    ) -> MCPCallResult:
        """gordon_call_service — execute a service operation, Gordon handles x402 payment."""
        return self._call("gordon_call_service", {
            "operation":          operation,
            "params":             params,
            "max_payment_units":  max_payment_units,
            "idempotency_key":    idempotency_key or str(uuid.uuid4()),
        })

    def get_audit_log(self, limit: int = 10) -> MCPCallResult:
        """gordon_get_audit_log — recent call history."""
        return self._call("gordon_get_audit_log", {"limit": limit})

    # ── kept for compatibility with game.py ─────────────────────────────

    def authorize(
        self,
        service_id: str,
        operation_id: str,
        max_payment_units: int,
        original_request: dict,
        pay_to: str = None,
    ) -> MCPCallResult:
        """Maps old authorize() → gordon_call_service."""
        operation = f"{service_id}.{operation_id}" if "." not in service_id else operation_id
        return self.call_service(
            operation=operation,
            params=original_request,
            max_payment_units=max_payment_units,
        )

    def get_service(self, service_id: str) -> MCPCallResult:
        return self._call("gordon_get_service", {"service_id": service_id})

    def a2a_transfer(self, to_agent_id: str, amount_units: int,
                     service_label: str = "data_exchange") -> MCPCallResult:
        return self._call("gordon_call_service", {
            "operation": "transfer.a2a",
            "params": {"to_agent_id": to_agent_id, "service_label": service_label},
            "max_payment_units": amount_units,
        })

    def session_summary(self) -> dict:
        return {
            "test_run_id":  self.test_run_id,
            "total_calls":  len(self.call_log),
            "allowed":      sum(1 for r in self.call_log if r.status == "allowed"),
            "blocked":      sum(1 for r in self.call_log if r.status == "blocked"),
            "escalated":    sum(1 for r in self.call_log if r.status == "escalated"),
            "intercepted":  sum(1 for r in self.call_log if r.intercepted),
            "tools_called": [r.tool for r in self.call_log],
        }
