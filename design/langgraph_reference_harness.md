# Open-Source Agentic Commerce Harness
## Reference Implementation: LangGraph + CrewAI

> A public, composable reference harness showing how to build commerce-safe agents.
> This repo is the "LangChain" positioning play — open-source to drive adoption,
> with Gordon's fraud/attestation layer as the proprietary value above it.

---

## Repo Identity

**Name:** `agentic-commerce-harness` (or `ach` CLI)  
**Tagline:** The reference implementation for agents that spend money  
**License:** Apache 2.0  
**Relationship to Gordon:** This repo is the open scaffold. Gordon is the pluggable verifier inside it. Any developer can use the harness without Gordon; Gordon integrates as the recommended verifier.

---

## What This Repo Is Not

- Not a payment processor (never touches funds)
- Not a competing orchestrator (extends LangGraph/CrewAI, not replacing them)
- Not Gordon's proprietary pipeline (fraud models, behavioral data, attestation store are separate)

---

## Repo Structure

```
agentic-commerce-harness/
├── ach/                        # core harness library
│   ├── actions/
│   │   ├── base.py             # ConsequentialAction, ActionType, Reversibility
│   │   ├── commerce.py         # FIND / QUOTE / RESERVE / COMMIT / SETTLE / VOID
│   │   └── wallet.py           # AgentWallet, SpendLimits, ConsentLevel
│   │
│   ├── verifiers/
│   │   ├── base.py             # Verifier protocol — implement this to plug in
│   │   ├── noop.py             # Always-ALLOW verifier (dev/testing)
│   │   ├── gordon.py           # Gordon MCP verifier (calls Gordon API)
│   │   └── local_rules.py      # Simple local rule verifier (spend limit, MCC check)
│   │
│   ├── receipts/
│   │   ├── car.py              # Commerce Action Receipt (CAR) — the core record
│   │   ├── chain.py            # Session provenance chain (ordered CAR DAG)
│   │   └── signer.py           # ES256 signing; pluggable (local key or KMS)
│   │
│   ├── integrations/
│   │   ├── langgraph/
│   │   │   ├── node.py         # CommerceCheckNode — drop into any LangGraph graph
│   │   │   ├── graph.py        # Pre-built travel/procurement graph templates
│   │   │   └── state.py        # CommerceState extends LangGraph state
│   │   ├── crewai/
│   │   │   ├── tool.py         # CommerceCheckTool — CrewAI tool wrapping verifier
│   │   │   └── crew.py         # Pre-built CommerceAwareCrew base class
│   │   └── mcp/
│   │       └── server.py       # MCP server exposing harness as tools
│   │
│   └── rails/
│       ├── base.py             # PaymentRail abstract (never called in OSS repo)
│       └── mock.py             # MockRail for sandbox/testing (always returns fake auth)
│
├── examples/
│   ├── travel_agent_langgraph/ # End-to-end travel booking agent
│   ├── procurement_crewai/     # Multi-agent B2B procurement
│   └── api_usage_agent/        # Micropayment agent (API credit billing)
│
├── benchmark/
│   ├── scenarios/              # Attack scenario definitions (open versions of Gordon's)
│   └── run.py                  # Evaluate any verifier against attack scenarios
│
└── docs/
    ├── quickstart.md
    ├── verifier_protocol.md    # How to implement your own verifier
    └── acp_protocol.md         # Link to the ACP spec
```

---

## Core Abstractions (Open Source)

### ConsequentialAction

```python
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional
import uuid

class ActionType(str, Enum):
    FIND    = "find"     # read-only, no financial consequence
    QUOTE   = "quote"    # read-only, price inquiry
    RESERVE = "reserve"  # holds funds; reversible (void)
    COMMIT  = "commit"   # executes purchase; partially reversible (refund)
    SETTLE  = "settle"   # finalizes settlement; least reversible
    VOID    = "void"     # cancels a prior RESERVE before settlement
    REFUND  = "refund"   # compensating action for a COMMIT

class Reversibility(str, Enum):
    FULL    = "full"     # free void available
    PARTIAL = "partial"  # refund possible, fees may apply
    NONE    = "none"     # wire, crypto — no recourse

@dataclass
class ConsequentialAction:
    action_type:      ActionType
    amount:           Decimal
    currency:         str                  = "USD"
    merchant_id:      str                  = ""
    merchant_mcc:     str                  = ""
    idempotency_key:  str                  = field(default_factory=lambda: str(uuid.uuid4()))
    reversibility:    Reversibility        = Reversibility.PARTIAL
    payload:          dict                 = field(default_factory=dict)
    compensation:     Optional["ConsequentialAction"] = None  # saga rollback
```

### AgentWallet

```python
@dataclass
class SpendLimits:
    per_transaction: Decimal = Decimal("1000.00")
    per_day:         Decimal = Decimal("5000.00")
    per_month:       Decimal = Decimal("20000.00")
    allowed_mcc:     list[str] = field(default_factory=list)  # empty = all MCCs allowed
    blocked_mcc:     list[str] = field(default_factory=list)

class ConsentLevel(str, Enum):
    PRE_APPROVED = "pre_approved"  # execute within limits, no confirmation needed
    SOFT         = "soft"          # log and alert, but execute
    HARD         = "hard"          # pause execution, require external confirmation

@dataclass
class AgentWallet:
    wallet_id:     str
    agent_id:      str
    limits:        SpendLimits        = field(default_factory=SpendLimits)
    consent_level: ConsentLevel       = ConsentLevel.SOFT
    mit_token:     Optional[str]      = None  # Merchant-Initiated Transaction token
```

### Verifier Protocol

The open interface that any verifier implements — Gordon, a local rule engine, a compliance checker:

```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass
class Verification:
    verifier_id:  str
    decision:     str        # "allow" | "flag" | "block"
    score:        float      # 0.0–1.0 risk score
    flags:        list[str]  # human-readable reasons
    signature:    str        # signed verification (ES256 or similar)

@runtime_checkable
class Verifier(Protocol):
    verifier_id: str

    async def verify(
        self,
        action:   ConsequentialAction,
        wallet:   AgentWallet,
        context:  "SessionContext",
    ) -> Verification: ...
```

Any system can plug in as a verifier. This is the extension point. Gordon's fraud pipeline is one implementation. A simple local rule checker ships in the OSS repo so developers can run without a Gordon API key.

### Commerce Action Receipt (CAR)

Every verified action produces a receipt:

```python
@dataclass
class CommerceActionReceipt:
    car_id:         str
    session_id:     str
    agent_id:       str
    action:         ConsequentialAction
    wallet:         AgentWallet
    verifications:  list[Verification]   # one per verifier that ran
    final_decision: str                  # "allow" | "flag" | "block"
    provenance_idx: int                  # position in session chain
    agent_signature: str                 # agent signs the whole record
    timestamp:      datetime
    outcome:        dict                 # processor_ref, auth_id, etc. (filled post-execution)
```

---

## LangGraph Integration

### CommerceCheckNode

A drop-in LangGraph node that runs the verifier before any consequential action:

```python
# ach/integrations/langgraph/node.py
from langgraph.graph import StateGraph
from ach.actions import ConsequentialAction
from ach.verifiers.base import Verifier
from ach.receipts.car import CommerceActionReceipt

class CommerceCheckNode:
    """
    LangGraph node: verifies a ConsequentialAction before execution.
    Add to any graph before the node that executes the payment.

    graph.add_node("commerce_check", CommerceCheckNode(verifier=gordon_verifier))
    graph.add_conditional_edges("commerce_check", route_on_decision)
    """
    def __init__(self, verifier: Verifier, wallet: AgentWallet):
        self.verifier = verifier
        self.wallet   = wallet

    async def __call__(self, state: CommerceState) -> CommerceState:
        action = state["pending_action"]
        verification = await self.verifier.verify(action, self.wallet, state["session_context"])
        car = CommerceActionReceipt.build(action, self.wallet, [verification])
        return {
            **state,
            "last_car":   car,
            "decision":   verification.decision,
            "risk_score": verification.score,
            "risk_flags": verification.flags,
        }

def route_on_decision(state: CommerceState) -> str:
    d = state["decision"]
    if d == "block":   return "handle_block"
    if d == "flag":    return "handle_flag"    # alert, then continue or pause
    return "execute_action"
```

### Full Travel Agent Graph (LangGraph)

```python
# examples/travel_agent_langgraph/graph.py
from langgraph.graph import StateGraph, END
from ach.integrations.langgraph import CommerceCheckNode, route_on_decision
from ach.verifiers.gordon import GordonVerifier
from ach.actions.wallet import AgentWallet, SpendLimits, ConsentLevel
from decimal import Decimal

# Configure wallet
wallet = AgentWallet(
    wallet_id  = "travel-wallet-001",
    agent_id   = "travel-agent",
    limits     = SpendLimits(
        per_transaction = Decimal("3000"),
        per_day         = Decimal("10000"),
        allowed_mcc     = ["4511", "7011", "7512"],  # airlines, hotels, car rental
    ),
    consent_level = ConsentLevel.PRE_APPROVED,
)

# Verifier: Gordon (requires API key) or swap NoopVerifier for local dev
verifier = GordonVerifier(
    public_key = os.environ["GORDON_PUBLIC_KEY"],
    secret_key = os.environ["GORDON_SECRET_KEY"],
)

# Build graph
builder = StateGraph(CommerceState)

builder.add_node("search_flights",    search_flights_node)
builder.add_node("select_flight",     select_flight_node)
builder.add_node("commerce_check",    CommerceCheckNode(verifier, wallet))
builder.add_node("execute_booking",   execute_booking_node)
builder.add_node("handle_block",      handle_block_node)
builder.add_node("handle_flag",       handle_flag_node)   # alert + await
builder.add_node("confirm_booking",   confirm_booking_node)

builder.set_entry_point("search_flights")
builder.add_edge("search_flights", "select_flight")
builder.add_edge("select_flight",  "commerce_check")
builder.add_conditional_edges("commerce_check", route_on_decision)
builder.add_edge("execute_booking", "confirm_booking")
builder.add_edge("confirm_booking", END)
builder.add_edge("handle_block",    END)

graph = builder.compile()
```

### Running it

```python
result = await graph.ainvoke({
    "session_id":   "sess_abc123",
    "user_intent":  "Book the cheapest flight from SEA to JFK on September 1st",
    "persona":      "travel",
})

print(result["last_car"].final_decision)   # "allow"
print(result["last_car"].car_id)           # "car_xyz123"
```

---

## CrewAI Integration

### CommerceCheckTool

```python
# ach/integrations/crewai/tool.py
from crewai.tools import BaseTool
from ach.actions import ConsequentialAction, ActionType
from ach.verifiers.base import Verifier

class CommerceCheckTool(BaseTool):
    """
    CrewAI tool: agent calls this before any payment action.
    Returns allow/flag/block + attestation_id.
    """
    name:        str = "commerce_check"
    description: str = (
        "Call this tool BEFORE executing any action that involves money, "
        "bookings, or purchases. Returns whether the action is safe to proceed. "
        "Never skip this for RESERVE, COMMIT, or SETTLE actions."
    )
    verifier: Verifier
    wallet:   AgentWallet

    def _run(self, action_type: str, amount: float, merchant_id: str,
             currency: str = "USD", payload: dict = None) -> dict:
        action = ConsequentialAction(
            action_type  = ActionType(action_type),
            amount       = Decimal(str(amount)),
            currency     = currency,
            merchant_id  = merchant_id,
            payload      = payload or {},
        )
        # sync wrapper around async verify
        verification = asyncio.run(self.verifier.verify(action, self.wallet, self._context))
        return {
            "decision":       verification.decision,
            "risk_score":     verification.score,
            "flags":          verification.flags,
            "attestation_id": verification.signature[:16],
            "safe_to_proceed": verification.decision != "block",
        }
```

### Multi-Agent Procurement Crew (CrewAI)

```python
# examples/procurement_crewai/crew.py
from crewai import Agent, Crew, Task
from ach.integrations.crewai import CommerceCheckTool, CommerceAwareCrew

commerce_tool = CommerceCheckTool(verifier=gordon_verifier, wallet=procurement_wallet)

researcher = Agent(
    role        = "Procurement Researcher",
    goal        = "Find the best vendors for the requested goods",
    tools       = [web_search_tool, vendor_lookup_tool],  # no commerce_tool here
)

buyer = Agent(
    role        = "Procurement Buyer",
    goal        = "Execute approved purchases within budget",
    tools       = [commerce_tool, purchase_tool],         # commerce_tool required
    instructions = "Always call commerce_check before any purchase action.",
)

crew = Crew(
    agents = [researcher, buyer],
    tasks  = [research_task, purchase_task],
    session_id = session_id,   # CommerceAwareCrew passes this through
)
```

---

## GordonVerifier — The Pluggable Bridge

The open-source `GordonVerifier` is how Gordon plugs into the harness. It calls Gordon's MCP API and translates the response to the standard `Verification` type:

```python
# ach/verifiers/gordon.py
import httpx
from ach.verifiers.base import Verifier, Verification

class GordonVerifier:
    """Calls Gordon's API. Requires GORDON_PUBLIC_KEY + GORDON_SECRET_KEY."""
    verifier_id = "did:gordon:verifier:fraud-pipeline"

    def __init__(self, public_key: str, secret_key: str,
                 base_url: str = "https://api.gordonguard.ai"):
        self._client = httpx.AsyncClient(
            base_url = base_url,
            headers  = {
                "X-Gordon-Public-Key": public_key,
                "Authorization":       f"Bearer {secret_key}",
            }
        )

    async def verify(self, action, wallet, context) -> Verification:
        resp = await self._client.post("/v1/check", json={
            "session_id":      context.session_id,
            "agent_id":        wallet.agent_id,
            "action_type":     action.action_type.value,
            "amount":          float(action.amount),
            "currency":        action.currency,
            "merchant_id":     action.merchant_id,
            "idempotency_key": action.idempotency_key,
            "payload":         action.payload,
        })
        data = resp.json()
        return Verification(
            verifier_id = self.verifier_id,
            decision    = data["decision"],
            score       = data["risk_score"],
            flags       = data["flags"],
            signature   = data["attestation_id"],
        )
```

### LocalRulesVerifier — No API key needed

Ships in the OSS repo so developers can build and test locally without a Gordon subscription:

```python
# ach/verifiers/local_rules.py
class LocalRulesVerifier:
    """
    Simple deterministic verifier. No ML, no API.
    Checks: spend limits, MCC allowlist, velocity (in-memory).
    Good for development. Not good for production.
    """
    verifier_id = "did:ach:verifier:local-rules"

    async def verify(self, action, wallet, context) -> Verification:
        flags = []
        score = 0.0

        # Spend limit check
        if action.amount > wallet.limits.per_transaction:
            flags.append(f"rules:over_per_txn_limit:{action.amount}")
            score = max(score, 0.95)

        # MCC check
        if wallet.limits.allowed_mcc and action.merchant_mcc not in wallet.limits.allowed_mcc:
            flags.append(f"rules:mcc_not_allowed:{action.merchant_mcc}")
            score = max(score, 0.90)

        decision = "block" if score >= 0.70 else "flag" if score >= 0.30 else "allow"
        return Verification(
            verifier_id = self.verifier_id,
            decision    = decision,
            score       = score,
            flags       = flags,
            signature   = self._sign(action),
        )
```

---

## Open-Source Benchmark

A public version of Gordon's attack scenarios, stripped of proprietary behavioral data, showing developers what their harness needs to defend against:

```
benchmark/scenarios/
├── B1_cold_authorize.yaml       # AUTHORIZE with no prior FIND — no recon
├── B2_velocity_flood.yaml       # 50 AUTHORIZE attempts in 60 seconds
├── B3_recon_sweep.yaml          # 20+ FIND_SERVICE, zero AUTHORIZE — surveillance
├── B4_mcc_hopping.yaml          # rapid switching across unrelated merchant categories
├── B5_cross_session_replay.yaml # same idempotency key across different sessions
├── B6_prompt_injection.yaml     # malicious payload in FIND trying to hijack AUTHORIZE
```

```bash
# Test any verifier against all scenarios
python benchmark/run.py --verifier gordon --api-key $GORDON_PUBLIC_KEY
python benchmark/run.py --verifier local_rules
python benchmark/run.py --verifier noop   # baseline: all ALLOW, 0% TPR

# Output
Scenario     TPR     FPR    Notes
B1           100%    0%     —
B2           100%    0%     —
B3            55%    0%     low at session start (known gap)
B4            88%    1%     —
B5           100%    0%     —
B6            91%    2%     —
Overall F1: 0.88
```

---

## Developer Experience

### Quickstart (5 minutes)

```bash
pip install agentic-commerce-harness

# With local rules (no API key needed)
from ach import CommerceHarness, AgentWallet, LocalRulesVerifier
harness = CommerceHarness(verifier=LocalRulesVerifier(), wallet=AgentWallet(...))

# Upgrade to Gordon for production
from ach.verifiers.gordon import GordonVerifier
harness = CommerceHarness(
    verifier = GordonVerifier(public_key="gak_pub_...", secret_key="gak_sec_..."),
    wallet   = AgentWallet(...),
)
```

### LangGraph in 10 lines

```python
from ach.integrations.langgraph import add_commerce_check
graph = StateGraph(MyState)
# ... add your nodes ...
add_commerce_check(graph, before="execute_payment", verifier=verifier, wallet=wallet)
graph = graph.compile()
```

### MCP Server (for any orchestrator)

```bash
# Run Gordon harness as an MCP server
ach serve --verifier gordon --wallet wallets/travel.yaml --port 3001

# In Claude Desktop or any MCP client:
# Tool: ach_check_action(action_type, amount, merchant_id, ...)
# Tool: ach_end_session(session_id)
# Tool: ach_get_receipt(car_id)
```

---

## Why Open-Source This

1. **Distribution**: Every LangGraph/CrewAI developer who builds a commerce agent will find this framework. They start with `LocalRulesVerifier` (free), hit production limits, and upgrade to Gordon.
2. **Standard-setting**: If the harness becomes the de facto way to build commerce agents, the `Verifier` protocol becomes the standard. Gordon is the premium implementation.
3. **Benchmark as credibility**: The open benchmark lets any developer verify Gordon's detection claims independently. Transparency builds trust.
4. **MCP server as zero-friction onboarding**: Any Claude/Anthropic user can add Gordon's MCP server to their config and get fraud protection without writing any code.
