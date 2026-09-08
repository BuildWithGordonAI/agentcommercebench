# Gordon Agentic Commerce Harness — Holistic Design

> Full system design: offline training → model deployment → MCP/SDK integration → customer-facing detection → adversarial sandboxing

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [How Fraud Detection Works via MCP](#2-how-fraud-detection-works-via-mcp)
3. [SDK Telemetry — Agent Owner Integration](#3-sdk-telemetry--agent-owner-integration)
4. [Dual Signal: MCP + SDK Together](#4-dual-signal-mcp--sdk-together)
5. [Model Deployment Architecture](#5-model-deployment-architecture)
6. [Adversarial Design & Sandboxing](#6-adversarial-design--sandboxing)
7. [Multi-Tenant Architecture](#7-multi-tenant-architecture)
8. [Customer-Facing Components](#8-customer-facing-components)
9. [End-to-End Flow: Offline → Production](#9-end-to-end-flow-offline--production)

---

## 1. System Overview

Gordon sits between an agent orchestrator and payment rails. It has two integration surfaces (MCP and SDK), a shared fraud intelligence layer, and three deployment environments (sandbox, shadow, production).

```
┌───────────────────────────────────────────────────────────────────┐
│                    Agent Ecosystem (Customer)                      │
│                                                                    │
│   ┌─────────────────────┐       ┌─────────────────────────────┐   │
│   │  Agent Orchestrator  │       │  Agent Codebase (Python/TS) │   │
│   │  LangGraph / CrewAI  │       │  Custom agent loop          │   │
│   └──────────┬──────────┘       └──────────────┬──────────────┘   │
│              │ MCP tool call                    │ SDK instrumentation│
└──────────────┼──────────────────────────────────┼──────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Gordon API Layer                                │
│                                                                    │
│   ┌──────────────┐  ┌─────────────────┐  ┌────────────────────┐  │
│   │  MCP Server  │  │ Telemetry Intake │  │  Session Store     │  │
│   │  (FastMCP)   │  │ (Kafka / HTTP)  │  │  (Redis + Postgres)│  │
│   └──────┬───────┘  └────────┬────────┘  └────────┬───────────┘  │
│          │                   │                     │              │
│          └──────────┬────────┘                     │              │
│                     ▼                              │              │
│          ┌──────────────────────────────────────────┐             │
│          │           Fraud Pipeline                  │             │
│          │  Rules → Behavioral ML → LLM Classifier  │◄────────────┘
│          └──────────────────┬───────────────────────┘             │
│                             │                                      │
│          ┌──────────────────▼───────────────────────┐             │
│          │         Decision + Attestation            │             │
│          │  ALLOW / FLAG / BLOCK + signed record     │             │
│          └──────────────────┬───────────────────────┘             │
└─────────────────────────────┼────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Payment Rails    Customer Alert   Audit Log
        (Stripe/ACH)    (Webhook/Slack)  (Postgres/Chain)
```

### Three integration modes

| Mode | What Gordon sees | Latency | Coverage |
|---|---|---|---|
| **MCP only** | Action + amount at decision point | Synchronous | Payment actions only |
| **SDK only** | Full event stream (all tool calls, LLM calls) | Async stream | Everything, but no inline blocking |
| **MCP + SDK** | Full context at every decision point | Sync decision + async enrichment | Complete |

Production recommendation: MCP for blocking decisions, SDK for behavioral telemetry. MCP blocks on a decision; SDK enriches it.

---

## 2. How Fraud Detection Works via MCP

### What Gordon sees through MCP

When an agent orchestrator connects to Gordon's MCP server, every consequential action the agent takes passes through a tool call before execution. Gordon sees:

- `session_id`, `agent_id`, `customer_id`
- `action_type` (authorize, settle, void, refund)
- `amount`, `currency`, `merchant_id`, `rail`
- `idempotency_key`
- Optionally: `payload` (the full action payload for prompt injection scanning)

Gordon does **not** see, via MCP alone: prior tool calls, LLM reasoning steps, intermediate search results, or anything the agent did before deciding to call this tool. That is the SDK's job.

### Session state — Gordon holds it server-side

Because agents are stateless between tool calls, Gordon maintains session context server-side. The first MCP call for a `session_id` creates a `TransactionContext`. Subsequent calls append to it. This means Gordon can run the full sequence model and trajectory guards even through the MCP path, as long as the `session_id` is consistent across calls.

```python
# Gordon MCP server tools
@mcp.tool()
async def gordon_check_action(
    session_id:      str,
    agent_id:        str,
    action_type:     str,    # "authorize" | "settle" | "void"
    amount:          float,
    currency:        str,
    merchant_id:     str,
    idempotency_key: str,
    payload:         dict = None,   # optional: full request for payload scan
) -> dict:
    """
    Synchronous fraud check. Call BEFORE executing any consequential action.
    Returns decision immediately. Agent should not proceed on BLOCK.
    """
    context = await session_store.get_or_create(session_id, agent_id)
    event   = build_event(action_type, amount, merchant_id, payload)
    context, signals = pipeline.process_event(event, context)
    await session_store.save(session_id, context)

    decision = resolve_decision(context, signals)
    attestation_id = await attester.sign_and_store(session_id, event, decision, signals)

    return {
        "decision":         decision.value,         # "allow" | "flag" | "block"
        "risk_score":       context.risk_score,
        "flags":            context.risk_flags,
        "attestation_id":   attestation_id,
        "reversibility_window": "24h",
    }

@mcp.tool()
async def gordon_end_session(session_id: str) -> dict:
    """
    Run settlement guards. Call after the agent's session is complete.
    """
    context = await session_store.get(session_id)
    context, signals = pipeline.end_session(context)
    return {"session_risk": context.risk_score, "flags": context.risk_flags}
```

### What the agent orchestrator does with the response

The orchestrator pattern should be:

```python
# In the agent's tool executor (LangGraph node, CrewAI step, etc.)
async def execute_action(action: dict) -> dict:
    check = await gordon_mcp.check_action(
        session_id      = current_session_id,
        agent_id        = agent_id,
        action_type     = action["type"],
        amount          = action["amount"],
        merchant_id     = action["merchant"],
        idempotency_key = f"{current_session_id}:{action_index}",
    )

    if check["decision"] == "block":
        raise AgentBlockedError(f"Gordon blocked action: {check['flags']}")

    if check["decision"] == "flag":
        await alert_channel.send(f"Flagged: {check['flags']}")
        # Proceed but log — or pause for human confirmation depending on policy

    # Execute the actual payment action
    result = await payment_rail.authorize(action)
    return result
```

### MCP limitations without SDK

Without SDK telemetry, Gordon has no context about what happened before the agent called `gordon_check_action`. The sequence model has only the events Gordon has seen via MCP this session. If the agent did 10 recon searches before the authorize, Gordon sees a cold-start authorize — and may not flag it.

This is why the SDK path exists.

---

## 3. SDK Telemetry — Agent Owner Integration

### What the SDK captures

The Gordon SDK wraps the agent's existing tool calls and LLM calls. It captures every event automatically, not just payment actions:

```
Event types captured by SDK:
  TOOL_CALL          tool name, arguments (sanitized), response summary
  LLM_CALL           model, truncated prompt/response, token count
  ACTION_INTENT      agent's declared intent from its reasoning
  CONSEQUENTIAL_ACTION  full payment action (same as MCP)
  ERROR              exceptions, retries, timeouts
  SESSION_START / SESSION_END
```

Each event goes to Gordon's telemetry intake (Kafka topic or HTTPS endpoint), stamped with `session_id`, `agent_id`, `customer_id`, and `timestamp`.

### Instrumentation — what the agent owner does

**Option A: Decorator wrapping (lowest effort)**

```python
from gordon import Gordon

gordon = Gordon(
    public_key  = os.environ["GORDON_PUBLIC_KEY"],
    secret_key  = os.environ["GORDON_SECRET_KEY"],
    env         = "production",   # or "sandbox"
)

# Wrap any tool function — Gordon captures calls automatically
@gordon.instrument_tool
async def search_flights(origin: str, destination: str, date: str) -> list:
    return await amadeus_api.search(origin, destination, date)

@gordon.instrument_tool
async def book_flight(flight_id: str, passenger: dict) -> dict:
    return await amadeus_api.book(flight_id, passenger)

# Wrap LLM calls
@gordon.instrument_llm
async def run_agent_step(messages: list) -> str:
    return await anthropic_client.messages.create(...)
```

**Option B: Context manager (explicit sessions)**

```python
async with gordon.session(session_id=session_id, agent_id=agent_id, persona="travel") as gs:
    # All tool/LLM calls inside this block are captured automatically
    flights = await search_flights("SEA", "JFK", "2026-09-01")
    booking = await book_flight(flights[0]["id"], passenger)

    # For consequential actions, SDK can do inline check + telemetry together
    result = await gs.authorize(
        amount      = booking["total"],
        merchant_id = "amadeus-001",
        rail        = "card",
    )
```

**Option C: Manual event emission (most control)**

```python
gordon.emit(Event(
    session_id  = session_id,
    agent_id    = agent_id,
    event_type  = EventType.TOOL_CALL,
    tool_name   = "search_flights",
    args        = {"origin": "SEA", "destination": "JFK"},
    response    = {"count": 12, "min_price": 320.00},
))
```

### Telemetry privacy and what gets sanitized

The SDK sanitizes before transmission:

- **Truncated**: LLM prompt/response (first 500 chars of user content, no system prompt)
- **Hashed**: PII in tool args (card numbers → last4, email → hash, name → omit)
- **Omitted**: Raw tool responses that contain credentials, full card PANs, SSNs
- **Kept**: Action types, amounts, merchant IDs, timing, sequence structure

Customers can configure additional field-level redaction via SDK config.

---

## 4. Dual Signal: MCP + SDK Together

When both are active, the flow is:

```
1. SDK captures TOOL_CALL (search_flights) → async to telemetry
2. SDK captures TOOL_CALL (search_flights x3) → async to telemetry
3. Agent decides to book → calls gordon_check_action via MCP (synchronous)
4. Gordon receives MCP call:
   a. Loads session context from session_store
   b. Session context is already enriched with 3 search events from SDK telemetry
   c. SequenceModelGuard sees: 3x find_service before authorize → normal travel pattern
   d. Without SDK: cold-start authorize → ambiguous
   e. With SDK: full behavioral context → confident ALLOW or FLAG
5. Gordon returns decision in <50ms
6. Agent proceeds or blocks
7. SDK captures CONSEQUENTIAL_ACTION result → async to telemetry
8. At session end: SDK calls gordon_end_session → settlement guards run
```

The key: SDK telemetry arrives async (fire-and-forget), pre-loading the session context. MCP calls are sync, consuming that context for decisions. The two streams share the same `session_id` as the join key.

---

## 5. Model Deployment Architecture

### The three tiers and where they run

```
Tier 1: Rules Engine
  Deployment: In-process (Gordon API gateway, no network hop)
  Latency:    <1ms
  Examples:   REPLAYED- key prefix, velocity hard limits, OFAC check
  Update:     Config change, no redeploy

Tier 2: Behavioral ML
  Deployment: Gordon API servers (in-memory ONNX or pickled models)
  Latency:    2–10ms
  Examples:   Markov sequence model, PriceOracleGuard, TrajectoryGuard
  Update:     Rolling deploy of new model artifact

Tier 3: LLM Classifier
  Deployment: SageMaker real-time endpoint (existing: acb-guard-qwen25-7b-graph)
  Latency:    200–800ms
  Trigger:    Only when Tier 1+2 confidence < threshold (currently <12% of sessions)
  Update:     SageMaker endpoint swap with traffic shifting
```

### Offline training pipeline

```
Customer telemetry (consented)
    + Gordon red-team dataset (synthetic attacks)
    + Benchmark sessions (benchmark/data/)
          │
          ▼
    Data pipeline (scripts/build_training_data.py)
    - Deduplicate sessions
    - Label: clean / attack class (B1-B6, D1-D2, P1-P6)
    - Build session graphs (graph statistics for LLM tier)
    - Train/validation/test split (80/10/10, time-based)
          │
          ▼
    ┌─────────────────────────────────┐
    │  Tier 2: Markov model refit     │  → serialize to models/sequence_v{n}.pkl
    │  (fit_all on clean sessions)    │
    └─────────────────────────────────┘
          │
    ┌─────────────────────────────────┐
    │  Tier 3: LLM fine-tune          │  → SageMaker training job
    │  (QLoRA on Qwen2.5-3B)          │     using sagemaker_train/
    └─────────────────────────────────┘
          │
          ▼
    Evaluate on holdout (benchmark/evaluate.py)
    Target: F1 ≥ 0.88, FPR ≤ 5%
          │
          ▼
    Promote to shadow if metrics pass
```

### Shadow mode → production promotion

```
Shadow deployment:
  - New model runs alongside production
  - Makes decisions but does not enforce them
  - Both decisions logged: production_decision, shadow_decision
  - Compare: where do they diverge?

Promotion criteria:
  - Shadow F1 ≥ production F1
  - Shadow FPR ≤ production FPR + 2%
  - No regressions on known-clean sessions (false positive spike)
  - 48-hour shadow soak with ≥ 10,000 sessions

Traffic shifting (SageMaker):
  - 10% → shadow endpoint
  - Monitor for 24h
  - 50% → shadow endpoint
  - Monitor for 24h
  - 100% → new endpoint becomes production
```

### Model versioning

Every fraud decision references the model versions that made it:

```json
{
  "attestation_id": "att_abc123",
  "decision": "block",
  "model_versions": {
    "rules": "v1.4.2",
    "sequence_model": "v3.1",
    "llm_classifier": "qwen25-3b-graph-v2"
  },
  "tier_that_decided": "sequence_model"
}
```

This makes decisions auditable and reproducible — a specific session can be replayed against the exact model versions that saw it.

---

## 6. Adversarial Design & Sandboxing

### The adversary model

The attacker is not necessarily a human. Three adversary types:

1. **Compromised agent**: a legitimate agent that has been prompt-injected and is now executing attacker-controlled actions. Looks like a normal agent until it hits an unusual AUTHORIZE.
2. **Malicious agent**: an agent deliberately built to commit fraud — carding, replay, velocity flooding.
3. **Insider / misconfigured agent**: not malicious, but badly configured — loops, duplicate bookings, over-spending. Looks like fraud from the detector's perspective.

The harness must handle all three. The key invariant: **Gordon must never be exploitable as a vector to break production payment rails.**

### Sandbox environment

A full Gordon environment where:
- **Payment rails are simulated**: Gordon returns fake auth IDs, no real Stripe calls
- **Fraud pipeline is real**: same models, same guards, same thresholds
- **Session state is real**: sequences, velocity, cross-session guards all run
- **Attestation is real**: signed records written to sandbox store (not production)

```python
gordon_sandbox = Gordon(
    public_key = os.environ["GORDON_PUBLIC_KEY"],
    secret_key = os.environ["GORDON_SECRET_KEY"],
    env        = "sandbox",     # ← all payment calls are no-ops, pipeline is real
)

# In sandbox: gordon_check_action returns real fraud decisions
# but gordon_execute_payment returns a fake authorization
# {"auth_id": "sandbox_auth_abc123", "status": "authorized_simulated"}
```

### What the sandbox prevents

The sandbox enforces:
- No real Stripe/ACH/wire calls — all payment operations are intercepted and mocked
- No real merchant accounts are charged
- Sandbox sessions never contaminate production session state or cross-session graphs
- Rate limits still apply (prevents using sandbox to DoS the production pipeline)

A developer can run a full replay attack scenario in sandbox, hit the REPLAYED- guard, get a BLOCK decision, and see the attestation record — without any real money moving or any production signal being polluted.

### Adversarial test runner

The benchmark harness (`benchmark/evaluate.py`) already does this. It can be run against any Gordon environment:

```bash
# Run the full attack benchmark against sandbox endpoint
python benchmark/evaluate.py \
  --detector gordon_mcp \
  --env sandbox \
  --gordon-key $GORDON_PUBLIC_KEY \
  --scenarios B1,B2,B3,B4,B5,B6,D1,D2

# Output: per-attack TPR, FPR, F1 — same format as offline eval
```

This is the adversarial test loop: develop an attack scenario → run it in sandbox → observe detection → adjust attack → iterate. The harness does not break real production on any iteration of this loop.

### Replay attack isolation

The replay guard specifically checks idempotency keys within a session's prior events. In sandbox, the session store is namespaced by `env=sandbox`, so a replay attack in sandbox:
- Is correctly detected by the pipeline
- Does NOT pollute the production session store
- Does NOT create a real duplicate authorization

If a real-world attacker somehow calls Gordon's sandbox endpoint: the same detection runs, but the execution is mocked. They cannot use the sandbox to probe for gaps they could then exploit in production, because the detection in both environments is identical.

### Shadow mode for adversarial validation

Before promoting a new model to production, run the full red-team attack suite against it in shadow mode:

```bash
# Replay historical attack sessions against shadow model
python scripts/shadow_eval.py \
  --shadow-endpoint $SHADOW_ENDPOINT \
  --dataset benchmark/data/attack_sessions.jsonl \
  --compare-to production
```

If shadow F1 drops on any attack class, the model does not promote.

---

## 7. Multi-Tenant Architecture

### Customer isolation

Each customer has:
- Their own `customer_id` namespace
- Their own API keys (`GORDON_PUBLIC_KEY`, `GORDON_SECRET_KEY`)
- Their own session store partition (Redis keyspace by `customer_id`)
- Their own wallet definitions and spend limits
- Their own attestation records (Postgres row-level security by `customer_id`)

Sessions, signals, and attestation records from Customer A are never accessible to Customer B.

### Shared threat intelligence

Anonymized cross-customer signals improve the behavioral models for everyone:

- **What is shared**: attack pattern fingerprints (not session content), merchant fraud rates, idempotency key format anomalies
- **What is not shared**: session content, agent identities, amounts, merchant names
- **How**: nightly aggregation job extracts anonymized signal vectors, updates the shared Markov model priors and LLM training data
- **Opt-in**: customers can opt out of contributing telemetry to the shared pool (they still benefit from others' signal, but don't contribute)

This is the network effect moat: Customer 1's attack data improves detection for Customer 50, without Customer 1's data being visible to Customer 50.

### Different agent types per customer

Each customer may have multiple agent types with different risk profiles:

```python
# Customer: AcmeCorp
wallets = {
    "travel-booker": AgentWallet(
        spend_limits   = SpendLimits(per_txn=3000, per_day=10000),
        allowed_rails  = [CardRail, ACHRail],
        allowed_mcc    = ["4511", "7011"],   # Airlines, Hotels
        consent_level  = ConsentLevel.PRE_APPROVED,
    ),
    "procurement-agent": AgentWallet(
        spend_limits   = SpendLimits(per_txn=50000, per_day=200000),
        allowed_rails  = [ACHRail, WireRail],
        allowed_mcc    = ["5065", "5045"],   # Electronics, Computers (wholesale)
        consent_level  = ConsentLevel.SOFT,  # Requires secondary approval above $10k
    ),
    "research-agent": AgentWallet(
        spend_limits   = SpendLimits(per_txn=50, per_day=200),
        allowed_rails  = [CardRail],
        allowed_mcc    = ["7372"],           # Software
        consent_level  = ConsentLevel.PRE_APPROVED,
    ),
}
```

The fraud pipeline uses the wallet's agent type to select the right behavioral model (travel persona, procurement persona, research persona) — matching what we already have in the Markov model.

---

## 8. Customer-Facing Components

### Real-time session dashboard

Customers see a live view of their agents' sessions:

```
Session: sess_abc123  Agent: travel-booker  Status: ACTIVE
Risk: 0.12 ████░░░░░░ LOW

Events:
  10:42:01  FIND_SERVICE    amadeus/flights     ALLOW  0.04
  10:42:08  FIND_SERVICE    amadeus/flights     ALLOW  0.06
  10:42:15  GET_SERVICE     amadeus/flight/UA789 ALLOW  0.09
  10:42:31  AUTHORIZE       $450.00 United →   ALLOW  0.12  ← current

Guards fired:  seq:travel_pattern ✓  price_oracle ✓  replay ✓
```

### Alert webhooks

```json
POST https://customer.example.com/gordon-webhook
{
  "event":          "session.flagged",
  "session_id":     "sess_def456",
  "agent_id":       "travel-booker",
  "decision":       "flag",
  "risk_score":     0.74,
  "flags":          ["seq:unusual_transition authorize/finance→authorize/travel", "velocity:3x_normal"],
  "attestation_id": "att_ghi789",
  "timestamp":      "2026-08-23T10:43:00Z"
}
```

Customers configure webhooks per decision level (flag-only, block-only, or both). Gordon retries with exponential backoff on 5xx responses.

### Audit log and chargeback defense

When a customer receives a chargeback, they need to prove the agent was authorized. Gordon provides:

```
GET /v1/attestations/{attestation_id}
→ {
    "record_id":      "att_ghi789",
    "session_id":     "sess_abc123",
    "agent_id":       "travel-booker",
    "action":         { "type": "authorize", "amount": 450.00, "merchant": "United Airlines" },
    "decision":       "allow",
    "fraud_score":    0.12,
    "consent_proof":  "eyJhbGci...",   # signed JWT: wallet_id, limits, agent_id, timestamp
    "wallet_limits":  { "per_txn": 3000, "allowed_mcc": ["4511"] },
    "processor_ref":  "auth_stripe_xyz123",
    "timestamp":      "2026-08-23T10:42:31Z"
  }
```

The signed consent_proof JWT is the chargeback defense document: it proves that a pre-configured wallet with explicit limits authorized this action before execution, and that Gordon's fraud pipeline cleared it.

### Analytics

```
/v1/analytics/summary?period=30d
→ {
    "sessions":          4821,
    "actions_checked":   18340,
    "allow":             17892,  (97.6%)
    "flag":              340,    ( 1.9%)
    "block":             108,    ( 0.6%)
    "top_flags":         ["velocity", "seq:unusual_transition", "price_oracle"],
    "attack_breakdown":  { "replay": 23, "velocity": 41, "recon": 18, "prompt_injection": 12 },
    "false_positive_est": 0.8%,
    "chargeback_deflections": 14
  }
```

---

## 9. End-to-End Flow: Offline → Production

### Full lifecycle

```
OFFLINE
  ①  Red-team team generates attack scenarios (B1-B6, D1-D2, P1-P6)
  ②  Customer telemetry (opted-in) + synthetic clean sessions
  ③  Data pipeline: label, graph, split
  ④  Train: Markov model refit + LLM fine-tune (QLoRA / SageMaker)
  ⑤  Evaluate: benchmark/evaluate.py → F1, FPR per attack class
  ⑥  Pass threshold? → promote to shadow

SHADOW
  ⑦  Shadow endpoint receives 10% of production traffic
  ⑧  Both old + new decisions logged, compared nightly
  ⑨  48h soak, no regressions → traffic shift to 50% → 100%
  ⑩  Old endpoint decommissioned

PRODUCTION
  ⑪  Customer agent calls gordon_check_action (MCP) before each consequential action
  ⑫  SDK telemetry pre-loads session context asynchronously
  ⑬  Confidence cascade: Rules → ML → LLM (LLM fires <12% of sessions)
  ⑭  Decision returned: ALLOW / FLAG / BLOCK
  ⑮  Attestation record signed and stored
  ⑯  Flagged sessions trigger customer webhook
  ⑰  Blocked sessions: agent receives AgentBlockedError, must not proceed

FEEDBACK LOOP
  ⑱  Periodically sample ALLOW sessions for human review (spot-check FPR)
  ⑲  Blocked sessions reviewed: true positive or false positive?
  ⑳  Labels fed back into ① for next training cycle
  ㉑  Cross-customer anonymized signal aggregated nightly → shared priors updated
```

### Session lifecycle in code

```python
# 1. Session starts (SDK)
async with gordon.session(session_id, agent_id="travel-booker", persona="travel") as gs:

    # 2. Agent browses (SDK captures automatically, no MCP call needed)
    flights = await search_flights("SEA", "JFK", "2026-09-01")
    details = await get_flight_details(flights[0]["id"])

    # 3. Agent decides to book — MCP checkpoint (synchronous)
    check = await gordon_mcp.check_action(
        session_id      = session_id,
        agent_id        = "travel-booker",
        action_type     = "authorize",
        amount          = details["price"],
        merchant_id     = "united-airlines",
        idempotency_key = f"{session_id}:flight:0",
    )
    if check["decision"] == "block":
        raise AgentBlockedError(check["flags"])

    # 4. Agent executes (real payment)
    auth = await stripe.authorize(details["price"], customer_card_token)

    # 5. Confirm and settle
    await gordon_mcp.check_action(..., action_type="settle")
    await stripe.capture(auth["id"])

# 6. Session ends → settlement guards run automatically (SDK calls gordon_end_session)
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| MCP vs SDK | Both, complementary | MCP blocks inline; SDK provides behavioral context |
| Session state location | Gordon server-side (Redis) | Agents are stateless; Gordon must hold context |
| LLM tier trigger | <12% of sessions | Cost control; rules+ML handles the clear cases |
| Sandbox vs production isolation | Separate Redis keyspace + mocked rails | Real pipeline, no real money, no cross-contamination |
| Cross-customer signal | Anonymized, opt-in contribution | Moat without privacy violation |
| Chargeback defense | Signed JWT per attestation | Tamper-evident proof of pre-authorization |
| Model promotion | Shadow mode → traffic shifting | Gradual risk, regression-safe |
| Adversary model | Compromised agent, malicious agent, misconfigured agent | Covers full threat surface |
