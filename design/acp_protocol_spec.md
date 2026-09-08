# ACP — Agentic Commerce Protocol
## Specification v0.1

> A verifiability-first protocol for agent-initiated commerce.
> Fraud detection is a subset. Provenance is the primitive.

---

## Why a New Protocol

MCP gives agents access to tools. A2A gives agents the ability to delegate to other agents. Neither protocol has any concept of:

- **Economic consequence** — there is no semantic difference between "search for flights" and "buy a flight"
- **Verifiability** — no cryptographic proof that a tool was actually called with those arguments and produced that outcome
- **Provenance** — no chain of custody: who authorized what, under what scope, at what time
- **Reversibility** — no mechanism to declare that an action can or cannot be undone
- **Commerce identity** — no standard way for an agent to assert who it is, who it acts for, and what it is permitted to do

ACP extends both MCP and A2A by adding these as first-class protocol concepts. It does not replace them — it layers on top. An MCP tool that handles money adds an ACP wrapper. An A2A agent that delegates purchasing power adds an ACP scope grant.

**The core insight:** fraud detection is just one verifier answering one question about a verifiable record. Once you have verifiability, fraud detection, compliance checking, price validation, audit, billing, and dispute resolution all become different queries against the same data structure.

---

## Design Principles

1. **Verifiability over detection** — build the record first; detection is a query on top of it
2. **Provenance as a DAG** — agent spawns sub-agents; authority flows down; receipts flow up
3. **Tiered ceremony** — not every action needs a full cryptographic ceremony; cost scales with consequence
4. **Pluggable verifiers** — Gordon, compliance engines, price oracles all implement the same interface
5. **Backwards compatible** — an ACP-unaware MCP tool can still be called; the CAR marks it as unverified
6. **Identity via DIDs** — agents, verifiers, and merchants are identified by Decentralized Identifiers

---

## Core Data Structures

### Commerce Action Receipt (CAR)

The atomic unit of ACP. Every consequential action produces exactly one CAR. The CAR is the thing that gets signed, stored, shared, and verified.

```json
{
  "car_version":    "1.0",
  "car_id":         "car_a1b2c3d4",
  "session_id":     "sess_e5f6g7h8",

  "agent": {
    "id":           "did:acp:agent:travel-booker-001",
    "issued_by":    "did:acp:platform:acmecorp",
    "public_key":   "ES256:MFkwEwYHKoZIzj0C..."
  },

  "action": {
    "type":              "COMMIT",
    "commerce_primitive": "book_flight",
    "amount":            450.00,
    "currency":          "USD",
    "merchant": {
      "id":              "did:acp:merchant:united-airlines",
      "mcc":             "4511",
      "name":            "United Airlines"
    },
    "reversibility":     "PARTIAL",
    "reversibility_window": "2026-09-01T10:42:31Z",
    "idempotency_key":   "sess_e5f6:flight:0",
    "payload_hash":      "sha256:9f86d0..."
  },

  "authorization": {
    "scope_id":      "scope_xyz123",
    "wallet_id":     "wallet_travel_001",
    "limits": {
      "per_transaction": 3000.00,
      "per_day":         10000.00,
      "allowed_mcc":     ["4511", "7011", "7512"]
    },
    "consent_level":   "PRE_APPROVED",
    "issued_by":       "did:acp:platform:acmecorp",
    "issued_at":       "2026-08-23T09:00:00Z",
    "expires_at":      "2026-08-23T18:00:00Z",
    "scope_signature": "ES256:eyJhbGci..."
  },

  "provenance": [
    {
      "step":       0,
      "type":       "SESSION_START",
      "agent":      "did:acp:agent:acmecorp-orchestrator",
      "timestamp":  "2026-08-23T10:40:00Z"
    },
    {
      "step":       1,
      "type":       "FIND",
      "tool":       "search_flights",
      "args_hash":  "sha256:abc123",
      "timestamp":  "2026-08-23T10:41:00Z"
    },
    {
      "step":       2,
      "type":       "FIND",
      "tool":       "search_flights",
      "args_hash":  "sha256:def456",
      "timestamp":  "2026-08-23T10:41:30Z"
    },
    {
      "step":       3,
      "type":       "QUOTE",
      "tool":       "get_flight_details",
      "args_hash":  "sha256:ghi789",
      "timestamp":  "2026-08-23T10:42:00Z"
    },
    {
      "step":       4,
      "type":       "COMMIT",
      "car_id":     "car_a1b2c3d4",
      "timestamp":  "2026-08-23T10:42:31Z"
    }
  ],

  "verifications": [
    {
      "verifier_id": "did:gordon:verifier:fraud-pipeline",
      "decision":    "allow",
      "score":       0.12,
      "flags":       [],
      "model_versions": {
        "rules":          "v1.4.2",
        "sequence_model": "v3.1",
        "llm_classifier": "qwen25-3b-graph-v2"
      },
      "timestamp":   "2026-08-23T10:42:31Z",
      "signature":   "ES256:verifier_sig_abc..."
    }
  ],

  "final_decision":   "allow",
  "agent_signature":  "ES256:agent_sig_xyz...",
  "timestamp":        "2026-08-23T10:42:31Z",

  "outcome": {
    "status":         "AUTHORIZED",
    "processor_ref":  "auth_stripe_abc123",
    "settled":        false
  }
}
```

### What the CAR enables

| Use case | How the CAR answers it |
|---|---|
| **Fraud detection** | Verifier checks action vs. authorization scope and session provenance |
| **Compliance audit** | Full provenance chain shows every step taken before the action |
| **Chargeback defense** | `authorization.scope_signature` proves pre-authorized limits; `verifications[0].decision = allow` proves fraud check passed |
| **Dispute resolution** | `agent.id` + `authorization.issued_by` establishes the chain of responsibility |
| **Billing / metering** | `action.amount` + `action.timestamp` + `agent.id` = billing record |
| **Model accountability** | `verifications[0].model_versions` makes every decision reproducible |
| **Replay detection** | `idempotency_key` + `car_id` uniquely identifies each action |

---

## Protocol Extensions

### ACP over MCP

ACP extends MCP with two new concepts:

**1. Commerce Context Header**

Every MCP message involving a consequential tool carries a `commerce_context` extension:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "book_flight",
    "arguments": { "flight_id": "UA789", "passenger": "..." },
    "commerce_context": {
      "session_id":  "sess_e5f6g7h8",
      "agent_did":   "did:acp:agent:travel-booker-001",
      "wallet_id":   "wallet_travel_001",
      "action_type": "COMMIT",
      "amount":      450.00,
      "currency":    "USD",
      "merchant_id": "did:acp:merchant:united-airlines",
      "idempotency_key": "sess_e5f6:flight:0"
    }
  }
}
```

**2. CAR in MCP Tool Response**

When a consequential MCP tool returns, it includes the CAR:

```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [{ "type": "text", "text": "Booking confirmed: PNR-789" }],
    "car": {
      "car_id":         "car_a1b2c3d4",
      "final_decision": "allow",
      "attestation_url": "https://api.gordonguard.ai/v1/cars/car_a1b2c3d4"
    }
  }
}
```

An MCP-unaware tool that doesn't include a `car` field still works — the client marks it as `unverified` in the provenance chain.

### ACP over A2A

ACP extends A2A agent delegation with commerce scope:

```json
{
  "a2a_version": "1.0",
  "task_id":     "task_abc",
  "agent":       "did:acp:agent:travel-booker-001",
  "delegated_by": "did:acp:agent:acmecorp-orchestrator",
  "commerce_scope": {
    "wallet_id":    "wallet_travel_001",
    "limits": {
      "per_transaction": 3000.00,
      "allowed_mcc":     ["4511", "7011"]
    },
    "consent_level": "PRE_APPROVED",
    "scope_signature": "ES256:parent_agent_sig..."
  }
}
```

The sub-agent's CARs reference this `commerce_scope`. A verifier can check that the action was within the delegated scope — and trace back to the original human authorization if needed.

---

## Provenance DAG

When agents spawn sub-agents, the provenance forms a directed acyclic graph:

```
Human authorization
    │
    ▼  scope_grant: $50,000/day, all MCCs
did:acp:agent:acmecorp-orchestrator
    │
    ├──► scope_grant: $10,000/day, MCCs [4511, 7011]
    │   did:acp:agent:travel-booker
    │       ├── CAR: FIND / search_flights       (no financial consequence, no verifier needed)
    │       ├── CAR: FIND / search_flights       (same)
    │       ├── CAR: QUOTE / get_flight_details  (same)
    │       └── CAR: COMMIT / book_flight        ← verifiers run here
    │               verifiers: [gordon-fraud, price-oracle]
    │               decision: ALLOW, score: 0.12
    │
    └──► scope_grant: $200/day, MCCs [7372]
        did:acp:agent:research-agent
            ├── CAR: FIND / web_search           (no verifier)
            └── CAR: COMMIT / buy_report         ← verifiers run here
                    verifiers: [gordon-fraud]
                    decision: ALLOW, score: 0.04
```

### Scope inheritance rules

- Sub-agents can never exceed their parent's scope
- Spend against a sub-agent's wallet counts against all parent wallets in the chain
- A BLOCK at any level blocks the sub-agent; parent is notified
- A FLAG propagates up the chain (parent sees aggregated session risk)

---

## Tiered Ceremony

Not every action warrants full cryptographic overhead. ACP defines four ceremony levels based on action consequence:

| Level | Action Types | Ceremony | Latency Overhead |
|---|---|---|---|
| **L0** | FIND, QUOTE | No verifier, no CAR — just provenance hash appended to session | ~0ms |
| **L1** | RESERVE < $50 | Local rules verifier only; CAR with local signature | ~2ms |
| **L2** | COMMIT $50–$5,000 | Required verifier (Gordon or equivalent); signed CAR | ~50ms |
| **L3** | COMMIT > $5,000 or SETTLE or WIRE | Required verifier + second verifier (or human approval); CAR + blockchain anchor | ~500ms |

The wallet's `consent_level` determines whether L2/L3 actions are pre-approved or require pause.

---

## Verifier Registry

ACP maintains an open verifier registry (like DNS for verifiers). Any system can register as a verifier with a DID and public key. Wallets specify which verifiers are required for which action classes:

```yaml
# wallet.yaml
wallet_id: wallet_travel_001
agent_id:  did:acp:agent:travel-booker-001
limits:
  per_transaction: 3000
  allowed_mcc: ["4511", "7011"]
consent_level: PRE_APPROVED

required_verifiers:
  L2:
    - did:gordon:verifier:fraud-pipeline   # required
    - did:acp:verifier:price-oracle         # required
  L3:
    - did:gordon:verifier:fraud-pipeline
    - did:acp:verifier:price-oracle
    - did:acp:verifier:compliance-engine
    - human_approval: true                  # pause for human at L3
```

---

## Proving Value — the ACP Dashboard View

One of the hardest problems in fraud infrastructure is showing ROI. ACP makes this tractable because every action is recorded:

```
ACP Session Analytics (30 days)
────────────────────────────────────────────────────────────
Total sessions:          4,821    Total actions:    62,847
L0 (FIND/QUOTE):        48,312   (76.9%)  — no verifier needed
L1 (local rules):        7,423   (11.8%)  — 2ms overhead
L2 (full verify):        6,892   (11.0%)  — 48ms avg overhead
L3 (elevated):             220   ( 0.4%)  — human approval required

Verifier outcomes:
  ALLOW:    13,805  (97.6%)
  FLAG:        262   (1.9%)
  BLOCK:        45   (0.5%)

Estimated fraud prevented:    $84,300  (across 45 blocked sessions)
Chargeback deflections:           14   (avg $1,200 each → $16,800 saved)
False positive rate:            0.8%   (estimated from spot-check)
────────────────────────────────────────────────────────────
ROI: $101,100 protected / [your Gordon plan cost]
```

This is possible because the CAR records the amount, decision, and outcome for every action. You can reconstruct "what would have happened" counterfactually — sessions that were blocked, with their attempted amounts, are the fraud prevented metric.

---

## Identity: Agent DIDs

Every agent in ACP has a Decentralized Identifier (DID):

```
did:acp:agent:travel-booker-001
         │       │
         │       └── customer-scoped agent identifier
         └── ACP DID method
```

DID Document (resolvable via ACP registry or well-known URL):

```json
{
  "@context":     "https://www.w3.org/ns/did/v1",
  "id":           "did:acp:agent:travel-booker-001",
  "controller":   "did:acp:platform:acmecorp",
  "verificationMethod": [{
    "id":           "did:acp:agent:travel-booker-001#key-0",
    "type":         "JsonWebKey2020",
    "publicKeyJwk": { "kty": "EC", "crv": "P-256", "x": "...", "y": "..." }
  }],
  "commerce": {
    "wallet_id":     "wallet_travel_001",
    "persona":       "travel",
    "created_at":    "2026-08-01T00:00:00Z"
  }
}
```

Why DIDs:
- Decentralized — no central registry required, no lock-in
- Cryptographically verifiable — the DID resolves to a public key, signatures can be verified by anyone
- Portable — an agent DID is valid across different orchestrators and processors
- Standard — W3C DID spec is already widely implemented

---

## Comparison to MCP and A2A

| Capability | MCP | A2A | ACP |
|---|---|---|---|
| Tool calling | ✓ | — | ✓ (extends MCP) |
| Agent delegation | — | ✓ | ✓ (extends A2A) |
| Commerce semantics | — | — | ✓ |
| Cryptographic verifiability | — | — | ✓ |
| Provenance chain | — | partial | ✓ |
| Reversibility metadata | — | — | ✓ |
| Pluggable verifiers | — | — | ✓ |
| Agent identity (DID) | — | — | ✓ |
| Fraud detection | — | — | ✓ (as a verifier) |
| Chargeback defense | — | — | ✓ (via CAR) |

ACP does not fork MCP or A2A. It is additive. An MCP server that does not handle money ignores ACP extensions entirely. An A2A agent that does not delegate commerce scope behaves exactly as before.

---

## Reference Implementations

| Implementation | Language | Status |
|---|---|---|
| `acp-py` | Python | reference (this repo) |
| `acp-ts` | TypeScript | planned |
| Gordon verifier | Python | implemented (fraud pipeline) |
| Price oracle verifier | Python | planned |
| Local rules verifier | Python | implemented (open source) |
| MCP server extension | Python (FastMCP) | implemented |
| A2A scope grant extension | Python | planned |

---

## What Gordon Owns in This Picture

ACP is the open protocol. Gordon's proprietary value is:

1. **The fraud verifier** (`did:gordon:verifier:fraud-pipeline`) — the behavioral ML + LLM classifier behind the Verification interface. This is the part no one else can replicate without the training data.
2. **The cross-customer threat network** — anonymized signals across all ACP sessions running through Gordon, improving the fraud verifier for everyone.
3. **The attestation store** — production-grade signed CAR storage with chargeback defense exports, audit APIs, and analytics.
4. **The behavioral dataset** — the moat that compounds with volume.

The protocol is open. The intelligence is Gordon's.
