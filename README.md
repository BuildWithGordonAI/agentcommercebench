# Agentic Commerce Harness (ACH)

**Fraud-semantic verification for autonomous payment agents.**

Existing agent frameworks (LangChain, LangGraph, CrewAI) execute tools — they don't
know whether a payment is fraudulent. ACH adds the missing layer: a pluggable
verification protocol that intercepts every consequential action, produces a signed
Commerce Action Receipt (CAR), and routes it through a tiered confidence cascade
before settlement.

```
Agent  →  ConsequentialAction  →  [LocalRules → ML → LLM]  →  CAR (signed)  →  PaymentRail
```

---

## Benchmark Results

Evaluated on 520 synthetic sessions (200 clean + 40 × 8 attack classes, seed=42):

| Approach | Framework | F1 | Recall | FPR | B1 | B2 | B5 | B6 | B8 |
|---|---|---|---|---|---|---|---|---|---|
| No verification | LangChain | 0.000 | 0.000 | 0.000 | ✗ | ✗ | ✗ | ✗ | ✗ |
| Schema only | LangGraph | 0.000 | 0.000 | 0.000 | ✗ | ✗ | ✗ | ✗ | ✗ |
| Rules only | CrewAI-style | 0.689 | 0.525 | 0.000 | ✗ | ✗ | ✓ | ✗ | ✓ |
| ML only | — | 0.768 | 0.881 | 0.660 | ✓ | ~ | ✓ | ✓ | ~ |
| **ACP (ours)** | — | **0.780** | **0.906** | 0.670 | ✓ | ~ | ✓ | ✓ | **✓** |

Attack key: B1=cold-start, B2=velocity, B5=replay, B6=prompt injection, B8=spend limit.  
✓=detected (≥80%), ~=partial (50–79%), ✗=missed (<20%).

---

## Install

```bash
pip install -e .                          # core harness, no ML deps
pip install -e ".[gordon]"                # + Gordon fraud pipeline (requires API key)
pip install -e ".[benchmark]"             # + benchmark tools
pip install -e ".[langgraph]"             # + LangGraph integration
pip install -e ".[crewai]"                # + CrewAI integration
```

---

## Quick Start

### Standalone (no framework)

```python
from ach import (
    ConsequentialAction, ActionType, Reversibility,
    AgentWallet, SpendLimits, ConsentLevel,
    SessionContext, LocalRulesVerifier,
    CommerceActionReceipt,
)
from decimal import Decimal

wallet  = AgentWallet(
    wallet_id="wallet-001", agent_id="my-agent", persona="travel",
    consent_level=ConsentLevel.PRE_APPROVED,
    limits=SpendLimits(per_transaction=Decimal("3000"), per_day=Decimal("10000"),
                       allowed_mcc=["4511", "7011"]),
)
action  = ConsequentialAction(
    action_type=ActionType.COMMIT, amount=Decimal("450"), currency="USD",
    merchant_id="booking.com", merchant_name="Booking.com",
    merchant_mcc="7011", category="travel",
)
context = SessionContext(session_id="sess-001", agent_id="my-agent", persona="travel")

verifier = LocalRulesVerifier()
v        = verifier.verify(action, wallet, context)
car      = CommerceActionReceipt.build(action, wallet, [v], session_id="sess-001")

print(car.final_decision)      # "allow"
print(car.is_valid())          # True
print(v.uncertainty)           # epistemic uncertainty estimate
```

### LangGraph Integration

```python
from ach.integrations.langgraph.node import CommerceCheckNode, route_on_decision
from ach import LocalRulesVerifier, GordonVerifier
from langgraph.graph import StateGraph

verifier = GordonVerifier()   # or LocalRulesVerifier() for dev
node     = CommerceCheckNode(verifier, wallet)

graph = StateGraph(dict)
graph.add_node("commerce_check", node)
graph.add_node("execute_payment", execute_payment_fn)
graph.add_node("handle_block", handle_block_fn)
graph.add_conditional_edges("commerce_check", route_on_decision)

# For structural atomicity, compile with interrupt_before:
app = graph.compile(
    checkpointer=your_checkpointer,
    interrupt_before=["execute_payment"],   # structural gate
)
```

---

## Architecture

```
ach/
  actions/
    base.py           ConsequentialAction, ActionType, Reversibility
    wallet.py         AgentWallet, SpendLimits, ConsentLevel

  verifiers/
    base.py           Verifier protocol, Verification (with uncertainty), SessionContext
    local_rules.py    Deterministic: replay, MCC, velocity, spend limits
    gordon.py         Behavioral ML: Gordon fraud pipeline bridge
    uncertainty.py    UncertaintyEstimator, disagreement_uncertainty

  receipts/
    car.py            CommerceActionReceipt, ProvenanceStep
    signer.py         HMAC-SHA256 signing

  anchoring/
    chain.py          Hash-chained CARLog (tamper-evident audit trail)
    merkle.py         Merkle tree + inclusion proofs for batch anchoring

  ceremony.py         CeremonyRouter: (score, uncertainty) → L0-L3 + AnchoringLevel

  integrations/
    langgraph/        CommerceCheckNode, route_on_decision
    crewai/           CommerceCheckTool (BaseTool)
    mcp/              MCP server wrapper (gordon_check_action tool)

benchmark/
  synthetic.py        Reproducible session generator (B1-B8 attack taxonomy)
  acp_baselines.py    NoVerify / SchemaOnly / RulesOnly / MLOnly / ACP
  run_experiment.py   Full comparison table with calibration + chain integrity

paper/
  acp_paper_outline.md  Abstract, contributions, experiment design
```

---

## Verifier Protocol

Any system implements the `Verifier` protocol:

```python
class Verifier(Protocol):
    verifier_id: str
    def verify(self, action: ConsequentialAction,
               wallet: AgentWallet, context: SessionContext) -> Verification: ...
```

`Verification` now includes calibrated uncertainty:

```python
@dataclass
class Verification:
    verifier_id:    str
    decision:       str             # "allow" | "flag" | "block"
    score:          float           # 0–1 point estimate
    confidence:     float           # 1 - uncertainty
    uncertainty:    float           # epistemic: model knows it doesn't know
    score_interval: tuple[float, float]   # 90% CI [low, high]
    flags:          list[str]
    metadata:       dict

    @property
    def should_escalate(self) -> bool:   # route to next cascade tier
        return self.uncertainty > 0.30 or (score_interval[1] - score_interval[0]) > 0.40
```

---

## Commerce Action Receipt (CAR)

Every consequential action produces one CAR. The CAR is the atomic unit of ACP:

```json
{
  "car_id": "car_931f290d",
  "final_decision": "allow",
  "score": 0.12,
  "confidence": 0.94,
  "uncertainty": 0.06,
  "score_interval": [0.07, 0.17],
  "ceremony_level": "L2",
  "anchoring": {
    "log_entry": "sha256:a3f...",
    "prev_hash": "sha256:b7c...",
    "merkle_root": "sha256:d2e...",
    "chain_tx": null
  },
  "signature": "hmac256:..."
}
```

CAR signatures are HMAC-SHA256 over all decision-relevant fields. Tampering is
detected immediately via `car.is_valid()`. The hash-chained log makes retroactive
modification of any past entry detectable.

---

## Ceremony Tiers

| Level | Trigger | Anchoring | Cost |
|---|---|---|---|
| L0 | FIND / QUOTE | None | $0 |
| L1 | Commit < $50, low uncertainty | Local hash-chain | $0 |
| L2 | Commit $50–$5000 or uncertainty > 0.25 | Merkle batch | ~$0.0001 |
| L3 | Commit > $5000 or uncertainty > 0.45 or disagreement | On-chain | ~$0.10 |

---

## Run the Benchmark

```bash
# Quick validation (260 sessions)
python benchmark/run_experiment.py --n-clean 100 --n-per-attack 20

# Full paper benchmark (520 sessions)
python benchmark/run_experiment.py --n-clean 200 --n-per-attack 40 --seed 42

# Specific attack classes
python benchmark/run_experiment.py --attacks B1 B2 B5 B6
```

Results are saved to `benchmark/results/acp_benchmark.json`.

---

## Paper

> ACP: Agentic Commerce Protocol — Verifiable, Uncertainty-Aware Fraud Detection
> for Autonomous Payment Agents.
> Target: ICLR 2027 Workshop on Agentic AI Systems / NeurIPS 2026 Workshop on
> Safe and Trustworthy Agents.

Full outline: [paper/acp_paper_outline.md](paper/acp_paper_outline.md)

---

## License

Apache 2.0. The `ach/` package and benchmark infrastructure are open-source.
Gordon fraud pipeline (`fraud/`) requires a separate API key.
