# Commerce-Native Agentic Orchestration: Grounding LLM Execution in Transaction and Payment Theory

**Workshop on Foundations of Agentic Systems Theory (FAST @ NeurIPS 2026)**

---

## Abstract

Existing LLM agent orchestration frameworks — LangGraph, AutoGen, CrewAI — are built from graph execution primitives. They provide typed state, conditional routing, and checkpointing, but their state models are semantically neutral: they know nothing about financial commitments, action reversibility, or audit obligations. When agents are authorized to execute real financial transactions, this semantic gap produces systems that are difficult to reason about formally and vulnerable to a class of attacks we call *payment-protocol violations* — attacks that produce valid LLM outputs but violate the state machine of payment processing infrastructure. We present Commerce-Native Agentic Orchestration (CAO), a theoretical framework that grounds LLM agent execution in four decades of distributed transaction theory — the Saga pattern (Garcia-Molina & Salem, 1987), two-phase commit (Gray, 1978), ISO 8583 payment semantics, and CRDT-style monotonic state (Shapiro et al., 2011). Our central contributions are: (1) a formal finite automaton M over commerce action states grounded in ISO 8583, the x402 AI-payment protocol, and modern payment APIs, whose acceptance language defines the set of valid agent session trajectories; (2) *Commerce-Typed State (CTS)*: a channel-plus-reducer abstraction in which the reducer encodes the safety invariant as a monotone semilattice join, guaranteeing that safety decisions are irreversible under pipeline composition; (3) a *Structural Atomic Gate* that formalizes what `interrupt_before` (LangGraph) cannot provide; and (4) empirical demonstration on a 520-session benchmark. A standalone behavioral ML baseline achieves 74.7% recall at 36.5% hard FPR — the hard FPR driven by genuine distributional overlap between high-value legitimate purchases and marginal overspend attacks. ACP reduces hard FPR to 25.5% while achieving 88.4% recall, using deterministic rules to constrain uncertain ML verdicts — a direct consequence of the CTS monotone semilattice composability property.

---

## 1. Introduction

LLM agents are transitioning from advisory roles to *consequential actors*: systems authorized to book travel, subscribe to software, pay for API services, and execute wire transfers without per-action human approval. This transition demands that agent orchestration frameworks develop a formal account of *what it means to take a financial action* — not just what it means to route between graph nodes.

The gap is not merely pragmatic. Three questions current frameworks cannot answer formally:

1. **Atomicity**: If an agent reserves a hotel and then crashes before committing the flight, is the hotel hold refundable? Under what conditions and within what timeout?
2. **Monotonicity**: If a fraud verifier flags an action as high-risk, can a subsequent verifier in the same pipeline *reduce* the risk score? Should it be able to?
3. **Provenance**: If a disputed transaction is charged back three months later, can the agent operator prove what information the agent acted on, which verifier approved it, and at what confidence level?

Distributed systems theory has answered analogous questions for databases and payment rails for decades. The Saga pattern [1] handles atomicity for long-running transactions. Two-phase commit [2] provides distributed consensus before irreversible commitment. ISO 8583 [3] defines the message lifecycle for financial transactions. Certificate Transparency [4] provides tamper-evident audit logs via hash-chained, Merkle-anchored structures.

These solutions are not immediately applicable to LLM agents — they were designed for deterministic systems. But the formal properties they guarantee (atomicity, monotonicity, tamper-evidence, provenance) are exactly what agentic commerce requires. The theoretical contribution of this paper is to build the bridge.

**Contributions**:

1. **Commerce State Automaton** (§3.0): a formal finite automaton M over ISO 8583-grounded action states, whose valid transitions define the security invariant for agentic payment sessions. Every payment-protocol attack is a trajectory rejected by M or a field-constraint violation on a valid transition.

2. **Commerce-Typed State (§3.1)**: typed channels whose reducers are monotone semilattice joins, encoding safety invariants in the type system rather than in ad hoc application logic.

3. **Structural Atomic Gate (§3.2)**: a formal argument that LangGraph's `interrupt_before` is insufficient as an atomic gate, and how the Commerce Action Receipt (CAR) abstraction achieves proper 2PC-style atomicity.

4. **Ceremony Tiers (§3.3)**: a principled mapping from transaction risk to commitment protocol (Saga vs. 2PC) with a formal cost model.

5. **Empirical validation (§4)**: ACP outperforms rules-only and ML-only baselines on a 520-session benchmark, with mechanistic interpretations grounded in the formal framework.

---

## 2. Background and Related Work

### 2.1 LLM Agent Orchestration Frameworks

**ReAct (Yao et al., 2022)** [5]: alternates reasoning trace generation with tool execution. No state management beyond context window. Each step is a fresh inference over accumulated trajectory. No safety semantic on tool calls.

**LangGraph** [6]: stateful graph execution based on Pregel [7]. State is a typed dictionary with per-channel reducers; `interrupt_before` pauses for human review. Key limitation: reducers are user-specified — the framework imposes no constraints on reducer semantics. `interrupt_before` is a *workflow gate*, not a *fraud gate*: it pauses execution but applies no automated safety signal.

**AutoGen (Wu et al., 2023)** [8]: multi-agent conversational actors with handoff primitives. Safety mechanism: role-based prompt engineering. Does not model financial action lifecycle.

**CrewAI** [9]: role-based orchestration with task schema validation. Provides syntactic validation, no semantic or behavioral verification.

**OpenAI Agents SDK** [10]: guardrails on agent inputs and outputs; handoffs between specialized agents. Guardrails observe text, not payment-rail state.

**Summary**: all existing frameworks operate at the reasoning and routing layer. None models the lifecycle of a financial commitment or the invariants that lifecycle imposes.

### 2.2 Payment Protocol Foundations

**ISO 8583** [3]: international standard for financial transaction message format and lifecycle. Core message types: 0100 (authorization request), 0110 (authorization response), 0200 (financial transaction / commit), 0220 (financial advice / settle), 0420 (reversal). The lifecycle enforces ordering: 0200 cannot precede 0110. This is precisely the invariant that cold-start attacks (B1) violate. The x402 AI-payment protocol [11] maps this lifecycle to HTTP headers for AI-to-AI payments, and documents the revert-grant attack class (B9) that occurs when agents can issue REFUND after SETTLE.

**Idempotency in payment rails**: Stripe, Adyen, and ISO 20022 require idempotency keys on all mutation requests. A request with a previously-used key returns the prior response without re-executing. If the agent verifier does not have access to the settled-key registry, replay attacks (B5) are invisible.

**The agentic gap**: in all prior payment systems, the ISO 8583 state machine was enforced by terminal hardware and issuing bank logic. The human cardholder presses "Pay" on a UI — they cannot skip the authorization step. An LLM agent issuing structured API calls directly to a payment rail can. There is no terminal enforcing M. The agent is the terminal.

### 2.3 Distributed Transaction Theory

**Two-Phase Commit (Gray, 1978)** [2]: coordinator asks all participants to prepare (phase 1); commits if all agree, aborts otherwise (phase 2). Guarantees no participant commits if any participant aborts. Cost: synchronous blocking.

**The Saga Pattern (Garcia-Molina & Salem, 1987)** [1]: sequence T₁, ..., Tₙ where each Tᵢ has compensating transaction Cᵢ. Failure at Tⱼ triggers Cⱼ₋₁, ..., C₁. Trades atomicity for availability. Cost: intermediate states are visible; compensating transactions must be idempotent.

**CRDTs (Shapiro et al., 2011)** [12]: join-semilattice data structures whose operations satisfy commutativity, associativity, and idempotency. Convergence guaranteed: all replicas with the same updates reach the same state regardless of order. The join operation *encodes the invariant* — a G-Counter's join is `max` (can only increase); a 2P-Set's join is `union` (elements only added). The algebraic structure of the join *is* the safety guarantee.

### 2.4 Uncertainty Quantification in Safety-Critical Systems

**Selective Prediction (Geifman & El-Yaniv, 2017)** [13]: a classifier uncertain beyond threshold θ should abstain and defer to a human expert. Applied to agent safety: a verifier that cannot confidently classify an action should escalate rather than approve.

**Conformal Prediction (Angelopoulos & Bates, 2022)** [14]: distribution-free coverage guarantees without distributional assumptions. Note: our current uncertainty estimates are *heuristic* (boundary proximity, OOD boost, cold-start boost) — not true conformal intervals. Production deployment requires a held-out calibration set from the deployment distribution before activating conformal guarantees.

---

## 3. Commerce-Native Agentic Orchestration (CAO)

### 3.0 The Commerce State Automaton

Before defining CTS, we establish the formal grammar that CTS enforces.

**Definition 3.0 (Commerce State Automaton)**: Let M = (Q, Σ, δ, q₀, F) where:
- Q = {START, FIND, QUOTE, RESERVE, COMMIT, SETTLE, VOID, END}
- Σ = {find, quote, reserve, commit, settle, void, refund}
- q₀ = START, F = {END}
- δ: Q × Σ → Q defines valid transitions (Table 1)

**Table 1: Valid Transitions in M**

| From | Action | To | Payment Protocol Grounding |
|------|--------|----|---------------------------|
| START | find | FIND | Session initialization |
| FIND | find | FIND | Iterative discovery (ISO 8583 pre-auth inquiry loop) |
| FIND | quote | QUOTE | Proceed to pricing |
| QUOTE | reserve | RESERVE | Hold funds — 2PC prepare phase |
| QUOTE | commit | COMMIT | Direct commit (Saga, low-risk) |
| RESERVE | commit | COMMIT | Execute after hold — 2PC commit phase |
| RESERVE | void | VOID | Cancel hold — Saga compensating transaction |
| COMMIT | settle | SETTLE | ISO 8583 0220 financial advice |
| COMMIT | refund | VOID | Saga compensating transaction |
| Any | — | END | Session close |

**Definition 3.1 (ConsequentialAction)**: A *consequential action* is a tuple (a, m, r, amt, ccy, ctx, k, p) where a ∈ Σ, m is a merchant identifier with associated MCC, r ∈ {FULL, PARTIAL, NONE} is reversibility, amt ∈ ℝ₊ is the transaction amount, ccy is currency, ctx = (session\_id, agent\_id, persona) is session context, k is an idempotency key, and p is a payload dict. An action is *consequential* iff a ∈ {reserve, commit, settle}.

**Definition 3.2 (Valid Session)**: A session trajectory s = a₁, a₂, ..., aₙ is *protocol-valid* iff:
1. The state sequence q₀, δ(q₀, a₁), δ(q₁, a₂), ... ∈ F is accepted by M (state-transition validity)
2. For all aᵢ with aᵢ.action = commit: aᵢ.amt ≤ wallet.per\_transaction (field validity — spend limit)
3. For all aᵢ with aᵢ.action = commit: aᵢ.mcc ∈ wallet.allowed\_mcc (field validity — MCC allowlist)
4. All idempotency keys k in committed actions are unique in the settled-key registry (field validity — idempotency)
5. ctx.agent\_id matches the registered agent for ctx.session\_id (identity validity)

**Every payment-protocol attack in the taxonomy is a violation of exactly one of these five conditions.** B1 violates condition 1 (START→COMMIT is not in δ). B5 violates condition 4. B7 violates condition 3. B8 violates condition 2. B4 violates condition 5. B2, B3, B6 violate condition 1 or 2 in more subtle ways (count overflow, persona-merchant mismatch, malicious payload).

### 3.1 Commerce-Typed State (CTS)

LangGraph state channels are (type, reducer) pairs. The default reducer is `operator.add` (lists) or last-writer-wins (scalars). We observe that **the reducer can encode a safety invariant as a monotone semilattice join**.

**Definition 3.3 (Commerce-Typed Channel)**: A *commerce-typed channel* is a (type, reducer) pair where the reducer f is a monotone semilattice join over the value domain:
- Associative: f(f(x,y),z) = f(x,f(y,z))
- Commutative: f(x,y) = f(y,x)
- Idempotent: f(x,x) = x
- Monotone: x ≤ f(x,y) for all y

**Proposition 3.1 (CTS channels are CRDTs)**: Every CTS channel is a join-semilattice CRDT. Any two replicas receiving the same updates converge to the same state regardless of order; the channel state is a lower bound that can only grow more committed, never less.

**Concrete CTS channels** for agentic commerce:

```python
spend:      Annotated[Decimal, operator.add]      # monotone: spend_new ≥ spend_old
decision:   Annotated[str, block_wins]             # block > flag > allow; once blocked, stays blocked
provenance: Annotated[list[CARRef], operator.add]  # append-only tamper-evident audit trail
flags:      Annotated[frozenset, frozenset.union]  # monotone: flags never removed
```

where `block_wins(a, b) = "block" if "block" in {a,b} else "flag" if "flag" in {a,b} else "allow"`.

**Theorem 3.1 (Safety monotonicity)**: In a CTS-typed graph state, if any verifier produces decision="block" for any action in a session, the session-level decision is irreversibly "block" regardless of subsequent verifier outputs.

*Proof*: By definition of `block_wins`, f(x, "block") = "block" for all x ∈ {allow, flag, block}. By idempotency, f("block", "block") = "block". By monotonicity, once the decision state reaches "block", no subsequent join can reduce it. □

**Corollary 3.1**: The `block_wins` reducer is a formal mechanism for *fail-closed safety* — the agent pipeline cannot unblock itself by producing subsequent allow decisions. LangGraph's last-writer-wins scalar semantics allows exactly this: a subsequent node can overwrite a prior block if the state is not carefully protected.

**Proposition 3.2 (Rules-certainty veto under CTS)**: In the ACP cascade, the protocol-layer veto (block→flag downgrade when LocalRules score=0 and ML uncertainty>0.20) does not violate Theorem 3.1. Proof: the veto fires only when LocalRules decision="allow" (score=0). By block\_wins, the merged pre-veto decision is ML's block. The veto applies before the session-level reducer, producing a flag output from the ACP node. The session-level block\_wins reducer then joins this flag with any prior decisions — if LocalRules had previously blocked an earlier action, the session-level state remains "block". The veto cannot reduce a decision that is already in the log. □

### 3.2 Structural Atomic Gate

**Proposition 3.3 (`interrupt_before` is not an atomic gate)**: An agent using `interrupt_before=["execute_payment"]` can reach `execute_payment` after: (a) prior nodes executed and modified external state; (b) a human approved the checkpoint — but the checkpoint contains no fraud-semantic signal (no wallet state, no idempotency check, no action sequence history); (c) the graph resumes with no further safety check. The human sees graph state, not protocol-validity state.

**Definition 3.4 (Structural Atomic Gate)**: A node G in the execution graph is a *structural atomic gate* iff:
- (a) G executes synchronously before any consequential action reaches the payment rail
- (b) G produces a signed Commerce Action Receipt (CAR) with a `final_decision` field
- (c) The graph has no path from G to the payment rail bypassing `final_decision`
- (d) G's output is persisted to the tamper-evident CARLog before the payment rail executes

Properties (a)–(d) formalize the 2PC coordinator role: G is the coordinator, the payment rail is the participant. No participant commits without coordinator approval; the decision is durably logged before participant execution.

**Definition 3.5 (Commerce Action Receipt, CAR)**: A CAR is a tuple (id, action, verifications, decision, level, anchoring, sig, ts) where: `verifications` lists (verifier\_id, score, confidence, uncertainty, score\_interval); `decision` ∈ {allow, flag, block}; `level` ∈ {L0, L1, L2, L3}; `anchoring` = (log\_hash, prev\_hash, merkle\_root, chain\_tx); `sig` = HMAC-SHA256 over decision-relevant fields; `ts` = wall clock at decision time.

**Proposition 3.4**: The ACP ceremony node satisfies Definition 3.4: (a) ceremony routing is synchronous (<10ms); (b) CAR is signed before payment routing; (c) routing is conditional on `final_decision`; (d) CAR is appended to the hash-chained CARLog before the payment node executes.

### 3.3 Ceremony Tiers: Risk-Adjusted Commitment Protocols

| Level | Action / Condition | Protocol | Anchoring | Justification |
|-------|--------------------|----------|-----------|---------------|
| L0 | FIND, QUOTE | None | None | Read-only; no financial consequence |
| L1 | COMMIT, amt < $50, unc < 0.25 | Saga | Local hash-chain | Compensable; REFUND available |
| L2 | COMMIT, $50–$5000, or unc ≥ 0.25 | 2PC-lite (RESERVE before COMMIT) | Merkle batch | Non-trivial commitment; prepare phase required |
| L3 | COMMIT, amt > $5000, or unc ≥ 0.45, or multi-verifier disagreement | 2PC + multi-verifier consensus | On-chain | High-finality; single-coordinator failure unacceptable |

**Cost model**: L0/L1 verification <$0.0001/action; L2 Merkle anchoring ~$0.0001/batch; L3 on-chain ~$0.10/transaction. Flat L3 on all actions would cost ~$0.10 × (daily action count) — prohibitive at scale. The tier system is a risk-adjusted verification budget.

### 3.4 Tamper-Evident Provenance via Hash-Chained CARLog

Adapting Certificate Transparency [4] to the CAR audit problem:

```
entry_n = {
    seq:             n,
    car_id:          unique identifier,
    car_hash:        SHA256(CAR_payload),
    prev_hash:       SHA256(entry_{n-1}) if n > 0 else "genesis",
    entry_hash:      SHA256(seq || car_id || car_hash || prev_hash || ts),
    entry_signature: HMAC-SHA256(entry_hash, signing_key)
}
```

**Theorem 3.2 (Tamper-evidence)**: Any modification of entry_k changes entry_k.entry\_hash, which changes entry_{k+1}.prev\_hash, breaking all subsequent entry signatures. A chain scan detects modification in O(N) time.

**Merkle batch anchoring**: every B CARs are batched into a Merkle tree; the root is written to a public blockchain (Ethereum OP\_RETURN or Solana memo). An inclusion proof for CAR\_k consists of O(log B) sibling hashes sufficient to recompute the root. The proof is (a) bound to a specific block timestamp, (b) verifiable by any third party without trusting the operator, (c) private with respect to other CARs in the batch.

**Chargeback application**: for a disputed transaction, the operator produces (CAR, inclusion proof, blockchain anchor). This establishes "decision D was made by verifier V at confidence C at time T, before this dispute was filed" — cryptographically binding and publicly verifiable.

### 3.5 Confidence Cascade as Selective Prediction

```
Tier 1 (LocalRulesVerifier):  confidence = 1.0, uncertainty = 0.0
   → rule fires: block, no delegation
   → rule passes (score=0): delegate to Tier 2

Tier 2 (BehavioralML):        confidence = 1 - uncertainty_heuristic
   → uncertainty_heuristic: 0.30 if |score - threshold| < 0.12, else 0.08
   → decision: block / flag / allow at μ+1.1σ threshold
   → if score=0 in Tier 1 and uncertainty > 0.20: downgrade block → flag (rules-certainty veto)

Tier 3 (LLM Verifier):        invoked for <3% of actions (high uncertainty)
   → natural language reasoning over session summary
   → structured decision with reasoning trace
```

**Proposition 3.5 (Cascade monotonicity)**: Under the CTS `block_wins` reducer, adding Tier 2 or Tier 3 cannot reduce a block from a prior tier, and can only add flags or blocks. The cascade is monotone in the safety direction.

---

## 4. Experiments

We evaluate CAO/ACP against four baselines on 520 synthetic sessions (seed=42, 200 clean + 40×8 attacks). Full experimental details in the companion benchmark paper [15].

### 4.1 Formal Property Verification

**Safety monotonicity (Theorem 3.1)**: In 1,000 random session orderings (permuting verifier execution order), session-level decisions are identical in 100% of cases. The `block_wins` reducer is order-independent by construction.

**Tamper-evidence (Theorem 3.2)**: Random byte flips at 10 positions in a 172-entry CARLog. `log.verify()` detects 10/10 corruptions in O(N) time.

**Inclusion proof validity**: Merkle inclusion proofs verified for 172/172 entries. Proof generation and verification both O(log N).

### 4.2 Attack Detection

| Approach | F1 | Recall | Hard FPR | EscRate | Protocol Attacks (B1,B7,B8) |
|---|---|---|---|---|---|
| No verification (LangChain) | 0.000 | 0.000 | 0.000 | 0.000 | 0% |
| Schema only (LangGraph) | 0.000 | 0.000 | 0.000 | 0.000 | 0% |
| Rules only (CrewAI-equivalent) | 0.584 | 0.412 | **0.000** | **0.000** | **100%** |
| ML only | 0.690 | 0.747 | 0.365 | 0.305 | 64% |
| **ACP (CAO)** | **0.768** | **0.884** | 0.255 | 0.415 | **100%** |

*Hard FPR = auto-block on clean; EscRate = flag-for-review on clean. Benchmark includes ~14% high-value legitimate sessions creating genuine distributional overlap with B8 marginal attacks.*

**Rules and ML are complementary — neither alone is deployable.**

Rules: 0% hard FPR, 0% escalation, 58.8% FNR. B2, B4, B5, B6 are invisible to deterministic rules without session-level behavioral signal.

ML: 74.7% recall, 36.5% hard FPR. The hard FPR stems from distributional overlap: high-value legitimate travel purchases ($1,800–$2,800) overlap with marginal B8 overspend attacks ($3,010–$4,500) in the log-amount feature space. The ML's OOD amount detector cannot perfectly separate them at any threshold without simultaneously degrading recall.

ACP: 88.4% recall, 25.5% hard FPR. The Pareto improvement follows from the CTS composability property: when LocalRules score=0 (Proposition 3.2), the rules-certainty veto converts ML hard-blocks to flags without violating safety monotonicity. The 22 converted sessions have LocalRules score=0 — "no deterministic protocol violation observed" — providing the certainty evidence the ML lacks. The residual 51 hard-blocks have LocalRules score>0 (near-limit amounts) with both verifiers agreeing — genuine overlap, not calibration artifact.

### 4.3 Uncertainty Calibration

| Bucket | N | Escalation | FP | FN |
|--------|---|------------|----|----|
| Low unc (<0.20) | 15 | 0% | 0 | 0 |
| Mid unc (0.20–0.40) | 75 | 24% | 0 | 37 |
| High unc (>0.40) | 430 | 100% | 134 | 0 |

All 37 missed fraud sessions (FN) concentrate in the mid-uncertainty bucket — the model signals uncertainty on exactly the hard cases, consistent with selective prediction. All 134 false positives concentrate in high uncertainty — the distributional-overlap cases are flagged as uncertain rather than confidently mislabeled. In production, Tier 3 LLM review of mid-uncertainty sessions would substantially reduce effective FNR.

---

## 5. Discussion

### 5.1 Theoretical Relationship to Prior Work

**CAO vs. Saga**: Saga says "how to undo." CAO says "whether to proceed" — the ceremony gate must approve before any action is committed, even one with a compensating transaction.

**CAO vs. 2PC**: 2PC provides consensus among participants. CAO adds a *pre-commit verification gate*: the agent verifier acts as coordinator, approving before the payment rail (participant) executes its own 2PC. CAO does not replace 2PC in the rail — it layers above it.

**CTS vs. LangGraph state**: LangGraph reducers are unconstrained. CTS constrains reducers to monotone semilattice joins, providing the formal guarantee that safety decisions are irreversible — a strict strengthening of LangGraph's model.

**M vs. prior state machine work**: the commerce state automaton M is not a new theoretical construct — it is a formalization of what ISO 8583 has enforced implicitly for decades. The novelty is applying it as a security invariant for agentic systems where no terminal hardware enforces the grammar.

### 5.2 Limitations

**Heuristic uncertainty**: current uncertainty estimates are boundary-proximity heuristics, not true conformal intervals. Production deployment requires a held-out calibration set from the deployment distribution to activate conformal coverage guarantees.

**Residual hard FPR (25.5%)**: cases where both verifiers detect signal — LocalRules score>0 (near-limit amounts) and ML above threshold. Eliminating this requires distribution-aware calibration that incorporates amount-threshold geometry directly into the uncertainty estimate. We frame this as future work.

**B4/B5 partial detection**: B4 (session hijack) at 68% — cross-session identity tracking requires external agent-ID registry not in this release. B5 (replay) at 70% via ML behavioral signals — the robust mechanism requires a settled-key registry; current detection is via correlated sequence anomalies.

**L3 consensus**: the multi-verifier consensus protocol for L3 actions is specified but not implemented. A Byzantine-fault-tolerant protocol for the multi-verifier case is future work.

### 5.3 Extensions

**A2A delegation and sub-agent scope**: as agents delegate to sub-agents (OpenAI Agents SDK handoffs, AutoGen delegation), the CAR provenance model should track the full delegation chain. A sub-agent's COMMIT should be CARed under the delegating agent's wallet with explicit scope constraints — preventing a sub-agent from exercising authorization scope it was not granted.

**Financial trading and crypto**: the automaton M and the CTS framework are asset-class agnostic. Equity trading agents have an analogous lifecycle (QUOTE\_PRICE → PLACE\_ORDER → FILL → CONFIRM → CLEAR → SETTLE) with a different field-constraint vocabulary (instrument allowlist, position size, leverage ratio). Crypto agents require `Reversibility=NONE` at COMMIT time (irreversibility at broadcast, not at settlement) and a chain-ID field for cross-chain replay prevention.

**Observer-mode deployment**: ACP currently assumes it is inline in the payment path (it can block). A future extension defines an observer mode for deployments where the MCP server observes but does not intermediate — producing risk scores, informing the agent builder via webhook, and generating evidence packages for chargeback support without requiring inline interception.

---

## 6. Conclusion

Agentic commerce requires orchestration frameworks to develop the formal guarantees that distributed transaction systems have provided for four decades. We have shown that the commerce state automaton M — grounded in ISO 8583, x402, and modern payment APIs — defines the security invariant for agentic payment sessions; that CTS reducers encode this invariant in the type system via monotone semilattice joins; that the structural atomic gate formalizes what `interrupt_before` cannot provide; and that the confidence cascade implements selective prediction as a risk-adjusted verification budget.

The transition from semantically neutral graph execution to commerce-typed execution with invariant-enforcing reducers is a qualitative shift in what can be formally guaranteed about agent behavior. As agents are authorized to take increasingly consequential financial actions — and as the gap between reasoning-layer tools and protocol-layer threats becomes operationally important — the theoretical foundations developed here become not optional but necessary.

---

## References

[1] Garcia-Molina, H., & Salem, K. (1987). Sagas. *ACM SIGMOD Record*, 16(3), 249–259.

[2] Gray, J. N. (1978). Notes on data base operating systems. *Operating Systems: An Advanced Course*, 393–481.

[3] ISO/IEC 8583:2021. *Financial transaction card originated messages: Interchange message specifications*. ISO.

[4] Laurie, B., Langley, A., & Kasper, E. (2013). Certificate Transparency. *RFC 6962*. IETF.

[5] Yao, S., et al. (2022). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*.

[6] LangChain Inc. LangGraph. https://langchain-ai.github.io/langgraph/, 2024.

[7] Malewicz, G., et al. (2010). Pregel: A system for large-scale graph processing. *ACM SIGMOD*, 135–146.

[8] Wu, Q., et al. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv:2308.08155*.

[9] CrewAI Inc. CrewAI. https://github.com/crewAIInc/crewAI, 2024.

[10] OpenAI. OpenAI Agents SDK. https://platform.openai.com/docs/guides/agents, 2025.

[11] Li, et al. "Security Analysis of the x402 AI Payment Protocol." *arXiv:2605.11781*, 2026.

[12] Shapiro, M., et al. (2011). Conflict-free replicated data types. *SSS 2011*, 386–400.

[13] Geifman, Y., & El-Yaniv, R. (2017). Selective prediction in deep neural networks. *NeurIPS 2017*.

[14] Angelopoulos, A. N., & Bates, S. (2022). A gentle introduction to conformal prediction. *arXiv:2107.07511*.

[15] [Authors]. ACP-Bench: Measuring Safety at the Payment-Protocol Layer of Agentic Commerce. *AIWILD @ NeurIPS 2026*. (companion paper)

[16] Hewitt, C., Bishop, P., & Steiger, R. (1973). A universal modular ACTOR formalism. *IJCAI*, 235–245.

[17] Pritchett, D. (2008). BASE: An acid alternative. *ACM Queue*, 6(3), 48–55.

[18] Debenedetti, E., et al. (2024). AgentDojo: A dynamic environment to evaluate attacks and defenses for LLM agents. *NeurIPS 2024*.
