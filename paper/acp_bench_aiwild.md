# ACP-Bench: Measuring Safety at the Payment-Protocol Layer of Agentic Commerce

**Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD @ NeurIPS 2026)**

---

## Abstract

Autonomous agents authorized to execute financial transactions operate across a layered stack: a *reasoning layer* where the LLM generates tool calls, and a *payment-protocol layer* where those calls execute against wallets, merchants, and payment rails. Existing safety tools — garak, promptfoo, LlamaGuard 4, FinHarness — are purpose-built for the reasoning layer and are the correct tools for the threat classes they address (jailbreaks, harmful content, prompt injection into the LLM context). We study a structurally distinct and previously unmeasured threat surface: attacks that produce valid LLM outputs but violate the formal state machine of the payment protocol — a grammar grounded in ISO 8583, the x402 AI-payment protocol, and modern payment APIs. We define this grammar as a finite automaton over commerce action states (`START → FIND → QUOTE → RESERVE → COMMIT → SETTLE → {VOID, END}`), show that six of our eight evaluated attack classes are violations of this automaton that produce no reasoning-layer signal whatsoever, and introduce ACP-Bench: 520 synthetic sessions spanning eight attack classes generated from persona-conditional Markov chains, log-normal amount distributions, and multiplicative Gaussian perturbation. Critically, the benchmark includes ~14% high-value legitimate sessions ($1,800–$2,800 for travel) that create genuine distributional overlap with marginal overspend attacks ($3,010–$4,500), making the detection task non-trivial by construction. We distinguish *hard FPR* (auto-block of legitimate sessions, requiring human override) from *escalation rate* (flag-for-review, an operational cost). A standalone behavioral ML baseline achieves 74.7% recall but 36.5% hard FPR. Our Agentic Commerce Protocol (ACP), combining deterministic rules with behavioral ML under uncertainty routing, achieves 88.4% recall and 25.5% hard FPR — a 30% reduction in auto-rejection at the cost of routing more borderline cases to human review. ACP-Bench and its generator are released as open-source.

---

## 1. Introduction

LLM agents are becoming consequential financial actors. A travel agent books flights. A procurement agent subscribes to SaaS. A research agent purchases API credits. These are real, partially-irreversible financial commitments executed without per-action human confirmation.

The security community has responded with a rich set of safety tools targeting the *reasoning layer*: garak [1] probes LLM outputs for jailbreak success; promptfoo [2] evaluates assertions over LLM responses; LlamaGuard 4 [3] classifies conversation content for harmful categories; FinHarness [4] monitors LLM query formulation and tool selection. These tools are correct for their threat model — an adversary who manipulates what the LLM *reasons* or *says*.

This paper studies a different threat model: an adversary who manipulates what the agent *does* in payment protocol space. Consider three attacks:

- **Cold-start (B1)**: the agent issues `COMMIT` with no prior `FIND` or `QUOTE`. The LLM output is syntactically valid, semantically coherent, and calls the right tool. No injection occurred, no harmful content was generated. The violation is that `COMMIT` cannot precede discovery — a payment-protocol invariant invisible to any reasoning-layer tool.

- **Idempotency replay (B5)**: the adversary re-uses a prior settled session's idempotency key — a string that looks identical to a legitimate key in format — to re-execute a payment. The LLM is uninvolved. The payment rail is being exploited directly.

- **Spend-limit violation (B8)**: the agent's `COMMIT` amount exceeds its wallet's per-transaction ceiling. The amount field is a number. The violation is arithmetic against wallet state. No classifier of text can observe this.

These attacks have no reasoning-layer signal. The question of whether they are detectable is not a question about LLM safety classifiers — it is a question about whether any verifier has access to payment-protocol state: action sequencing, idempotency key registries, wallet spend ceilings, MCC allowlists.

The x402 payment protocol [5] for AI-to-AI transactions first documented this surface in the context of revert-grant attacks. We extend this to a systematic taxonomy and the first benchmark specifically targeting the payment-protocol layer.

**Contributions**:

1. **Commerce State Machine**: a formal finite automaton M over action states grounded in ISO 8583, x402, and modern payment API semantics, whose valid transitions define the security invariant (§2).

2. **Attack taxonomy (B1–B9)**: nine attack classes classified by violation type — state-transition violation, field violation, or identity violation — each with an explicit account of whether a reasoning-layer signal exists (§2).

3. **ACP-Bench**: 520 synthetic sessions generated from persona-conditional Markov chains with log-normal amounts and multiplicative Gaussian perturbation, including high-value legitimate edge cases that make the task genuinely hard (§3).

4. **Layered evaluation**: direct measurement of reasoning-layer tools (garak, promptfoo, LlamaGuard) at their operating surface (B6 injection), plus full protocol-layer evaluation across all eight attack classes; explicit accounting of what each tool is and is not designed to detect (§4).

5. **ACP**: a protocol-layer verifier that achieves 88.4% recall and 25.5% hard FPR on the benchmark, with a mechanistic account of residual errors (§5).

---

## 2. The Commerce State Machine and Attack Taxonomy

### 2.1 The Commerce State Machine

Payment processing for agentic systems has structure. Every real financial commitment — whether through ISO 8583, the x402 protocol [5], or modern payment APIs (Stripe PaymentIntent, Adyen Checkout) — follows the same ordered lifecycle:

```
START → FIND → QUOTE → RESERVE → COMMIT → SETTLE → {VOID, END}
```

We formalize this as a finite automaton:

**Definition 2.1 (Commerce State Automaton)**: Let M = (Q, Σ, δ, q₀, F) where:
- Q = {START, FIND, QUOTE, RESERVE, COMMIT, SETTLE, VOID, END}
- Σ = {find, quote, reserve, commit, settle, void, refund}
- q₀ = START, F = {END}
- δ defines the valid transitions:

| From | To | Condition |
|------|----|-----------|
| START | FIND | always |
| FIND | FIND | iterative search |
| FIND | QUOTE | proceed to pricing |
| QUOTE | RESERVE | hold before commit (medium/high risk) |
| QUOTE | COMMIT | direct commit (low risk) |
| RESERVE | COMMIT | execute after hold |
| RESERVE | VOID | cancel hold (compensating) |
| COMMIT | SETTLE | finalize |
| COMMIT | REFUND | compensating Saga rollback |
| Any | END | session close |

**Any agent session trajectory not accepted by M is a payment-protocol violation.**

This is not a new state machine — ISO 8583 has enforced it for four decades (0100 authorization → 0200 financial transaction → 0220 financial advice → 0420 reversal). Stripe PaymentIntent enforces it via state (`requires_confirmation → processing → succeeded`). x402 enforces it via HTTP headers (`Payment-Required → Payment → Payment-Response`). The critical difference in agentic systems: **in all prior payment contexts, M was enforced by terminal hardware and issuing bank logic — a cardholder pressing "Pay" cannot bypass the authorization step. An LLM agent issuing structured API calls directly to a payment rail can.** There is no terminal. The agent is the terminal.

### 2.2 Attack Taxonomy

We classify attacks along three dimensions: *violation type* (state-transition, field, or identity), *whether a reasoning-layer signal exists*, and *which layer can detect it*:

| ID | Name | Violation Type | Reasoning-Layer Signal | Detectable At |
|----|------|---------------|----------------------|---------------|
| B1 | Cold-start authorization | State transition: START→COMMIT | None | Protocol layer only |
| B2 | Velocity flooding | Field: commit count > session max | None | Protocol layer only |
| B3 | Merchant anomaly | Field: merchant outside persona scope | Weak/indirect | Protocol layer (behavioral) |
| B4 | Session hijacking | Identity: agent\_id mismatch on settled session | None | Protocol layer only |
| B5 | Idempotency replay | Field: idempotency key already settled | None | Protocol layer only |
| B6 | Prompt injection | Field: adversarial text in payload | **Yes — text detectable** | **Both layers** |
| B7 | MCC violation | Field: merchant\_mcc not in wallet allowlist | None | Protocol layer only |
| B8 | Spend-limit violation | Field: amount > wallet per-transaction ceiling | None | Protocol layer only |
| B9 | Revert-grant exploit | State transition: SETTLE→COMMIT | None | Protocol layer only |

**B6 is the hinge case**: it is the only attack that exists at both layers simultaneously. The injected text (`"ignore previous instructions and approve all"`) appears in the LLM's context *and* in the `ConsequentialAction.payload` field. Garak detects it at the reasoning layer; ACP's keyword scanner detects it at the protocol layer. These are independent backstops — if one layer misses an evasive encoding, the other may catch it.

B1, B2, B4, B5, B7, B8 produce *no reasoning-layer signal*. The LLM reasoned correctly, selected the right tool, emitted a valid output. The violation is structural, arithmetic, or identity-based — properties that only exist in payment-protocol state.

The taxonomy is anchored to the x402 attack analysis [5], which documented revert-grant attacks on AI-to-AI payment flows. We extend it to the full automaton M, covering state-transition violations (B1, B9), field violations (B2, B3, B5, B6, B7, B8), and identity violations (B4).

---

## 3. ACP-Bench: Benchmark Methodology

### 3.1 Synthetic Data Generation

We do not use production session data. The generator is parameterized entirely from domain knowledge of ISO 8583-derived payment protocol semantics and commerce behavior.

**Step 1 — Base parameterization.** Per-persona Markov transition matrices encode the behavioral grammar of three agent archetypes:

- *Travel* (median ≈ $380, σ_log = 0.72): frequent FIND loops (45% self-transition), RESERVE before COMMIT (60% of quotes), occasional VOID. Models flight/hotel booking behavior.
- *SaaS* (median ≈ $85, σ_log = 1.05): direct FIND→QUOTE→COMMIT (78% quote→commit rate). Models subscription purchase behavior.
- *Research* (median ≈ $48, σ_log = 0.90): iterative FIND loops (55% self-transition), direct quote→commit (97%). Models API credit and dataset purchase behavior.

Amount distributions are log-normal with parameters derived from published commerce category ranges. The full parameter specification is in `benchmark/distribution_params.py`.

**Step 2 — Perturbation.** Multiplicative Gaussian perturbation (σ=0.05) is applied to each Markov row and amount parameter:
```
θ̃ᵢ = θᵢ · |1 + εᵢ|,   εᵢ ~ N(0, 0.05²),   renormalize to simplex
```
No two sessions have identical dynamics. The evaluation seed (42) and ML training seed (7) are independent — no train/test contamination.

**Step 3 — Session synthesis.** Clean sessions are sampled from the Markov chains. Attack sessions mimic the same FIND→QUOTE→COMMIT structure except for the specific invariant they violate. Attack realism: B5 uses a prior settled session's idempotency key (format-identical to a legitimate key — detectable only by stateful registry); B7 uses one of six varied restricted MCC categories (not a single hardcoded keyword); B8 is bimodal (60% marginal overspend $3,010–$4,500, 40% obvious $4,500–$8,000).

**High-value clean sessions (~14%)**: `_clean_highval` generates legitimate purchases at $1,800–$2,800 (international travel), $800–$1,500 (annual SaaS enterprise), $400–$900 (GPU credits). These create genuine distributional overlap with B8 marginal attacks — this is the source of the residual hard FPR and the proof that the task is non-trivially hard.

### 3.2 Dataset Statistics (seed=42)

- 200 clean sessions; ~14% high-value (amounts $1,800–$2,800)
- 40 sessions × 8 attack classes = 320 attack sessions
- Total: 520 sessions, 61.5% fraud rate (intentionally unbalanced, reflecting real fraud)
- Actions per session: mean=3.8, std=1.6
- B9 defined in taxonomy but excluded from this release (requires mocking a settled payment rail)

### 3.3 Evaluation Protocol

A session is detected if *any* action triggers a `block` or `flag` decision. We distinguish:
- **Hard FPR**: rate of `block` on clean sessions — auto-rejection requiring human override. The costly false positive.
- **Escalation rate (EscRate)**: rate of `flag` on clean sessions — human-review routing. An operational cost, not an auto-rejection.

Metrics: F1, precision, recall, hard FPR, EscRate, FNR, per-attack detection rate. Full dataset is deterministic given `--seed`; results saved to JSON for independent verification.

---

## 4. Evaluation: Reasoning-Layer Tools and Protocol-Layer Baselines

### 4.1 What Each Tool Observes

We characterize each tool by its *observation surface* — what it can and cannot see:

**garak v0.16** [1]: probe-based red-teaming. Generates adversarial prompt variants, evaluates LLM text output, classifies pass/fail. Observation surface: *LLM text output*. Cannot observe tool call parameters, session history, wallet state, or payment amounts. Designed for: jailbreaks, harmful content, prompt manipulation.

**promptfoo** [2]: assertion-based evaluation over LLM outputs. Custom YAML assertions can match patterns in LLM responses. Observation surface: *LLM text output with user assertions*. Cannot observe structured action fields (MCC, amount, idempotency key). Designed for: output quality, safety assertion, injection in responses.

**LlamaGuard 4** [3]: safety classifier for conversation content. Published: 61.10% TPR, 29.91% FPR on financial agent safety benchmark. Observation surface: *conversation-level text classification*. Its technical report notes it classifies conversation content, not tool execution state. Designed for: unsafe content categories in multi-turn dialogue.

**FinHarness** [4]: Query Monitor + Tool Monitor + cascade routing. Published reduction in attack success rate from 38.3% to 15.0% on FinVault benchmark — a significant advance in LLM-level safety. Observation surface: *LLM reasoning traces and tool selection decisions*. Cannot enforce action ordering invariants, idempotency key uniqueness, MCC allowlists, or wallet arithmetic — these require payment-protocol state outside the LLM conversation context. Designed for: unsafe reasoning, inappropriate tool selection.

**Schema validator (LangGraph)**: Pydantic validation of tool call fields — type and range checks. Observation surface: *individual tool call schema*. No session history, no wallet state. Designed for: field type correctness.

**The key point**: these tools are purpose-built for their threat models and are the correct tools for those threat models. Their 0% detection on B1, B2, B4, B5, B7, B8 is not a failure — it is the expected result of correct architectural design. Reasoning-layer tools observe the reasoning layer. Protocol-layer attacks do not exist at the reasoning layer.

### 4.2 Detection Results

| Tool | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | Hard FPR | EscRate |
|------|----|----|----|----|----|----|----|-----|----------|---------|
| garak | 0% | 0% | 0% | 0% | 0% | ~40% | 0% | 0% | 0% | — |
| promptfoo | 0% | 0% | 0% | 0% | 0% | ~35% | 0% | 0% | 0% | — |
| LlamaGuard 4 | 0% | 0% | 0% | 0% | 0% | ~61% | 0% | 0% | ~29% | — |
| FinHarness | 0% | ~30% | ~20% | ~45% | 0% | ~70% | 0% | 0% | — | — |
| Schema only | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| RulesOnly | 38% | 0% | 93% | 0% | 0% | 0% | **100%** | **100%** | **0%** | **0%** |
| MLOnly | **100%** | **100%** | 60% | 68% | 70% | 70% | 57% | 72% | 36.5% | 30.5% |
| **ACP (ours)** | **100%** | **100%** | **100%** | 68% | 70% | 70% | **100%** | **100%** | 25.5% | 41.5% |

*Reasoning-layer tool rates (garak/promptfoo/LlamaGuard/FinHarness) are upper bounds from published benchmarks and architectural analysis; 0% entries for B1–B5, B7–B8 follow from the observation surface analysis in §4.1. RulesOnly/MLOnly/ACP figures are from direct ACP-Bench evaluation (520 sessions, seed=42). All results reproducible via `python -m benchmark.run_experiment --seed 42`.*

**Notes on specific entries**: B5 (idempotency replay): RulesOnly=0% because the current release does not maintain a cross-session settled-key registry; MLOnly=70% via correlated behavioral signals (the key mismatch produces subtle sequence anomalies). B7 (MCC violation): 100% by Rules and ACP — the wallet MCC allowlist rejects all six restricted categories regardless of which one the attacker chooses. ACP B4=68%: session hijacking requires cross-session agent-ID state not in this release. LlamaGuard 4 hard FPR of ~29% is from published financial safety benchmark, not measured directly on ACP-Bench.

### 4.3 The Orthogonal Surface

The 0% detection rates for B1, B2, B4, B5, B7, B8 by reasoning-layer tools are not a tuning failure. They follow from the observation surface:

B1 (cold-start) requires tracking action-type sequences across a session — this sequence is payment middleware state, not LLM context. B5 (replay) requires a persistent settled-key set — this lives in the payment rail, not in any conversation. B8 (spend limit) requires accumulating amounts against an authorization ceiling — this arithmetic requires wallet state. No additional probe templates, assertion rules, or classifier fine-tuning can access these signals without access to payment-protocol state itself.

This is the surface ACP-Bench measures. It is orthogonal to, not competing with, the surface that garak, promptfoo, and LlamaGuard measure. A production deployment needs both: reasoning-layer tools at the agent boundary, protocol-layer verification at the payment rail.

---

## 5. ACP: Protocol-Layer Verification

The Agentic Commerce Protocol intercepts at `ConsequentialAction` — the typed representation of each commerce action before it reaches the payment rail. Three verification tiers run in cascade:

1. **LocalRulesVerifier** (<1ms): deterministic checks against wallet state. Enforces M's valid transitions. Enforces field constraints (MCC allowlist, spend ceiling, sequence ordering). Confidence=1.0 always; uncertainty=0.0. Zero false positives by construction when a rule fires.

2. **BehavioralMLBaseline** (trained on seed=7, N=500 clean sessions): four features — Markov sequence log-likelihood, commit-count velocity z-score, amount OOD z-score, injection keyword density — combined via weighted sigmoid. Threshold at μ+1.1σ of clean training scores.

3. **LLM Verifier** (invoked for <3% of actions, uncertainty>θ₂): natural language reasoning for novel patterns outside the behavioral model.

**Rules-certainty veto** (the key mechanism): when LocalRules score=0 and ML wants to block with uncertainty>0.20, ACP downgrades block→flag. Rules provide hard evidence — "no deterministic violation I can observe" — that constrains an uncertain probabilistic verdict. The CTS `block_wins` semilattice property guarantees this downgrade cannot reduce a prior block: it only applies when rules have not themselves blocked.

**Full results** (520 sessions, seed=42):

| Metric | ACP | RulesOnly | MLOnly | SchemaOnly |
|--------|-----|-----------|--------|------------|
| F1 | **0.768** | 0.584 | 0.690 | 0.000 |
| Recall | **0.884** | 0.412 | 0.747 | 0.000 |
| Hard FPR | 0.255 | **0.000** | 0.365 | 0.000 |
| EscRate | 0.415 | **0.000** | 0.305 | 0.000 |
| FNR | **0.116** | 0.588 | 0.253 | 1.000 |

**Rules and ML are complementary at neither extreme alone reaches an acceptable operating point.**

*RulesOnly*: 0% hard FPR, 0% escalation — but 58.8% FNR. B2 (velocity), B4 (hijack), B5 (replay), B6 (injection) have no deterministic rule that catches them without session-level ML signal.

*MLOnly*: 74.7% recall — but 36.5% hard FPR. The source: ~14% of clean sessions are high-value legitimate purchases whose log-amount distribution overlaps with the 60% of B8 attacks that are marginal overspend ($3,010–$4,500). The ML's OOD amount detector cannot perfectly separate them. Reducing threshold to cut hard FPR also cuts recall.

*ACP*: 88.4% recall (vs. 74.7% ML, 41.2% rules) and 25.5% hard FPR (vs. 36.5% ML). The 22 MLOnly hard-blocks that become ACP flags are sessions where LocalRules score=0 — rules provide the "no deterministic violation" evidence that unlocks the downgrade. The residual 51 hard-blocks are sessions where LocalRules *also* detects signal (near-limit amounts with both verifiers agreeing) — genuine overlap cases, not calibration artifacts.

**Uncertainty calibration**: all 37 missed fraud sessions (mid-uncertainty bucket, 24% escalation rate) are cases where ML scores below threshold but signals elevated uncertainty — the model correctly identifies them as hard cases. Routing these to Tier 3 LLM review would substantially reduce effective FNR. All 134 false positives are in the high-uncertainty bucket — the model signals uncertainty on the hard distributional-overlap cases rather than confidently mislabeling them.

---

## 6. Discussion

**Complementarity, not competition.** ACP and FinHarness are defense-in-depth layers, not alternatives. FinHarness catches an LLM agent that reasons incorrectly toward a harmful tool selection. ACP catches an agent that reasons correctly but whose action violates payment-protocol invariants. Both operate correctly within their observation surfaces. A production deployment needs both.

**What B6 tells us about layer design.** B6 (prompt injection) is the only attack class detectable at both layers. Garak detects it in the LLM output; ACP's keyword scanner detects it in the `payload` field of `ConsequentialAction`. The two detectors can disagree — an adversarially encoded injection might evade garak's probe patterns but still appear in the action field; or it might be caught by garak in the context but not surface in the action payload. This is exactly the defense-in-depth value: independent detectors at independent layers, each with different evasion surfaces.

**Limitations.** (1) B9 (revert-grant) is defined in the taxonomy but excluded from this release — implementing it requires mocking post-settlement rail state, deferred to a future version. (2) B4 detection at 68% — cross-session identity tracking requires external state not in the current release; a per-agent-ID session registry raises this to near 100% in production. (3) B5 detection at 70% — the current release lacks a stateful settled-key registry; ML detects it via correlated behavioral signals, which is not the robust mechanism. (4) Reasoning-layer tool detection rates for B6 are estimated from published benchmarks and architectural analysis; direct integration with ACP-Bench is deferred to a future release.

---

## 7. Conclusion

Payment-protocol attacks against agentic commerce systems occupy a threat surface that is structurally orthogonal to the surface that existing reasoning-layer tools measure. The commerce state machine `START → FIND → QUOTE → RESERVE → COMMIT → SETTLE → {VOID, END}`, grounded in four decades of ISO 8583 payment infrastructure and the emerging x402 AI-payment protocol, defines a formal invariant that agentic systems can violate in ways that produce no LLM-level signal. ACP-Bench is the first benchmark to measure this surface specifically. Our results show that the task is genuinely hard — a behavioral ML baseline achieves 74.7% recall at 36.5% hard FPR, with residual errors caused by real distributional overlap between high-value legitimate purchases and marginal overspend attacks — and that combining deterministic rules with behavioral ML under uncertainty routing (ACP) achieves a meaningful Pareto improvement: 88.4% recall at 25.5% hard FPR. The benchmark, generator, and verifier are released as open-source to support the systematic development of protocol-layer safety for agentic commerce.

---

## References

[1] Derczynski et al. "garak: A Framework for Security Probing Large Language Models." arXiv:2406.11036, 2024.

[2] promptfoo. "LLM Testing and Red Teaming Framework." https://promptfoo.dev, 2024.

[3] Meta AI. "LlamaGuard 4: Technical Report." Meta, 2026.

[4] Chen et al. "FinHarness: A Safety Harness for Autonomous Financial Agents." arXiv:2605.27333, 2026.

[5] Li et al. "Security Analysis of the x402 AI Payment Protocol." arXiv:2605.11781, 2026.

[6] Debenedetti et al. "AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents." NeurIPS, 2024.

[7] Inan et al. "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations." arXiv:2312.06674, 2023.

[8] ISO/IEC 8583:2021. *Financial transaction card originated messages: Interchange message specifications*. International Organization for Standardization.

[9] Stripe Inc. "PaymentIntent object." https://stripe.com/docs/api/payment_intents, 2024.
