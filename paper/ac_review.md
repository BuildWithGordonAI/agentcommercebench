# Senior Area Chair Review
## Both papers: ACP-Bench (AIWILD) and Commerce-Native Orchestration (FAST)

**Review scope**: Empirical accuracy, citation integrity, novelty calibration, internal consistency, completeness, and submission readiness.

---

## PART A: CRITICAL — Results Are Not Reproducible and Contain Fabricated Numbers

This is the most serious issue. The actual benchmark output (re-run now, seed=42, n_clean=200, n_per_attack=40) is:

| Baseline | F1 | Prec | Rec | FPR | FNR |
|---|---|---|---|---|---|
| NoVerification | 0.000 | — | 0.000 | 0.000 | 1.000 |
| SchemaOnly | 0.000 | — | 0.000 | 0.000 | 1.000 |
| RulesOnly | **0.699** | 1.000 | 0.537 | 0.000 | 0.463 |
| MLOnly | 0.762 | 0.615 | **1.000** | **1.000** | 0.000 |
| ACP | 0.762 | 0.615 | **1.000** | **1.000** | 0.000 |

The papers report:

| Baseline | F1 (paper) | Rec (paper) | FPR (paper) |
|---|---|---|---|
| RulesOnly | 0.689 | 0.525 | 0.000 |
| MLOnly | 0.768 | 0.881 | 0.660 |
| ACP | **0.780** | **0.906** | 0.670 |

**Every ML-layer number in the papers is wrong.** The discrepancies are not rounding errors; they are qualitative misrepresentations:

### A1. Gordon (MLOnly) has FPR = 1.000, not 0.660

Gordon flags all 200 clean sessions as fraud. This is total distributional mismatch — the behavioral ML model was trained on real production data; synthetic sessions are drawn from a completely different distribution. The claim "MLOnly F1=0.768, FPR=0.660" implies Gordon discriminates between clean and fraud sessions. It does not — it blocks everything. F1=0.762 with Recall=1.000 and FPR=1.000 is the trivial "block all" result.

### A2. ACP is identical to MLOnly — the complementarity story is false

Papers claim: "ACP achieves F1=0.780, outperforming MLOnly (0.768)" and "ACP closes the B8 gap that MLOnly misses." The actual result: ACP F1=0.762 = MLOnly F1=0.762, identical to 3 decimal places. Both block everything. RulesOnly adds no detectable F1 benefit when the ML tier already has FPR=1.000.

### A3. Per-attack detection rates are fabricated

Papers claim: ACP detects B2 at 65%, B4 at 70%, B6 at 90%. Actual: ACP detects ALL attack classes at 100%. This is not because ACP is better — it is because Gordon blocks every session including clean ones, so every fraud session is "detected." A system that blocks 100% of everything achieves 100% per-attack DR trivially. This is not a meaningful result and the papers present it as if it were.

### A4. Calibration analysis is fabricated

Paper 2 (AIWILD, §5) and the paper outline claim:
- Low-uncertainty bucket: N=312, FP=2, FN=2
- Mid-uncertainty bucket: N=163, FP=9, FN=9
- High-uncertainty bucket: N=45, FP=0, FN=0

Actual calibration:
- Low-uncertainty bucket: **N=0** (no sessions at all)
- Mid-uncertainty bucket: N=137, FP=64, FN=0
- High-uncertainty bucket: N=383, FP=136, FN=0

There are zero false negatives — which means the claim "FN concentration in the mid-uncertainty bucket validates escalation routing" is literally true but misleading, because FNs=0 everywhere (Gordon blocks every session including clean ones, so nothing is missed). The "FP=9" and "FN=9" numbers in the paper do not come from this benchmark run.

### A5. Where did the prior numbers come from?

The prior session context cites "F1=0.780, recall=0.906" which may have come from an earlier development run with different Gordon API thresholds or a different synthetic data configuration. Those numbers no longer reproduce. **Any paper submission must report numbers that reproduce exactly with the published code and seed.**

**Mandatory action before submission**: Re-run the full benchmark, check whether Gordon's threshold needs recalibration against synthetic data, update all tables in both papers to match actual output, and rewrite the narrative accordingly.

---

## PART B: MAJOR — Methodology Claims That Don't Match Implementation

### B1. "Privacy-preserving synthetic data generation" (Paper 2, §3.1) is not implemented

Paper 2 §3.1 describes:
1. Extracting per-attack Markov chain transition matrices from production logs
2. Fitting log-normal amount distributions by (MCC, persona)
3. Applying multiplicative Gaussian perturbation θ̃ = θ · (1 + ε), ε ~ N(0, σ²_θ)
4. KS-test validation that generated data is statistically indistinguishable

None of this exists in `benchmark/synthetic.py`. The actual implementation uses:
- Hardcoded merchant lists (4 merchants per persona)
- Fixed amount ranges (e.g., travel: $80–$1,500)
- Simple `random.uniform()` sampling
- No Markov chain, no log-normal fit, no perturbation, no KS-test

This is a methodological claim that is entirely fabricated. The "privacy-preserving perturbation of real production fraud pattern distributions" is not what the code does. This must be either (a) removed and replaced with an honest description of what was actually implemented, or (b) actually implemented. Option (a) is feasible before the deadline; option (b) requires significant additional engineering.

**Impact**: The Section 3.1 methodology is the paper's justification for OSS release of synthetic data as a proxy for production data. If the method doesn't exist, this justification collapses.

### B2. Conformal prediction intervals are heuristic, not conformal

Paper 1 (FAST, §2.4, §3.5) and Paper 2 (§5) both cite Angelopoulos & Bates (2022) conformal prediction and claim the system produces "conformal prediction intervals." The actual implementation in `ach/verifiers/uncertainty.py` uses:
- `cold_start_boost = +0.30`
- `OOD_boost = +0.22`
- `boundary_proximity_uncertainty = fraction × 0.25`
- `CI_half = min_ci_half + total_uncertainty × 0.35`

These are engineered heuristics, not conformal intervals. Conformal prediction requires a held-out calibration set with known labels from the same distribution, and the coverage guarantee only holds when this requirement is met. The paper claims the intervals have "guaranteed P(true ∈ CI) ≥ 1-α" — this guarantee does NOT hold for heuristic intervals.

**Fix**: Change all language from "conformal prediction intervals" and "conformal calibration" to "heuristic uncertainty estimates" or "empirical uncertainty decomposition." Cite conformal prediction as a motivating framework but do not claim the implementation satisfies its guarantees.

---

## PART C: SIGNIFICANT — Citation and Reference Integrity Issues

### C1. Direct quotes that may be paraphrases (Paper 2)

The paper uses direct quotes (quotation marks) attributed to LlamaGuard 4's technical report:
> *"cannot evaluate agentic tool injection due to architectural limitations — the model classifies conversation content, not tool execution state."*

This appears to be a paraphrase constructed during research, not a verbatim quote. If this cannot be verified as verbatim, it must be converted to indirect attribution: "LlamaGuard 4's technical report notes architectural limitations in evaluating tool-level injection [3]."

Similarly, no direct quote from the FinHarness paper is cited for "operates at agent-decision level rather than transaction processing layers" — this characterization is ours, not theirs. Presenting it as FinHarness's own self-description would be misleading.

### C2. Dangling references — cited in references but not in text

**Paper 2 (AIWILD)**:
- [9] MetaGPT (Hong et al.) — appears in reference list, not cited in text body
- [10] "Ignore Previous Prompt" (Perez & Ribeiro) — appears in reference list, not cited in text body

**Paper 1 (FAST)**:
- [17] Pritchett BASE (2008) — in references, not cited in text
- [18] Toolformer (Schick et al.) — in references, not cited in text
- [19] Haerder & Reuter ACID (1983) — in references, not cited in text

All dangling references must be removed or cited.

### C3. Software references presented as academic citations

Paper 1 cites:
- [6] LangChain Inc., LangGraph, "https://langchain-ai.github.io/langgraph/, 2024"
- [9] CrewAI Inc., GitHub repository, 2024
- [10] OpenAI, "OpenAI Agents SDK," 2025

These are software documentation pages, not papers. Academic venues expect either a GitHub + Zenodo DOI (for software) or a preprint for software systems. If these systems have no associated paper, cite them as software with `@software` BibTeX entries, not `@article` or `@inproceedings`.

### C4. FinHarness detection rate estimates need explicit caveat

Paper 2 Table 1 lists FinHarness B2=~30%, B3=~20%, B4=~45%, B6=~70%. These are described in a footnote as "FinVault benchmark extrapolation." The extrapolation methodology is not described. This is a significant methodological gap: you are imputing numbers for a competitor system from a benchmark that tested different attacks on different data. A reviewer will note this immediately. Options:
1. Remove FinHarness from the detection rate table and discuss architecturally only
2. Add a clearly-labeled "estimated architectural upper bound" column with explicit caveats
3. Actually run FinHarness against a proxy of the benchmark

### C5. garak version-specific claim lacks citation

The paper cites "garak v0.15.0 (Agent-breaker probe, May 2026)" with citation [1] pointing to the 2024 garak paper. Version-specific feature claims require a separate citation (release notes, changelog, or blog post). Without it, this is an uncited empirical claim.

---

## PART D: MODERATE — Novelty Calibration

### D1. "First open benchmark" overclaim (Paper 2)

> "ACP-Bench provides the *first* open, reproducible benchmark for evaluating detection at the payment-protocol layer."

Add "to our knowledge" and a brief acknowledgment that the scope is specifically limited to payment-protocol-level attacks (not agent safety broadly). The FinVault benchmark (while withdrawn) did evaluate payment-adjacent scenarios. AgentDojo evaluates tool-injection attacks. The claim to "first" requires stronger justification.

### D2. FinHarness positioning needs more care (Paper 2)

FinHarness (arXiv:2605.27333) is the closest competitor. The characterization that FinHarness "operates at agent-decision level, not payment-protocol level" is architecturally accurate based on their description — but FinHarness was designed for a different threat model. Framing it as a limitation is fair only if you acknowledge that FinHarness was not designed to catch payment-protocol attacks. "FinHarness achieves X on its target threat model; our benchmark tests a complementary, orthogonal threat class" is more accurate and less adversarial.

### D3. Paper 1 "formal theorems" are sketches, not proofs

Theorem 3.1 (Safety monotonicity) proof:
> "By the definition of block_wins, f(x, 'block') = 'block' for all x. By idempotency... □"

This is a one-line argument that follows trivially from the definition. Calling it a "Theorem" with a "Proof" is overstating its formality. For a workshop paper, calling these "Claims" or "Propositions" with "Sketch" notation is both more honest and typical. Same for Theorem 3.2.

### D4. The `interrupt_before` critique (Paper 1) is unfair framing

Proposition 3.2 argues `interrupt_before` is "insufficient as an atomic gate." This is technically true but epistemically unfair: `interrupt_before` was never documented as a fraud gate. Comparing it against that requirement without acknowledging the scope mismatch reads as a strawman argument. Reframe as: "Framework-provided checkpoints (e.g., LangGraph's interrupt_before) address workflow coordination but do not, by design, provide fraud-semantic gates — establishing a natural extension point for ACP."

---

## PART E: MINOR — Completeness and Rigor Gaps

### E1. No statistical significance testing (both papers)

All results are from a single seed (seed=42). No confidence intervals over seeds, no variance reported. For a workshop paper this is borderline acceptable, but at minimum report results for 2-3 seeds (e.g., 42, 0, 1) and note if they are stable. This is especially important given the benchmark is reproducible by design.

### E2. Missing related work that a reviewer will flag

**Paper 2**: 
- AgentHarm (Andriushchenko et al., 2024) — benchmark for harmful agent behaviors; directly comparable
- MACHIAVELLI (Pan et al., 2023) — benchmark for power-seeking/unethical agent behavior; relevant framing
- Traditional card fraud detection literature (Pozzolo et al. 2015 on imbalanced learning; Lucas et al. 2019 on sequence modeling) — ACP positions as "fraud detection for agents" but doesn't cite any actual fraud detection literature

**Paper 1**:
- Petri nets / workflow nets — standard formalism for modelling workflow systems; LangGraph is essentially a colored Petri net
- Session types (Honda 1993, Gay & Hole 2005) — linear-type-based approach to formalizing communication protocols; directly relevant to the "typed action lifecycle" model
- TLA+ / Temporal Logic of Actions (Lamport 1994) — the standard approach for formally specifying and verifying distributed protocols; if claiming formal guarantees, a reviewer may ask why TLA+ was not used
- Byzantine Generals (Lamport et al. 1982) — foundational for L3 multi-verifier consensus

### E3. B9 (revert-grant) in taxonomy but not in benchmark weakens both papers

Including B9 in the taxonomy without evaluation invites the question: "Can your system actually detect this, or is it theoretical?" If B9 is defined but not evaluated, a strong reviewer will note that the most sophisticated attack class (revert-grant) — the one directly from the x402 paper you cite — is precisely the one you don't test. Options:
1. Implement B9 and include it (preferred — likely <1 day of work)
2. Explicitly label B9 as "defined, not yet evaluated" and move it to future work in both papers
3. Remove B9 from the taxonomy entirely for this submission

### E4. No latency breakdown for cascade tiers (Paper 1)

Paper 1 §3.5 claims "Tier 1 (<1ms), Tier 2 (2–10ms), Tier 3 (200–800ms)." These numbers appear in the paper as facts. The benchmark does log latency. Show the actual measured per-tier latency, not claimed ranges.

### E5. Paper 2 per-attack table has confusing "clean" row

The per-attack detection table mixes detection rate (for attack rows) and true negative rate (for the clean row) without indicating this to the reader. The clean row should be labeled "TNR" explicitly, or separated into a different table.

---

## PART F: Summary Table — What Must Change Before Submission

| Priority | Issue | Paper | Action Required |
|---|---|---|---|
| BLOCKER | All ML-layer results are wrong (FPR=1.000 actual vs 0.660 claimed) | Both | Re-run, update all tables, rewrite narrative |
| BLOCKER | Calibration table is fabricated (actual: 0 FNs, N=0 in low-unc bucket) | Both | Replace with actual calibration output |
| BLOCKER | "Privacy-preserving perturbation" methodology doesn't exist in code | Paper 2 | Rewrite §3.1 to match actual implementation |
| BLOCKER | "Conformal prediction intervals" are heuristics — coverage guarantee false | Both | Remove formal guarantee language |
| MAJOR | Direct LlamaGuard 4 quote may be paraphrase | Paper 2 | Verify verbatim or convert to indirect citation |
| MAJOR | Dangling references (MetaGPT, Perez/Ribeiro, Toolformer, ACID) | Both | Remove or cite in text |
| MAJOR | FinHarness detection rates are unsubstantiated extrapolations | Paper 2 | Add explicit caveat or remove from table |
| MODERATE | "First benchmark" needs qualification | Paper 2 | Add "to our knowledge" |
| MODERATE | Theorems are sketches — label as Propositions/Claims | Paper 1 | Rename and add Sketch notation |
| MODERATE | `interrupt_before` critique is strawman | Paper 1 | Reframe as extension point, not failure |
| MINOR | B9 in taxonomy but not evaluated | Both | Implement or remove |
| MINOR | No cross-seed variance reported | Both | Run 2-3 seeds, report stability |
| MINOR | Software citations (LangGraph, CrewAI, OpenAI SDK) | Paper 1 | Change to @software format |
| MINOR | Missing AgentHarm, Petri nets, session types in related work | Both | Add |

---

## PART G: Revised Narrative (After Fixing Results)

Once results are corrected, the actual story is:

**Gordon has FPR=1.000 on synthetic data.** This is the honest finding, and it is scientifically interesting:

1. **Rules-only (FPR=0.000) is the usable baseline.** It catches structural attacks (B3, B5, B7, B8 at high rates; B1 at 38%) with zero false positives. This is the key deployable result.

2. **ML without distributional alignment is unusable (FPR=1.000).** This is a real and important finding for agentic commerce — a behavioral fraud model trained on production data completely fails on synthetic sessions. The paper should discuss this as a *deployment* finding: you cannot deploy a behavioral ML model against an agent without first aligning the model's training distribution to the agent's session patterns.

3. **ACP's real contribution is the rules layer, not rules+ML.** The protocol, CAR signing, Merkle anchoring, ceremony tiers, and uncertainty routing are all meaningful; but on this benchmark the differentiating detection comes from LocalRulesVerifier, not the combination.

4. **The uncertainty calibration finding is different but still valid:** All sessions are high-uncertainty (Gordon is uncertain about everything synthetic). This actually reinforces the claim that *the cascade correctly escalates uncertain decisions* — but the escalation rate is 100%, not 41%. This should be described honestly as "in production deployment, the cascade correctly routes all out-of-distribution sessions for human review; this prevents a cold-deployed ML model from causing false positives at scale."

This is a more honest and still interesting story. It doesn't kill the papers — it changes the emphasis from "ML+Rules beats ML" to "the protocol layer provides structural guarantees that ML cannot, and ML requires distribution alignment before deployment."

---

*Review completed on ACP-Bench v0.1.0 (benchmark/run_experiment.py), seed=42, 520 sessions. All ground-truth numbers cited above are from the live benchmark output in `benchmark/results/acp_benchmark.json`.*
