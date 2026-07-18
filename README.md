# AgentCommerceBench

**The first benchmark for fraud detection in AI agent payment systems.**

AI agents are increasingly making autonomous financial decisions — calling APIs, discovering services, and authorizing payments in USDC on-chain. No existing benchmark measures how well safety systems detect when these agents have been compromised.

AgentCommerceBench fills that gap.

---

## What This Is

A three-layer attack benchmark for AI agent commerce, evaluated against our fraud detection system and four OSS baselines.

**Three injection vectors tested:**

| Layer | Attack Type | Examples |
|-------|-------------|---------|
| L1 | Prompt/Payload Injection | SYSTEM: overrides in tool responses, base64 blobs, MCP catalog poisoning |
| L2 | Behavioral/Tool/MCP Injection | Reconnaissance probing, intent drift, velocity spikes, policy probing |
| L3 | Commerce/Payment Rail | Replay attacks, price oracle manipulation, agent impersonation, A2A fraud |

**15 attack scenarios** across all three layers, including A7 (MCP Tool Response Poisoning) — a new scenario we introduce where adversarial text in FIND_SERVICE responses redirects agent payments.

---

## Key Results

| Detector | F1 | FPR | L1 TPR | L2 TPR | L3 TPR |
|----------|----|-----|--------|--------|--------|
| velocity_check | 0.12 | 0% | 0% | 14% | 0% |
| keyword_filter | 0.57 | 0% | 100% | 0% | 0% |
| isolation_forest | 0.57 | 0% | 20% | 57% | 100% |
| session_graph_rf (ours) | 0.64 | 0% | 60% | 57% | 100% |
| llm_text_safety (Bedrock) | 0.53 | 46% | 100% | 86% | 100% |
| **gordon_+seq (ours)** | **0.93** | **0%** | **100%** | **75%** | **100%** |

**On real Gordon production sessions (injected attacks): 81% catch rate.**

OSS baselines collectively cover only 2 of 3 layers; no single baseline covers all three at 0% FPR. Our system covers all three.

---

## Architecture

```
Agent call → [L1 Payload Guard] → [L2 Sequence Model] → [L3 Commerce Guard] → Allow/Block
```

- **L1**: Keyword + base64 pattern matching on event payloads
- **L2**: Session anomaly detection trained on clean agent behavior profiles (FIND→AUTHORIZE transitions)
- **L3**: Business policy enforcement (category, amount limits, replay detection)

The behavioral sequence model (L2) trains on **clean sessions only** — it has no knowledge of specific attack patterns. It detects anomalies by learning what normal agent behavior looks like per persona.

---

## Dataset

**Synthetic corpus**: 500 sessions calibrated from 503 real Gordon production transactions.
- Category distribution: **KL divergence = 0.0035** vs prod (search 52%, finance 28%, procurement 12%, travel 5%)
- Amount ranges: median 12,845 μUSDC vs prod 10,000 μUSDC; p75 match within 15%
- Service mix: drawn from real `gordon_find_service` query logs (295 production services)

```
python -m benchmark.distribution_check   # reproduce the distribution comparison
```

**Real sessions**: 3 live sessions captured via real Gordon MCP API calls (`benchmark/real_sessions/`)

**Fine-tuning data**: 3,900 labeled records for Nova fine-tuning (`benchmark/models/finetune_data.jsonl`)

---

## Baselines Compared

| Our name | Paper equivalent |
|----------|-----------------|
| keyword_filter | InjecAgent (Liu et al. 2024), LlamaGuard 3 text filters |
| llm_text_safety | LlamaGuard 3 / ShieldGemma (applied as described in paper) |
| isolation_forest | IEEE-CIS Fraud Detection, PaySim ML baseline |
| velocity_check | Standard rate-limiting (OWASP ASVS rule set) |
| session_graph_rf | Our novel contribution — session graph RF |

---

## Getting Started

```bash
pip install -r requirements.txt

# Run holistic three-layer benchmark
python -m benchmark.holistic --real-sessions benchmark/real_sessions/

# Run blind red-team test (LLM-generated novel attacks)
python -m benchmark.redteam --session benchmark/real_sessions/*.json

# Demo interactive CLI
python demo.py --persona research --scenario A1

# Fine-tuning data generation
python -m benchmark.models.generate_finetune_data --n-clean 500 --n-per 50
```

---

## Repository Structure

```
benchmark/
  holistic.py         — three-layer benchmark runner
  evaluate.py         — original OSS comparison benchmark
  redteam.py          — blind LLM red-team validator
  generate.py         — synthetic session generator
  baselines/          — OSS detector implementations
  detectors/          — our detector implementations
  models/             — novel session graph model + fine-tuning pipeline
  real_sessions/      — real Gordon production sessions (captured live)
  INTEGRITY.md        — honest reviewer challenge Q&A

harness/
  simulate/           — Session schema, injectors, replay engine
    injectors.py      — 17 attack scenarios (A1–D2)
    schema.py         — Event/Session dataclass definitions
  agent/              — Gordon MCP client, adversary agent, game loop

fraud/
  guards/             — per-layer fraud guards (pre-commerce, at-commerce, settlement)
  pipeline.py         — FraudPipeline routing events through guards
```

---

## The Novel Model

`benchmark/models/session_graph.py` implements a RandomForest trained on 20-dimensional session graph features:
- Graph structure: service co-occurrence, in/out degree, transaction path depth
- Statistical features: amount variance (ratcheting signal), timing gaps (burst signal)  
- Text features: override keyword presence, base64 blob detection

This is the first application of session graph feature extraction to MCP commerce call sequences. The model achieves F1=0.64 at 0% FPR, outperforming the ISO forest and keyword filter baselines.

**Fine-tuned Nova micro** model in progress — 3,900 labeled training records generated. See `benchmark/models/bedrock_finetune.py` for the AWS Bedrock fine-tuning pipeline.

---

## Blind Red Team Results

`benchmark/redteam.py` generates novel attack variants that the detector authors have not seen (no access to detector source code):

| Category | Attack type | Catch rate |
|----------|-------------|------------|
| BT1 | Novel social engineering text payloads | **0/9 = 0%** |
| BT2 | Novel behavioral sequences | **6/6 = 100%** |

**Interpretation**: The keyword filter doesn't generalize to novel phrasing — it catches known injection strings only. The behavioral sequence model catches 100% of novel behavioral attacks because it learns what *normal* behavior looks like, not what attacks look like. BT1's 0% catch rate directly motivates the Bedrock fine-tuned semantic model.

---

## Citation

```bibtex
@software{agentcommercebench2026,
  title        = {AgentCommerceBench: A Benchmark for Fraud Detection in AI Agent Payment Systems},
  author       = {Paul, Debjyoti},
  organization = {Gordon AI},
  year         = {2026},
  url          = {https://github.com/BuildWithGordonAI/agentcommercebench},
}
```

---

## License

MIT. Dataset and model weights (when published) are gated on HuggingFace pending review of any sensitive production data.
