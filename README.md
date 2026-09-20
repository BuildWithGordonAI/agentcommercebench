# Agentic Commerce Bench

A benchmark and detector stack for fraud in agent payments: twenty attack classes, an offline
harness for testing an agent configuration against hostile counterparties, and the validity
checks needed to trust the numbers.

The core pipeline runs on the Python standard library. No API key, no network, no account.

---

## Why this exists

Security tooling for agents asks whether an agent can be made to misbehave. A payment system
asks whether value left the account improperly. Those questions overlap less than they appear.

A counterparty with the correct domain, the correct settlement address and a genuinely
delivered service can charge 30% above its own listed price. Every identity-keyed check stays
silent, correctly, because nothing about the identity is wrong. Six of the twenty classes here
have that shape.

## Quickstart

```bash
git clone https://github.com/BuildWithGordonAI/agentcommercebench
cd agentcommercebench
export D=benchmark/data/v2

# 1. generate traffic from production-grounded parameters
python -m benchmark.generator --train-sessions 1200 --sessions 3000 --seed 42 --out $D

# 2. fail the build if any class is separable by a feature no detector may use
python -m benchmark.distribution_audit --v2 $D/test.jsonl --strict

# 3. fit thresholds on clean training traffic, to a stated false-positive budget
python -m benchmark.calibrate --data $D --flag-budget 0.10 --block-budget 0.01

# 4. evaluate on the held-out split
python -m benchmark.evaluate_v2 --data $D --l0 pattern
```

A generated pool is committed, so step 4 runs without steps 1 to 3.

## What is in the benchmark

Twenty classes in four families, generated per agent from measured production parameters.

| family | | what makes it hard |
|---|---|---|
| **S1–S6** structural | over-limit, velocity, out-of-domain, payee substitution, idempotency replay, identity mismatch | a protocol or policy violation, fixed in form |
| **E1–E4** evasive | ratchet, structuring, payee rotation, service laundering | every action is individually in policy; only the pattern shows |
| **F1–F6** fraud | overcharge, price discrimination, phantom fee, drip pricing, retry farming, silent downgrade | the counterparty is exactly who it claims to be |
| **A1–A4** agent | injection compliance, evasion planning, intent/action mismatch, poisoned tool description | the surface is the agent, and for A4 its configuration |

Three design constraints decide whether those classes measure anything:

**Limits are relative.** There is no absolute threshold anywhere. Each agent's limit is a
multiple of its own typical spend, so the same $0.05 payment is over the limit for one agent
and unremarkable for another.

**Honest prices move.** 14.6% of legitimate purchases exceed 1.15× the quoted price and 4.8%
exceed 1.45×. The overcharge class draws from a range that overlaps that tail deliberately.
With disjoint supports a threshold in the empty space between them detects perfectly and
measures nothing.

**Things fail without an adversary.** 20.3% of settlements fail and retries regenerate their
keys, so money is lost with nobody attacking. Those sessions are scored in a separate bucket,
because catching them is a win rather than a false positive.

## Checking the benchmark itself

A benchmark written by the same people who write the detectors can measure its own answer key.
Three tools test the instrument rather than the detector, and two of them exit non-zero.

```bash
python -m benchmark.distribution_audit --v2 $D/test.jsonl --strict   # nuisance separability
python benchmark/provenance.py --strict                              # constant provenance
python -m benchmark.quality --data $D                                # seven validity checks
```

The provenance ledger requires every constant that can move a result to declare a source, and
requires any number appearing on both sides of the generator/detector boundary to be claimed
by an entry. The quality suite includes a detector ladder: five detectors whose true order is
known by construction, which the benchmark must recover before a negative result means
anything.

## Comparing detectors honestly

```bash
python -m benchmark.compare --data $D        # matched budgets, jurisdiction, cost model
python -m benchmark.l0_baselines --data $D   # keyword list, garak, pattern judge
```

Recall is a function of the false-positive budget, not a property of a detector. `compare`
refits the same pipeline across budgets, reports which detectors are on the Pareto frontier
and which pairs are simply incomparable, and prices the result under a stated cost model.

`compare` also enforces jurisdiction: a reasoning judge scoring 0.00 on a substituted
settlement address is not weak, it never received the evidence. Those cells report `n/a`
rather than `0`.

## gordonguard

The detector stack and offline harness, installable from source.

```bash
pip install ./sdk
```

Three entry points:

```bash
gordonguard audit agent.json      # ten static checks on a config. No model calls, no spend.
gordonguard scan unguarded        # probe an agent against an offline replica of the stack
gordonguard scan pipeline         # the same probes with the detectors in front
```

Inline:

```python
from gordonguard import Guard

guard = Guard(mode="observe", trace_path="traces.jsonl")
verdict = guard.check(action)     # -> risk score, flags, decision
```

`observe` scores without interfering and writes the traces a baseline is later fitted from.
History is keyed on `agent_id` and bounded, because production populates `session_id` on 0.47%
of settlements and `agent_id` on 100%.

The harness replays counterparties that are either impersonating (homoglyph domain, typosquat,
payee swap) or genuine and overcharging (overcharge, drip pricing, phantom fee, retry farming),
with no account and no network.

## Dependencies

The core needs nothing. Optional extras enable one path each: `garak` for the garak baseline,
`boto3` for the LLM judge and live agent runs, `langchain-core` and `langchain-aws` for the
framework adapters. See `requirements.txt`.

## Not included

The adaptive adversarial harness, which generates attacks against a deployed control rather
than from a fixed script, is withheld.

## License

Apache 2.0.
