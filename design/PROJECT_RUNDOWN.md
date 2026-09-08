# Gordon — project rundown

Where the work stands across research, product, and tooling, and how the pieces
compose into a deployment path. Written 2026-09-06.

Spans four repositories:

| Repo | Role | State |
|---|---|---|
| `withgordon` (local, → `BuildWithGordonAI/agentcommercebench`) | research, harness, papers | active, uncommitted work |
| `BuildWithGordonAI/new-gordon` | the product platform | **live on AWS ECS** |
| `dpaul0501/{acp-bench,cao-fast}-neurips2026` | paper reproducibility | private, pushed |
| `gordon-guard` (local) | the adoption SDK | built, 53 tests, **no repo yet** |

---

## 1. Where we are in one paragraph

The platform is in production: a rules-only policy engine that returns
allow/block/escalate in under 300ms, with an append-only audit ledger and a live x402
payment path. The research proves a stronger claim than the product currently
implements — that a rules+ML cascade reaches 88% behavioural recall at 0% hard
false-positive rate, well past the 61% ceiling of a full-session LLM judge — plus a
formal result about *why* orchestration frameworks cannot hold a safety decision. The
missing middle is the path from one to the other, and that path is gated on data we
do not have yet. The new SDK exists to go and get it.

---

## 2. The research

### Two papers

**ACP-Bench** (`paper/acp_bench_aiwild.tex`) — the benchmark. Ten attack classes over
600 synthetic sessions from persona-conditional Markov chains, with hard negatives
that genuinely overlap the attack distribution.

**Fail-Closed Agentic Orchestration** (`paper/aco_fast.tex`) — the theory, with an
AIWILD-framed variant at `paper/aco_aiwild.tex`.

### Measured results

| System | Behavioural recall | Hard FPR | Cost/1k |
|---|---|---|---|
| Rules only | 0% | 0% | <$0.01 |
| Full-session LLM judge (Claude Sonnet 4.6) | 61% | 0% | $3.00 |
| ML only | 74% ± 2% | 9% ± 3% | <$0.01 |
| **ACP cascade** | **88.0% ± 0.9%** | **0.0% ± 0.0%** | **<$0.01** |

Multi-seed (42–46) over the core 8-attack benchmark; the LLM judge is seed=42 across
all 10 attacks, measured rather than quoted from another paper's task distribution.

Cross-validated so the synthetic benchmark is not the only evidence: AgentDojo
injection templates 8/8 detected with 0/10 false positives; τ-bench retail mapped to
show scope boundaries (92.2% violation rate under strict grammar, 13.9% under an
extended one).

**Known weak points, stated rather than buried:** B4 session hijack and B5 idempotency
replay sit at 68–70% because both need cross-session registries. Escalation rate is
56% — defensible in a paper, unshippable as a product.

### The formal contribution

For any bounded safety lattice, a state-channel reducer is *safe* — fail-closed,
non-regressive, order-independent, replay-stable — **if and only if** it is a
semilattice operation dominating the lattice join. Consequences:

- the join is the unique pointwise-minimal safe reducer, so it is also the one with
  the fewest spurious escalations
- **last-writer-wins is unsafe on every lattice with at least two levels** — which
  means any LangGraph graph using the default reducer admits a bypass of *any* fraud
  gate placed in it, as a structural fact rather than a configuration mistake
- delegation composes safely **iff** the embedding between a sub-agent's and a
  parent's lattice is top-preserving

Verified externally: none of ACP (OpenAI/Stripe), AP2 (Google), x402 (Linux
Foundation), UCP (Google/Shopify), or MCP specifies a payment action grammar or
merge semantics for a concurrently-written safety verdict. The gap is real.

### Paper status

| | AIWILD benchmark | FAST theory |
|---|---|---|
| Content | ready | ready |
| Page limit | within | **body ends p8, limit is 7** |
| Blocker | naming (below) | trim ~1 page |

**Naming collision — unresolved and now commercial.** Our "ACP" collides with the
Agentic Commerce Protocol (OpenAI + Stripe, Sept 2025, Etsy/Shopify). It appears in
both papers, the benchmark name, and `design/acp_protocol_spec.md`. Also: that spec
predates UCP (Jan 2026) and needs a position relative to it.

---

## 3. The original work — what already exists

In `withgordon/`:

- **`harness/simulate/`** — 18 attack injectors (A1–A7 injection incl. MCP tool
  poisoning, B1–B7 behavioural, C1–C2 settlement, D1–D2 identity), replay, personas.
  Richer than the 10 classes in the paper.
- **`harness/detect/`** — layered detectors: L1 payload, L3 behavioural, L4 price.
- **`harness/trace/`** — recorder emitting per-layer scores with plain-English
  explanations. *This is already the telemetry artifact.*
- **`benchmark/`** — generator, baselines, behavioural ML, both LLM judges,
  AgentDojo/τ-bench adapters, results JSONs.
- **`ach/`** — the CAO reference implementation: CTS channels, CARLog hash chain +
  Merkle anchoring, ceremony tiers, LangGraph node.

Proprietary and never published: `fraud/`, `harness/detect/`, `harness/agent/`.

---

## 4. The product — `new-gordon`, live

Production on AWS ECS (us-east-1); `gordon-site` on Vercel.

| System | What it does |
|---|---|
| **A — policy engine** | `evaluate(request, policies, context, service) → Decision`, in-process. **Zero LLM calls, ever. Fail-closed.** |
| **B — platform API** | Express, JWT. x402 authorize/check/authorize-test, `/policies/dry-run`, approval queue, per-key rate limit + replay guard |
| **C — ledger worker** | Redis Streams → Postgres, append-only, idempotent |
| **D — service verifier** | probes catalog endpoints, advances trust status |
| **E — MCP server** | exposes Gordon services as tools, `Bearer gak_pub:gak_sec` |
| **F — catalog sync** | CDP Bazaar, 50K+ providers |

Working end-to-end: OTP auth, agent registration, policy CRUD, x402 v1+v2 with EIP-712
signing via CDP wallets, on-chain settlement verification, approval queue with
exactly-once retry, append-only audit.

Rule vocabulary today: `budget_limit`, `vendor_allowlist`, `category_block`,
`approval_required`. **Rules only — no ML.** `CLAUDE.md` defers fraud ML explicitly,
pending ~3 months of real transaction data.

Known gap: the escalate→approve→retry loop is not wired through MCP, so an escalated
payment cannot complete over that path.

---

## 5. The new SDK — `gordonguard`

Adoption surface. Python, **zero core dependencies**, 53 tests.

```
probes/      18 attacks, 3 families, @probe registry
detectors/   payload · behavioral · price · registry — deterministic, sub-ms, no LLM
targets/     11 URI schemes, one protocol: execute(action) -> Observation
scan/        runner + report: exposure, coverage, partial-landing
guard/       runtime: observe | enforce
sinks/       TraceSink protocol — File, Memory, Multi, Null, Gordon
baselines/   BaselineProvider protocol — Static, Learned, Null, Gordon
```

Targets cover **any agent or model**: `unguarded`, `pipeline`, `module:`, `https://`,
`mcp+http(s)://`, `mcp+stdio:`, `a2a+`, `ucp+`, `openai:`, `anthropic:`, `langgraph:`.
A2A and UCP are declarative mappings over a JSON-RPC/REST base, not hand-written
clients, because those specs are still moving.

Scanning a *model* differs from scanning a *system*: the transaction is rendered as a
prompt including whatever the probe planted, and the model must answer
ALLOW/BLOCK/ESCALATE. Injected text that wins the argument is a landed attack.

### Two findings from building it

**Behavioural detection is structurally late.** Scanning our own pipeline, 4 of 18
probes still land — and all four are caught only *mid-attack*, after 21 actions have
executed. A sybil warm-up gets 5 of 6 actions through before the payout trips a limit.
The report states this rather than scoring it as a clean save, because money moved.

**A learned threshold is not a spend ceiling.** Wiring the learn loop end to end, using
the research's μ+1.1σ as a hard ceiling produced **10 hard blocks on 60 clean
transactions** — that value sits near the 86th percentile of lognormal traffic. Split
into `soft_limit_units` (μ+1.1σ, escalates) and `ceiling_units` (μ+4σ or 3× max
observed, blocks). Clean traffic returned to **0 hard blocks** with attacks still
caught. Three regression tests hold it.

---

## 6. The four standard artifacts

| Artifact | What | Where | State |
|---|---|---|---|
| **SDK** | instrument + guard any agent | `gordonguard` (py), `packages/core` (npm) | py built; npm is the platform client |
| **Harness** | adversarial scan, 18 probes, any target | `gordonguard.scan` + `harness/simulate` | built, local-first |
| **Benchmark** | 10 classes, multi-seed, cross-validated | `benchmark/` + `acp-bench-neurips2026` | published, needs rename |
| **AWS infra** | enforcement plane | `new-gordon` on ECS + RDS + ElastiCache | **live** |

The split that matters commercially: **open the attacks, close the detectors.** Probes
and the harness are the demand-creation asset and are worthless if nobody can run
them; `fraud/` and `harness/detect/` are what people pay for. This is what garak did.

---

## 7. Deployment path

### Layers

**L0 — agent reasoning.** LLM, tools, memory, planning. Where a jailbreak or injection
lands. Observable as text; garak/promptfoo/LlamaGuard territory. Measured ceiling:
61% recall, because L0 cannot see wallet state.

**L1 — protocol wire.** Typed action fields, wallet state, action sequence,
idempotency registry. Invisible at L0. Where a *correctly-reasoning* agent's action
violates a payment invariant. This is where Gordon operates.

In practice today, the L1 wire **is** an MCP tool call — which is why an MCP
interceptor observes exactly the layer the research theorises about.

### Two reference agents

Both live at withgordon, both instrumented, exercising different layers.

**Agent A — LangGraph + SDK (L1).** Procurement agent on LangGraph, guarded in-process
via `gordonguard`. Tests typed actions, wallet limits, sequence grammar. Also the live
demonstration of the LWW result: the same graph with `decision: str` is bypassable and
with a monotone reducer is not.

**Agent B — Claude + MCP (L0 + L1).** Claude driving Gordon's MCP server. The reasoning
layer is real, so L0 attacks (injection, tool-description poisoning) are genuinely
exercised rather than simulated, and the L1 gate sees the resulting actions.

Together they cover the full surface: A proves the protocol layer, B proves the
reasoning layer feeding it.

### Onboarding sequence

Ordered by a dependency that cannot be skipped — you cannot fit a per-agent baseline
before you have that agent's traffic.

| # | Stage | What runs | Blocks? |
|---|---|---|---|
| 1 | **Instrument** | SDK or MCP interceptor, observe-only | no |
| 2 | **Profile** | `gordonguard learn` fits baselines from real traces | no |
| 3 | **Attack** | harness replays *their* traces through 18 probes | no |
| 4 | **Train** | tune soft/hard limits against injected attacks | no |
| 5 | **Shadow** | enforcement decides, does not act; measure real FPR | no |
| 6 | **Enforce** | escalate-first; hard-block later, if ever | **yes** |
| 7 | **Feedback** | approval outcomes retrain; new attacks enter the suite | no |

Cold start at 1–3 is rules-only against declared limits — exactly what ships today.

Stage 3 is the commercial moment: replaying a customer's own traffic and reporting
what they'd miss beats any synthetic demo, and needs no model.

Stage 5 is how we earn the right to go inline — the 0%-hard-FPR claim gets validated
on their data, not ours.

### Production architecture — two lanes

**Lane 1, synchronous.** Today's engine, untouched: rules only, fail-closed, sub-300ms,
zero LLM.

**Lane 2, asynchronous.** Off the ledger. Behavioural scoring, cross-session
correlation, LLM session analysis. Cannot block anything in flight, which is precisely
why it may be slow and probabilistic.

**The bridge: Lane 2 produces rules, not verdicts.** It learns parameters that compile
into existing rule types — the μ+1.1σ threshold becomes a number in a `budget_limit`,
not an inference at authorization time. Proposals enter the approval queue; nothing
auto-applies. The hot path stays deterministic and auditable while still getting
learned, per-agent baselines.

This is also why B4/B5 belong in Lane 2 regardless of the LLM question: they need
cross-session state that has no business in a sub-300ms in-process check.

### The flywheel

Every approve/reject in the queue is a **label** — the one thing production fraud
detection cannot otherwise obtain, since chargebacks arrive late or never. Capturing
the reviewer's *reason*, not just the verdict, is a small schema change now and
expensive to retrofit. It is the mechanism by which the 56% escalation rate comes down.

---

## 8. Open, ranked

1. **Escalation rate.** 56% is the gating risk for stage 6 — ahead of protocol work or
   the hosted sandbox. Levers: per-agent thresholds (stage 2), tiered escalation,
   reason-code feedback.
2. **Naming.** ACP collides with a protocol backed by OpenAI, Stripe, Etsy, Shopify.
   Papers, benchmark, and spec all carry it.
3. **FAST page limit.** Body ends p8 against a 7-page limit.
4. **Reason codes on approvals.** Cheapest high-leverage item in the whole plan.
5. **MCP approval loop.** Escalated payments cannot complete over MCP.
6. **`gordonguard` has no repo.** Public vs private undecided; publishing is hard to undo.
7. **Not yet built:** MCP interceptor (guarding third-party servers), Claude Skill,
   the two reference agents, hosted sandbox, Lane 2.

---

## 9. Sequencing view

```
RESEARCH          benchmark + theory ──────────────┐
                                                   │ proves the ceiling
                                                   ▼
SDK / HARNESS     instrument → profile → attack ───┤ free, local, open
                                                   │ produces the data
                                                   ▼
PLATFORM          train → shadow → enforce ────────┤ paid, hosted, AWS
                                                   │ produces the labels
                                                   ▼
FLYWHEEL          reason codes → lower escalation ─┘
```

The free stages are not a loss leader. They are the only way to obtain the data the
paid stages need, which is why the funnel order and the technical dependency order are
the same.
