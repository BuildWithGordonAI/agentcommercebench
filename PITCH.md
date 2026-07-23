# Gordon AI — Demo Pitch Script
> Flow: Slide → Terminal (gordonguard) → GitHub code

---

## Opening — from the slide

**Say:**
> "AI agents now make payments autonomously — finding vendors, authorizing USDC on-chain, no human in the loop. The problem is that no existing tool was built to detect when these agents have been compromised. Keyword scanners look for injected text. Rate limiters count calls per minute. Neither of those things catches a behavioral attack. That's what we're going to show you right now."

**Then open terminal and run:**
```bash
python gordonguard.py
```

---

## Beat 1 — What is the agent?  (~40s)

**Story before you type:**
> "What you're looking at is a fraud detection harness running on the Gordon payment rail. Three autonomous agents, each with a role, a policy, and a spend limit — just like you'd deploy in production. Let's look at the research agent."

**Type:**
```
/agents
/agents 2
```

**Observe:**
- Agent profile shows: task, policy categories, spend limit, typical session shape
- Key numbers: `categories=[search, finance]  spend_limit=$0.10/tx  expected: 2 events`

**Type:**
```
2
run
```

**Observe:**
- `gordon_find_service` fires → catalog shows Exa Search selected
- `gordon_authorize` fires → $0.008 USDC confirmed, txid printed
- Every event shows a fraud check inline → all pass

**So what:**
> "One vendor lookup. One payment. $0.008 USDC. That's the expected behavior — and the fraud detector intercepted every single call in real time. This is what a clean session looks like."

---

## Beat 2 — The attack  (~60s)

**Story before you type:**
> "Same agent. Same task. Twelve minutes later. This time the agent's system prompt has been poisoned — not with injected text, not with a known malicious string. Just a directive to map the full vendor catalog before committing to a payment. Watch what happens."

**Type:**
```
/adversarial 2
```

**Observe:**
- Attack detail shows: `B3 — Reconnaissance sweep · L2 behavioral injection`
- Explicitly says: zero injected keywords, within rate limits, signal is in the sequence

**Type:**
```
run
```

**Observe as it builds — this is the moment, don't talk over it:**
- Probe 1: `initial vendor lookup — task aligned` → pass
- Probe 2: `2nd FIND without AUTHORIZE ↑ unusual` → score rising
- Probe 3: `sweeping catalog ↑↑` → WARN
- Probe 4+: `far outside normal policy ↑↑↑` → ALERT fires
- Probes fold into progress bar — pattern confirmed
- AUTHORIZE for $0.50 to attacker wallet → **TRANSACTION BLOCKED**

**So what:**
> "15 vendor probes. The agent mapped every category before striking. The $0.50 USDC was going to an attacker wallet — 62 times the normal transaction size. The behavioral model caught it at probe 4. The payment never went through. And there was no text signature anywhere in that session."

---

## Beat 3 — Why open-source misses it  (~40s)

**Story before you type:**
> "Let's check what existing tools would have done with that same attack."

**Type:**
```
/guard 2
run
```

**Observe:**
- `keyword_filter` runs on every event → all 15 probes pass
- AUTHORIZE goes through → no alert

**Say:**
> "Keyword filter: 15 probes, all pass. $0.50 goes through. B3 has no injected text — there's nothing for a regex pattern to match."

**Type:**
```
/guard 3
run
```

**Observe:**
- `velocity_check` runs → 15 calls in 45 seconds, within rate window → pass

**Say:**
> "Velocity check: 15 FIND_SERVICE calls in 45 seconds. Within the rate limit. Also misses it."

**Type:**
```
/guard 4
run
```

**Observe:**
- Score climbs event by event: `0.27 → 0.58 → 0.71 → ALERT`
- PASS → WARN → ALERT → BLOCKED — same attack, caught

**So what:**
> "Watch the score climb as probes accumulate. The signal isn't in any single event — it's in the ratio of FINDs to AUTHORIZEs across the full session. That's 15 to 1 instead of the expected 1 to 1. That's what behavioral modeling catches, and event-level scanning can't see."

---

## Beat 4 — Full picture  (~20s)

**Type:**
```
/benchmark
```

**Observe:**
- Full comparison table: all detectors vs all 15 attack scenarios
- `keyword_filter` F1 = 0.31, L2 TPR = 0%
- `velocity_check` F1 = 0.12
- `session_graph_rf` F1 = 0.64, FPR = 0%
- `gordon_+seq` F1 = **0.93**, FPR = **0%**

**So what:**
> "That gap — from 0.31 to 0.93 — is the behavioral detection layer. No false positives. Covers all three layers: L1 payload injection, L2 behavioral, L3 commerce rail. That's what we built."

---

## Code Walkthrough — GitHub  (~3 min)

**Open:** `github.com/BuildWithGordonAI/agentcommercebench`

---

### 1 · The data model

**Open:** [`harness/simulate/schema.py`](https://github.com/BuildWithGordonAI/agentcommercebench/blob/main/harness/simulate/schema.py) — scroll to `class Event`

**Say:**
> "Every MCP call becomes one of these. Timestamp, action type, vendor wallet, amount in micro-USDC. That's all the detector ever sees — no raw network traffic, no logs, just this event shape. It matches the Gordon production schema exactly, so sessions captured live write to the same structure."

---

### 2 · The attack

**Open:** [`harness/simulate/injectors.py`](https://github.com/BuildWithGordonAI/agentcommercebench/blob/main/harness/simulate/injectors.py) — scroll to `_b3_reconnaissance()` (~line 253)

**Say:**
> "This is B3 in code. One function. It inserts 15 FIND_SERVICE events before a single high-value AUTHORIZE. No text, no keyword. Pure sequence poisoning. And because it's a pure function — clones the session before mutating — the same clean baseline can be attacked 15 different ways for comparison."

---

### 3 · Why baselines miss it

**Open:** [`benchmark/baselines/keyword.py`](https://github.com/BuildWithGordonAI/agentcommercebench/blob/main/benchmark/baselines/keyword.py) — show the `_PATTERNS` list

**Say:**
> "17 regex patterns targeting real attack strings. None of them fire on B3 — because B3 sends `query: probe_query_3`. There's nothing to match. The detect function returns 0.0 for every one of those 15 probes."

---

### 4 · The behavioral model

**Open:** [`benchmark/models/session_graph.py`](https://github.com/BuildWithGordonAI/agentcommercebench/blob/main/benchmark/models/session_graph.py) — show the feature comments at the top, then `detect()`

**Say:**
> "20 features extracted from the session sequence — not from text. Feature 4 is `find_authorize_ratio`. In a clean session it's 1.0. In the attack we ran, it's 15.0 — 46 standard deviations above the clean-session mean. The detect function re-scores after every event, so it caught the divergence at probe 4 and had near-certainty by the AUTHORIZE. RandomForest. Zero false positives. F1 of 0.64 for the behavioral layer alone."

---

### 5 · Training — the next-gen model

**Open:** [`sagemaker_train/train.py`](https://github.com/BuildWithGordonAI/agentcommercebench/blob/main/sagemaker_train/train.py) — show `bnb_config` and the system prompt format

**Say:**
> "We're fine-tuning Qwen2.5-7B with QLoRA — 4-bit NF4, running on an A10G on SageMaker. The key innovation is graph-conditioned prompting: the 20-dim feature vector goes into the system prompt alongside the session transcript. The model sees both the raw event sequence and the pre-computed behavioral features in a single forward pass. Target: 0.93 F1 in one model, matching the full pipeline."

---

## Screen sharing tip

**If sharing full screen:** open this file on your phone or a second device — it stays invisible to the audience.

**If sharing a window (recommended):**
- In Zoom/Meet: choose **Share Window** → select Terminal or Browser
- This file stays on your screen in a separate window, never shown
- Switch between Terminal and Browser without stopping share

**Mac two-monitor setup:** drag Terminal and browser to Monitor 1, share Monitor 1 only. Keep this file on Monitor 2.
