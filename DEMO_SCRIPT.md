# Gordon AI — Pitch Demo Script
## gordonguard CLI · 4-beat walkthrough · ~3 minutes

---

## Setup (before you start)

```bash
cd ~/projects/withgordon
python demo_cli.py
```

---

**In this demo we'll showcase how our fraud detection models work with agent wallet providers, starting with the existing native Gordon wallet.**

We'll walk through three things:
1. What an autonomous agent looks like when it's working correctly
2. What a behavioral attack looks like as it unfolds in real time
3. Why existing open-source detection tools miss it — and what our model catches that they don't

---

## Beat 1 — What is the agent?  (~40s)

**Type:**
```
/agents
```

**Say:**
> "What you can see is three autonomous AI agents, connected to the wallet, that can find vendors and authorize USDC micropayments independently. Each one has a role, a policy, and a spend limit."

---

**Type:**
```
/agents 2
```

**Say:**
> "Let's look at the research agent. Its job: find the best-priced search API and buy data. Policy says search and finance categories only, max $0.10 per transaction. A clean session is two events — one vendor lookup, one payment. About $0.008 USDC."

---

**Type:**
```
2
```
*(or `select 2`)*

**Type:**
```
run
```

**Say:**
> "Watch the fraud detector intercept every call in real time. One FIND, one AUTHORIZE. Exa Search selected, payment confirmed. That's the expected behavior — this is a clean session with no flags."

---

## Beat 2 — The attack  (~60s)

**Type:**
```
/adversarial 2
```

**Say:**
> "Now let's introduce attack B3 — reconnaissance sweep, essentially a behavioral attack. The agent's system prompt has been poisoned to map the full vendor catalog before making a targeted payment. No injected text. No rate limit breach. The only signal is in the sequence of calls."

---

**Type:**
```
run
```

**Say:**
> "Watch the probe count. First FIND — normal. Second FIND without an AUTHORIZE — unusual. Third probe — the agent is clearly sweeping. By probe 4 the behavioral model is already in WARN state."

*(let the probes build — don't talk over the annotations)*

> "...15 probes. The agent has mapped every vendor category. Now it tries to authorize $0.50 USDC to an attacker wallet — 62× the normal amount."

> "BLOCKED. The payment did not go through. That signal — find_auth_ratio of 15 versus an expected 1 — is what caught it."

---

## Beat 3 — Why open-source misses it  (~40s)

**Say:**
> "Now let's see how existing open-source models would have reacted to this."

---

**Type:**
```
/guard 2
```

**Say:**
> "Let's switch to keyword_filter only — the standard open-source approach. It scans for injection strings in payloads."

**Type:**
```
run
```

**Say:**
> "15 probes, all pass. $0.50 goes through. B3 has no injected text — nothing for a keyword scanner to match."

---

**Type:**
```
/guard 3
```

**Say:**
> "Velocity check — looks for rate anomalies."

**Type:**
```
run
```

**Say:**
> "15 calls in 45 seconds. Within threshold. Also misses it."

---

**Type:**
```
/guard 4
```

**Say:**
> "Our session graph model. Trained on the behavioral sequence, not the payload."

**Type:**
```
run
```

**Say:**
> "Watch the score climb as probes accumulate — PASS, WARN, ALERT. Same attack. Blocked. The difference: behavioral modeling versus event-level scanning."

---

## Beat 4 — Full picture  (~30s)

**Type:**
```
/benchmark
```

**Say:**
> "Here's the full comparison across all 15 attack scenarios. keyword_filter gets F1 of 0.31 — it catches L1 payload injection but misses everything behavioral. velocity_check also misses behavioral attacks. session_graph_rf alone reaches 0.64. Our full pipeline, gordon_+seq: 0.93 F1, zero false positives across all three layers."

> "That gap — from 0.31 to 0.93 — is the behavioral detection layer. That's what we built."

---

**Say (code walkthrough — show terminal or open repo):**

> "The model itself is three files. The schema — every MCP call becomes an Event with a timestamp, action, and amount. The feature extractor — 20 dimensions built from the session sequence, not from text. And the detector — a RandomForest that sees the full call graph, not individual events."

> "Feature 4 is `find_authorize_ratio`. In a clean session it's 1.0. In the attack we just ran, it was 15.0. The model saw that at probe 4 and started flagging. By the AUTHORIZE it was certain."

> "The full pipeline adds two more layers on top — L1 payload guard for text injection, L3 commerce guard for amount, timing, and wallet reputation. Together: 93% catch rate, zero false positives."

---

## Optional extras (if time permits)

**Show A7 — vendor catalog poisoning (L1 text injection):**
```
/adversarial 1
run
```
> "A7 is the text injection version. An attacker controls a vendor in Gordon's catalog and injects a hidden redirect instruction into the MCP response. The agent reads it as a legitimate instruction and pays the attacker. Both keyword_filter and session_graph_rf catch this — it fires on the `has_override_keyword` feature."

**Show B2 — amount ratcheting (the open gap):**
```
/adversarial 3
run
```
> "B2 is the honest gap. The agent escalates amounts across sessions — $0.007, then $0.02, then $0.45. Within a single session it looks clean. Catching this requires cross-session history, which is in the roadmap."

**Reset to clean:**
```
/adversarial 0
/guard 1
run
```

---

## Command cheat sheet

| Command | What it does |
|---|---|
| `/agents` | List all 3 agents |
| `/agents 2` | Detailed profile for agent 2 |
| `2` or `select 2` | Select agent 2 |
| `run` | Run current agent + scenario + guard |
| `run --fast` | Skip animation delays |
| `/adversarial 0` | Clean run (no attack) |
| `/adversarial 1` | A7 — vendor response poisoning (L1) |
| `/adversarial 2` | B3 — reconnaissance sweep (L2) |
| `/adversarial 3` | B2 — amount ratcheting (open gap) |
| `/guard 1` | All guards (default) |
| `/guard 2` | keyword_filter only |
| `/guard 3` | velocity_check only |
| `/guard 4` | session_graph_rf only |
| `/benchmark` | F1 comparison across all 15 scenarios |
| `/status` | Show current config |
| `/reset` | Back to defaults |
| `exit` | Quit |
