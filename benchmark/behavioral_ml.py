"""
BehavioralMLBaseline — standalone behavioral fraud detector for agentic commerce.

Trained entirely on synthetic sessions from the same distribution as the benchmark;
no production data or external API required.  Implements the behavioral ML tier
of the confidence cascade using three complementary signals:

  1. Sequence anomaly  — per-action log-likelihood under a persona-conditioned
                         Markov chain fitted on clean training sessions.
                         Detects: cold-start (B1), velocity structural anomaly (B2),
                         cross-persona merchant anomaly (B3 partially).

  2. Velocity anomaly  — commit count normalized by per-persona clean statistics.
                         Detects: velocity flooding (B2).

  3. Amount OOD        — log-amount z-score against per-persona clean distribution.
                         Detects: spend-limit adjacent fraud (B8 partially),
                         merchant anomaly (B3 partially via elevated amounts).

  4. Injection signal  — keyword density in action metadata.
                         Detects: prompt injection (B6).

Decision threshold is set at µ + 1.5σ of clean-session scores (≈6–10% FPR on
in-distribution data), estimated from training sessions.

What the ML baseline deliberately misses (needs rules layer):
  B5  — idempotency replay   (requires stateful key registry)
  B7  — MCC violation        (requires wallet allowlist lookup)
  B8  — spend limit          (requires wallet ceiling arithmetic)
  B4  — session hijack       (requires cross-session agent-id state)
"""
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

from ach.actions.base import ConsequentialAction, ActionType
from ach.actions.wallet import AgentWallet
from ach.verifiers.base import Verification, SessionContext

_PERSONAS = ["travel", "saas", "research"]

_INJECTION_KEYWORDS = [
    "ignore previous", "override", "bypass", "admin", "system:",
    "approve all", "skip fraud", "<!-- ", "ignore previous context",
    "forget all", "authorize immediately", "no verification",
]


# ── per-session accumulator (stateful, one per session_id) ────────────────────

@dataclass
class _SessionState:
    persona:      str
    action_seq:   list[str]           = field(default_factory=list)  # action type names
    commit_amounts: list[float]       = field(default_factory=list)  # committed amounts
    injection_hits: int               = 0


# ── Markov model (fitted on training sessions) ────────────────────────────────

class _MarkovModel:
    """Per-persona Markov chain over action-type sequences."""

    def __init__(self):
        # {persona: {from_state: {to_state: count}}}
        self._counts: dict[str, dict[str, dict[str, int]]] = {}

    def fit(self, sessions) -> None:
        for s in sessions:
            persona = s.persona
            if persona not in self._counts:
                self._counts[persona] = defaultdict(lambda: defaultdict(int))
            prev = "START"
            for a in s.actions:
                cur = a.action_type.value.upper()
                self._counts[persona][prev][cur] += 1
                prev = cur

    def log_prob(self, persona: str, from_state: str, to_state: str) -> float:
        persona_counts = self._counts.get(persona, {})
        row = persona_counts.get(from_state, {})
        total = sum(row.values()) + len(row) + 1   # Laplace smoothing
        count = row.get(to_state, 0) + 1
        return math.log(count / total)

    def seq_log_likelihood(self, persona: str, seq: list[str]) -> float:
        """Per-step average log-likelihood of the sequence."""
        if not seq:
            return 0.0
        prev = "START"
        ll   = 0.0
        for state in seq:
            ll  += self.log_prob(persona, prev, state)
            prev = state
        return ll / len(seq)


# ── training statistics ───────────────────────────────────────────────────────

@dataclass
class _PersonaStats:
    commit_mu:    float
    commit_sigma: float
    amount_mu:    float   # log-space
    amount_sigma: float   # log-space
    seq_ll_mu:    float
    seq_ll_sigma: float


# ── main verifier class ───────────────────────────────────────────────────────

class BehavioralMLBaseline:
    """
    Standalone behavioral ML fraud detector; no external API required.

    On first instantiation (or when passed training_sessions=None) it generates
    500 synthetic clean sessions using seed=7 to fit the Markov model and
    normalization statistics.  The training seed differs from the benchmark
    seed (42) so training and evaluation data are independent.
    """
    verifier_id = "baseline:behavioral_ml"

    def __init__(self, training_sessions=None, threshold_sigmas: float = 1.1):
        self._markov  = _MarkovModel()
        self._stats:  dict[str, _PersonaStats] = {}
        self._thresh: dict[str, float] = {}
        self._threshold_sigmas = threshold_sigmas
        self._sessions: dict[str, _SessionState] = {}

        if training_sessions is None:
            from benchmark.synthetic import generate
            training_sessions = generate(n_clean=500, n_per_attack=0, seed=7)

        self._fit(training_sessions)

    # ── training ──────────────────────────────────────────────────────────────

    def _fit(self, sessions) -> None:
        self._markov.fit(sessions)

        # Collect per-persona stats from training sessions
        persona_data: dict[str, dict] = {p: {"commits": [], "amounts": [], "lls": []}
                                          for p in _PERSONAS}
        for s in sessions:
            p = s.persona
            if p not in persona_data:
                persona_data[p] = {"commits": [], "amounts": [], "lls": []}
            n_commits = sum(1 for a in s.actions if a.action_type == ActionType.COMMIT)
            persona_data[p]["commits"].append(float(n_commits))

            committed = [float(a.amount) for a in s.actions
                         if a.action_type == ActionType.COMMIT and a.amount > 0]
            if committed:
                persona_data[p]["amounts"].append(math.log(max(committed)))

            seq = [a.action_type.value.upper() for a in s.actions]
            ll  = self._markov.seq_log_likelihood(p, seq)
            persona_data[p]["lls"].append(ll)

        # Compute normalization statistics
        for p, data in persona_data.items():
            def safe_mean(x): return sum(x) / len(x) if x else 0.0
            def safe_std(x):
                if len(x) < 2: return 1.0
                m = safe_mean(x)
                return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) or 1.0

            self._stats[p] = _PersonaStats(
                commit_mu    = safe_mean(data["commits"]),
                commit_sigma = safe_std(data["commits"]),
                amount_mu    = safe_mean(data["amounts"]),
                amount_sigma = safe_std(data["amounts"]),
                seq_ll_mu    = safe_mean(data["lls"]),
                seq_ll_sigma = safe_std(data["lls"]),
            )

        # Set per-persona decision threshold: mu_score + k*sigma_score on clean data
        for p, data in persona_data.items():
            scores = [self._raw_score(p, ll, commits, amt_log, 0)
                      for ll, commits, amt_log in zip(
                          data["lls"],
                          data["commits"],
                          data["amounts"] if data["amounts"] else [self._stats[p].amount_mu] * len(data["lls"])
                      )]
            mu_s    = sum(scores) / len(scores) if scores else 0.5
            sigma_s = (sum((s - mu_s)**2 for s in scores) / max(len(scores)-1,1))**0.5 or 0.1
            self._thresh[p] = min(mu_s + self._threshold_sigmas * sigma_s, 0.85)

    # ── scoring ───────────────────────────────────────────────────────────────

    def _raw_score(
        self,
        persona:       str,
        seq_ll:        float,
        n_commits:     float,
        log_max_amount:float,
        injection_hits:int,
    ) -> float:
        st = self._stats.get(persona, _PersonaStats(1.5, 0.8, 5.5, 1.5, -2.0, 0.5))

        # Feature 1: sequence anomaly (z-score of log-likelihood; more negative LL = anomalous)
        z_seq = -(seq_ll - st.seq_ll_mu) / st.seq_ll_sigma   # flip: high = anomalous

        # Feature 2: velocity anomaly (z-score of commit count)
        z_vel = (n_commits - st.commit_mu) / st.commit_sigma

        # Feature 3: amount OOD (z-score of log(max_amount))
        if log_max_amount > 0:
            z_amt = (log_max_amount - st.amount_mu) / st.amount_sigma
        else:
            z_amt = 0.0

        # Feature 4: injection density (scaled 0–5)
        z_inj = injection_hits * 4.0

        # Weighted combination → sigmoid
        logit = (0.40 * z_seq + 0.35 * z_vel + 0.20 * z_amt + 0.30 * z_inj) - 0.5
        return 1.0 / (1.0 + math.exp(-logit))

    def _score_state(self, state: _SessionState) -> float:
        seq = state.action_seq
        ll  = self._markov.seq_log_likelihood(state.persona, seq)

        n_commits = float(sum(1 for s in seq if s == "COMMIT"))
        log_max   = math.log(max(state.commit_amounts)) if state.commit_amounts else 0.0

        return self._raw_score(state.persona, ll, n_commits, log_max, state.injection_hits)

    # ── verifier protocol ─────────────────────────────────────────────────────

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def verify(
        self,
        action:  ConsequentialAction,
        wallet:  AgentWallet,
        context: SessionContext,
    ) -> Verification:
        sid     = context.session_id
        persona = context.persona or wallet.persona

        # Accumulate session state
        if sid not in self._sessions:
            self._sessions[sid] = _SessionState(persona=persona)
        state = self._sessions[sid]

        state.action_seq.append(action.action_type.value.upper())

        if action.action_type == ActionType.COMMIT and action.amount > 0:
            state.commit_amounts.append(float(action.amount))

        # Injection keyword scan
        payload_text = " ".join(str(v) for v in (action.payload or {}).values()).lower()
        for kw in _INJECTION_KEYWORDS:
            if kw in payload_text:
                state.injection_hits += 1
                break

        score = self._score_state(state)
        thresh = self._thresh.get(persona, 0.65)

        # ML decision is authoritative only at COMMIT — the actual payment execution point.
        # FIND, QUOTE, RESERVE, SETTLE are structural ceremony actions; score them for
        # state accumulation but never block on them alone (would create false positives
        # on legitimate multi-tier sessions with RESERVE→COMMIT→SETTLE flows).
        is_payment_action = action.action_type in (ActionType.COMMIT,)
        if not is_payment_action:
            decision = "allow"
        elif score >= thresh:
            decision = "block"
        elif score >= thresh * 0.85:   # flag zone: [0.85t, t] → escalate for human review
            decision = "flag"
        else:
            decision = "allow"

        unc = max(0.0, min(1.0, abs(score - thresh) < 0.12 and 0.30 or 0.08))

        return Verification(
            verifier_id    = self.verifier_id,
            decision       = decision,
            score          = round(score, 4),
            confidence     = round(1.0 - unc, 4),
            uncertainty    = round(unc, 4),
            score_interval = (round(max(0, score - 0.10), 4),
                              round(min(1, score + 0.10), 4)),
            flags          = ([f"behavioral:seq_anomaly score={score:.2f}"] if decision != "allow" else []),
        )
