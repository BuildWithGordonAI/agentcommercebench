"""
SequenceModelGuard — Persona-conditioned Markov sequence model.

Source: benchmark/detectors/sequence_model.py (refactored for Guard interface).

Models (action_type, category) bigram transitions per persona.
Unusual transitions → high -log P → high risk score.

This is the only guard that fires purely from FIND_SERVICE sequences —
it can detect reconnaissance patterns (B3) before any AUTHORIZE.

Inter-layer: reads context.persona if available.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Optional
from fraud.guards.base import PreCommerceGuard
from fraud.context import FraudContext, GuardSignal, Phase


# Hand-crafted priors from persona behavioral sequences
_PRIOR_COUNTS: dict[str, dict] = {
    "research": {
        (None, None):                           {("find_service", "finance"): 80, ("find_service", "search"): 20},
        ("find_service", "finance"):            {("authorize", "finance"): 85, ("get_service", "finance"): 10, ("find_service", "finance"): 5},
        ("find_service", "search"):             {("authorize", "search"): 80, ("find_service", "search"): 15, ("authorize", "finance"): 5},
        ("authorize",    "finance"):            {("find_service", "finance"): 60, ("authorize", "finance"): 30, ("settle", "finance"): 10},
        ("authorize",    "search"):             {("find_service", "search"): 70, ("authorize", "search"): 20, ("settle", "search"): 10},
    },
    "procurement": {
        (None, None):                           {("find_service", "procurement"): 70, ("find_service", "search"): 30},
        ("find_service", "procurement"):        {("get_service", "procurement"): 60, ("authorize", "procurement"): 35, ("find_service", "procurement"): 5},
        ("get_service",  "procurement"):        {("authorize", "procurement"): 90, ("get_service", "procurement"): 10},
        ("authorize",    "procurement"):        {("find_service", "procurement"): 70, ("settle", "procurement"): 20, ("authorize", "procurement"): 10},
    },
    "travel": {
        (None, None):                           {("find_service", "travel"): 90, ("find_service", "search"): 10},
        ("find_service", "travel"):             {("get_service", "travel"): 80, ("authorize", "travel"): 15, ("find_service", "travel"): 5},
        ("get_service",  "travel"):             {("get_service", "travel"): 40, ("authorize", "travel"): 55, ("find_service", "travel"): 5},
        ("authorize",    "travel"):             {("settle", "travel"): 70, ("find_service", "travel"): 20, ("authorize", "travel"): 10},
    },
}

_SMOOTHING = 0.01
_MIN_SCORE = 0.50   # suppress borderline scores to keep FPR at 0%


class _MarkovModel:
    def __init__(self, persona: str):
        self.persona = persona
        self._counts: dict = defaultdict(lambda: defaultdict(float))
        for prev, dist in _PRIOR_COUNTS.get(persona, {}).items():
            for curr, cnt in dist.items():
                self._counts[prev][curr] += cnt

    def fit(self, sessions: list) -> None:
        for s in sessions:
            if not getattr(s, "is_clean", True):
                continue
            prev = (None, None)
            for e in s.events:
                at  = str(getattr(e, "action_type", "")).replace("ActionType.", "").lower()
                cat = getattr(e, "category", None) or "unknown"
                curr = (at, cat)
                self._counts[prev][curr] += 1.0
                prev = curr

    def neg_log_prob(self, prev: tuple, curr: tuple) -> float:
        dist  = self._counts.get(prev, {})
        total = sum(dist.values()) + _SMOOTHING * 1000
        count = dist.get(curr, 0) + _SMOOTHING
        return -math.log(max(count / total, 1e-9))


_MODELS: dict[str, _MarkovModel] = {}


def fit_all(sessions: list) -> None:
    for persona in ("research", "procurement", "travel"):
        m = _MarkovModel(persona)
        subset = [s for s in sessions if getattr(s, "persona", None) and
                  str(s.persona).replace("Persona.", "").lower() == persona]
        if subset:
            m.fit(subset)
        _MODELS[persona] = m


def _get_model(persona: str) -> _MarkovModel:
    if persona not in _MODELS:
        _MODELS[persona] = _MarkovModel(persona)
    return _MODELS[persona]


class SequenceModelGuard(PreCommerceGuard):
    """
    Persona-conditioned Markov model over (action_type, category) bigrams.
    Can detect reconnaissance (B3) before any payment fires.
    """
    name  = "sequence_model"
    phase = Phase.PRECOMMERCE

    def should_fire(self, action_type: str) -> bool:
        return True  # score every event in the session stream

    def detect(self, event, context: FraudContext) -> Optional[GuardSignal]:
        persona = (
            context.persona
            or getattr(event, "_persona", None)
            or "research"
        )

        at  = str(getattr(event, "action_type", "")).replace("ActionType.", "").lower()
        cat = getattr(event, "category", None) or "unknown"
        curr = (at, cat)

        if context.events:
            prev_e = context.events[-1]
            p_at   = str(getattr(prev_e, "action_type", "")).replace("ActionType.", "").lower()
            p_cat  = getattr(prev_e, "category", None) or "unknown"
            prev   = (p_at, p_cat)
        else:
            prev = (None, None)

        model = _get_model(persona)
        nlp   = model.neg_log_prob(prev, curr)
        score = min(1.0, nlp / 8.0)

        flags: list[str] = []
        if score >= 0.30:
            flags.append(
                f"seq:unusual_transition {prev[0] or 'start'}/{prev[1]}"
                f"→{curr[0]}/{curr[1]} p={math.exp(-nlp):.4f}"
            )

        # Recon signal: many FIND_SERVICE, zero AUTHORIZE yet
        n_finds = sum(1 for e in context.events
                      if str(getattr(e, "action_type", "")).lower().endswith("find_service"))
        n_auths = sum(1 for e in context.events
                      if str(getattr(e, "action_type", "")).lower().endswith("authorize"))
        if n_finds >= 5 and n_auths == 0:
            score = max(score, 0.55)
            flags.append(f"seq:recon_pattern finds={n_finds} auths={n_auths}")

        if score < _MIN_SCORE:
            return None

        return self._signal(event, score, flags)
