"""
Detector: Persona-conditioned Markov sequence model

NOVEL CONTRIBUTION
==================
Existing detectors (velocity, keyword, IsolationForest) score each
transaction in isolation. They cannot see that:
  - 15 catalog lookups with no purchase is a recon pattern (B3)
  - a research bot using a travel service is behavioral drift (B1)
  - progressively escalating amounts across a session is a probe (B7)

We model agent sessions as sequences of (action_type, category) bigrams
and learn their transition probabilities per persona from clean sessions.

Score(event) = -log P(action_t | action_{t-1}, category_{t-1}, persona)

Low probability transitions → suspicious. Novel sequence patterns the
agent has never shown → very suspicious.

This is the simplest form of what could become a full LLM-based session
model. It is:
  1. Stateless at inference (no external model required)
  2. Persona-aware (different expectations per agent role)
  3. Interpretable (flags exactly which transition was unusual)
  4. Trainable from clean sessions with no labels

The class supports two modes:
  - fit(sessions)    — learn from labeled clean sessions
  - from_prior()     — use hand-crafted priors from personas.py
                        (no training data needed for the benchmark)

Reference: inspired by n-gram language model perplexity scoring applied
to agent action sequences rather than token sequences.
"""
import math
from collections import defaultdict
from harness.simulate.schema import Event, ActionType, Persona, Session

name = "sequence_model"

# Prior transition counts derived from PERSONA_CONFIG sequences in personas.py.
# Format: {persona: {(prev_action, prev_category): {(next_action, next_category): count}}}
# These encode the "expected sequence" per persona.

_PRIOR_COUNTS: dict[str, dict[tuple, dict[tuple, float]]] = {
    # Research: find_service → authorize (finance/search), not much else
    "research": {
        (None, None): {
            (ActionType.FIND_SERVICE, "finance"): 80,
            (ActionType.FIND_SERVICE, "search"):  20,
        },
        (ActionType.FIND_SERVICE, "finance"): {
            (ActionType.AUTHORIZE, "finance"):    85,
            (ActionType.GET_SERVICE, "finance"):  10,
            (ActionType.FIND_SERVICE, "finance"): 5,
        },
        (ActionType.FIND_SERVICE, "search"): {
            (ActionType.AUTHORIZE, "search"):     80,
            (ActionType.FIND_SERVICE, "search"):  15,
            (ActionType.AUTHORIZE, "finance"):    5,
        },
        (ActionType.AUTHORIZE, "finance"): {
            (ActionType.FIND_SERVICE, "finance"): 60,
            (ActionType.AUTHORIZE, "finance"):    30,
            (ActionType.SETTLE, "finance"):       10,
        },
        (ActionType.AUTHORIZE, "search"): {
            (ActionType.FIND_SERVICE, "search"):  70,
            (ActionType.AUTHORIZE, "search"):     20,
            (ActionType.SETTLE, "search"):        10,
        },
    },
    # Procurement: find → get → authorize, tightly scoped categories
    "procurement": {
        (None, None): {
            (ActionType.FIND_SERVICE, "procurement"): 70,
            (ActionType.FIND_SERVICE, "search"):      30,
        },
        (ActionType.FIND_SERVICE, "procurement"): {
            (ActionType.GET_SERVICE, "procurement"):  60,
            (ActionType.AUTHORIZE,   "procurement"):  35,
            (ActionType.FIND_SERVICE, "procurement"): 5,
        },
        (ActionType.GET_SERVICE, "procurement"): {
            (ActionType.AUTHORIZE,   "procurement"):  90,
            (ActionType.GET_SERVICE, "procurement"):  10,
        },
        (ActionType.AUTHORIZE, "procurement"): {
            (ActionType.FIND_SERVICE, "procurement"): 70,
            (ActionType.SETTLE, "procurement"):       20,
            (ActionType.AUTHORIZE,   "procurement"):  10,
        },
    },
    # Travel: find → get → get → authorize, high amounts, travel category
    "travel": {
        (None, None): {
            (ActionType.FIND_SERVICE, "travel"):  90,
            (ActionType.FIND_SERVICE, "search"):  10,
        },
        (ActionType.FIND_SERVICE, "travel"): {
            (ActionType.GET_SERVICE, "travel"):   80,
            (ActionType.AUTHORIZE,   "travel"):   15,
            (ActionType.FIND_SERVICE, "travel"):   5,
        },
        (ActionType.GET_SERVICE, "travel"): {
            (ActionType.GET_SERVICE, "travel"):   40,
            (ActionType.AUTHORIZE,   "travel"):   55,
            (ActionType.FIND_SERVICE, "travel"):   5,
        },
        (ActionType.AUTHORIZE, "travel"): {
            (ActionType.SETTLE, "travel"):        70,
            (ActionType.FIND_SERVICE, "travel"):  20,
            (ActionType.AUTHORIZE,   "travel"):   10,
        },
    },
}

# Smoothing: probability mass given to unseen transitions
_SMOOTHING_ALPHA = 0.01


class MarkovSequenceModel:
    """
    Persona-conditioned first-order Markov model over (action_type, category)
    bigrams. Trained from clean sessions or initialised from priors.
    """

    def __init__(self, persona: str):
        self.persona = persona
        # counts[(prev_state, curr_state)] = n
        self._counts: dict[tuple, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
        self._fitted = False
        self._load_prior()

    def _load_prior(self):
        prior = _PRIOR_COUNTS.get(self.persona, {})
        for prev_state, next_dist in prior.items():
            for next_state, count in next_dist.items():
                self._counts[prev_state][next_state] += count
        self._fitted = True

    def fit(self, sessions: list[Session]):
        """
        Update counts from a list of clean sessions.
        Can be called repeatedly (incremental).
        """
        for session in sessions:
            if not session.is_clean:
                continue
            prev_state = (None, None)
            for event in session.events:
                curr_state = (event.action_type, event.category or "unknown")
                self._counts[prev_state][curr_state] += 1.0
                prev_state = curr_state
        self._fitted = True

    def transition_probability(
        self,
        prev_state: tuple,
        curr_state: tuple,
    ) -> float:
        """
        P(curr | prev) with Laplace smoothing.
        Returns probability in [0, 1].
        """
        dist = self._counts.get(prev_state, {})
        total = sum(dist.values()) + _SMOOTHING_ALPHA * 1000
        count = dist.get(curr_state, 0) + _SMOOTHING_ALPHA
        return count / total

    def neg_log_prob(self, prev_state: tuple, curr_state: tuple) -> float:
        p = self.transition_probability(prev_state, curr_state)
        return -math.log(max(p, 1e-9))

    def score_event(
        self,
        event: Event,
        history: list[Event],
    ) -> tuple[float, list[str]]:
        """
        Score one event given its history.
        Returns (risk_score: float, flags: list[str]).
        """
        curr_state = (event.action_type, event.category or "unknown")

        if not history:
            prev_state = (None, None)
        else:
            prev = history[-1]
            prev_state = (prev.action_type, prev.category or "unknown")

        nlp = self.neg_log_prob(prev_state, curr_state)

        # Calibrate: nlp=0 → score=0, nlp=log(1/0.01)=4.6 → score≈0.70
        # nlp=log(1/0.001)=6.9 → score≈0.90
        score = min(1.0, nlp / 8.0)

        flags = []
        if score >= 0.30:
            flags.append(
                f"seq:unusual_transition "
                f"{prev_state[0].value if prev_state[0] else 'start'}"
                f"/{prev_state[1]}→"
                f"{curr_state[0].value}/{curr_state[1]} "
                f"p={self.transition_probability(prev_state, curr_state):.4f}"
            )

        # Additional recon signal: many FIND_SERVICE with no AUTHORIZE
        n_finds = sum(1 for e in history if e.action_type == ActionType.FIND_SERVICE)
        n_auths = sum(1 for e in history if e.action_type == ActionType.AUTHORIZE)
        if n_finds >= 5 and n_auths == 0:
            score = max(score, 0.55)
            flags.append(f"seq:recon_pattern finds={n_finds} auths={n_auths}")

        # Suppress borderline scores — only fire when confident.
        # Recon signal (0.55) and unusual transitions (nlp > 4 → score > 0.50)
        # are kept; noisy borderline transitions are suppressed to avoid FPR.
        if score < 0.50:
            return 0.0, []

        return round(score, 4), flags


# ── Module-level registry (one model per persona) ─────────────────────────

_MODELS: dict[str, MarkovSequenceModel] = {}


def _get_model(persona: str) -> MarkovSequenceModel:
    if persona not in _MODELS:
        _MODELS[persona] = MarkovSequenceModel(persona)
    return _MODELS[persona]


def fit_all(train_sessions: list[Session]):
    """Call once with training data to refine priors."""
    for persona in ("research", "procurement", "travel"):
        subset = [s for s in train_sessions if s.persona.value == persona]
        _get_model(persona).fit(subset)


def detect(event: Event, history: list[Event]) -> tuple[float, list[str]]:
    """
    Replay-engine-compatible interface.
    Persona read from event._persona (injected by evaluate.py before replay).
    """
    persona = getattr(event, "_persona", None)
    if not persona:
        persona = next(
            (getattr(e, "_persona") for e in (history or [])
             if hasattr(e, "_persona")), None
        )
    if not persona:
        cats = [e.category for e in (history or []) if e.category]
        if cats:
            top = max(set(cats), key=cats.count)
            persona = ("travel" if top in ("travel","hotel","flight")
                       else "procurement" if top == "procurement"
                       else "research")
    persona = persona or "research"
    return _get_model(persona).score_event(event, history)
