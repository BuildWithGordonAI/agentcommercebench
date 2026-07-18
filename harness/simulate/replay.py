"""
Replay engine — feeds any recorded session through a detector pipeline.

Key properties:
  - Deterministic: same session file → same detector inputs every time
  - Composable: swap detector configs without touching session data
  - Comparable: run same session against Gordon vs. OSS baselines
  - Appendable: prod sessions can be loaded and replayed identically

Usage:
    session = Session.load("data/sessions/clean_001.json")

    # replay through Gordon L1-L5
    result = replay(session, pipeline=GORDON_PIPELINE)

    # replay the exact same session through OSS baseline
    result_oss = replay(session, pipeline=OSS_ISOLATION_FOREST)

    # replay with attack injected at event index 3
    attacked = inject(session, scenario="A1", at_event=3)
    result_attacked = replay(attacked, pipeline=GORDON_PIPELINE)
"""
from dataclasses import dataclass
from typing import Callable, Optional
from .schema import Session, Event, Decision
import time


@dataclass
class DetectorResult:
    event_id:       str
    risk_score:     float           # 0.0 = clean, 1.0 = certain fraud
    risk_flags:     list[str]
    decision:       Decision
    latency_ms:     float
    layer_scores:   dict[str, float]  # {"L1": 0.9, "L2": 0.1, ...}


@dataclass
class ReplayResult:
    session_id:         str
    scenario_id:        Optional[str]
    is_clean:           bool
    event_results:      list[DetectorResult]
    session_risk_score: float           # max risk across all events
    true_positive:      Optional[bool]  # attacked + caught
    false_positive:     Optional[bool]  # clean + fired
    latency_p99_ms:     float
    detector_name:      str


# ── Decision thresholds ────────────────────────────────────────────────────
ALLOW_THRESHOLD   = 0.30
ESCALATE_THRESHOLD = 0.70

def _make_decision(score: float) -> Decision:
    if score >= ESCALATE_THRESHOLD:
        return Decision.BLOCK
    if score >= ALLOW_THRESHOLD:
        return Decision.ESCALATE
    return Decision.ALLOW


# ── Replay core ────────────────────────────────────────────────────────────
def replay(
    session: Session,
    pipeline: list[Callable],           # ordered list of detector functions
    detector_name: str = "gordon",
    verbose: bool = False,
) -> ReplayResult:
    """
    Feed session events through the detector pipeline in order.
    Each detector receives (event, session_history_so_far) and returns
    a (score: float, flags: list[str]) tuple.

    Detectors are called in parallel conceptually but sequentially here
    (parallelism is the production concern, not the harness concern).
    """
    event_results = []
    history: list[Event] = []          # growing context window for L2/L3

    for event in session.events:
        t_start = time.perf_counter()
        layer_scores = {}
        all_flags = []

        raw_scores = []
        for detector in pipeline:
            score, flags = detector(event, history)
            det_key = f"{detector.__module__}.{detector.__name__}"
            layer_scores[det_key] = round(score, 4)
            raw_scores.append(score)
            all_flags.extend(flags)

        agg_score = max(raw_scores) if raw_scores else 0.0

        latency_ms = (time.perf_counter() - t_start) * 1000
        decision = _make_decision(agg_score)

        result = DetectorResult(
            event_id=event.event_id,
            risk_score=round(agg_score, 4),
            risk_flags=list(set(all_flags)),
            decision=decision,
            latency_ms=round(latency_ms, 2),
            layer_scores=layer_scores,
        )
        event_results.append(result)

        # Write scores back onto event for downstream use
        event.risk_score = agg_score
        event.risk_flags = list(set(all_flags))
        event.decision = decision
        event.detector_scores = layer_scores

        history.append(event)

        if verbose:
            print(f"  [{event.action_type.value:12}] score={agg_score:.3f} "
                  f"decision={decision.value} flags={all_flags} "
                  f"latency={latency_ms:.1f}ms")

    session_risk = max((r.risk_score for r in event_results), default=0.0)
    latencies = [r.latency_ms for r in event_results]
    latency_p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0.0

    # Ground truth evaluation
    attacked_events = [e for e in session.events if e.is_injected]
    caught_events = [e for e in session.events if e.is_injected and e.risk_score and e.risk_score >= ALLOW_THRESHOLD]

    true_positive = None
    false_positive = None
    if not session.is_clean:
        true_positive = len(caught_events) > 0
    else:
        false_positive = session_risk >= ALLOW_THRESHOLD

    return ReplayResult(
        session_id=session.session_id,
        scenario_id=session.scenario_id,
        is_clean=session.is_clean,
        event_results=event_results,
        session_risk_score=round(session_risk, 4),
        true_positive=true_positive,
        false_positive=false_positive,
        latency_p99_ms=latency_p99,
        detector_name=detector_name,
    )


# ── Multi-session replay ───────────────────────────────────────────────────
def replay_batch(
    sessions: list[Session],
    pipeline: list[Callable],
    detector_name: str = "gordon",
) -> list[ReplayResult]:
    """Replay a list of sessions and return all results."""
    return [replay(s, pipeline, detector_name) for s in sessions]


def compare(
    session: Session,
    pipelines: dict[str, list[Callable]],
) -> dict[str, ReplayResult]:
    """
    Run the same session through multiple detector configurations.
    This is how we produce the OSS comparison table.

    Example:
        compare(session, {
            "gordon":           GORDON_PIPELINE,
            "isolation_forest": OSS_ISOLATION_FOREST,
            "keyword_filter":   OSS_KEYWORD_FILTER,
            "gpt4o_zeroshot":   OSS_GPT4O,
        })
    """
    # Always clone before replay so detector writes don't cross-contaminate
    return {
        name: replay(session.clone(), pipeline, detector_name=name)
        for name, pipeline in pipelines.items()
    }
