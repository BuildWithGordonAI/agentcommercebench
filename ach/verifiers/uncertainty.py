"""
Uncertainty estimation for ACP verifiers.

Three epistemic uncertainty sources:
  cold_start   — agent_id or persona seen for first time in this estimator
  ood_category — action category outside the persona's observed distribution
  near_boundary — score within `boundary_width` of the decision threshold

Aleatory uncertainty (irreducible noise):
  verifier_disagreement — computed by CombinedVerifier across parallel verifiers

Confidence interval construction:
  Production: replace _heuristic_ci with conformal prediction (fit on calibration set).
  Here we use a conservative heuristic: CI half-width grows linearly with total uncertainty.

Usage:
    est = UncertaintyEstimator()
    ue  = est.estimate(score=0.73, context=ctx, action=act, is_new_session=True)
    # ue.total, ue.ci_low, ue.ci_high, ue.sources
"""
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field

DECISION_THRESHOLD = 0.50
BLOCK_THRESHOLD    = 0.70


@dataclass
class UncertaintyEstimate:
    epistemic:  float
    aleatory:   float
    total:      float
    ci_low:     float
    ci_high:    float
    confidence: float          # 1 - total
    sources:    list[str] = field(default_factory=list)


class UncertaintyEstimator:
    """
    Lightweight, stateful estimator.  One instance per verifier.

    State tracked:
      _seen_agents   — set of agent_ids ever observed
      _seen_cats     — per-persona set of categories ever observed
    """

    def __init__(
        self,
        cold_start_boost:  float = 0.30,
        ood_boost:         float = 0.22,
        boundary_width:    float = 0.18,   # distance from threshold that counts as "near"
        boundary_peak_unc: float = 0.25,   # max uncertainty added at exact boundary
        min_ci_half:       float = 0.08,   # minimum CI half-width even when certain
    ):
        self.cold_start_boost  = cold_start_boost
        self.ood_boost         = ood_boost
        self.boundary_width    = boundary_width
        self.boundary_peak_unc = boundary_peak_unc
        self.min_ci_half       = min_ci_half
        self._seen_agents: set[str] = set()
        self._seen_cats: defaultdict[str, set[str]] = defaultdict(set)

    # ── public API ─────────────────────────────────────────────────────────────

    def estimate(
        self,
        score:          float,
        agent_id:       str,
        persona:        str,
        category:       str | None = None,
        is_new_session: bool = False,
        aleatory:       float = 0.0,
    ) -> UncertaintyEstimate:
        sources: list[str] = []
        epistemic = 0.0

        # --- cold-start ---
        if is_new_session or agent_id not in self._seen_agents:
            epistemic += self.cold_start_boost
            sources.append("cold_start")
            self._seen_agents.add(agent_id)

        # --- out-of-distribution category ---
        if category:
            persona_cats = self._seen_cats[persona]
            if category not in persona_cats:
                epistemic += self.ood_boost
                sources.append(f"ood:{category}")
                persona_cats.add(category)

        # --- near decision boundary ---
        for threshold in (DECISION_THRESHOLD, BLOCK_THRESHOLD):
            dist = abs(score - threshold)
            if dist < self.boundary_width:
                fraction   = 1.0 - dist / self.boundary_width
                added      = fraction * self.boundary_peak_unc
                epistemic += added
                sources.append(f"near_boundary:{score:.2f}")
                break   # only count the closest threshold once

        epistemic = min(epistemic, 0.90)
        total     = min(math.sqrt(epistemic ** 2 + aleatory ** 2), 1.0)

        ci_low, ci_high = self._heuristic_ci(score, total)

        return UncertaintyEstimate(
            epistemic  = round(epistemic, 4),
            aleatory   = round(aleatory, 4),
            total      = round(total, 4),
            ci_low     = round(ci_low, 4),
            ci_high    = round(ci_high, 4),
            confidence = round(max(0.0, 1.0 - total), 4),
            sources    = sources,
        )

    def reset(self) -> None:
        """Clear all state (use between benchmark runs)."""
        self._seen_agents.clear()
        self._seen_cats.clear()

    # ── internal ───────────────────────────────────────────────────────────────

    def _heuristic_ci(self, score: float, total_uncertainty: float) -> tuple[float, float]:
        """
        Conservative heuristic CI.
        Production: replace with conformal prediction on a calibration set.

        Conformal recipe (for future):
          1. Run verifier on held-out calibration set, record nonconformity scores.
          2. Find q = quantile(nonconformity_scores, 1 - alpha).
          3. CI = [score - q, score + q], clipped to [0, 1].
          4. Guarantees marginal coverage: P(true_label in CI) >= 1 - alpha.
        """
        half = self.min_ci_half + total_uncertainty * 0.35
        return max(0.0, score - half), min(1.0, score + half)


# ── disagreement utility ───────────────────────────────────────────────────────

def disagreement_uncertainty(verifications: list) -> float:
    """
    Aleatory uncertainty from verifier ensemble disagreement.
    Returns score variance across verifiers, normalised to [0, 1].
    """
    if len(verifications) < 2:
        return 0.0
    scores = [v.score for v in verifications]
    mean   = sum(scores) / len(scores)
    var    = sum((s - mean) ** 2 for s in scores) / len(scores)
    # sqrt(var) = std dev; max possible std dev on [0,1] is 0.5
    return round(min(math.sqrt(var) / 0.5, 1.0), 4)
