"""
Detector pipeline — runs multiple detectors, returns max risk score.

Interface contract:
    detector(tool, params, response) → (risk_score: float, flags: list[str])
    risk_score ∈ [0, 1]
    0.00–0.29 → ALLOW
    0.30–0.69 → ESCALATE
    0.70–1.00 → BLOCK
"""
from dataclasses import dataclass, field


@dataclass
class DetectorResult:
    risk_score: float
    flags: list[str]
    layer_scores: dict[str, float]  # per-layer breakdown

    @property
    def decision(self) -> str:
        if self.risk_score >= 0.70:
            return "BLOCK"
        if self.risk_score >= 0.30:
            return "ESCALATE"
        return "ALLOW"

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 3),
            "decision":   self.decision,
            "flags":      self.flags,
            "layers":     {k: round(v, 3) for k, v in self.layer_scores.items()},
        }


class DetectorPipeline:
    """
    Runs N detectors in parallel and returns the highest risk score
    across all layers (conservative — any layer can block).
    """

    def __init__(self, detectors: list):
        self.detectors = detectors

    def __call__(self, tool: str, params: dict, response: dict):
        """Drop-in for GordonMCPClient.detector."""
        result = self.score(tool, params, response)
        return result.risk_score, result.flags

    def score(self, tool: str, params: dict, response: dict) -> DetectorResult:
        all_flags: list[str] = []
        layer_scores: dict[str, float] = {}

        for det in self.detectors:
            score, flags = det(tool, params, response)
            layer_scores[det.name] = score
            all_flags.extend(flags)

        max_score = max(layer_scores.values()) if layer_scores else 0.0
        return DetectorResult(
            risk_score=max_score,
            flags=all_flags,
            layer_scores=layer_scores,
        )

    def add(self, detector):
        self.detectors.append(detector)
        return self
