from .pipeline import DetectorPipeline, DetectorResult
from .l1_payload import L1PayloadClassifier
from .l4_price import L4PriceOracle
from .l3_behavioral import L3BehavioralFingerprint

def build_pipeline(persona: str = None) -> DetectorPipeline:
    """Build the default L1+L3+L4 pipeline for a given persona."""
    detectors = [
        L1PayloadClassifier(),
        L4PriceOracle(),
    ]
    if persona:
        detectors.append(L3BehavioralFingerprint(persona))
    return DetectorPipeline(detectors)
