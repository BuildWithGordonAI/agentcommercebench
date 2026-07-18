from fraud_detection.risk_aggregator import RiskAggregator


def test_aggregate_combines_layer_scores():
    aggregator = RiskAggregator()
    score = aggregator.aggregate({"L1": 0.1, "L2": 0.8, "L3": 0.7, "L4": 0.6, "L5": 0.2, "L6": 0.4})
    assert 0.0 <= score <= 1.0


def test_decision_returns_hitl_for_midrange_score():
    aggregator = RiskAggregator()
    decision = aggregator.decision(0.6)
    assert decision["decision"] == "HITL"
