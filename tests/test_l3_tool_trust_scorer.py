from fraud_detection.l3_tool_trust_scorer import ToolTrustScorer


def test_tool_baseline_registration_and_scoring():
    scorer = ToolTrustScorer()
    scorer.register_tool_baseline("price_lookup", {"price": 50.0, "currency": "USD"}, 100.0)
    result = scorer.score_call("price_lookup", {"price": 51.0, "currency": "USD"}, 105.0)
    assert 0.0 <= result["trust_score"] <= 1.0


def test_validate_response_constraints_detects_price_out_of_range():
    scorer = ToolTrustScorer()
    result = scorer.validate_response_constraints(
        "price_lookup",
        {"price": 120.0},
        {"price_range": (10.0, 100.0)},
    )
    assert result["valid"] is False
    assert "price" in result["errors"][0]
