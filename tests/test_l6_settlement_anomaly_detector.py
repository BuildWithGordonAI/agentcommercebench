from fraud_detection.l6_settlement_anomaly_detector import SettlementAnomalyDetector


def test_detect_circular_flow_finds_cycle():
    detector = SettlementAnomalyDetector()
    detector.add_transaction("A", "B", 10.0, "tx1", "2026-06-27T12:00:00Z", "confirmed")
    detector.add_transaction("B", "A", 10.0, "tx2", "2026-06-27T12:05:00Z", "confirmed")
    cycles = detector.detect_circular_flow()
    assert any(set(cycle) == {"A", "B"} for cycle in cycles)


def test_aggregate_anomaly_score_in_range():
    detector = SettlementAnomalyDetector()
    score = detector.aggregate_anomaly_score()
    assert 0.0 <= score <= 1.0
