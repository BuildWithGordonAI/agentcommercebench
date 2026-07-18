from fraud_detection.l2_trajectory_monitor import TrajectoryMonitor


def test_step_verdict_suspect_for_unauthorized_action():
    monitor = TrajectoryMonitor()
    result = monitor.step_verdict(
        thought="The user asked for a refund.",
        action={"tool": "wallet.pay", "args": {"amount": 500}},
        observation="The system confirms payment.",
        consent_envelope={"max_amount": 100, "block_amount": 250},
    )
    assert result["verdict"] in {"SUSPECT", "BLOCK"}


def test_spend_velocity_score_defaults_before_fit():
    monitor = TrajectoryMonitor()
    score = monitor.spend_velocity_score([10.0, 1.0, 1.0])
    assert 0.0 <= score <= 1.0
