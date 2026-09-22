import numpy as np

from branchmot import association_calibration


def test_perfect_predictions_have_zero_error() -> None:
    result = association_calibration(np.eye(3), np.array([0, 1, 2]))
    assert result == {"accuracy": 1.0, "ece": 0.0, "brier": 0.0, "nll": 0.0}


def test_metrics_detect_overconfident_errors() -> None:
    result = association_calibration(
        np.array([[0.9, 0.1], [0.9, 0.1]]), np.array([0, 1]), n_bins=5
    )
    assert result["accuracy"] == 0.5
    assert result["ece"] == 0.4
    assert result["brier"] > 0.8

