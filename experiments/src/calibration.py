"""Post-prediction probability calibration.

Addresses two key issues found in error analysis:
1. Strong favorites are underestimated (P>=0.70: pred=0.755, actual=0.798)
2. Model never predicts draws as most likely outcome

Uses isotonic regression per outcome for calibration.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """Calibrate 3-way probabilities using isotonic regression per outcome."""

    def __init__(self):
        self.calibrators = {}  # outcome_idx -> IsotonicRegression

    def fit(self, probs, outcomes):
        """Train calibrators on validation data.

        Args:
            probs: (N, 3) predicted probabilities [P(H), P(D), P(A)]
            outcomes: (N,) actual outcomes (0=home, 1=draw, 2=away)
        """
        for i, label in enumerate(["home", "draw", "away"]):
            y_binary = (outcomes == i).astype(float)
            cal = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
            cal.fit(probs[:, i], y_binary)
            self.calibrators[i] = cal

        return self

    def predict(self, probs):
        """Apply calibration to predicted probabilities.

        Args:
            probs: (N, 3) predicted probabilities

        Returns:
            (N, 3) calibrated and normalized probabilities
        """
        calibrated = np.zeros_like(probs)
        for i in range(3):
            calibrated[:, i] = self.calibrators[i].predict(probs[:, i])

        # Normalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        calibrated /= row_sums

        return calibrated
