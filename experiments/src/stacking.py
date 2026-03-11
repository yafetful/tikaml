"""Stacking ensemble: Dixon-Coles + LightGBM + Elo → Meta-Learner → Isotonic Calibration.

Meta-features (~27+ dimensions):
- 3 models × [P(H), P(D), P(A)] = 9
- 3 models × [λ_home, λ_away] = 6
- 3 models × std of probabilities = 3 (model agreement)
- 3 models × [P(>2.5 goals), P(BTTS), P(most_likely_score)] = 9
- Pairwise probability differences = 6
- Average probabilities = 3
Total: ~36 meta-features
"""

import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb


class StackedPredictor:

    def __init__(self, meta_learner="lgbm"):
        """Initialize stacked predictor.

        Args:
            meta_learner: "lgbm" (LightGBM multiclass) or "lr" (LogisticRegression)
        """
        self.meta_learner_type = meta_learner
        self.meta_model = None
        self.calibrators = {}

    @staticmethod
    def _score_matrix_from_lambdas(lh, la, max_goals=7):
        """Build score matrix from lambda parameters (no rho correction)."""
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    @staticmethod
    def _matrix_features(matrix):
        """Extract summary features from a score probability matrix."""
        # P(total goals > 2.5)
        over_25 = sum(
            matrix[i, j] for i in range(matrix.shape[0])
            for j in range(matrix.shape[1]) if i + j >= 3
        )

        # P(BTTS - both teams to score)
        btts = 1.0 - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0]

        # Confidence: probability of most likely score
        confidence = matrix.max()

        return [over_25, btts, confidence]

    @staticmethod
    def build_meta_features(dc_probs, dc_lambdas,
                            lgbm_probs, lgbm_lambdas,
                            elo_probs, elo_lambdas):
        """Build meta-feature matrix from base model predictions.

        Args:
            *_probs: (N, 3) arrays of [P(H), P(D), P(A)]
            *_lambdas: (N, 2) arrays of [λ_home, λ_away]

        Returns:
            (N, ~36) meta-feature array
        """
        n = len(dc_probs)
        features = []

        for i in range(n):
            row = []

            # 1. Base probabilities (9 dims)
            row.extend(dc_probs[i])
            row.extend(lgbm_probs[i])
            row.extend(elo_probs[i])

            # 2. Lambda parameters (6 dims)
            row.extend(dc_lambdas[i])
            row.extend(lgbm_lambdas[i])
            row.extend(elo_lambdas[i])

            # 3. Model agreement / disagreement (3 dims)
            all_probs = np.array([dc_probs[i], lgbm_probs[i], elo_probs[i]])
            row.extend(np.std(all_probs, axis=0))

            # 4. Score matrix summaries (9 dims: 3 per model)
            for lambdas in [dc_lambdas[i], lgbm_lambdas[i], elo_lambdas[i]]:
                matrix = StackedPredictor._score_matrix_from_lambdas(
                    lambdas[0], lambdas[1])
                row.extend(StackedPredictor._matrix_features(matrix))

            # 5. Pairwise probability differences (6 dims)
            # DC vs LGBM, DC vs Elo, LGBM vs Elo for home and away
            row.append(dc_probs[i][0] - lgbm_probs[i][0])  # home diff DC-LGBM
            row.append(dc_probs[i][2] - lgbm_probs[i][2])  # away diff DC-LGBM
            row.append(dc_probs[i][0] - elo_probs[i][0])    # home diff DC-Elo
            row.append(dc_probs[i][2] - elo_probs[i][2])    # away diff DC-Elo
            row.append(lgbm_probs[i][0] - elo_probs[i][0])  # home diff LGBM-Elo
            row.append(lgbm_probs[i][2] - elo_probs[i][2])  # away diff LGBM-Elo

            # 6. Average probabilities (3 dims)
            row.extend(np.mean(all_probs, axis=0))

            features.append(row)

        return np.array(features)

    def fit(self, meta_features, outcomes, calibration_split=0.3):
        """Train meta-learner and calibrators.

        Uses temporal split: last calibration_split fraction for calibration.
        """
        n = len(outcomes)
        split_idx = int(n * (1 - calibration_split))

        meta_train = meta_features[:split_idx]
        meta_cal = meta_features[split_idx:]
        y_train = outcomes[:split_idx]
        y_cal = outcomes[split_idx:]

        if self.meta_learner_type == "lgbm":
            # LightGBM multiclass meta-learner
            # Further split train for early stopping
            es_split = int(len(meta_train) * 0.85)
            X_fit = meta_train[:es_split]
            X_es = meta_train[es_split:]
            y_fit = y_train[:es_split]
            y_es = y_train[es_split:]

            self.meta_model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                verbose=-1,
            )
            self.meta_model.fit(
                X_fit, y_fit,
                eval_set=[(X_es, y_es)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        else:
            # Logistic Regression meta-learner
            self.meta_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
            )
            self.meta_model.fit(meta_train, y_train)

        # Calibrate on held-out set
        raw_probs_cal = self.meta_model.predict_proba(meta_cal)
        labels = ["home", "draw", "away"]
        for i, label in enumerate(labels):
            y_binary = (y_cal == i).astype(int)
            cal = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            cal.fit(raw_probs_cal[:, i], y_binary)
            self.calibrators[label] = cal

        return self

    def predict(self, meta_features):
        """Predict calibrated probabilities.

        Returns:
            (N, 3) array of [P(home), P(draw), P(away)]
        """
        raw_probs = self.meta_model.predict_proba(meta_features)
        calibrated = np.zeros_like(raw_probs)

        for i, label in enumerate(["home", "draw", "away"]):
            calibrated[:, i] = self.calibrators[label].predict(raw_probs[:, i])

        # Normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        calibrated /= row_sums

        return calibrated
