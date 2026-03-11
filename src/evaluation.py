"""Evaluation metrics for probability predictions."""

import numpy as np


def ranked_probability_score(probs, actual_outcome):
    """Compute RPS for a single prediction.

    Standard formula: RPS = 1/(K-1) * sum_{r=1}^{K-1} (F_r - O_r)^2
    where K=3 (home/draw/away), so we sum the first K-1=2 cumulative terms.

    Args:
        probs: array [P(home), P(draw), P(away)]
        actual_outcome: 0=home win, 1=draw, 2=away win

    Returns:
        RPS value in [0, 1], lower is better.
    """
    k = len(probs)
    cumulative_pred = np.cumsum(probs)
    cumulative_actual = np.cumsum([1.0 if i == actual_outcome else 0.0
                                   for i in range(k)])
    # Standard RPS: only first K-1 terms (last is always 1 vs 1 = 0)
    return np.sum((cumulative_pred[:k - 1] - cumulative_actual[:k - 1]) ** 2) / (k - 1)


def brier_score(probs, actual_outcome, n_classes=3):
    """Compute Brier score for a single prediction."""
    actual_onehot = np.zeros(n_classes)
    actual_onehot[actual_outcome] = 1
    return np.mean((np.asarray(probs) - actual_onehot) ** 2)


def log_loss_single(probs, actual_outcome):
    """Compute log loss for a single prediction."""
    p = np.clip(probs[actual_outcome], 1e-10, 1.0)
    return -np.log(p)


def match_outcome(home_goals, away_goals):
    """Convert goals to outcome: 0=home, 1=draw, 2=away."""
    if home_goals > away_goals:
        return 0
    elif home_goals == away_goals:
        return 1
    else:
        return 2


def evaluate_predictions(all_probs, all_outcomes):
    """Evaluate a batch of predictions.

    Args:
        all_probs: (N, 3) array of [P(H), P(D), P(A)]
        all_outcomes: (N,) array of outcomes (0/1/2)

    Returns:
        dict with rps, brier, log_loss, accuracy, n_matches
    """
    all_probs = np.asarray(all_probs)
    all_outcomes = np.asarray(all_outcomes)
    n = len(all_outcomes)

    rps_vals = [ranked_probability_score(all_probs[i], all_outcomes[i]) for i in range(n)]
    brier_vals = [brier_score(all_probs[i], all_outcomes[i]) for i in range(n)]
    ll_vals = [log_loss_single(all_probs[i], all_outcomes[i]) for i in range(n)]

    predicted = np.argmax(all_probs, axis=1)
    accuracy = (predicted == all_outcomes).mean()

    return {
        "rps": np.mean(rps_vals),
        "brier": np.mean(brier_vals),
        "log_loss": np.mean(ll_vals),
        "accuracy": accuracy,
        "n_matches": n,
    }
