"""Bivariate Poisson model for football score prediction.

Standard Poisson: home_goals ~ Poisson(λ_h), away_goals ~ Poisson(λ_a) — independent.
Bivariate Poisson: home_goals = X₁ + X₃, away_goals = X₂ + X₃
  where X₁ ~ Poisson(λ₁), X₂ ~ Poisson(λ₂), X₃ ~ Poisson(λ₃)

λ₃ is the "common scoring" component:
- Creates positive correlation between home and away goals
- Naturally boosts draw probabilities (especially 1-1, 2-2)
- When λ₃=0, reduces to standard independent Poisson

Reference: Karlis & Ntzoufras (2003), "Analysis of sports data by using
bivariate Poisson models"
"""

import numpy as np
from scipy.stats import poisson
from scipy.special import comb


def bivariate_poisson_pmf(x, y, lambda1, lambda2, lambda3):
    """P(home=x, away=y) under bivariate Poisson.

    home = X₁ + X₃, away = X₂ + X₃
    P(x,y) = exp(-(λ₁+λ₂+λ₃)) * Σ_{k=0}^{min(x,y)}
              [λ₁^(x-k) * λ₂^(y-k) * λ₃^k] / [(x-k)! * (y-k)! * k!]
    """
    total = 0.0
    for k in range(min(x, y) + 1):
        total += (poisson.pmf(x - k, lambda1) *
                  poisson.pmf(y - k, lambda2) *
                  poisson.pmf(k, lambda3))
    return total


def bivariate_score_matrix(lambda_h, lambda_a, lambda3, max_goals=7,
                           rho=0.0):
    """Build 7x7 score matrix from bivariate Poisson.

    Args:
        lambda_h: Expected home goals (total, including common component)
        lambda_a: Expected away goals (total, including common component)
        lambda3: Common scoring component (correlation parameter)
        max_goals: Maximum goals per team
        rho: Dixon-Coles low-score correction (applied on top)

    Returns:
        (max_goals, max_goals) probability matrix
    """
    # Decompose: λ₁ = λ_h - λ₃, λ₂ = λ_a - λ₃
    # Ensure λ₁, λ₂ > 0
    lambda3_eff = min(lambda3, lambda_h * 0.5, lambda_a * 0.5)
    lambda3_eff = max(lambda3_eff, 0.0)
    lambda1 = lambda_h - lambda3_eff
    lambda2 = lambda_a - lambda3_eff

    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i, j] = bivariate_poisson_pmf(i, j, lambda1, lambda2,
                                                  lambda3_eff)

    # Apply Dixon-Coles low-score correction
    if rho != 0:
        lh, la = lambda_h, lambda_a
        if matrix[0, 0] > 0:
            matrix[0, 0] *= max(0, 1 - lh * la * rho)
        if matrix[0, 1] > 0:
            matrix[0, 1] *= max(0, 1 + lh * rho)
        if matrix[1, 0] > 0:
            matrix[1, 0] *= max(0, 1 + la * rho)
        if matrix[1, 1] > 0:
            matrix[1, 1] *= max(0, 1 - rho)

    # Normalize
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


def matrix_to_1x2(matrix):
    """Convert score matrix to [P(home), P(draw), P(away)]."""
    p_home = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix)
    p_away = np.triu(matrix, 1).sum()
    p = np.array([p_home, p_draw, p_away])
    total = p.sum()
    if total > 0:
        p /= total
    return p
