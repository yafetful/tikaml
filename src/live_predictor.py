"""Live (in-play) match prediction engine.

Uses Bayesian updating on top of pre-match Poisson model predictions.
The pre-match λ_home and λ_away serve as priors; in-game events
(goals, red cards, time elapsed) update the remaining-time Poisson rates.

Usage:
    from src.live_predictor import LivePredictor

    lp = LivePredictor(lambda_home=1.69, lambda_away=1.08)
    lp.update(minute=25, home_goals=1, away_goals=0)
    result = lp.get_probabilities()
"""

import numpy as np
from scipy.stats import poisson


# Red card impact multipliers
# Optimized on 150 EPL+LL matches vs Opta livePredictions.
# Much stronger than literature values (0.85/1.10) — red cards have
# a larger effect on scoring rates than previously assumed.
RED_CARD_FACTOR_SELF = 0.61   # scoring rate multiplier for team with red card
RED_CARD_FACTOR_OPP = 1.46    # scoring rate multiplier for opponent


class LivePredictor:
    """Real-time match probability engine using Bayesian Poisson updating."""

    def __init__(self, lambda_home, lambda_away, rho=-0.10,
                 total_minutes=90, max_goals=7):
        """Initialize with pre-match model predictions.

        Args:
            lambda_home: Pre-match expected home goals (from LGBMPoissonModel).
            lambda_away: Pre-match expected away goals.
            rho: Dixon-Coles low-score correction parameter.
            total_minutes: Match length (90 for regular time).
            max_goals: Maximum goals to consider in score matrix.
        """
        self.lambda_home_full = lambda_home
        self.lambda_away_full = lambda_away
        self.rho = rho
        self.total_minutes = total_minutes
        self.max_goals = max_goals

        # Current match state
        self.minute = 0
        self.home_goals = 0
        self.away_goals = 0
        self.home_red_cards = 0
        self.away_red_cards = 0

        # Event log
        self.events = []

    def update(self, minute, home_goals=None, away_goals=None,
               home_red_cards=None, away_red_cards=None):
        """Update match state.

        Args:
            minute: Current match minute (0-90+).
            home_goals: Current home team score.
            away_goals: Current away team score.
            home_red_cards: Total home red cards so far.
            away_red_cards: Total away red cards so far.
        """
        self.minute = min(minute, self.total_minutes)
        if home_goals is not None:
            self.home_goals = home_goals
        if away_goals is not None:
            self.away_goals = away_goals
        if home_red_cards is not None:
            self.home_red_cards = home_red_cards
        if away_red_cards is not None:
            self.away_red_cards = away_red_cards

    def _remaining_lambdas(self):
        """Calculate expected remaining goals for each team.

        Adjusts the full-match λ by:
        1. Non-linear time decay (r^0.82)
        2. Score-aware momentum with 6 buckets
        3. Red card impact

        All parameters optimized via differential evolution on 150 EPL+LL
        matches against Opta livePredictions (MAE=0.0298).
        """
        if self.minute >= self.total_minutes:
            return 0.0, 0.0

        # Non-linear time decay: r^0.82
        r = (self.total_minutes - self.minute) / self.total_minutes
        r_adj = r ** 0.82

        # Base remaining expected goals
        lambda_h = self.lambda_home_full * r_adj
        lambda_a = self.lambda_away_full * r_adj

        # Score-aware momentum adjustment (6 buckets, optimized)
        score_diff = self.home_goals - self.away_goals
        if score_diff <= -3:       # home losing by 3+ → desperate push
            lambda_h *= 1.40
            lambda_a *= 0.85
        elif score_diff == -2:     # home losing by 2 → strong push
            lambda_h *= 1.13
            lambda_a *= 1.09
        elif score_diff == -1:     # home losing by 1 → moderate push
            lambda_h *= 1.00
            lambda_a *= 0.87
        elif score_diff == 1:      # home winning by 1 → slight caution
            lambda_h *= 0.88
            lambda_a *= 1.03
        elif score_diff == 2:      # home winning by 2 → sit back
            lambda_h *= 0.87
            lambda_a *= 1.06
        elif score_diff >= 3:      # home winning by 3+ → cruise control
            lambda_h *= 0.85
            lambda_a *= 1.01

        # Red card adjustments (cumulative per card)
        for _ in range(self.home_red_cards):
            lambda_h *= RED_CARD_FACTOR_SELF
            lambda_a *= RED_CARD_FACTOR_OPP

        for _ in range(self.away_red_cards):
            lambda_a *= RED_CARD_FACTOR_SELF
            lambda_h *= RED_CARD_FACTOR_OPP

        return max(lambda_h, 0.01), max(lambda_a, 0.01)

    def _remaining_score_matrix(self):
        """Build probability matrix for REMAINING goals (not total)."""
        lambda_h, lambda_a = self._remaining_lambdas()
        mg = self.max_goals

        matrix = np.zeros((mg, mg))
        for i in range(mg):
            for j in range(mg):
                matrix[i, j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)

        # Dixon-Coles correction on remaining goals (only for low scores)
        # Linear decay: ρ effect diminishes as match progresses
        r = (self.total_minutes - self.minute) / self.total_minutes
        rho_adj = self.rho * r
        for i in range(min(2, mg)):
            for j in range(min(2, mg)):
                tau = self._tau(i, j, lambda_h, lambda_a, rho_adj)
                matrix[i, j] *= tau

        matrix /= matrix.sum()
        return matrix

    @staticmethod
    def _tau(i, j, lh, la, rho):
        """Dixon-Coles low-score correction."""
        if i == 0 and j == 0:
            return 1 - lh * la * rho
        elif i == 0 and j == 1:
            return 1 + lh * rho
        elif i == 1 and j == 0:
            return 1 + la * rho
        elif i == 1 and j == 1:
            return 1 - rho
        return 1.0

    def get_probabilities(self):
        """Calculate current match probabilities.

        Returns:
            dict with:
                - minute: current minute
                - score: (home_goals, away_goals)
                - probs_1x2: [P(home_win), P(draw), P(away_win)]
                - over_under: {2.5: {over, under}, 3.5: {over, under}}
                - next_goal: {home, away, none}
                - remaining_matrix: probability matrix for remaining goals
                - lambda_remaining: (λ_home_remaining, λ_away_remaining)
        """
        if self.minute >= self.total_minutes:
            # Match is over - deterministic result
            if self.home_goals > self.away_goals:
                p1x2 = np.array([1.0, 0.0, 0.0])
            elif self.home_goals == self.away_goals:
                p1x2 = np.array([0.0, 1.0, 0.0])
            else:
                p1x2 = np.array([0.0, 0.0, 1.0])

            total = self.home_goals + self.away_goals
            return {
                "minute": self.minute,
                "score": (self.home_goals, self.away_goals),
                "probs_1x2": p1x2,
                "over_under": {
                    2.5: {"over": float(total > 2.5), "under": float(total <= 2.5)},
                    3.5: {"over": float(total > 3.5), "under": float(total <= 3.5)},
                },
                "next_goal": {"home": 0.0, "away": 0.0, "none": 1.0},
                "remaining_matrix": np.zeros((self.max_goals, self.max_goals)),
                "lambda_remaining": (0.0, 0.0),
            }

        # Remaining goals matrix
        rem_matrix = self._remaining_score_matrix()
        mg = self.max_goals

        # 1x2 probabilities based on current score + remaining goals
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0

        for i in range(mg):
            for j in range(mg):
                final_h = self.home_goals + i
                final_a = self.away_goals + j
                p = rem_matrix[i, j]
                if final_h > final_a:
                    p_home += p
                elif final_h == final_a:
                    p_draw += p
                else:
                    p_away += p

        p1x2 = np.array([p_home, p_draw, p_away])
        p1x2 /= p1x2.sum()

        # Over/Under
        current_total = self.home_goals + self.away_goals
        over_under = {}
        for line in [2.5, 3.5]:
            needed = line - current_total
            if needed <= 0:
                # Already over
                over_under[line] = {"over": 1.0, "under": 0.0}
            else:
                # P(remaining goals >= needed)
                p_over = 0.0
                for i in range(mg):
                    for j in range(mg):
                        if i + j >= needed:
                            p_over += rem_matrix[i, j]
                over_under[line] = {"over": p_over, "under": 1.0 - p_over}

        # Next goal probabilities
        lambda_h, lambda_a = self._remaining_lambdas()
        total_lambda = lambda_h + lambda_a

        if total_lambda > 0:
            # P(no more goals) = P(0,0) in remaining matrix
            p_no_goals = rem_matrix[0, 0]
            p_home_next = (lambda_h / total_lambda) * (1 - p_no_goals)
            p_away_next = (lambda_a / total_lambda) * (1 - p_no_goals)
        else:
            p_no_goals = 1.0
            p_home_next = 0.0
            p_away_next = 0.0

        next_goal = {
            "home": p_home_next,
            "away": p_away_next,
            "none": p_no_goals,
        }

        return {
            "minute": self.minute,
            "score": (self.home_goals, self.away_goals),
            "probs_1x2": p1x2,
            "over_under": over_under,
            "next_goal": next_goal,
            "remaining_matrix": rem_matrix,
            "lambda_remaining": (lambda_h, lambda_a),
        }

    def simulate_timeline(self, events=None):
        """Simulate a full match timeline and return probabilities at each point.

        Args:
            events: List of dicts with keys: minute, home_goals, away_goals,
                    home_red_cards (optional), away_red_cards (optional).
                    If None, simulates 0-0 draw with no events.

        Returns:
            List of probability snapshots at each minute.
        """
        if events is None:
            # Default: snapshot every 5 minutes, no events
            events = [{"minute": m, "home_goals": 0, "away_goals": 0}
                      for m in range(0, 91, 5)]

        timeline = []
        for event in events:
            self.update(
                minute=event["minute"],
                home_goals=event.get("home_goals", self.home_goals),
                away_goals=event.get("away_goals", self.away_goals),
                home_red_cards=event.get("home_red_cards", self.home_red_cards),
                away_red_cards=event.get("away_red_cards", self.away_red_cards),
            )
            probs = self.get_probabilities()
            # Remove matrix from timeline (too large)
            snapshot = {k: v for k, v in probs.items()
                        if k != "remaining_matrix"}
            timeline.append(snapshot)

        return timeline


def format_live(result):
    """Format live prediction for display."""
    m = result["minute"]
    h, a = result["score"]
    p = result["probs_1x2"]
    lh, la = result["lambda_remaining"]
    ou = result["over_under"]
    ng = result["next_goal"]

    lines = [
        f"  {m}' | {h}-{a} | λ_rem: {lh:.2f}-{la:.2f}",
        f"  1x2: 主胜 {p[0]:.1%} | 平局 {p[1]:.1%} | 客胜 {p[2]:.1%}",
        f"  大小球: O2.5 {ou[2.5]['over']:.1%} | O3.5 {ou[3.5]['over']:.1%}",
        f"  下一球: 主 {ng['home']:.1%} | 客 {ng['away']:.1%} | 无 {ng['none']:.1%}",
    ]
    return "\n".join(lines)
