# TikaML — Football Match Prediction

A machine learning system for predicting football match outcomes, corners, and yellow cards using LightGBM Poisson regression. Three independent models output full probability distributions for goals, corners, and cards — both pre-match and live (in-play).

Trained on 17,469 matches across Europe's top 5 leagues (2016–2026), the goal model achieves an average **RPS of 0.197** under strict forward-chain temporal validation.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Why LightGBM + Poisson?](#why-lightgbm--poisson)
- [Feature Engineering](#feature-engineering)
- [Live Prediction](#live-prediction)
- [Evaluation](#evaluation)
- [Comparison of Approaches](#comparison-of-approaches)
- [Known Limitations](#known-limitations)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [References](#references)

---

## Model Architecture

### Three-Model System

```
Feature Vector (84–91 features per match)
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Model 1: Goals (84 features)                               │
│  LightGBM Poisson × 2 → λ_home, λ_away                     │
│  → 7×7 score matrix → 1x2 + O/U 1.5, 2.5, 3.5             │
├─────────────────────────────────────────────────────────────┤
│  Model 2: Corners (89 features)                              │
│  LightGBM Poisson × 2 → λ_corners_home, λ_corners_away     │
│  → O/U 8.5, 9.5, 10.5, 11.5                                │
├─────────────────────────────────────────────────────────────┤
│  Model 3: Yellow Cards (91 features)                         │
│  LightGBM Poisson × 2 → λ_yellows_home, λ_yellows_away     │
│  → O/U 2.5, 3.5, 4.5, 5.5                                  │
└─────────────────────────────────────────────────────────────┘
```

Each model consists of **two independent Poisson regressors** predicting the expected rates (λ) for home and away teams separately. The goal model additionally constructs a 7×7 score probability matrix with a **Dixon-Coles low-score correction** (ρ = -0.108) and **temperature scaling** (T = 0.90) to correct conservative bias at high confidence levels.

All three models share the same 84-feature base, with corners adding 5 corner-specific rolling features and yellows adding 7 card/foul-specific rolling features.

### Live Prediction Engine

```
Pre-match λ_home, λ_away
    ↓
Bayesian Poisson Updating
    ├─ Non-linear time decay (r^0.82)
    ├─ 6-bucket score-aware momentum
    ├─ Red card impact (0.61× self / 1.46× opponent)
    └─ Dixon-Coles ρ decay over time
    ↓
Updated 1x2, O/U, next goal, remaining λ
```

The live predictor applies Bayesian updating to pre-match Poisson rates, adjusting for elapsed time, current score momentum, and red cards. Parameters were optimized via differential evolution on 150 matches against Opta live predictions (MAE = 0.030).

## Why LightGBM + Poisson?

We evaluated multiple modeling approaches. LightGBM with Poisson regression was selected for the following reasons:

### 1. Poisson distribution is the natural model for goal counts

Football goals are discrete, non-negative, relatively rare events — exactly what the Poisson distribution models. By predicting λ (expected goals) rather than directly predicting win/draw/loss probabilities, we:

- Enforce a **physically meaningful output** (goal rates must be positive)
- Generate a **full score matrix** from just 2 parameters, enabling rich analysis
- Naturally handle the **asymmetry** between home and away teams

### 2. LightGBM excels on structured tabular data

| Property | LightGBM | Neural Networks | Linear Models |
|----------|----------|-----------------|---------------|
| Tabular data performance | Excellent | Moderate | Good |
| Required data volume | ~10K rows | ~100K+ rows | ~1K rows |
| Feature interaction capture | Automatic (tree splits) | Manual architecture design | Requires explicit features |
| Missing value handling | Native | Requires imputation | Requires imputation |
| Training speed | Seconds | Minutes–Hours | Seconds |
| Interpretability | Feature importance | Black box | Coefficients |

With ~17K training samples and 84 engineered features, LightGBM sits in the sweet spot: enough data for gradient boosting to learn meaningful patterns, but far too little for deep learning to outperform. This finding is consistent with extensive benchmarks in the literature (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022).

### 3. The combination preserves the best of both worlds

Traditional Poisson regression (`log(λ) = Xβ`) captures the right distributional form but is limited to linear feature interactions. Pure LightGBM classification (predicting 1x2 directly) loses the structured score matrix. **LightGBM + Poisson objective** combines:

- The **distributional structure** of Poisson models (score matrix, expected goals)
- The **representational power** of gradient boosting (non-linear interactions, robustness)

## Feature Engineering

The base model uses **84 features** across 12 categories, all computable from pre-match information only (no data leakage). Corner and yellow card models extend this with domain-specific rolling features.

| Category | Features | Count | Description |
|----------|----------|-------|-------------|
| Rolling xG | xg, xgot, conceded (home/away) | 8 | Exponentially weighted moving averages of expected goals metrics |
| Rolling Tactical | shots, ppda, prog_passes, prog_carries, touches_in_box, etc. | 16 | Team tactical profile from recent matches |
| Form | goals scored/conceded rolling | 4 | Recent goal-scoring form |
| Venue-specific | xg, xgot, goals, shots (home-only / away-only) | 8 | Performance split by venue |
| Derived | xg_overperformance, shot_accuracy, clean_sheet_pct | 6 | Composite metrics |
| H2H | win_pct, goal_diff, matches | 3 | Head-to-head historical record |
| Relative Strength | xg, xgot, shots vs league average | 6 | Team strength relative to competition |
| Momentum | short-term vs long-term form delta | 2 | Recent trajectory |
| Draw Tendency | team/league draw rates, xg_diff, defensive_strength | 5 | Draw-proneness indicators |
| Match Context | days_rest, is_midweek, season_stage, points/position_diff, league_avg_goals | 8 | Schedule and standings context |
| Match Importance | relegation_battle, title_race, points_to_safety/leader, composite | 5 | Stakes and competitive pressure |
| Lineup Rotation | changes, stability, formation_change, rotation_rate (home/away) | 8 | Squad management patterns |
| Market Odds | implied probabilities (home/draw/away) | 3 | Bookmaker market consensus (optional) |
| **Base Total** | | **84** | |
| Corner-specific | corners_rolling, corners_conceded, referee_corners | +5 | 10-match rolling corner stats |
| Yellow-specific | yellows_rolling, yellows_conceded, fouls_rolling, referee_yellows | +7 | 10-match rolling card/foul stats |

### Data Scale

- **Leagues**: Premier League (EPL), La Liga, Serie A, Bundesliga, Ligue 1
- **Time span**: 12 seasons (2014–2026)
- **Total matches**: 21,121 (17,469 used for training from 2016+)
- **Feature table dimensions**: 21,121 rows × 253 columns
- **Odds coverage**: ~51% overall, ~77% for 2018+ seasons
- **Corner data coverage**: 17,207 matches (98.5%)
- **Yellow card data coverage**: 14,902 matches (85.3%)

The processed feature table (`data/opta/processed/features.csv`) is included in this repository for research purposes. Raw match event data is not distributed.

## Live Prediction

The live prediction engine (`src/live_predictor.py`) provides real-time probability updates during matches. It takes the pre-match Poisson rates as priors and applies Bayesian updating based on:

- **Non-linear time decay** (r^0.82): accounts for goals clustering in late periods
- **Score-aware momentum** (6 buckets): teams losing by 3+ attack harder (1.40× multiplier), winning teams sit back (0.85×)
- **Red card factors**: a red card reduces the team's scoring rate to 0.61× and boosts the opponent to 1.46×
- **Dixon-Coles decay**: ρ correction diminishes linearly as match progresses

All parameters were optimized via differential evolution on 150 EPL/La Liga matches against Opta live predictions and validated with 3-fold cross-validation (MAE = 0.030, consistent across folds).

Corner and yellow card live predictions use simple linear time decay (no score momentum), as these events lack the strong score-dependency of goals.

## Evaluation

### Methodology

We use **forward-chain temporal validation** (also known as expanding window):

- For test season *S*, train on all seasons before *S* (starting from 2016-2017)
- The last training season serves as validation for early stopping
- **No future data leakage**: the model never sees future matches during training

### Results

| Test Season | Matches | RPS ↓ | Accuracy | High-Confidence (≥60%) Acc |
|-------------|---------|-------|----------|---------------------------|
| 2022-2023 | 1,827 | 0.2021 | 53.1% | 68.0% |
| 2023-2024 | 1,752 | 0.1908 | 54.2% | 71.1% |
| 2024-2025 | 1,750 | 0.1974 | 53.5% | 67.5% |
| 2025-2026 | 1,290 | 0.1990 | 52.4% | 70.3% |
| **Average** | | **0.1973** | **53.3%** | **69.2%** |

**RPS (Ranked Probability Score)** is the primary metric — it measures the quality of the full probability distribution, not just the top prediction. Lower is better.

### Probability Calibration

| Predicted P(Home Win) | Actual Home Win Rate | Bias |
|-----------------------|---------------------|------|
| 35–45% | 40.5% | -0.6% |
| 45–55% | 52.0% | -2.2% |
| 55–65% | 60.2% | -0.3% |
| 65–75% | 72.1% | -2.4% |
| 75–90% | 87.0% | -7.9% |

The model is well-calibrated overall. Temperature scaling (T=0.90) partially corrects the conservative bias at high confidence levels.

## Comparison of Approaches

All methods evaluated under the same forward-chain validation protocol:

| Approach | RPS | Notes |
|----------|-----|-------|
| Dixon-Coles (baseline) | 0.2060 | Traditional Poisson model with MLE |
| Elo ratings → Poisson | ~0.205 | Elo strength → λ conversion |
| **LightGBM Poisson (68 features)** | 0.1997 | Our base model with Optuna-tuned hyperparameters |
| + Venue-specific & derived features | 0.1993 | 8 additional venue rolling + 6 derived |
| Stacking (DC + LGBM + Elo → LR meta) | 0.1994 | LGBM dominates; other models add noise |
| 4-model weighted ensemble | 0.1993 | DC + Poisson + Multiclass + Elo blend |
| Neural network (team embeddings) | 0.1993 | Insufficient data; 53 draw predictions but no RPS gain |
| + Market odds implied probabilities | 0.1979 | 3 odds features; strongest single addition |
| + Temperature scaling (T=0.90) | 0.1976 | Corrects conservative bias at extremes |
| **+ Match importance + Lineup rotation** | **0.1973** | **Final model: 84 features** |
| Bookmaker consensus (reference) | ~0.185 | Target benchmark |

### Methods that did NOT improve results

| Method | Impact | Reason |
|--------|--------|--------|
| Isotonic probability calibration | +0.001 (worse) | Calibration set too small, overfits |
| LightGBM stacking meta-learner | +0.002 (worse) | Meta-learner overfits on limited folds |
| Temporal weighting (recent data upweighted) | ±0.000 | More data > time decay for generalization |
| LightGBM 5x bagging | -0.0002 | Marginal and unstable |
| Bivariate Poisson (λ₃ correlation) | ±0.000 | λ₃ unstable across seasons (0.024 → 0.000 → 0.269) |
| Score matrix post-processing | ±0.000 | Draw boost, adaptive ρ, 1x2 adjustments — all ineffective |
| Draw specialist classifier | ±0.000 | P(draw) for actual draws ≈ P(draw) for non-draws (0.270 vs 0.269) |
| Opta pre-match win probability | ±0.000 | Redundant with bookmaker odds (r² ≈ 0.98) |
| Squad depth (roster size/position counts) | +0.0003 (worse) | Static season-level feature; too coarse to capture match-level dynamics |
| Player-level XI features (27 variants) | ±0.0001 | Team rolling stats already encode player contributions; odds already price lineups |
| UCL combined training (leagues + UCL) | +0.0002 (worse) | UCL only 433 matchable games; zero-shot transfer (league model → UCL) already achieves RPS=0.1946 on group stage |

### Key insight

Single-model enrichment (adding features to LightGBM) consistently outperformed multi-model ensembling. When the primary model has access to rich features including market odds, secondary models provide negligible orthogonal information.

## Known Limitations

### 1. Draw prediction as argmax is impossible

The model never predicts "Draw" as the most likely 1x2 outcome. This is a **mathematical property of the Poisson distribution**, not a model deficiency: when λ ≥ 0.89 (all real football matches exceed this), the sum of draw probabilities (0-0, 1-1, 2-2, ...) is always less than the sum of home win or away win probabilities.

However, the model's aggregate P(Draw) is well-calibrated (predicted 25.6% vs actual 25.3%), and individual draw scorelines (especially 1-1) frequently appear as the single most probable score in the 7×7 matrix.

### 2. Upset sensitivity

Away wins contribute 41% of total RPS loss despite being only 31% of outcomes. Upsets (strong home team loses) incur very high per-match RPS penalties. This is partially mitigated by market odds features but remains the largest error source.

### 3. Gap to bookmaker performance

| | RPS |
|--|-----|
| TikaML | 0.197 |
| Bookmakers | ~0.185 |
| Gap | +0.012 |

The gap is primarily driven by:
- **Real-time information** (~60%): injuries, confirmed lineups, transfers, player fitness — bookmakers have dedicated teams tracking these
- **Market wisdom** (~25%): aggregated judgment from thousands of bettors
- **Model structure** (~15%): Poisson distributional constraints on low-score matches

## Getting Started

### Prerequisites

```bash
python >= 3.10
pip install numpy scipy pandas lightgbm scikit-learn optuna
```

### Quick Start

```python
from src.inference import MatchPredictor

# Load pre-trained models (goals + corners + yellows)
predictor = MatchPredictor()
predictor.load_model()

# Pre-match prediction
result = predictor.predict(
    home_team="Arsenal",
    away_team="Chelsea",
    league="EPL",
    season="2025-2026",
    match_date="2026-03-15",
)

# Access results
print(result["probs_1x2"])            # [P(Home), P(Draw), P(Away)]
print(result["recommended_score"])     # {"label": "2-1", "prob": 0.10, ...}
print(result["goals_over_under"])      # {1.5: {over, under}, 2.5: ..., 3.5: ...}
print(result["corners"])               # {lambda_home, lambda_away, over_under: {8.5, 9.5, ...}}
print(result["yellows"])               # {lambda_home, lambda_away, over_under: {2.5, 3.5, ...}}
```

### Live (In-Play) Prediction

```python
# Live prediction at 35', score 1-0, 3 corners each side
result = predictor.predict_live(
    home_team="Arsenal", away_team="Chelsea",
    league="EPL", season="2025-2026", match_date="2026-03-15",
    minute=35, home_goals=1, away_goals=0,
    home_corners=3, away_corners=1,
    home_yellows=1, away_yellows=0,
    home_red_cards=0, away_red_cards=0,
)

print(result["probs_1x2"])        # Updated 1x2 probabilities
print(result["next_goal"])        # {home: 0.51, away: 0.33, none: 0.16}
print(result["lambda_remaining"]) # Remaining expected goals
print(result["corners"])          # Corners with remaining λ and updated O/U
```

### CLI Prediction

```bash
# Predict a specific match
python scripts/predict.py --matches "Arsenal vs Chelsea"

# With bookmaker odds (improves accuracy)
python scripts/predict.py --matches "Arsenal vs Chelsea" --odds 1.85 3.60 4.50

# Demo mode (classic matchups)
python scripts/predict.py
```

### Output Format

```
Arsenal vs Chelsea
λ_home=1.96  λ_away=1.08
主胜 56.5%  |  平局 24.0%  |  客胜 19.5%
预测结果: 主胜  |  推荐比分: 2-1 (10.0%)

三组比分预测:
  主胜 [56.5%]: 2-1(10.0%), 2-0(9.2%), 1-0(8.3%)
  平局 [24.0%]: 1-1(11.3%), 0-0(5.9%), 2-2(5.4%)
  客胜 [19.5%]: 1-2(5.5%), 0-1(4.1%), 0-2(2.8%)

进球大小: O1.5 81.8% | O2.5 58.5% | O3.5 36.0%

角球: λ_home=6.1  λ_away=4.1
角球大小: O8.5 67.9% | O9.5 55.6% | O10.5 43.1% | O11.5 31.6%

黄牌: λ_home=1.9  λ_away=2.3
黄牌大小: O2.5 79.5% | O3.5 61.2% | O4.5 41.8% | O5.5 25.3%
```

### Retraining

To retrain all three models on updated data:

```python
predictor = MatchPredictor()
predictor.train(save=True)  # Trains goals + corners + yellows, saves to models/
```

### Hyperparameter Tuning

```bash
python scripts/tune_lgbm.py  # Optuna-based Bayesian optimization
```

### Model Validation

```bash
python scripts/run_lgbm.py   # Forward-chain validation with detailed metrics
```

## Project Structure

```
tikaml/
├── src/
│   ├── lgbm_poisson.py      # Core model: LightGBM Poisson regression + Dixon-Coles
│   ├── inference.py          # Inference engine: pre-match + live prediction
│   ├── live_predictor.py     # Live (in-play) Bayesian Poisson updating engine
│   └── evaluation.py         # Metrics: RPS, accuracy, calibration
├── scripts/
│   ├── predict.py            # CLI prediction tool
│   ├── run_lgbm.py           # Forward-chain validation
│   └── tune_lgbm.py          # Optuna hyperparameter optimization
├── models/
│   ├── lgbm_home.txt         # Trained home goals model
│   ├── lgbm_away.txt         # Trained away goals model
│   ├── meta.json             # Goals model metadata
│   ├── corners/              # Corner prediction model
│   │   ├── lgbm_home.txt
│   │   ├── lgbm_away.txt
│   │   └── meta.json
│   └── yellows/              # Yellow card prediction model
│       ├── lgbm_home.txt
│       ├── lgbm_away.txt
│       └── meta.json
├── data/
│   └── opta/processed/
│       └── features.csv      # Processed feature table (21,121 × 253)
├── experiments/              # Experimental code (stacking, neural net, etc.)
├── doc/
│   ├── model_report.md       # Detailed model progress report
│   └── score_heatmap.svg     # Score matrix visualization example
└── README.md
```

## References

- Dixon, M. J., & Coles, S. G. (1997). *Modelling association football scores and inefficiencies in the football betting market.* Journal of the Royal Statistical Society: Series C, 46(2), 265-280.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS 2022.
- Constantinou, A. C., & Fenton, N. E. (2012). *Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models.* Journal of Quantitative Analysis in Sports, 8(1).

## License

This project is for research and educational purposes. The processed feature table is derived from proprietary match event data and is provided for academic use only.
