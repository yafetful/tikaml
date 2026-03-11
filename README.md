# TikaML — Football Match Outcome Prediction

A machine learning system for predicting football match outcomes using LightGBM Poisson regression. The model outputs a full 7×7 score probability matrix for each match, enabling rich downstream analysis including 1x2 outcome probabilities, most likely scorelines, and expected goals.

Trained on 17,469 matches across Europe's top 5 leagues (2016–2026), the model achieves an average **RPS of 0.197** under strict forward-chain temporal validation.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Why LightGBM + Poisson?](#why-lightgbm--poisson)
- [Feature Engineering](#feature-engineering)
- [Evaluation](#evaluation)
- [Comparison of Approaches](#comparison-of-approaches)
- [Known Limitations](#known-limitations)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Model Architecture

```
84 features (per-match)
    ↓
LightGBM (objective=poisson) × 2
    ├─ Model A → λ_home (expected home goals)
    └─ Model B → λ_away (expected away goals)
    ↓
Poisson PMF:  P(k | λ) = λ^k · e^(-λ) / k!
    ↓
7×7 score matrix  P(i, j) = P(home=i) · P(away=j) · τ(i,j,λ_h,λ_a,ρ)
    ↓
Dixon-Coles correction (ρ = -0.108) for low-scoring matches
    ↓
Temperature scaling (T = 0.90) to correct conservative bias
    ↓
Final outputs:
    ├─ 1x2 probabilities: P(Home Win), P(Draw), P(Away Win)
    ├─ Recommended score (from predicted outcome group)
    └─ Full 7×7 score probability matrix
```

**Two independent Poisson models** predict the expected goal rates (λ) for home and away teams separately. The score probability matrix is then constructed analytically via the Poisson distribution, with a Dixon-Coles low-score correction factor τ that adjusts probabilities for 0-0, 0-1, 1-0, and 1-1 scorelines.

**Temperature scaling** (T=0.90) sharpens the 1x2 probability distribution, correcting a systematic conservative bias observed at high-confidence predictions (model predicts P(H)=75–90%, actual win rate ≈87%).

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

The model uses **84 features** across 12 categories, all computable from pre-match information only (no data leakage):

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
| **Total** | | **84** | |

### Data Scale

- **Leagues**: Premier League (EPL), La Liga, Serie A, Bundesliga, Ligue 1
- **Time span**: 12 seasons (2014–2026)
- **Total matches**: 21,121 (17,469 used for training from 2016+)
- **Feature table dimensions**: 21,121 rows × 225 columns (84 used by model)
- **Odds coverage**: ~51% overall, ~77% for 2018+ seasons
- **Lineup feature coverage**: 82.4%

The processed feature table (`data/opta/processed/features.csv`) is included in this repository for research purposes. Raw match event data is not distributed.

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

# Load pre-trained model (included in models/)
predictor = MatchPredictor()
predictor.load_model()

# Predict a match
result = predictor.predict(
    home_team="Arsenal",
    away_team="Chelsea",
    league="EPL",
    season="2025-2026",
    match_date="2026-03-15",
)

# Access results
print(result["probs_1x2"])          # [P(Home), P(Draw), P(Away)]
print(result["recommended_score"])   # {"label": "2-1", "prob": 0.10, ...}
print(result["lambda_home"])         # Expected home goals
print(result["score_matrix"])        # 7×7 numpy array
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
λ_home=2.03  λ_away=1.02
主胜 59.4%  |  平局 23.2%  |  客胜 17.4%
预测结果: 主胜  |  推荐比分: 2-1 (10.0%)

三组比分预测:
  主胜 [59.4%]: 2-1(10.0%), 1-0(9.8%), 2-0(9.5%)
  平局 [23.2%]: 1-1(11.5%), 0-0(6.0%), 2-2(5.3%)
  客胜 [17.4%]: 0-1(5.0%), 1-2(4.8%), 0-2(3.2%)
```

### Retraining

To retrain the model on updated data:

```python
predictor = MatchPredictor()
predictor.train(save=True)  # Trains on all data, saves to models/
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
│   ├── inference.py          # Inference engine: feature extraction + prediction
│   └── evaluation.py         # Metrics: RPS, accuracy, calibration
├── scripts/
│   ├── predict.py            # CLI prediction tool
│   ├── run_lgbm.py           # Forward-chain validation
│   └── tune_lgbm.py          # Optuna hyperparameter optimization
├── models/
│   ├── lgbm_home.txt         # Trained home goals model
│   ├── lgbm_away.txt         # Trained away goals model
│   └── meta.json             # Model metadata (features, parameters, medians)
├── data/
│   └── opta/processed/
│       └── features.csv      # Processed feature table (21,121 × 225)
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
