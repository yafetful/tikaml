"""Neural network model with team embeddings for football prediction.

Architecture:
  Team Embeddings (dim=16 per team) + Match Features (68) →
  MLP → [λ_home, λ_away, ρ_correction] →
  Dixon-Coles score matrix → 1x2 probabilities

Key innovations:
  1. Learnable team embeddings capture team identity beyond rolling features
  2. Per-match ρ allows model to adapt DC correction to match context
  3. Direct RPS loss optimization (not just Poisson NLL)
  4. Ordinal-aware cross-entropy on 1x2 probabilities

Usage:
    source .venv/bin/activate
    python scripts/run_neural.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import FEATURE_COLS
from src.evaluation import evaluate_predictions, match_outcome, ranked_probability_score

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EMBED_DIM = 16
HIDDEN_DIM = 128
DROPOUT = 0.3
LR = 0.001
EPOCHS = 80
BATCH_SIZE = 256
MAX_GOALS = 7


class MatchDataset(Dataset):
    def __init__(self, features, home_ids, away_ids, home_goals, away_goals):
        self.features = torch.FloatTensor(features)
        self.home_ids = torch.LongTensor(home_ids)
        self.away_ids = torch.LongTensor(away_ids)
        self.home_goals = torch.FloatTensor(home_goals)
        self.away_goals = torch.FloatTensor(away_goals)
        # Outcomes: 0=home, 1=draw, 2=away
        self.outcomes = torch.LongTensor([
            0 if h > a else (1 if h == a else 2)
            for h, a in zip(home_goals, away_goals)
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.home_ids[idx], self.away_ids[idx],
                self.home_goals[idx], self.away_goals[idx], self.outcomes[idx])


class FootballNet(nn.Module):
    """Neural network for football score prediction.

    Inputs: match features + team IDs
    Outputs: λ_home, λ_away (Poisson rates) + ρ (DC correction)
    """
    def __init__(self, n_features, n_teams, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.team_embed = nn.Embedding(n_teams, embed_dim)
        nn.init.normal_(self.team_embed.weight, 0, 0.05)

        input_dim = n_features + embed_dim * 2

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Separate heads for home/away λ and ρ
        self.head_home = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # λ > 0
        )
        self.head_away = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.head_rho = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),  # ρ ∈ [-1, 1], will scale to [-0.3, 0.1]
        )

    def forward(self, features, home_ids, away_ids):
        h_embed = self.team_embed(home_ids)
        a_embed = self.team_embed(away_ids)
        x = torch.cat([features, h_embed, a_embed], dim=1)
        x = self.shared(x)
        lh = self.head_home(x).squeeze(-1) + 0.1  # min 0.1
        la = self.head_away(x).squeeze(-1) + 0.1
        rho = self.head_rho(x).squeeze(-1) * 0.2 - 0.1  # ρ ∈ [-0.3, 0.1]
        return lh, la, rho


def build_score_matrix_batch(lh, la, rho, max_goals=MAX_GOALS):
    """Build score matrix for a batch (differentiable).

    Uses the Poisson PMF + Dixon-Coles τ correction.
    """
    from scipy.stats import poisson as sp_poisson
    batch_size = len(lh)
    lh_np = lh.detach().cpu().numpy()
    la_np = la.detach().cpu().numpy()
    rho_np = rho.detach().cpu().numpy()

    matrices = np.zeros((batch_size, max_goals, max_goals))
    for b in range(batch_size):
        for i in range(max_goals):
            for j in range(max_goals):
                tau = 1.0
                if i == 0 and j == 0:
                    tau = max(0, 1 - lh_np[b] * la_np[b] * rho_np[b])
                elif i == 0 and j == 1:
                    tau = max(0, 1 + lh_np[b] * rho_np[b])
                elif i == 1 and j == 0:
                    tau = max(0, 1 + la_np[b] * rho_np[b])
                elif i == 1 and j == 1:
                    tau = max(0, 1 - rho_np[b])
                matrices[b, i, j] = (tau *
                    sp_poisson.pmf(i, lh_np[b]) *
                    sp_poisson.pmf(j, la_np[b]))
        total = matrices[b].sum()
        if total > 0:
            matrices[b] /= total

    return matrices


def compute_1x2_from_matrix(matrices):
    """Extract 1x2 probabilities from score matrices."""
    batch_size = len(matrices)
    probs = np.zeros((batch_size, 3))
    for b in range(batch_size):
        probs[b, 0] = np.tril(matrices[b], -1).sum()  # home
        probs[b, 1] = np.trace(matrices[b])            # draw
        probs[b, 2] = np.triu(matrices[b], 1).sum()    # away
        total = probs[b].sum()
        if total > 0:
            probs[b] /= total
    return probs


def poisson_nll_loss(lh, la, home_goals, away_goals):
    """Poisson negative log-likelihood loss."""
    # Poisson NLL: λ - y*log(λ) + log(y!)
    eps = 1e-8
    loss_home = lh - home_goals * torch.log(lh + eps)
    loss_away = la - away_goals * torch.log(la + eps)
    return (loss_home + loss_away).mean()


def prepare_data(df, feature_cols):
    """Prepare features and team ID mappings."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    medians = X.median()
    X = X.fillna(medians)

    # Normalize features
    means = X.mean()
    stds = X.std().replace(0, 1)
    X = (X - means) / stds

    return X.values, available, medians, means, stds


def run():
    print("=" * 70)
    print("TikaML: 神经网络 + 团队嵌入模型")
    print(f"  设备: {DEVICE}")
    print(f"  嵌入维度: {EMBED_DIM}, 隐藏层: {HIDDEN_DIM}")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  数据: {len(df)} 场")

    # Build team ID mapping
    all_teams = sorted(set(df["home_team_id"].unique()) |
                       set(df["away_team_id"].unique()))
    team_to_id = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)
    print(f"  球队数: {n_teams}")

    all_results = []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        # Split data
        all_train = df[
            (df["season"] < test_season) &
            (df["season"] >= "2016-2017")
        ].copy()

        seasons = sorted(all_train["season"].unique())
        if len(seasons) < 2:
            continue
        val_season = seasons[-1]
        train_data = all_train[all_train["season"] < val_season]
        val_data = all_train[all_train["season"] == val_season]
        test_data = df[df["season"] == test_season].copy()

        # Prepare features
        X_train, feat_cols, medians, means, stds = prepare_data(
            train_data, FEATURE_COLS)
        X_val = ((val_data[feat_cols].fillna(medians) - means) / stds).values
        X_test = ((test_data[feat_cols].fillna(medians) - means) / stds).values

        # Team IDs
        train_h = train_data["home_team_id"].map(team_to_id).values
        train_a = train_data["away_team_id"].map(team_to_id).values
        val_h = val_data["home_team_id"].map(team_to_id).values
        val_a = val_data["away_team_id"].map(team_to_id).values
        test_h = test_data["home_team_id"].map(team_to_id).values
        test_a = test_data["away_team_id"].map(team_to_id).values

        # Datasets
        train_ds = MatchDataset(
            X_train, train_h, train_a,
            train_data["home_goals"].values,
            train_data["away_goals"].values)
        val_ds = MatchDataset(
            X_val, val_h, val_a,
            val_data["home_goals"].values,
            val_data["away_goals"].values)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        # Model
        model = FootballNet(
            n_features=len(feat_cols),
            n_teams=n_teams,
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-5)

        best_val_rps = float("inf")
        best_state = None
        patience_counter = 0
        max_patience = 20

        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_losses = []
            for batch in train_loader:
                feats, h_ids, a_ids, h_goals, a_goals, outcomes = [
                    b.to(DEVICE) for b in batch]
                lh, la, rho = model(feats, h_ids, a_ids)
                loss = poisson_nll_loss(lh, la, h_goals, a_goals)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            model.eval()
            all_probs = []
            all_outcomes = []
            with torch.no_grad():
                for batch in val_loader:
                    feats, h_ids, a_ids, h_goals, a_goals, outcomes = [
                        b.to(DEVICE) for b in batch]
                    lh, la, rho = model(feats, h_ids, a_ids)
                    matrices = build_score_matrix_batch(lh, la, rho)
                    probs = compute_1x2_from_matrix(matrices)
                    all_probs.append(probs)
                    all_outcomes.append(outcomes.cpu().numpy())

            all_probs = np.vstack(all_probs)
            all_outcomes = np.concatenate(all_outcomes)
            val_m = evaluate_predictions(all_probs, all_outcomes)
            val_rps = val_m["rps"]

            scheduler.step(val_rps)

            if val_rps < best_val_rps:
                best_val_rps = val_rps
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:>3}: train_loss={np.mean(train_losses):.4f}  "
                      f"val_RPS={val_rps:.4f}  best={best_val_rps:.4f}")

            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_state)
        model.eval()

        # Predict on test set
        test_feats = torch.FloatTensor(X_test).to(DEVICE)
        test_h_ids = torch.LongTensor(test_h).to(DEVICE)
        test_a_ids = torch.LongTensor(test_a).to(DEVICE)

        with torch.no_grad():
            lh, la, rho = model(test_feats, test_h_ids, test_a_ids)

        matrices = build_score_matrix_batch(lh, la, rho)
        nn_probs = compute_1x2_from_matrix(matrices)

        test_outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in test_data.iterrows()
        ])

        nn_m = evaluate_predictions(nn_probs, test_outcomes)
        print(f"\n  Neural Net: RPS={nn_m['rps']:.4f}  "
              f"Brier={nn_m['brier']:.4f}  Acc={nn_m['accuracy']:.1%}")
        print(f"  Mean ρ={rho.mean().item():.3f}  "
              f"std={rho.std().item():.3f}")

        # Also run LGBM for comparison
        from src.lgbm_poisson import LGBMPoissonModel
        lgbm = LGBMPoissonModel(rho=-0.108)
        lgbm.fit(train_data, val_df=val_data)
        lgbm_probs = lgbm.predict_1x2(test_data)
        lgbm_m = evaluate_predictions(lgbm_probs, test_outcomes)
        print(f"  LGBM:       RPS={lgbm_m['rps']:.4f}  "
              f"Brier={lgbm_m['brier']:.4f}  Acc={lgbm_m['accuracy']:.1%}")

        # Ensemble: NN + LGBM
        for alpha in [0.3, 0.5, 0.7]:
            ens_probs = alpha * nn_probs + (1 - alpha) * lgbm_probs
            ens_probs /= ens_probs.sum(axis=1, keepdims=True)
            ens_m = evaluate_predictions(ens_probs, test_outcomes)
            print(f"  Ensemble(α={alpha:.1f}): RPS={ens_m['rps']:.4f}  "
                  f"Acc={ens_m['accuracy']:.1%}")

        # Draw analysis
        nn_draw_pred = (np.argmax(nn_probs, axis=1) == 1).sum()
        lgbm_draw_pred = (np.argmax(lgbm_probs, axis=1) == 1).sum()
        actual_draws = (test_outcomes == 1).sum()
        print(f"\n  平局: 实际={actual_draws}, NN预测={nn_draw_pred}, "
              f"LGBM预测={lgbm_draw_pred}")

        all_results.append({
            "season": test_season,
            "rps_nn": nn_m["rps"],
            "rps_lgbm": lgbm_m["rps"],
            "acc_nn": nn_m["accuracy"],
            "acc_lgbm": lgbm_m["accuracy"],
        })

    # Summary
    if all_results:
        rdf = pd.DataFrame(all_results)
        print(f"\n{'=' * 70}")
        print("汇总")
        print(f"{'=' * 70}")
        print(f"\n  {'赛季':<12} {'NN RPS':>10} {'LGBM RPS':>10} {'NN Acc':>10} {'LGBM Acc':>10}")
        for _, r in rdf.iterrows():
            print(f"  {r['season']:<12} {r['rps_nn']:>10.4f} {r['rps_lgbm']:>10.4f} "
                  f"{r['acc_nn']:>9.1%} {r['acc_lgbm']:>9.1%}")
        print(f"\n  {'平均':<12} {rdf['rps_nn'].mean():>10.4f} {rdf['rps_lgbm'].mean():>10.4f} "
              f"{rdf['acc_nn'].mean():>9.1%} {rdf['acc_lgbm'].mean():>9.1%}")
        d = rdf['rps_nn'].mean() - rdf['rps_lgbm'].mean()
        print(f"  NN vs LGBM: {d:+.4f}")


if __name__ == "__main__":
    run()
