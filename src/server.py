"""Prediction service API.

FastAPI server that loads trained models and serves predictions
for the tikaml-data-service. Accepts pre-computed feature vectors,
runs LightGBM inference, and returns structured prediction results.

Usage:
    uvicorn src.server:app --host 0.0.0.0 --port 8001
"""

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from scipy.stats import poisson

from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS, CORNER_FEATURE_COLS, YELLOW_FEATURE_COLS
from src.live_predictor import LivePredictor

# ─── Config ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tika-server")

MODEL_DIR = Path("models")
CORNER_MODEL_DIR = Path("models/corners")
YELLOW_MODEL_DIR = Path("models/yellows")
BACKFILL_DIR = Path("models/backfill_20260131")
MAX_GOALS = 7

# API Key — set via environment variable, or auto-generate on first run
API_KEY = os.environ.get("TIKA_API_KEY", "")
if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    log.warning(f"No TIKA_API_KEY set, generated: {API_KEY}")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)):
    """Validate the API key from request header."""
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


# ─── Models (loaded once at startup) ──────────────────────────────

class Models:
    goals: LGBMPoissonModel | None = None
    corners: LGBMPoissonModel | None = None
    yellows: LGBMPoissonModel | None = None
    version: str = "unknown"

models = Models()
backfill_models = Models()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    log.info("Loading models...")
    t0 = time.time()

    models.goals = LGBMPoissonModel.load(str(MODEL_DIR))
    log.info(f"  Goals model loaded ({len(models.goals.feature_cols)} features)")

    if CORNER_MODEL_DIR.exists():
        models.corners = LGBMPoissonModel.load(str(CORNER_MODEL_DIR))
        log.info(f"  Corners model loaded ({len(models.corners.feature_cols)} features)")

    if YELLOW_MODEL_DIR.exists():
        models.yellows = LGBMPoissonModel.load(str(YELLOW_MODEL_DIR))
        log.info(f"  Yellows model loaded ({len(models.yellows.feature_cols)} features)")

    # Read version from meta.json
    meta_path = MODEL_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        models.version = f"lgbm-poisson-{len(meta.get('feature_cols', []))}f"

    # Load backfill models (optional)
    if BACKFILL_DIR.exists():
        bf_goals_dir = BACKFILL_DIR / "goals"
        bf_corners_dir = BACKFILL_DIR / "corners"
        bf_yellows_dir = BACKFILL_DIR / "yellows"

        if bf_goals_dir.exists():
            backfill_models.goals = LGBMPoissonModel.load(str(bf_goals_dir))
        if bf_corners_dir.exists():
            backfill_models.corners = LGBMPoissonModel.load(str(bf_corners_dir))
        if bf_yellows_dir.exists():
            backfill_models.yellows = LGBMPoissonModel.load(str(bf_yellows_dir))
        backfill_models.version = "lgbm-poisson-backfill-20260131"
        log.info("  Backfill models loaded")

    log.info(f"  All models loaded in {time.time() - t0:.1f}s")
    yield
    log.info("Shutting down")


# ─── App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="TikaML Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Request / Response schemas ────────────────────────────────────

class MatchContext(BaseModel):
    minute: int | None = None
    second: int = 0
    period: str | None = None
    status: str | None = None
    home_score: int = 0
    away_score: int = 0
    home_team_id: str | None = None
    away_team_id: str | None = None
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_yellows: int = 0
    away_yellows: int = 0


class PredictionRequest(BaseModel):
    match_id: int
    opta_match_id: str = ""
    prediction_type: str = "prematch"  # "prematch" or "live"
    trigger: str = ""
    feature_vector: dict[str, float | int | None]
    match_context: MatchContext | None = None
    models: list[str] = Field(default_factory=lambda: ["goals", "corners", "yellows"])


class PredictionResponse(BaseModel):
    predictions: dict
    model_metadata: dict


# ─── Helpers ───────────────────────────────────────────────────────

def _build_feature_df(feature_vector: dict, feature_list: list[str]) -> pd.DataFrame:
    """Build a single-row DataFrame from the feature vector."""
    row = {}
    for col in feature_list:
        val = feature_vector.get(col)
        row[col] = float(val) if val is not None else np.nan
    return pd.DataFrame([row])


def _goals_over_under(matrix: np.ndarray) -> dict:
    """Compute goals over/under from score matrix."""
    ou = {}
    for line in [1.5, 2.5, 3.5]:
        p_over = sum(
            matrix[i, j]
            for i in range(MAX_GOALS)
            for j in range(MAX_GOALS)
            if i + j > line
        )
        ou[str(line)] = {"over": round(p_over, 4), "under": round(1 - p_over, 4)}
    return ou


def _poisson_over_under(lambda_total: float, lines: list[float]) -> dict:
    """Compute over/under from total Poisson lambda."""
    ou = {}
    for line in lines:
        p_over = float(1 - poisson.cdf(int(line), lambda_total))
        ou[str(line)] = {"over": round(p_over, 4), "under": round(1 - p_over, 4)}
    return ou


def _live_over_under(lambda_remaining: float, current_total: int, lines: list[float]) -> dict:
    """Compute live over/under given remaining lambda and current count."""
    ou = {}
    for line in lines:
        needed = line - current_total
        if needed <= 0:
            ou[str(line)] = {"over": 1.0, "under": 0.0}
        else:
            p_over = float(1 - poisson.cdf(int(needed), lambda_remaining))
            ou[str(line)] = {"over": round(p_over, 4), "under": round(1 - p_over, 4)}
    return ou


# ─── Prediction logic ─────────────────────────────────────────────

def predict_prematch(feature_vector: dict, requested_models: list[str], m: Models | None = None) -> dict:
    """Run pre-match prediction using the given model set."""
    m = m or models
    predictions = {}

    # Goals model
    if "goals" in requested_models and m.goals:
        feat_df = _build_feature_df(feature_vector, m.goals.feature_cols)
        lh, la = m.goals.predict_lambdas(feat_df)
        lh, la = float(lh[0]), float(la[0])

        matrix = m.goals.predict_score_matrix(lh, la, MAX_GOALS)
        p_home = float(np.tril(matrix, -1).sum())
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, 1).sum())
        total = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

        # Score matrix as nested list (7x7) and recommended score
        score_matrix = [[round(float(matrix[i, j]), 4) for j in range(MAX_GOALS)] for i in range(MAX_GOALS)]
        best_i, best_j = divmod(int(np.argmax(matrix)), MAX_GOALS)
        recommended_score = {
            "home_goals": best_i,
            "away_goals": best_j,
            "prob": round(float(matrix[best_i, best_j]), 4),
            "label": f"{best_i}-{best_j}",
        }

        predictions["goals"] = {
            "home_win": round(p_home, 4),
            "draw": round(p_draw, 4),
            "away_win": round(p_away, 4),
            "expected_home": round(lh, 4),
            "expected_away": round(la, 4),
            "predicted_total": round(lh + la, 4),
            "over_under": _goals_over_under(matrix),
            "score_matrix": score_matrix,
            "recommended_score": recommended_score,
        }

    # Corners model
    if "corners" in requested_models and m.corners:
        feat_df = _build_feature_df(feature_vector, m.corners.feature_cols)
        clh, cla = m.corners.predict_lambdas(feat_df)
        clh, cla = float(clh[0]), float(cla[0])

        predictions["corners"] = {
            "expected_home": round(clh, 4),
            "expected_away": round(cla, 4),
            "predicted_total": round(clh + cla, 4),
            "over_under": _poisson_over_under(clh + cla, [8.5, 9.5, 10.5, 11.5]),
        }

    # Yellows model
    if "yellows" in requested_models and m.yellows:
        feat_df = _build_feature_df(feature_vector, m.yellows.feature_cols)
        ylh, yla = m.yellows.predict_lambdas(feat_df)
        ylh, yla = float(ylh[0]), float(yla[0])

        predictions["yellows"] = {
            "expected_home": round(ylh, 4),
            "expected_away": round(yla, 4),
            "predicted_total": round(ylh + yla, 4),
            "over_under": _poisson_over_under(ylh + yla, [2.5, 3.5, 4.5, 5.5]),
        }

    return predictions


def predict_live(feature_vector: dict, ctx: MatchContext, requested_models: list[str], m: Models | None = None) -> dict:
    """Run live (in-play) prediction using the given model set."""
    m = m or models
    predictions = {}

    minute = ctx.minute or 0

    # Goals model — Bayesian live update
    if "goals" in requested_models and m.goals:
        feat_df = _build_feature_df(feature_vector, m.goals.feature_cols)
        lh, la = m.goals.predict_lambdas(feat_df)
        lh, la = float(lh[0]), float(la[0])

        lp = LivePredictor(lh, la, rho=m.goals.rho, max_goals=MAX_GOALS)
        lp.update(
            minute=minute,
            home_goals=ctx.home_score,
            away_goals=ctx.away_score,
            home_red_cards=ctx.home_red_cards,
            away_red_cards=ctx.away_red_cards,
        )
        live = lp.get_probabilities()

        # Build final score matrix from remaining matrix
        rem = live["remaining_matrix"]
        matrix = np.zeros((MAX_GOALS, MAX_GOALS))
        for i in range(MAX_GOALS):
            for j in range(MAX_GOALS):
                ri, rj = i - ctx.home_score, j - ctx.away_score
                if 0 <= ri < MAX_GOALS and 0 <= rj < MAX_GOALS:
                    matrix[i, j] = rem[ri, rj]
        if matrix.sum() > 0:
            matrix /= matrix.sum()

        score_matrix = [[round(float(matrix[i, j]), 4) for j in range(MAX_GOALS)] for i in range(MAX_GOALS)]
        best_i, best_j = divmod(int(np.argmax(matrix)), MAX_GOALS)
        recommended_score = {
            "home_goals": best_i,
            "away_goals": best_j,
            "prob": round(float(matrix[best_i, best_j]), 4),
            "label": f"{best_i}-{best_j}",
        }

        predictions["goals"] = {
            "home_win": round(float(live["probs_1x2"][0]), 4),
            "draw": round(float(live["probs_1x2"][1]), 4),
            "away_win": round(float(live["probs_1x2"][2]), 4),
            "expected_home": round(lh, 4),
            "expected_away": round(la, 4),
            "predicted_total": round(lh + la, 4),
            "over_under": {
                str(k): {"over": round(v["over"], 4), "under": round(v["under"], 4)}
                for k, v in live["over_under"].items()
            },
            "lambda_remaining_home": round(float(live["lambda_remaining"][0]), 4),
            "lambda_remaining_away": round(float(live["lambda_remaining"][1]), 4),
            "next_goal": {
                k: round(float(v), 4) for k, v in live["next_goal"].items()
            },
            "score_matrix": score_matrix,
            "recommended_score": recommended_score,
        }

    # Corners — simple time-decay for live
    r = max(0, (90 - minute) / 90)

    if "corners" in requested_models and m.corners:
        feat_df = _build_feature_df(feature_vector, m.corners.feature_cols)
        clh, cla = m.corners.predict_lambdas(feat_df)
        clh, cla = float(clh[0]), float(cla[0])

        predictions["corners"] = {
            "expected_home": round(clh, 4),
            "expected_away": round(cla, 4),
            "predicted_total": round(clh + cla, 4),
            "lambda_remaining_home": round(clh * r, 4),
            "lambda_remaining_away": round(cla * r, 4),
            "over_under": _live_over_under(
                (clh + cla) * r,
                ctx.home_corners + ctx.away_corners,
                [8.5, 9.5, 10.5, 11.5],
            ),
        }

    # Yellows — simple time-decay for live
    if "yellows" in requested_models and m.yellows:
        feat_df = _build_feature_df(feature_vector, m.yellows.feature_cols)
        ylh, yla = m.yellows.predict_lambdas(feat_df)
        ylh, yla = float(ylh[0]), float(yla[0])

        predictions["yellows"] = {
            "expected_home": round(ylh, 4),
            "expected_away": round(yla, 4),
            "predicted_total": round(ylh + yla, 4),
            "lambda_remaining_home": round(ylh * r, 4),
            "lambda_remaining_away": round(yla * r, 4),
            "over_under": _live_over_under(
                (ylh + yla) * r,
                ctx.home_yellows + ctx.away_yellows,
                [2.5, 3.5, 4.5, 5.5],
            ),
        }

    return predictions


# ─── Routes ────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(req: PredictionRequest):
    """Main prediction endpoint.

    Accepts a feature vector (pre-computed by data service) and returns
    predictions from all requested models (goals, corners, yellows).
    """
    if models.goals is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    t0 = time.time()

    if req.prediction_type == "live" and req.match_context:
        predictions = predict_live(req.feature_vector, req.match_context, req.models)
    else:
        predictions = predict_prematch(req.feature_vector, req.models)

    elapsed = round((time.time() - t0) * 1000, 1)

    log.info(
        f"predict match_id={req.match_id} type={req.prediction_type} "
        f"trigger={req.trigger} models={list(predictions.keys())} "
        f"elapsed={elapsed}ms"
    )

    return PredictionResponse(
        predictions=predictions,
        model_metadata={
            "version": models.version,
            "prediction_type": req.prediction_type,
            "elapsed_ms": elapsed,
        },
    )


@app.post("/backfill", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def backfill(req: PredictionRequest):
    """Backfill prediction endpoint (model trained up to 2026-01-31).

    Same interface as /predict but uses the backfill model set.
    Only supports prematch predictions.
    """
    if backfill_models.goals is None:
        raise HTTPException(status_code=503, detail="Backfill models not loaded")

    t0 = time.time()
    predictions = predict_prematch(req.feature_vector, req.models, m=backfill_models)
    elapsed = round((time.time() - t0) * 1000, 1)

    log.info(
        f"backfill match_id={req.match_id} "
        f"models={list(predictions.keys())} elapsed={elapsed}ms"
    )

    return PredictionResponse(
        predictions=predictions,
        model_metadata={
            "version": backfill_models.version,
            "prediction_type": "prematch",
            "elapsed_ms": elapsed,
        },
    )


@app.get("/model-status")
async def model_status():
    """Model status check endpoint."""
    return {
        "status": "ok",
        "models_loaded": {
            "goals": models.goals is not None,
            "corners": models.corners is not None,
            "yellows": models.yellows is not None,
        },
        "version": models.version,
        "backfill": {
            "loaded": backfill_models.goals is not None,
            "version": backfill_models.version,
        },
    }
