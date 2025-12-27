"""
CatBoost Hyperparameter Tuning.

This script performs hyperparameter optimization for CatBoost using Optuna.
Key features:
- Optimizes for R2 score.
- Supports GPU acceleration.
- Utilizes early stopping.
- Saves optimal hyperparameters in JSON format.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

import config

# --- CONFIGURATION ---
DATA_FILE_PATH = Path(config.train_data_dir) / "data_100.csv"
RESULTS_DIR = Path(getattr(config, "hyperparameters_cb_dir", Path(config.ROOT_DIR) / "results" / "hyperparams" / "cb"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTED_PARAMETERS = list(config.PREDICTED_PARAMETERS)
SEED = int(getattr(config, "RANDOM_STATE", 42))
TEST_SIZE = float(getattr(config, "VALIDATION_SIZE_RATIO", 0.25))

N_TRIALS = int(os.environ.get("CB_TUNING_TRIALS", 10_000))
TIMEOUT = int(os.environ.get("CB_TUNING_TIMEOUT", 2 * 3600))
TASK_TYPE = "GPU" if os.environ.get("CB_DEVICE", "cpu").lower() == "gpu" else "CPU"

INPUT_FEATURES = [
    'X', 'Y', 'mean_width', 'mean_height', 'aspect1', 'aspect2',
    'dist_norm', 'shift_norm', 'area_ratio',
    'Vertical_Projection', 'Signed_Dist_Norm', 'Curvature',
    'Density_Excavated_Distances', 'Overlap_Index'
]

logs_dir = Path(config.ROOT_DIR) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_path = logs_dir / "hyperparams_cb_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ],
)
logger = logging.getLogger(__name__)


def make_params(trial: optuna.trial.Trial):
    """Generates CatBoost hyperparameters from the search space."""
    p = {
        "loss_function": "RMSE",
        "eval_metric": "R2",
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
        "depth": trial.suggest_int("depth", 5, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 8.0),
        "border_count": trial.suggest_int("border_count", 64, 254),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "iterations": trial.suggest_int("iterations", 1500, 6000),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
    }
    if TASK_TYPE == "GPU":
        p.update({"task_type": "GPU", "devices": "0"})
    return p


def objective(trial, X_tr, y_tr, X_val, y_val):
    params = make_params(trial)
    model = CatBoostRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=100,
        verbose=False
    )
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)

    try:
        best_it = int(model.get_best_iteration())
    except Exception:
        best_it = getattr(model, "tree_count_", params["iterations"])

    trial.set_user_attr("best_iteration_", best_it)
    return r2


if __name__ == "__main__":
    if not DATA_FILE_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE_PATH}")

    df = pd.read_csv(DATA_FILE_PATH)
    feats = [c for c in INPUT_FEATURES if c in df.columns]

    if not feats:
        raise SystemExit("No valid input features found in dataset.")

    X_full = df[feats].to_numpy(dtype=np.float32, copy=False)
    combined: Dict[str, Dict[str, Any]] = {}

    for target in PREDICTED_PARAMETERS:
        if target not in df.columns:
            logger.warning(f"Target '{target}' missing. Skipping.")
            continue

        logger.info(f"\n--- CatBoost | Target: {target} | Device: {TASK_TYPE} ---")
        y_full = df[target].to_numpy(dtype=np.float32, copy=False)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_full, y_full, test_size=TEST_SIZE, random_state=SEED, shuffle=True
        )

        sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True, n_startup_trials=15)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"cb_{target}",
        )

        t0 = time.time()
        try:
            study.optimize(
                lambda tr: objective(tr, X_tr, y_tr, X_val, y_val),
                n_trials=N_TRIALS, timeout=TIMEOUT, n_jobs=1, show_progress_bar=False
            )
        except Exception as e:
            logger.error(f"Optuna error for {target}: {e}")

        best_params = dict(study.best_params)
        best_it = int(study.best_trial.user_attrs.get("best_iteration_", best_params.get("iterations", 2000)))
        best_params["iterations"] = best_it
        elapsed = time.time() - t0

        logger.info(f"Best R2 (Val) for {target}: {study.best_value:.6f} (Time={elapsed / 60:.1f} min)")
        logger.info(f"Best Hyperparameters: {best_params}")

        out_path = RESULTS_DIR / f"hyperparameters_cb_{target}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"best_hyperparameters": best_params, "best_r2_on_validation": float(study.best_value)},
                f, ensure_ascii=False, indent=4
            )
        logger.info(f"Saved: {out_path}")

        combined[target] = {"r2": float(study.best_value), "params": best_params}

    best_file = getattr(config, "BEST_HYPERPARAMS_CB_FILE", "best_hyperparams_cb.json")
    with (RESULTS_DIR / best_file).open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    logger.info(f"\nSummary file saved: {(RESULTS_DIR / best_file)}")