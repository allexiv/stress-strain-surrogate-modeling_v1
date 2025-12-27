"""
Memory-Aware Random Forest Hyperparameter Optimization (HPO).

Features:
- Safe search space definition (bootstrap=True, controlled sampling).
- Step-wise forest growth (warm_start) with Optuna pruning.
- Memory mapping (memmap) via joblib to handle large datasets.
- Graceful handling of MemoryError (TrialPruned).
- Controlled parallelism (RF_N_JOBS).

Environment Variables:
  JOBLIB_TEMP_FOLDER: Directory for joblib memmap files.
  JOBLIB_MAX_NBYTES: Threshold for memmap (default "10M").
  RF_TUNING_TRIALS: Number of Optuna trials (default 10,000).
  RF_TUNING_TIMEOUT: Timeout in seconds (default 7200).
  RF_N_JOBS: Number of parallel workers (default ~50% CPU).
  RF_USE_MAX_LEAF_NODES: Toggle tuning strategy (1=max_leaf_nodes, 0=max_depth).
"""

import os
import json
import logging
import time
import gc
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from joblib import parallel_config

import config

# ---------- PATHS & CONSTANTS ----------
DATA_FILE_PATH = os.path.join(config.train_data_dir, "data_100.csv")
RESULTS_DIR = getattr(config, "hyperparameters_rf_dir",
                      str(Path(getattr(config, "ROOT_DIR", ".")) / "results" / "hyperparams" / "rf"))
os.makedirs(RESULTS_DIR, exist_ok=True)

PRED_PARAMS = list(config.PREDICTED_PARAMETERS)
SEED = int(getattr(config, "RANDOM_STATE", 42))
TEST_SIZE = float(getattr(config, "VALIDATION_SIZE_RATIO", 0.25))

N_TRIALS = int(os.environ.get("RF_TUNING_TRIALS", 10_000))
TIMEOUT = int(os.environ.get("RF_TUNING_TIMEOUT", 2 * 3600))
USE_LEAFS = bool(int(os.environ.get("RF_USE_MAX_LEAF_NODES", "0")))

# Concurrency & Memory Settings
CPU = os.cpu_count() or 2
RF_N_JOBS = int(os.environ.get("RF_N_JOBS", max(1, min(8, CPU // 2))))
JOBLIB_TEMP_FOLDER = os.environ.get("JOBLIB_TEMP_FOLDER", None)
JOBLIB_MAX_NBYTES = os.environ.get("JOBLIB_MAX_NBYTES", "10M")

# Forest Growth Strategy
START_TREES = 100
STEP_TREES = 100

# Search Space Configuration
if USE_LEAFS:
    PARAM_RANGES = {
        'n_estimators': (200, 600),
        'max_leaf_nodes': (64, 2048),
        'min_samples_leaf': (2, 8),
        'min_samples_split': (2, 20),
        'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0],
        'bootstrap': [True],
        'max_samples': (0.60, 0.95),
    }
else:
    PARAM_RANGES = {
        'n_estimators': (200, 600),
        'max_depth': (12, 28),
        'min_samples_leaf': (2, 8),
        'min_samples_split': (2, 20),
        'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0],
        'bootstrap': [True],
        'max_samples': (0.60, 0.95),
    }

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(RESULTS_DIR, "optimization_rf.log"), encoding="utf-8"),
              logging.StreamHandler()],
)
logger = logging.getLogger("rf_hpo")


def _rf_params_from_trial(trial: optuna.Trial) -> Dict[str, Any]:
    """Samples hyperparameters within safe bounds."""
    p = {
        'n_estimators': trial.suggest_int('n_estimators', *PARAM_RANGES['n_estimators']),
        'min_samples_split': trial.suggest_int('min_samples_split', *PARAM_RANGES['min_samples_split']),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', *PARAM_RANGES['min_samples_leaf']),
        'max_features': trial.suggest_categorical('max_features', PARAM_RANGES['max_features']),
        'bootstrap': True,
        'random_state': SEED,
        'n_jobs': RF_N_JOBS,
        'warm_start': True,
    }
    p['max_samples'] = trial.suggest_float('max_samples', *PARAM_RANGES['max_samples'])

    if USE_LEAFS:
        p['max_leaf_nodes'] = trial.suggest_int('max_leaf_nodes', *PARAM_RANGES['max_leaf_nodes'])
    else:
        p['max_depth'] = trial.suggest_int('max_depth', *PARAM_RANGES['max_depth'])
    return p


def _fit_safely(model: RandomForestRegressor, X, y):
    """Fits the model using memory-mapped resources, catching MemoryError."""
    with parallel_config(max_nbytes=JOBLIB_MAX_NBYTES, temp_folder=JOBLIB_TEMP_FOLDER):
        model.fit(X, y)


def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    params = _rf_params_from_trial(trial)
    n_estimators_max = params.pop('n_estimators')
    n_cur = min(START_TREES, n_estimators_max)

    model = RandomForestRegressor(n_estimators=n_cur, **params)

    try:
        _fit_safely(model, X_train, y_train)
    except MemoryError:
        del model;
        gc.collect()
        raise optuna.TrialPruned()

    y_pred = model.predict(X_val)
    best_r2 = float(r2_score(y_val, y_pred))
    best_n = n_cur
    trial.report(best_r2, step=n_cur)

    if trial.should_prune():
        raise optuna.TrialPruned()

    # Step-wise growth (warm start)
    try:
        while n_cur < n_estimators_max:
            n_cur = min(n_cur + STEP_TREES, n_estimators_max)
            model.set_params(n_estimators=n_cur)
            _fit_safely(model, X_train, y_train)

            y_pred = model.predict(X_val)
            r2_val = float(r2_score(y_val, y_pred))
            trial.report(r2_val, step=n_cur)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if r2_val > best_r2:
                best_r2 = r2_val
                best_n = n_cur
    except MemoryError:
        del model;
        gc.collect()
        raise optuna.TrialPruned()

    trial.set_user_attr("n_best", int(best_n))
    return best_r2


if __name__ == '__main__':
    logger.info(f"--- RF HPO Initialized | Strategy: {'Max Leaf Nodes' if USE_LEAFS else 'Max Depth'} | "
                f"Workers: {RF_N_JOBS} | Temp Folder: {JOBLIB_TEMP_FOLDER} ---")

    if not os.path.exists(DATA_FILE_PATH):
        raise SystemExit(f"Critical Error: Data file not found at {DATA_FILE_PATH}")

    df = pd.read_csv(DATA_FILE_PATH)
    INPUT_FEATURES = [
        'X', 'Y', 'mean_width', 'mean_height', 'aspect1', 'aspect2',
        'dist_norm', 'shift_norm', 'area_ratio',
        'Vertical_Projection', 'Signed_Dist_Norm', 'Curvature',
        'Density_Excavated_Distances', 'Overlap_Index'
    ]
    feats = [c for c in INPUT_FEATURES if c in df.columns]
    if not feats:
        raise SystemExit("Critical Error: No valid input features found in dataset.")

    Xfull = df[feats].to_numpy(dtype=np.float32, copy=False)

    for target in PRED_PARAMS:
        if target not in df.columns:
            logger.warning(f"Target column '{target}' missing. Skipping.")
            continue

        logger.info(f"\n--- Optimizing Random Forest for Target: {target} ---")
        yfull = df[target].to_numpy(dtype=np.float32, copy=False)

        Xtr, Xval, ytr, yval = train_test_split(
            Xfull, yfull, test_size=TEST_SIZE, random_state=SEED, shuffle=True
        )

        sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True, n_startup_trials=10)
        pruner = optuna.pruners.HyperbandPruner(min_resource=START_TREES, reduction_factor=3)

        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner,
                                    study_name=f"rf_{target}")

        t0 = time.time()
        try:
            study.optimize(lambda tr: objective(tr, Xtr, ytr, Xval, yval),
                           n_trials=N_TRIALS, timeout=TIMEOUT, n_jobs=1, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Optuna execution error for {target}: {e}")

        if study.best_trial is None:
            logger.error(f"Optimization failed: No valid hyperparameters found for {target}.")
            continue

        best = study.best_trial
        best_params = dict(study.best_params)

        # Retrieve optimal n_estimators
        n_best = int(best.user_attrs.get("n_best", best_params.get("n_estimators", PARAM_RANGES['n_estimators'][0])))
        best_params['n_estimators'] = n_best
        best_params.update({
            'bootstrap': True,
            'n_jobs': RF_N_JOBS,
            'warm_start': True
        })

        elapsed = time.time() - t0
        out = {
            "best_hyperparameters": best_params,
            "best_r2_on_validation": float(best.value),
            "elapsed_sec": float(elapsed),
            "notes": {
                "use_leaf_nodes": USE_LEAFS,
                "memmap_max_nbytes": JOBLIB_MAX_NBYTES,
                "memmap_temp_folder": JOBLIB_TEMP_FOLDER,
                "constraint_strategy": "bootstrap=True, max_samples in [0.60..0.95], "
                                       + ("max_leaf_nodes" if USE_LEAFS else "max_depth")
            }
        }
        out_path = os.path.join(RESULTS_DIR, f"hyperparameters_rf_{target}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=4)

        logger.info(f"Optimization Complete for {target} | Best R2 (Val): {best.value:.6f} | "
                    f"Optimal Estimators: {n_best} | Results saved to: {out_path}")

        del Xtr, Xval, ytr, yval
        gc.collect()