"""
LightGBM Hyperparameter Tuning.

This script optimizes LightGBM hyperparameters using Optuna.
Key features:
- Uses TPE (Tree-structured Parzen Estimator) for sampling.
- Implements Hyperband pruning for efficiency.
- Optimizes for R2 score on validation set.
- Supports GPU acceleration if configured.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb
import config

# --- CONFIGURATION ---
DATA_FILE_PATH = os.path.join(config.train_data_dir, "data_100.csv")
RESULTS_DIR = config.hyperparameters_gb_dir
PREDICTED_PARAMETERS = config.PREDICTED_PARAMETERS

N_TRIALS = int(os.environ.get("GB_TUNING_TRIALS", 10_000))
TIMEOUT_SEC = int(os.environ.get("GB_TUNING_TIMEOUT", 2 * 3600))
DEVICE = os.environ.get("GB_DEVICE", "gpu")

INPUT_FEATURES = [
    'X', 'Y', 'mean_width', 'mean_height', 'aspect1', 'aspect2',
    'dist_norm', 'shift_norm', 'area_ratio',
    'Vertical_Projection', 'Signed_Dist_Norm', 'Curvature',
    'Density_Excavated_Distances', 'Overlap_Index'
]

PARAM_RANGES = {
    'n_estimators': (500, 1500),
    'learning_rate': (0.005, 0.1),
    'num_leaves': (30, 200),
    'max_depth': (5, 25),
    'min_child_samples': (20, 100),
    'subsample': (0.7, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (1e-5, 10.0),
    'reg_lambda': (1e-5, 10.0),
}

os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "optimization_gb.log"), encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)


class OptunaPruningCallback:
    """Optuna callback for pruning unpromising LightGBM trials."""

    def __init__(self, trial: optuna.trial.Trial, metric_name: str = "l2"):
        self.trial = trial
        self.metric_name = metric_name
        self.before_iteration = False
        self.after_iteration = True

    def __call__(self, env: lgb.callback.CallbackEnv):
        for data_name, eval_name, value, _ in env.evaluation_result_list:
            if data_name == "valid_0" and eval_name == self.metric_name:
                self.trial.report(-float(value), step=env.iteration)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()


def make_params(trial: optuna.trial.Trial):
    """Generates LightGBM parameters from the search space."""
    return {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'device_type': DEVICE,
        'force_col_wise': True,
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'n_estimators': trial.suggest_int('n_estimators', *PARAM_RANGES['n_estimators']),
        'learning_rate': trial.suggest_float('learning_rate', *PARAM_RANGES['learning_rate'], log=True),
        'num_leaves': trial.suggest_int('num_leaves', *PARAM_RANGES['num_leaves']),
        'max_depth': trial.suggest_int('max_depth', *PARAM_RANGES['max_depth']),
        'min_child_samples': trial.suggest_int('min_child_samples', *PARAM_RANGES['min_child_samples']),
        'subsample': trial.suggest_float('subsample', *PARAM_RANGES['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *PARAM_RANGES['colsample_bytree']),
        'reg_alpha': trial.suggest_float('reg_alpha', *PARAM_RANGES['reg_alpha'], log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', *PARAM_RANGES['reg_lambda'], log=True),
    }


def objective(trial, X_train, y_train, X_val, y_val):
    params = make_params(trial)
    model = lgb.LGBMRegressor(**params)
    callbacks = [
        OptunaPruningCallback(trial, metric_name="l2"),
        lgb.early_stopping(stopping_rounds=50, verbose=False)
    ]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration_)
    r2 = r2_score(y_val, y_pred_val)

    trial.set_user_attr("best_iteration_", int(model.best_iteration_ or params['n_estimators']))
    return r2


if __name__ == "__main__":
    logger.info("--- Starting LightGBM Hyperparameter Optimization ---")

    if not os.path.exists(DATA_FILE_PATH):
        raise SystemExit(f"Data file not found: {DATA_FILE_PATH}")

    df = pd.read_csv(DATA_FILE_PATH)
    feats = [c for c in INPUT_FEATURES if c in df.columns]

    if not feats:
        raise SystemExit("No valid input features found in dataset.")

    X_full = df[feats].astype(np.float32, copy=False).values

    for param in PREDICTED_PARAMETERS:
        if param not in df.columns:
            logger.warning(f"Target '{param}' missing. Skipping.")
            continue

        logger.info(f"\n--- Target: {param} ---")
        y_full = df[param].astype(np.float32, copy=False).values

        X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=0.25, random_state=42, shuffle=True)

        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True, n_startup_trials=15)
        pruner = optuna.pruners.HyperbandPruner(min_resource=50, reduction_factor=3)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=f"lgbm_{param}")

        try:
            study.optimize(lambda tr: objective(tr, X_tr, y_tr, X_val, y_val),
                           n_trials=N_TRIALS, timeout=TIMEOUT_SEC, n_jobs=1, show_progress_bar=False)

            best_params = dict(study.best_params)
            best_params.update({
                'objective': 'regression', 'metric': 'l2', 'boosting_type': 'gbdt', 'device_type': DEVICE,
                'force_col_wise': True, 'verbosity': -1, 'n_jobs': -1, 'random_state': 42,
                'gpu_platform_id': 0, 'gpu_device_id': 0
            })

            best_params['n_estimators'] = int(
                study.best_trial.user_attrs.get("best_iteration_", best_params['n_estimators']))

            logger.info(f"Best R2 (Val) for {param}: {study.best_value:.4f}")

            out = {'best_hyperparameters': best_params, 'best_r2_on_validation': float(study.best_value)}
            out_path = os.path.join(RESULTS_DIR, f"hyperparameters_gb_{param}.json")

            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=4)

            logger.info(f"Hyperparameters saved to: {out_path}")

        except optuna.exceptions.TrialPruned:
            logger.info(f"Trial pruned for {param}.")
        except Exception as e:
            logger.error(f"Optuna error for {param}: {e}")