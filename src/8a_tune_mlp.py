"""
MLP (Multi-Layer Perceptron) Hyperparameter Tuning.

This script optimizes PyTorch-based MLP hyperparameters using Optuna.
Key features:
- Defines search space for architecture (layers, units), regularization (dropout, weight_decay), and optimization (LR, optimizer).
- Uses TPE sampler and Hyperband pruning.
- Supports Mixed Precision Training (AMP) for efficiency.
- Validates on R2 score.
"""

import os
import json
import logging
import warnings
import math
from pathlib import Path
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import config

# --- CONFIGURATION ---
DATA_FILE_PATH = os.path.join(config.train_data_dir, "data_100.csv")
RESULTS_DIR = config.hyperparameters_mpl_dir
PREDICTED_PARAMS = config.PREDICTED_PARAMETERS

N_TRIALS = int(os.environ.get("MLP_TUNING_TRIALS", 10_000))
TIMEOUT = int(os.environ.get("MLP_TUNING_TIMEOUT", 3 * 3600))  # 3 hours per target
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 300
PATIENCE = 10
LOG_EVERY = 0

INPUT_FEATURES = [
    'X', 'Y', 'mean_width', 'mean_height', 'aspect1', 'aspect2',
    'dist_norm', 'shift_norm', 'area_ratio',
    'Vertical_Projection', 'Signed_Dist_Norm', 'Curvature',
    'Density_Excavated_Distances', 'Overlap_Index'
]

PARAM_RANGES = {
    'num_hidden_layers': (1, 4),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
    'learning_rate': (1e-5, 1e-3),
    'dropout': (0.0, 0.5),
    'weight_decay': (1e-7, 1e-3),
    'optimizer': ['Adam', 'AdamW', 'RMSprop'],
    'layer_1_units': (64, 512),
    'layer_2_units': (32, 256),
    'layer_3_units': (16, 128),
    'layer_4_units': (8, 64),
}

os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "optimization_mpl.log"), encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# Seed Initialization
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def create_mlp(params, in_dim: int) -> nn.Sequential:
    """Constructs the MLP architecture based on hyperparameters."""
    layers, cur = [], in_dim
    act = getattr(nn, params['activation'])()
    for i in range(params['num_hidden_layers']):
        units = params[f'layer_{i + 1}_units']
        layers += [nn.Linear(cur, units), act]
        if params['dropout'] > 0:
            layers.append(nn.Dropout(params['dropout']))
        cur = units
    layers.append(nn.Linear(cur, 1))
    return nn.Sequential(*layers)


def train_one_trial(trial, model, optimizer, Xtr_t, ytr_t, Xval_t, yval_t, use_amp=True):
    """Executes training loop for a single Optuna trial."""
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and DEVICE.type == "cuda"))
    best_val, no_improve, best_state = math.inf, 0, None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=("cuda" if DEVICE.type == "cuda" else "cpu"),
                            dtype=torch.float16, enabled=(use_amp and DEVICE.type == "cuda")):
            pred = model(Xtr_t)
            loss = criterion(pred, ytr_t)

        if torch.isnan(loss):
            raise optuna.TrialPruned()

        if scaler and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t)
            val_loss = criterion(val_pred, yval_t).item()

        trial.report(-val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early Stopping check
        if val_loss + 1e-7 < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if LOG_EVERY and epoch % LOG_EVERY == 0:
            logger.info(f"Epoch {epoch:04d} | Val MSE={val_loss:.6f}")

        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return best_val


def objective(trial, Xtr_t, ytr_t, Xval_t, yval_t, in_dim):
    """Optuna objective function."""
    p = {
        'num_hidden_layers': trial.suggest_int('num_hidden_layers', *PARAM_RANGES['num_hidden_layers']),
        'activation': trial.suggest_categorical('activation', PARAM_RANGES['activation']),
        'learning_rate': trial.suggest_float('learning_rate', *PARAM_RANGES['learning_rate'], log=True),
        'dropout': trial.suggest_float('dropout', *PARAM_RANGES['dropout']),
        'weight_decay': trial.suggest_float('weight_decay', *PARAM_RANGES['weight_decay'], log=True),
        'optimizer': trial.suggest_categorical('optimizer', PARAM_RANGES['optimizer']),
    }
    for i in range(1, p['num_hidden_layers'] + 1):
        p[f'layer_{i}_units'] = trial.suggest_int(f'layer_{i}_units', *PARAM_RANGES[f'layer_{i}_units'])

    model = create_mlp(p, in_dim).to(DEVICE)
    opt_cls = getattr(optim, p['optimizer'])
    optimizer = opt_cls(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])

    _ = train_one_trial(trial, model, optimizer, Xtr_t, ytr_t, Xval_t, yval_t, use_amp=True)

    model.eval()
    with torch.no_grad():
        y_hat_val = model(Xval_t).detach().cpu().numpy().ravel()
    y_val_np = yval_t.detach().cpu().numpy().ravel()
    return r2_score(y_val_np, y_hat_val)


if __name__ == "__main__":
    logger.info(f"--- Starting MLP Hyperparameter Optimization | Device: {DEVICE} ---")

    if not os.path.exists(DATA_FILE_PATH):
        raise SystemExit(f"Data file not found: {DATA_FILE_PATH}")

    df = pd.read_csv(DATA_FILE_PATH)
    feats = [c for c in INPUT_FEATURES if c in df.columns]
    X_full = df[feats].values.astype(np.float32, copy=False)

    for param in PREDICTED_PARAMS:
        if param not in df.columns:
            logger.warning(f"Target '{param}' missing. Skipping.")
            continue

        logger.info(f"\n--- Target: {param} ---")
        y_full = df[param].values.astype(np.float32, copy=False)

        # Standardize data for neural network training
        X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=0.25, random_state=SEED, shuffle=True)
        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1, 1))

        X_tr_s = sx.transform(X_tr).astype(np.float32, copy=False)
        X_val_s = sx.transform(X_val).astype(np.float32, copy=False)
        y_tr_s = sy.transform(y_tr.reshape(-1, 1)).astype(np.float32, copy=False).ravel()
        y_val_s = sy.transform(y_val.reshape(-1, 1)).astype(np.float32, copy=False).ravel()

        Xtr_t = torch.from_numpy(X_tr_s).to(DEVICE, non_blocking=True)
        ytr_t = torch.from_numpy(y_tr_s).unsqueeze(1).to(DEVICE, non_blocking=True)
        Xval_t = torch.from_numpy(X_val_s).to(DEVICE, non_blocking=True)
        yval_t = torch.from_numpy(y_val_s).unsqueeze(1).to(DEVICE, non_blocking=True)

        sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True, n_startup_trials=15)
        pruner = optuna.pruners.HyperbandPruner(min_resource=20, reduction_factor=3)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=f"mlp_{param}")

        try:
            study.optimize(lambda tr: objective(tr, Xtr_t, ytr_t, Xval_t, yval_t, in_dim=X_tr_s.shape[1]),
                           n_trials=N_TRIALS, timeout=TIMEOUT, n_jobs=1, show_progress_bar=False)

            best_params = dict(study.best_params)
            out = {
                "best_hyperparameters": best_params,
                "best_r2_on_validation": float(study.best_value),
                "note": "R2 is scale-invariant; optimization performed on standardized X, y."
            }
            out_path = os.path.join(RESULTS_DIR, f"hyperparameters_mpl_{param}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=4)

            logger.info(f"Hyperparameters saved: {out_path}")

        except Exception as e:
            logger.error(f"Optuna error for {param}: {e}")