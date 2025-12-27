"""
Permutation Feature Importance (PFI) Evaluation.

This script evaluates and visualizes feature importance for:
- LightGBM, MLP, RandomForest, and CatBoost models.

Functionality:
- Loads the first available feature CSV file.
- Loads trained models for specified targets (handling .pkl, .zst, .cbm, .pt formats).
- Computes Permutation Feature Importance (PFI) or applies manual overrides for heavy models.
- Generates normalized feature importance heatmaps.
- Outputs raw and normalized importance data to CSV.
"""

from __future__ import annotations
from pathlib import Path
import os
import re
import json
import warnings
import gc
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

import joblib
import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostRegressor

import config

# ================== CONFIGURATION & CONSTANTS ==================
FIG_DIR = Path(config.figures_dir)
FIG_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR = Path(config.features_dir)
MODELS_ROOT = Path(config.trained_models_rf_dir).parent

INPUT_FEATURES: List[str] = list(config.INPUT_FEATURES)
PREDICTED_PARAMETERS: List[str] = list(config.PREDICTED_PARAMETERS)

# PFI Configuration
SAMPLE_N = 20_000  # Max rows from CSV for calculation
N_REPEATS = 5  # Number of permutation repeats
N_JOBS_PI = 1  # RF parallelization (1 = sequential)
RANDOM_SEED = 42

# Models & Visualization
MODEL_KEYS = ['lightgbm', 'mlp', 'random_forest', 'catboost']
MODEL_LABELS = {
    'lightgbm': 'LightGBM',
    'mlp': 'MLP',
    'random_forest': 'RandomForest',
    'catboost': 'CatBoost'
}

# Active Models Toggle
MODEL_ENABLED = {
    'lightgbm': False,
    'mlp': False,
    'random_forest': True,
    'catboost': False,
}
ACTIVE_MODEL_KEYS = [k for k in MODEL_KEYS if MODEL_ENABLED.get(k, False)]

# ---- MANUAL FI MODE FOR COMPUTATIONALLY EXPENSIVE MODELS ----
USE_MANUAL_FI = True  # If True, utilizes values from MANUAL_FI dictionary

# Dictionary structure: (model_key, target) -> {feature_name: importance_value}
# Features not listed will be zero-filled.
MANUAL_FI: Dict[tuple, Dict[str, float]] = {
    ('random_forest', 'Sigxy'): {
        'X': 44.9,
        'Y': 3.9,
        'Vertical_Projection': 32.1,
        'Signed_Dist_Norm': 5.1,
        'Curvature': 6.9,
        'Overlap_Index': 6.9,
    }
}

# LaTeX labels for targets
TARGET_TEX_LABELS = {
    "SigxxE": r"$\sigma_{xx}$",
    "SigyyE": r"$\sigma_{yy}$",
    "Sigxy": r"$\tau_{xy}$",
    "Epsxx": r"$\varepsilon_{xx}$",
    "Epsyy": r"$\varepsilon_{yy}$",
    "Gamxy": r"$\gamma_{xy}$",
    "Utot": r"$U_{\mathrm{tot}}$",
}

TOP_K_FEATURES = None  # None -> all features; else top-N by global |FI|
CELL_SIZE_IN = 0.4
WRITE_VALUES = True
VALUE_FMT = "{:+.1f}"
VALUE_DISPLAY_THRESHOLD = 1.0  # Absolute value threshold for displaying text
FONT_MIN_PT = 6
FONT_MAX_PT = 12

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 12,
    'axes.titlesize': 16, 'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.titlesize': 18
})


# ================== DATA UTILITIES ==================
def _first_csv(features_dir: Path) -> Path:
    cands = sorted(features_dir.glob("*.csv"))
    if not cands:
        raise FileNotFoundError(f"No *.csv files found in {features_dir}")
    return cands[0]


def _read_df(csv_path: Path, sample_n: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    miss = [f for f in INPUT_FEATURES if f not in df.columns]
    if miss:
        raise SystemExit(f"CSV missing features: {miss}")
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ================== MODEL LOADING ==================
def _joblib_load_from_zst(zst_path: Path):
    """
    Decompresses <name>.pkl.zst to a temporary <name>.tmp.pkl,
    loads via joblib, then removes the temporary file.
    """
    zst_path = Path(zst_path)
    try:
        import zstandard as zstd
    except ImportError:
        raise RuntimeError("Package 'zstandard' is required for reading *.pkl.zst files.")

    base_no_zst = zst_path.with_suffix('')
    tmp_pkl = base_no_zst.with_suffix('.tmp.pkl')

    dctx = zstd.ZstdDecompressor()
    CHUNK = 8 * 1024 * 1024

    try:
        with open(zst_path, "rb") as fin, open(tmp_pkl, "wb") as fout:
            reader = dctx.stream_reader(fin)
            while True:
                chunk = reader.read(CHUNK)
                if not chunk:
                    break
                fout.write(chunk)

        model = joblib.load(tmp_pkl)
    finally:
        try:
            if tmp_pkl.exists():
                os.remove(tmp_pkl)
        except Exception:
            pass

    return model


def _find_model_file(model_type: str, target: str) -> Optional[Path]:
    # ИСПОЛЬЗУЕМ ПУТИ ИЗ CONFIG, ЧТОБЫ ОНИ СОВПАДАЛИ С ПАПКАМИ
    base = {
        'random_forest': Path(config.trained_models_rf_dir),
        'lightgbm': Path(config.trained_models_gb_dir),
        'mlp': Path(config.trained_models_mpl_dir),
        'catboost': Path(config.trained_models_cb_dir),
    }[model_type]

    if not base.exists():
        return None

    prefix = {
        'random_forest': 'RF',
        'lightgbm': 'GB',
        'mlp': 'MLP',
        'catboost': 'CB'
    }[model_type]

    exts_priority = {
        'random_forest': ['.pkl.zst', '.pkl', '.joblib'],
        'lightgbm': ['.pkl', '.txt'],
        'mlp': ['.pt', '.pth'],
        'catboost': ['.cbm'],
    }[model_type]

    pat = re.compile(rf"^{prefix}_(\d+)_({re.escape(target)})($|[_\-\.])", re.IGNORECASE)

    candidates = []
    for p in base.iterdir():
        if not p.is_file():
            continue
        name_l = p.name.lower()
        name_norm = name_l.replace('.pkl.pkl', '.pkl')
        if not any(name_norm.endswith(ext) for ext in exts_priority):
            continue
        used_ext = next(ext for ext in exts_priority if name_norm.endswith(ext))
        stem_wo_ext = p.name[:-len(used_ext)]
        if not pat.match(stem_wo_ext):
            continue

        # Priority sort: (File Extension Priority, Sample Size k, File Size)
        pr = len(exts_priority) - 1 - exts_priority.index(used_ext)
        mnum = re.match(rf"^{prefix}_(\d+)_", stem_wo_ext, re.IGNORECASE)
        num = int(mnum.group(1)) if mnum else -1
        size = p.stat().st_size
        candidates.append((pr, num, size, p))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3]


def _create_mlp_architecture(hp: Dict, input_dim: int) -> nn.Module:
    act_map = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'Tanh': nn.Tanh()}
    layers = []
    cur = input_dim
    activation_fn = act_map.get(hp.get('activation', 'ReLU'), nn.ReLU())
    for i in range(hp.get('num_hidden_layers', 1)):
        units = hp.get(f'layer_{i + 1}_units', 128)
        layers.extend([nn.Linear(cur, units), activation_fn])
        if hp.get('dropout', 0) > 0:
            layers.append(nn.Dropout(hp['dropout']))
        cur = units
    layers.append(nn.Linear(cur, 1))
    net = nn.Sequential(*layers)
    net.eval()
    return net


def _load_mlp_predictor(model_path: Path, target: str):
    """Returns a Scikit-Learn compatible estimator wrapper for the MLP model."""
    obj = torch.load(model_path, map_location='cpu')

    sx_cands = sorted(model_path.parent.glob(f"scaler_X_*_{target}.pkl"))
    sy_cands = sorted(model_path.parent.glob(f"scaler_Y_*_{target}.pkl"))
    scaler_X = joblib.load(sx_cands[-1]) if sx_cands else None
    scaler_Y = joblib.load(sy_cands[-1]) if sy_cands else None

    def _extract_state_dict(o):
        if isinstance(o, nn.Module):
            return None, o
        if isinstance(o, dict):
            for k in ('state_dict', 'model_state_dict', 'model', 'net', 'state', 'weights'):
                if k in o and isinstance(o[k], dict):
                    return o[k], None
            if all(hasattr(v, 'shape') for v in o.values()):
                return o, None
        raise RuntimeError(f"Unknown .pt format: {model_path.name}")

    state_dict, net_obj = _extract_state_dict(obj)

    if net_obj is None:
        hp_path = Path(config.hyperparameters_mpl_dir) / f"hyperparameters_mpl_{target}.json"
        with open(hp_path, 'r', encoding='utf-8') as f:
            hp_obj = json.load(f)
            hp = hp_obj.get('best_hyperparameters', hp_obj)
        net = _create_mlp_architecture(hp, len(INPUT_FEATURES))

        prefixes = ('net.', 'module.', 'model.', 'mlp.', 'seq.')
        sd_norm = {}
        for k, v in state_dict.items():
            kk = k
            for pfx in prefixes:
                if kk.startswith(pfx):
                    kk = kk[len(pfx):]
                    break
            sd_norm[kk] = v

        net.load_state_dict(sd_norm, strict=False)
        net.eval()
    else:
        net = net_obj
        net.eval()

    class _MLPEstimator:
        def __init__(self, net, sx, sy):
            self.net, self.sx, self.sy = net, sx, sy

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

        def fit(self, X, y=None):
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            XX = X if self.sx is None else self.sx.transform(X)
            with torch.no_grad():
                y = self.net(torch.from_numpy(XX).float()).numpy()
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            if self.sy is not None:
                y = self.sy.inverse_transform(y.reshape(-1, 1)).ravel()
            return y

        def score(self, X, y):
            y_pred = self.predict(X)
            return r2_score(y, y_pred)

    if scaler_X is None or scaler_Y is None:
        print(f"[Warn] MLP: Scalers X/Y not found for {target}. Scaling may be incorrect.")
    return _MLPEstimator(net, scaler_X, scaler_Y)


def _wrap_lgb_booster(booster: lgb.Booster):
    """Wraps a LightGBM Booster into a Scikit-Learn compatible estimator."""

    class _LGBBoosterEstimator:
        def __init__(self, booster):
            self.booster = booster

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return self.booster.predict(X)

        def score(self, X, y):
            y_pred = self.predict(X)
            return r2_score(y, y_pred)

    return _LGBBoosterEstimator(booster)


def _load_model(model_path: Path, model_type: str, target: str):
    """Loads a model file and returns an estimator with predict/score methods."""
    if model_type == 'random_forest':
        name = model_path.name.lower()
        if name.endswith('.pkl.zst'):
            return _joblib_load_from_zst(model_path)
        else:
            return joblib.load(model_path)

    if model_type == 'lightgbm':
        name = model_path.name.lower()
        if name.endswith('.pkl'):
            return joblib.load(model_path)
        if name.endswith('.txt'):
            booster = lgb.Booster(model_file=str(model_path))
            return _wrap_lgb_booster(booster)
        raise RuntimeError(f"LightGBM: Unsupported extension: {model_path.name}")

    if model_type == 'catboost':
        m = CatBoostRegressor()
        m.load_model(str(model_path))
        return m

    if model_type == 'mlp':
        return _load_mlp_predictor(model_path, target)

    return None


# ================== FI NORMALIZATION ==================
def _normalize_fi_signed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes Feature Importance while preserving the sign:
    fi_norm_i = fi_i / sum_j(|fi_j|) * 100
    """
    df_norm = df.copy()
    for col in df_norm.columns:
        vals = df_norm[col].to_numpy(dtype=float)
        vals[~np.isfinite(vals)] = 0.0
        S = float(np.sum(np.abs(vals)))
        if S > 0:
            vals = vals / S * 100.0
        else:
            vals[:] = 0.0
        df_norm[col] = vals
    return df_norm


# ================== PFI CALCULATION ==================
def compute_fi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Feature Importance Matrix.
    Index: INPUT_FEATURES
    Columns: <model>_<target>
    Values: Mean permutation importance (R^2 drop), signed.
    Utilizes manual overrides from MANUAL_FI if USE_MANUAL_FI is active.
    """
    df_pi = pd.DataFrame(index=INPUT_FEATURES, dtype=float)
    X = df[INPUT_FEATURES].values

    for target in PREDICTED_PARAMETERS:
        if target not in df.columns:
            print(f"[Skip] Target {target} missing in CSV.")
            continue
        y = df[target].values

        for mk in ACTIVE_MODEL_KEYS:
            manual_key = (mk, target)

            # 1. Manual Mode
            if USE_MANUAL_FI and manual_key in MANUAL_FI:
                col_name = f"{mk}_{target}"
                df_pi[col_name] = 0.0
                for feat, val in MANUAL_FI[manual_key].items():
                    if feat not in df_pi.index:
                        print(f"[Warn] Manual FI: Feature '{feat}' not in INPUT_FEATURES.")
                        continue
                    df_pi.at[feat, col_name] = float(val)
                print(f"[OK] Manual FI applied: {col_name}")
                continue

            # 2. Permutation Importance Calculation
            mpath = _find_model_file(mk, target)
            if not mpath:
                print(f"[Warn] Model not found: {mk}_{target}")
                continue

            try:
                model = _load_model(mpath, mk, target)
            except Exception as e:
                print(f"[Warn] Model load failed {mpath.name}: {e}")
                continue

            if model is None:
                print(f"[Warn] Estimator creation failed for {mpath.name}")
                continue

            try:
                pi = permutation_importance(
                    estimator=model,
                    X=X,
                    y=y,
                    scoring='r2',
                    n_repeats=N_REPEATS,
                    random_state=RANDOM_SEED,
                    n_jobs=N_JOBS_PI
                )
                col_name = f"{mk}_{target}"
                df_pi[col_name] = pi.importances_mean
                print(f"[OK] PFI Calculated: {mpath.name}")
            except Exception as e:
                print(f"[Warn] PFI Calculation error for {mpath.name}: {e}")
            finally:
                del model
                if 'pi' in locals():
                    del pi
                gc.collect()

    return df_pi


# ================== VISUALIZATION ==================
def _wrap_feat_label(s: str, max_len: int = 14) -> str:
    """Wraps feature labels for plot readability."""
    parts = str(s).split('_')
    if len(parts) == 1:
        return s
    lines, cur = [], ""
    for p in parts:
        add = p if not cur else cur + "_" + p
        if len(add) <= max_len:
            cur = add
        else:
            if cur:
                lines.append(cur)
            cur = p
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _plot_heatmaps(df_fi: pd.DataFrame):
    targets = sorted({c.rsplit('_', 1)[1] for c in df_fi.columns if '_' in c})

    # Filter top features if required
    if TOP_K_FEATURES is None:
        features = list(df_fi.index)
    else:
        absmax = df_fi.abs().max(axis=1)
        features = absmax.sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist()

    # Determine global scale
    vals = []
    for mk in ACTIVE_MODEL_KEYS:
        cols = [f"{mk}_{t}" for t in targets if f"{mk}_{t}" in df_fi.columns]
        if cols:
            vals.append(df_fi.loc[features, cols].values)
    vals = np.concatenate(vals, axis=None) if vals else np.array([0.0])
    vals = vals[~np.isnan(vals)]
    amax = float(np.max(np.abs(vals))) if vals.size else 1.0
    vmax = amax if amax > 0 else 1.0
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    for mk in ACTIVE_MODEL_KEYS:
        cols = [f"{mk}_{t}" for t in targets if f"{mk}_{t}" in df_fi.columns]
        if not cols:
            print(f"[Skip] No FI data for {mk}")
            continue

        mat = df_fi.loc[features, cols].copy()
        n_rows, n_cols = mat.shape

        side = CELL_SIZE_IN * max(n_rows, n_cols) + 2.6
        fig, ax = plt.subplots(figsize=(side, side))
        ax.set_box_aspect(1)
        ax.set_aspect('auto')

        Xg, Yg = np.meshgrid(np.arange(n_cols + 1), np.arange(n_rows + 1))
        ax.pcolormesh(
            Xg - 0.5, Yg - 0.5,
            np.ma.masked_invalid(mat.values),
            cmap='RdGy', norm=norm,
            edgecolors='lightgray', linewidth=0.3
        )

        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)

        # X Axis: Targets
        xtick_labels = []
        for c in cols:
            tname = c.rsplit('_', 1)[1]
            xtick_labels.append(TARGET_TEX_LABELS.get(tname, tname))
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(xtick_labels)

        # Y Axis: Features
        ylabels = [_wrap_feat_label(f) for f in features]
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(ylabels)

        ax.tick_params(axis='y', which='major', direction='out', length=4, width=0.8, color='black')
        ax.tick_params(axis='x', which='major', direction='out', length=4, width=0.8, color='black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

        ax.set_title(MODEL_LABELS.get(mk, mk), pad=20)

        if WRITE_VALUES:
            cmap = plt.get_cmap('RdGy')
            base = min(FONT_MAX_PT, max(FONT_MIN_PT, int(14 - 0.12 * max(n_rows, n_cols))))
            for i in range(n_rows):
                for j in range(n_cols):
                    v = mat.iat[i, j]
                    if pd.isna(v) or abs(v) < VALUE_DISPLAY_THRESHOLD:
                        continue
                    rgba = cmap(norm(v))
                    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    txt = (0, 0, 0) if lum > 0.6 else (1, 1, 1)
                    ax.text(j, i, VALUE_FMT.format(v), ha='center', va='center',
                            fontsize=base, color=txt)

        max_len_label = max(len(str(f)) for f in features) if features else 10
        left_pad = 0.16 + 0.012 * max(10, min(max_len_label, 26))
        fig.subplots_adjust(left=left_pad, right=0.96, top=0.92, bottom=0.10)

        out = FIG_DIR / f"fi_heatmap_{mk}.png"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved: {out}")


# ================== MAIN EXECUTION ==================
def main():
    csv_path = _first_csv(FEATURES_DIR)
    print(f"[Data] Selected: {csv_path.name}")
    print(f"[Models] Active: {ACTIVE_MODEL_KEYS}")
    print(f"[Manual FI] Enabled: {USE_MANUAL_FI}")
    df = _read_df(csv_path, SAMPLE_N)

    df_fi_raw = compute_fi_matrix(df)
    if df_fi_raw.empty:
        raise SystemExit("Failed to aggregate any FI data. Check model/target names.")

    df_fi_norm = _normalize_fi_signed(df_fi_raw)

    df_fi_raw.to_csv(FIG_DIR / "fi_raw.csv", index=True, encoding="utf-8-sig")
    df_fi_norm.to_csv(FIG_DIR / "fi_norm_signed.csv", index=True, encoding="utf-8-sig")

    _plot_heatmaps(df_fi_norm)
    print("[Done] FI heatmaps generated.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()