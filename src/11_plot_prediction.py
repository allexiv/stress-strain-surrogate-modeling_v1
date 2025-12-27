"""
Triptych Generation (FEM / Model / Error).

This script generates comparative visualizations (triptychs) comprising:
1. Ground Truth (FEM data).
2. Model Prediction.
3. Absolute Error distribution.

Operational Logic:
- Selects a random CSV file from `config.features_dir` for evaluation.
- Iterates through defined parameters (PREDICTED_PARAMETERS) and model types.
- Loads the best available model for each target (handling .pkl, .zst, etc.).
- Computes predictions and generates high-resolution heatmaps.
- Supports handling of compressed Random Forest models via temporary files.
"""

from __future__ import annotations
import os
import re
import json
import random
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import griddata
import numpy.ma as ma
from scipy.spatial import KDTree

import torch
import torch.nn as nn
import joblib

try:
    import zstandard as zstd
except ImportError:
    zstd = None

import lightgbm as lgb
from catboost import CatBoostRegressor

import config

# =========================
# 1. VISUALIZATION STYLE
# =========================
def _journal_style():
    """Configures Matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.autolayout": False,
    })

# =========================
# 2. CONFIGURATION & PATHS
# =========================

_ALL_PREDICTED_PARAMETERS = list(config.PREDICTED_PARAMETERS)

# Select specific targets for visualization (leave empty to use all from config)
MANUAL_TARGETS = [
    "SigxxE",
]

if MANUAL_TARGETS:
    PREDICTED_PARAMETERS = [p for p in _ALL_PREDICTED_PARAMETERS if p in MANUAL_TARGETS]
else:
    PREDICTED_PARAMETERS = _ALL_PREDICTED_PARAMETERS

INPUT_FEATURES = [
    'X', 'Y',
    'mean_width', 'mean_height',
    'aspect1', 'aspect2',
    'dist_norm', 'shift_norm',
    'area_ratio',
    'Vertical_Projection',
    'Signed_Dist_Norm',
    'Curvature',
    'Density_Excavated_Distances',
    'Overlap_Index'
]

MODEL_TYPES = ['lightgbm', 'mlp', 'random_forest', 'catboost']
MODEL_NAME_MAP = {
    'lightgbm': 'LightGBM',
    'mlp': 'MLP',
    'random_forest': 'RandomForest',
    'catboost': 'CatBoost'
}

PREDICTED_PARAM_SYMBOLS_RU = {
    "SigxxE": r'$\sigma_{xx}$',
    "SigyyE": r'$\sigma_{yy}$',
    "Sigxy":  r'$\tau_{xy}$',
    "Epsxx":  r'$\varepsilon_{xx}$',
    "Epsyy":  r'$\varepsilon_{yy}$',
    "Gamxy":  r'$\gamma_{xy}$',
    "Utot":   r'$U_{tot}$'
}
PARAMETER_UNITS = {
    "SigxxE": "kPa",
    "SigyyE": "kPa",
    "Sigxy":  "kPa",
    "Epsxx": "",
    "Epsyy": "",
    "Gamxy": "",
    "Utot": "m"
}

GRID_SIZE = 300
INTERPOLATION_METHOD = 'linear'
SAVE_DPI = 600
FIGURES_DIR = Path(config.figures_dir)

TRAINED_MODELS_DIRS = {
    'lightgbm': Path(config.trained_models_gb_dir),
    'mlp': Path(config.trained_models_mpl_dir),
    'random_forest': Path(config.trained_models_rf_dir),
    'catboost': Path(config.trained_models_cb_dir),
}

# =========================
# 3. FEM FILE SELECTION
# =========================
_features_dir = Path(config.features_dir)
_csvs = [p for p in _features_dir.glob("*.csv") if p.is_file()]
if not _csvs:
    raise FileNotFoundError(f"No CSV files found in directory: {config.features_dir}")

FEATURES_FILE_FOR_VIZ = random.choice(_csvs)

# =========================
# 4. UTILITIES
# =========================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def create_mlp_architecture(params, input_dim: int):
    """Constructs MLP architecture dynamically based on configuration."""
    layers = []
    current_dim = input_dim
    act_map = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'Tanh': nn.Tanh()
    }
    activation_fn = act_map.get(params.get('activation', 'ReLU'), nn.ReLU())
    for i in range(params.get('num_hidden_layers', 1)):
        units = int(params.get(f'layer_{i + 1}_units', 64))
        layers.extend([nn.Linear(current_dim, units), activation_fn])
        if params.get('dropout', 0) > 0:
            layers.append(nn.Dropout(params['dropout']))
        current_dim = units
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)

# Regex patterns for model filenames
_PATTERNS = {
    'lightgbm': re.compile(r"^GB_(\d+)_([A-Za-z0-9]+)\.pkl(?:\.pkl)?$", re.IGNORECASE),
    'mlp':      re.compile(r"^MLP_(\d+)_([A-Za-z0-9]+)\.(pt|pth)$", re.IGNORECASE),
    'random_forest': re.compile(r"^RF_(\d+)_([A-Za-z0-9]+)\.pkl(?:\.zst)?$", re.IGNORECASE),
    'catboost': re.compile(r"^CB_(\d+)_([A-Za-z0-9]+)\.cbm$", re.IGNORECASE),
}

def find_model_path(model_type: str, target: str) -> Path | None:
    """Identifies the model file with the largest training set size (n) for a given target."""
    d = TRAINED_MODELS_DIRS[model_type]
    if not d.exists():
        return None

    best = None
    for p in d.iterdir():
        if not p.is_file():
            continue
        m = _PATTERNS[model_type].match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        t = m.group(2)
        if t != target:
            continue

        # Priority: prefer .pkl.zst over .pkl for Random Forest due to storage efficiency
        priority = 0
        name_low = p.name.lower()
        if model_type == 'random_forest':
            if name_low.endswith(".pkl.zst"):
                priority = 2
            elif name_low.endswith(".pkl"):
                priority = 1

        key = (n, priority)
        if best is None or key > best[0]:
            best = (key, p)

    return best[1] if best else None

# =========================
# 5. MODEL LOADING
# =========================
def _load_rf(path: Path):
    """
    Loads Random Forest models.
    Handles standard .pkl files and ZSTD compressed .pkl.zst files via temporary expansion.
    """
    name = path.name.lower()

    if name.endswith(".pkl.zst") or name.endswith(".zst"):
        if zstd is None:
            raise RuntimeError("Package 'zstandard' is required for reading RF *.zst files.")

        dctx = zstd.ZstdDecompressor()
        CHUNK = 8 * 1024 * 1024

        with tempfile.NamedTemporaryFile(suffix=".pkl", dir=path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with open(path, "rb") as f_in, dctx.stream_reader(f_in) as zr:
                while True:
                    chunk = zr.read(CHUNK)
                    if not chunk:
                        break
                    tmp.write(chunk)

        try:
            model = joblib.load(tmp_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        return model

    return joblib.load(path)

def load_model_and_scalers(model_path: Path, model_type: str, target: str):
    """Loads the model artifact and associated scalers (if applicable)."""
    try:
        if model_type == 'random_forest':
            return _load_rf(model_path), None, None

        elif model_type == 'lightgbm':
            return joblib.load(model_path), None, None

        elif model_type == 'catboost':
            model = CatBoostRegressor()
            model.load_model(str(model_path))
            return model, None, None

        elif model_type == 'mlp':
            hp_path = Path(config.hyperparameters_mpl_dir) / f"hyperparameters_mpl_{target}.json"
            with open(hp_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                hp = obj.get('best_hyperparameters', obj)

            net = create_mlp_architecture(hp, len(INPUT_FEATURES))
            sd = torch.load(model_path, map_location='cpu')

            prefixes = ('net.', 'module.', 'model.', 'mlp.', 'seq.')
            sd_norm = {}
            for k, v in sd.items():
                kk = k
                for pfx in prefixes:
                    if kk.startswith(pfx):
                        kk = kk[len(pfx):]
                        break
                sd_norm[kk] = v
            net.load_state_dict(sd_norm, strict=False)
            net.eval()

            m = _PATTERNS['mlp'].match(model_path.name)
            k = int(m.group(1)) if m else None
            sx = sy = None
            if k is not None:
                sx_path = model_path.parent / f"scaler_X_{k}_{target}.pkl"
                sy_path = model_path.parent / f"scaler_Y_{k}_{target}.pkl"
                if sx_path.exists() and sy_path.exists():
                    sx = joblib.load(sx_path)
                    sy = joblib.load(sy_path)
                else:
                    print(f"[Warn] MLP: Scalers missing for {target} (k={k}). Skipping.")
                    return None, None, None

            return net, sx, sy

        return None, None, None

    except Exception as e:
        print(f"[Warn] Error loading {model_type} for {target}: {e}")
        return None, None, None

def predict_unified(model, X_df: pd.DataFrame, model_type: str, scaler_X, scaler_Y):
    """Unified interface for model inference."""
    X_values = X_df[INPUT_FEATURES].values

    if model_type == 'mlp':
        X_scaled = scaler_X.transform(X_values)
        X_tensor = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            pred_scaled = model(X_tensor).numpy()
        return scaler_Y.inverse_transform(pred_scaled).ravel()

    elif model_type in ('random_forest', 'catboost', 'lightgbm'):
        return np.asarray(model.predict(X_values)).ravel()

    raise NotImplementedError(f"predict_unified: Not implemented for {model_type}")

# =========================
# 6. HEATMAP HELPERS
# =========================
def _get_cbar_formatter(vmin, vmax):
    abs_max = max(abs(vmin), abs(vmax))
    if np.isclose(vmin, vmax) or abs_max < 1e-9:
        return FormatStrFormatter('%.2f')
    if abs_max < 0.01:
        return FormatStrFormatter('%.4f')
    if abs_max < 1:
        return FormatStrFormatter('%.3f')
    if abs_max < 10:
        return FormatStrFormatter('%.2f')
    if abs_max < 100:
        return FormatStrFormatter('%.1f')
    return FormatStrFormatter('%.0f')

def _percentile_limits(arr_list, lo=2, hi=98):
    data = np.concatenate([a[np.isfinite(a)] for a in arr_list if a is not None])
    if data.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(data, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if not np.isfinite(vmin) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
    return float(vmin), float(vmax)

# =========================
# 7. TRIPTYCH GENERATION
# =========================
def create_heatmap_triptych(df_all, param, model_type, y_true, y_pred, output_dir: Path):
    _journal_style()
    print(f"    Triptych: {param} ({model_type})")

    df_plot = df_all.copy()
    df_plot['true'] = y_true
    df_plot['pred'] = y_pred
    df_plot['error'] = np.abs(y_true - y_pred)

    soil_mask = (df_plot.get('Excavated_soil', 0) == 0)
    df_soil = df_plot[soil_mask]
    if df_soil.empty:
        print("    [Warn] No soil data found (empty mask). Skipping.")
        return

    grid_x = np.linspace(df_all['X'].min(), df_all['X'].max(), GRID_SIZE)
    grid_y = np.linspace(df_all['Y'].min(), df_all['Y'].max(), GRID_SIZE)
    Grid_X, Grid_Y = np.meshgrid(grid_x, grid_y)
    points = df_soil[['X', 'Y']].values

    z_true = griddata(points, df_soil['true'].values, (Grid_X, Grid_Y), method=INTERPOLATION_METHOD)
    z_pred = griddata(points, df_soil['pred'].values, (Grid_X, Grid_Y), method=INTERPOLATION_METHOD)
    z_error = griddata(points, df_soil['error'].values, (Grid_X, Grid_Y), method=INTERPOLATION_METHOD)

    excavation_mask = None
    if 'Excavated_soil' in df_all.columns:
        kdtree = KDTree(df_all[['X', 'Y']].values)
        _, indices = kdtree.query(np.vstack((Grid_X.ravel(), Grid_Y.ravel())).T)
        exc_flags = df_all['Excavated_soil'].values[indices].reshape(Grid_X.shape)
        excavation_mask = exc_flags > 0.5

    def apply_mask(z):
        return ma.masked_where(excavation_mask, z) if excavation_mask is not None else z

    vmin, vmax = _percentile_limits([z_true, z_pred], 2, 98)
    if np.any(np.isfinite(z_error)):
        err_max = np.nanpercentile(z_error[np.isfinite(z_error)], 99)
    else:
        err_max = 1.0
    if not np.isfinite(err_max) or err_max <= 1e-9:
        err_max = 1.0

    fig_width_mm = 180.0
    inch_per_mm = 1.0 / 25.4
    fig_w = fig_width_mm * inch_per_mm
    fig_h = (fig_w / 3.0) * 1.3

    fig, axes = plt.subplots(
        1, 3,
        figsize=(fig_w, fig_h),
        dpi=SAVE_DPI,
        constrained_layout=True,
        gridspec_kw={'wspace': 0.05}
    )

    model_name = MODEL_NAME_MAP[model_type]
    symbol = PREDICTED_PARAM_SYMBOLS_RU.get(param, param)
    units = f" ({PARAMETER_UNITS.get(param)})" if PARAMETER_UNITS.get(param) else ""

    plots_data = [
        {'ax': axes[0], 'data': z_true,  'title': 'FEM',         'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax},
        {'ax': axes[1], 'data': z_pred,  'title': model_name,    'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax},
        {'ax': axes[2], 'data': z_error, 'title': 'Abs. Error',  'cmap': 'Reds',    'vmin': 0.0,  'vmax': err_max},
    ]

    # Error label formatting
    m = re.match(r'\$(.*)_{(.*)}\$', symbol)
    if m:
        symbol_base, symbol_sub = m.groups()
        error_label = (
            f"$|{symbol_base}_{{{symbol_sub}, \\mathrm{{FEM}}}} - "
            f"{symbol_base}_{{{symbol_sub}, \\mathrm{{{model_name}}}}}|${units}"
        )
    else:
        base = symbol.strip('$')
        error_label = (
            f"$|{base}_{{\\mathrm{{FEM}}}} - "
            f"{base}_{{\\mathrm{{{model_name}}}}}|${units}"
        )

    cbar_labels = [f"{symbol}{units}", f"{symbol}{units}", error_label]
    extent = (grid_x[0], grid_x[-1], grid_y[0], grid_y[-1])

    for i, p in enumerate(plots_data):
        ax = p['ax']
        im = ax.imshow(
            apply_mask(p['data']),
            extent=extent,
            origin='lower',
            cmap=p['cmap'],
            aspect='equal',
            vmin=p['vmin'],
            vmax=p['vmax'],
            interpolation='nearest'
        )

        if excavation_mask is not None:
            ax.contour(
                Grid_X, Grid_Y, excavation_mask,
                levels=[0.5],
                colors='black',
                linewidths=0.7
            )

        ax.set_title(p['title'])
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlabel("X, m")
        ax.set_ylim(df_all['Y'].min(), df_all['Y'].max())

        if i == 0:
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.set_ylabel("Y, m")
        else:
            ax.tick_params(axis='y', labelleft=False)
            ax.set_ylabel("")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="7%", pad=0.4)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(cbar_labels[i], fontsize=9)
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.6)

        fmt = _get_cbar_formatter(p['vmin'], p['vmax'])
        cbar.ax.xaxis.set_major_formatter(fmt)
        cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

    ensure_dir(output_dir)
    fname = f"triptych_{param}_{model_type}"
    fig.savefig(output_dir / f"{fname}.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [OK] {fname}.png")

# =========================
# 8. MAIN EXECUTION
# =========================
def main_triptychs():
    print("\n--- Generating Triptychs (Fact / Prediction / Error) ---")
    print(f"[Data] Selected FEM file: {FEATURES_FILE_FOR_VIZ.name}")

    df_viz = pd.read_csv(FEATURES_FILE_FOR_VIZ)
    missing = [f for f in INPUT_FEATURES if f not in df_viz.columns]
    if missing:
        raise SystemExit(f"Missing features in FEM CSV: {missing}")

    triptych_dir = FIGURES_DIR / "triptychs"
    ensure_dir(triptych_dir)

    for param in PREDICTED_PARAMETERS:
        if param not in df_viz.columns:
            print(f"[Warn] Parameter '{param}' not found in CSV. Skipping.")
            continue

        y_true = df_viz[param].values

        for model_type in MODEL_TYPES:
            mpath = find_model_path(model_type, param)
            if not mpath:
                print(f"[Warn] Model not found: {model_type}_{param}")
                continue

            model, sx, sy = load_model_and_scalers(mpath, model_type, param)
            if model is None:
                print(f"[Warn] Failed to load: {mpath.name}. Skipping.")
                continue

            try:
                y_pred = predict_unified(model, df_viz, model_type, sx, sy)
                create_heatmap_triptych(df_viz, param, model_type, y_true, y_pred, triptych_dir)
            except Exception as e:
                print(f"[Warn] Inference/Plotting error for {mpath.name}: {e}")


if __name__ == '__main__':
    _journal_style()
    ensure_dir(FIGURES_DIR)
    print(f"Features file for visualization (random selection): {FEATURES_FILE_FOR_VIZ.name}")
    print("Selected Targets:", PREDICTED_PARAMETERS)
    main_triptychs()
    print("\nCompleted. Check output directory:", FIGURES_DIR / "triptychs")