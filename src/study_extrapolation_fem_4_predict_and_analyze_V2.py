"""
Extrapolation Performance Evaluation.

This script evaluates the performance of trained ML models on the extrapolation dataset.
It performs the following steps:
1. Loads feature CSVs from `data/evaluation_extrapolation/02_features`.
2. Filters nodes based on plastic point indicators and (optional) radius.
3. Loads trained models (LightGBM, RandomForest, CatBoost, MLP) for specified targets.
4. Performs inference and calculates Mean Absolute Error (MAE = |ML - FEM|).
5. Aggregates results into summary Excel/CSV files.
"""

import re
import os
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn

import config

warnings.filterwarnings("ignore", category=UserWarning)

# -------- PARAMETERS --------
RADIUS_M = 0.0  # 0 -> Select only nodes with is_plastic > 0 (no radius)

ROOT = Path(config.ROOT_DIR)
EVAL_DIR = ROOT / "data" / "evaluation_extrapolation" / "02_features"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

PRED_XLSX = ROOT / "predictions.xlsx"
ERR_XLSX = ROOT / "error_summary.xlsx"

INPUT_FEATURES = [
    'X', 'Y', 'mean_width', 'mean_height', 'aspect1', 'aspect2', 'dist_norm',
    'shift_norm', 'area_ratio', 'Vertical_Projection', 'Signed_Dist_Norm',
    'Curvature', 'Density_Excavated_Distances', 'Overlap_Index'
]

TARGETS = ["SigxxE", "SigyyE", "Sigxy"]  # Note: Sigxy in CSV usually lacks "E"
PP_COL = "is_plastic"

# Target alias mapping for model directory search
FOLDER_ALIAS = {
    "Sigxy": ["Sigxy", "SigxyE"]
}

MODEL_SPECS = {
    "LightGBM": {
        "dir": Path(config.trained_models_gb_dir),
        "prefix": "GB",
        "exts": [".pkl", ".txt"],
    },
    "RandomForest": {
        "dir": Path(config.trained_models_rf_dir),
        "prefix": "RF",
        "exts": [".pkl.zst", ".pkl", ".joblib"],
    },
    "CatBoost": {
        "dir": Path(config.trained_models_cb_dir),
        "prefix": "CB",
        "exts": [".cbm"],
    },
    "MLP": {
        "dir": Path(config.trained_models_mpl_dir),
        "prefix": "MLP",
        "exts": [".pt", ".pth"],
    },
}

HYPERPARAMS_MLP_DIR = Path(getattr(config, "hyperparameters_mpl_dir",
                                   ROOT / "results" / "02_trained_models" / "mpl"))


# -------- MODEL LOADING UTILITIES --------
def _joblib_load_from_zst(zst_path: Path):
    """
    Decompresses a .zst file to a temporary .pkl, loads it via joblib,
    and ensures cleanup.
    """
    import zstandard as zstd
    import tempfile

    dctx = zstd.ZstdDecompressor()

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        CHUNK = 8 * 1024 * 1024
        with open(zst_path, "rb") as fin, dctx.stream_reader(fin) as zr, open(tmp_path, "wb") as fout:
            while True:
                chunk = zr.read(CHUNK)
                if not chunk:
                    break
                fout.write(chunk)
        model = joblib.load(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return model


def _find_model_file(model_name: str, target: str) -> Path | None:
    """Locates the specific model file based on naming conventions and priority."""
    spec = MODEL_SPECS[model_name]
    base = spec["dir"]
    if not base.exists():
        return None

    prefix = spec["prefix"]
    exts = spec["exts"]
    aliases = [target] + FOLDER_ALIAS.get(target, [])
    candidates = []

    for p in base.iterdir():
        if not p.is_file():
            continue
        name = p.name

        # Determine extension priority
        used_ext = None
        for ext in exts:
            if name.endswith(ext):
                used_ext = ext
                break
        if used_ext is None:
            continue

        # Parse filename: PREFIX_num_Target
        m = re.match(rf"^{prefix}_(\d+)_([A-Za-z0-9]+)", name, re.IGNORECASE)
        if not m:
            continue
        num = int(m.group(1))
        tag = m.group(2)
        if tag not in aliases:
            continue

        pr = len(exts) - exts.index(used_ext)  # Priority based on extension order
        size = p.stat().st_size
        candidates.append((pr, num, size, p))

    if not candidates:
        return None

    # Sort candidates: Ext Priority -> File Index -> Size
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3]


def _load_mlp_predictor(model_path: Path, target: str):
    """Wraps PyTorch MLP model with Scikit-Learn interface, handling scalers."""
    obj = torch.load(model_path, map_location="cpu")

    # Locate scalers
    sx_cands = sorted(model_path.parent.glob(f"scaler_X_*_{target}.pkl"))
    sy_cands = sorted(model_path.parent.glob(f"scaler_Y_*_{target}.pkl"))
    scaler_X = joblib.load(sx_cands[-1]) if sx_cands else None
    scaler_Y = joblib.load(sy_cands[-1]) if sy_cands else None

    def _extract_state_dict(o):
        if isinstance(o, nn.Module):
            return None, o
        if isinstance(o, dict):
            for k in ("state_dict", "model_state_dict", "model", "net", "state", "weights"):
                if k in o and isinstance(o[k], dict):
                    return o[k], None
            if all(hasattr(v, "shape") for v in o.values()):
                return o, None
        raise RuntimeError(f"Unknown .pt format: {model_path.name}")

    state_dict, net_obj = _extract_state_dict(obj)

    if net_obj is None:
        hp_path = HYPERPARAMS_MLP_DIR / f"hyperparameters_mpl_{target}.json"
        with open(hp_path, "r", encoding="utf-8") as f:
            hp_obj = json.load(f)
        hp = hp_obj.get("best_hyperparameters", hp_obj)

        act_map = {
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "Tanh": nn.Tanh,
        }
        act_cls = act_map.get(hp.get("activation", "ReLU"), nn.ReLU)

        layers = []
        cur = len(INPUT_FEATURES)
        for i in range(hp.get("num_hidden_layers", 1)):
            units = int(hp.get(f"layer_{i + 1}_units", 64))
            layers.append(nn.Linear(cur, units))
            layers.append(act_cls())
            if hp.get("dropout", 0) > 0:
                layers.append(nn.Dropout(hp["dropout"]))
            cur = units
        layers.append(nn.Linear(cur, 1))
        net = nn.Sequential(*layers)

        # Normalize state dict keys
        prefixes = ("net.", "module.", "model.", "mlp.", "seq.")
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

    class _MLPWrapper:
        def __init__(self, net, sx, sy):
            self.net = net
            self.sx = sx
            self.sy = sy

        def fit(self, X, y=None):
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            X_use = X
            if self.sx is not None:
                X_use = self.sx.transform(X)
            with torch.no_grad():
                y = self.net(torch.from_numpy(X_use.astype(np.float32))).numpy()
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            if self.sy is not None:
                y = self.sy.inverse_transform(y.reshape(-1, 1)).ravel()
            return y

    if scaler_X is None or scaler_Y is None:
        print(f"[Warn] MLP {model_path.name}: Scalers missing. Predictions will be unscaled.")

    return _MLPWrapper(net, scaler_X, scaler_Y), scaler_X, scaler_Y


def _load_model(model_name: str, target: str):
    mpth = _find_model_file(model_name, target)
    if mpth is None:
        print(f"[Skip] {model_name}/{target}: Model not found.")
        return None, None, None, None

    name_l = mpth.name.lower()

    if model_name == "RandomForest":
        if name_l.endswith(".pkl.zst"):
            model = _joblib_load_from_zst(mpth)
        elif name_l.endswith(".pkl") or name_l.endswith(".joblib"):
            model = joblib.load(mpth)
        else:
            print(f"[Skip] RF: Unknown extension {mpth.name}")
            return None, None, None, None
        return model, None, None, "RandomForest"

    if model_name == "LightGBM":
        if name_l.endswith(".pkl"):
            model = joblib.load(mpth)
            return model, None, None, "LightGBM"
        if name_l.endswith(".txt"):
            booster = lgb.Booster(model_file=str(mpth))
            return booster, None, None, "LightGBM"
        print(f"[Skip] LightGBM: Unsupported format {mpth.name}")
        return None, None, None, None

    if model_name == "CatBoost":
        model = CatBoostRegressor()
        model.load_model(str(mpth))
        return model, None, None, "CatBoost"

    if model_name == "MLP":
        try:
            wrapper, sx, sy = _load_mlp_predictor(mpth, target)
            return wrapper, sx, sy, "MLP"
        except Exception as e:
            print(f"[Skip] MLP {mpth.name}: {e}")
            return None, None, None, None

    return None, None, None, None


_model_cache = {}  # Cache: (model_name, target) -> (model, sX, sY, type)


def load_model_cached(model_name: str, target: str):
    key = (model_name, target)
    if key in _model_cache:
        return _model_cache[key]

    model, sX, sY, mtype = _load_model(model_name, target)
    _model_cache[key] = (model, sX, sY, mtype)
    return _model_cache[key]


def predict_unified(model, X_df: pd.DataFrame, mtype: str, sX=None, sY=None) -> np.ndarray:
    X = X_df[INPUT_FEATURES].to_numpy(dtype=np.float32, copy=False)

    if model is None:
        return np.full(X.shape[0], np.nan, dtype=float)

    if mtype == "MLP":
        return model.predict(X)

    # LightGBM Booster, sklearn wrappers, RF, CatBoost all support .predict(X)
    return np.asarray(model.predict(X), dtype=float).ravel()


def _row_is_excavated(row: pd.Series) -> bool:
    """Checks various column indicators to determine if a row represents excavated soil."""
    for col in ("Excavated_soil", "excavated_soil", "is_excavated", "Excavated_soill"):
        if col in row:
            v = row[col]
            if isinstance(v, (bool, np.bool_)) and bool(v):
                return True
            if isinstance(v, (int, float, np.integer, np.floating)) and v != 0:
                return True
    for col in ("Soil", "SoilName", "Material", "Phase", "MaterialName"):
        if col in row and isinstance(row[col], str) and "excavated" in row[col].lower():
            return True
    return False


# -------- MAIN EXECUTION --------
if __name__ == "__main__":
    print("--- Quality Evaluation: MAE = mean(|ML - FEM|) ---")
    if RADIUS_M == 0.0:
        print("[Info] RADIUS_M=0.0 -> Only nodes with is_plastic > 0 (no radius expansion).")

    files = sorted(
        EVAL_DIR.glob("*.csv"),
        key=lambda p: float(re.search(r"dist_([\d.]+)", p.name).group(1)) if re.search(r"dist_([\d.]+)",
                                                                                       p.name) else 1e9
    )
    if not files:
        raise FileNotFoundError(f"No CSV files found in {EVAL_DIR}")

    per_model_rows = {m: [] for m in MODEL_SPECS.keys()}
    sums = {t: {m: defaultdict(float) for m in MODEL_SPECS} for t in TARGETS}
    counts = {t: {m: defaultdict(int) for m in MODEL_SPECS} for t in TARGETS}

    for i, fp in enumerate(files, 1):
        m = re.search(r"dist_([\d.]+)", fp.name)
        if not m:
            print(f"[Warn] Skipping file without dist_*: {fp.name}")
            continue
        dist = float(m.group(1))
        print(f"\n({i}/{len(files)}) dist={dist:.3f} m :: {fp.name}")

        df = pd.read_csv(fp)

        # Exclude excavated soil
        if any(c in df.columns for c in (
                "Excavated_soil", "excavated_soil", "is_excavated", "Excavated_soill",
                "Soil", "SoilName", "Material", "Phase", "MaterialName"
        )):
            exc_mask = df.apply(_row_is_excavated, axis=1)
            df = df.loc[~exc_mask].reset_index(drop=True)

        if PP_COL not in df.columns:
            print("  [Warn] is_plastic column missing. Skipping file.")
            continue

        pp_idx = np.where(df[PP_COL].to_numpy() > 0)[0]
        if pp_idx.size == 0:
            print("  [Warn] No plastic nodes found. Skipping file.")
            continue

        XY = df[["X", "Y"]].to_numpy(dtype=float, copy=False)
        sel = set()
        R2 = RADIUS_M * RADIUS_M

        for idx_pp in pp_idx:
            if RADIUS_M == 0.0:
                sel.add(int(idx_pp))
                continue
            dx = XY[:, 0] - XY[idx_pp, 0]
            dy = XY[:, 1] - XY[idx_pp, 1]
            mask = (dx * dx + dy * dy) <= R2
            for j in np.nonzero(mask)[0]:
                sel.add(int(j))

        sel_idx = np.array(sorted(sel), dtype=int)
        if sel_idx.size == 0:
            print("  [Warn] Plastic zone empty. Skipping file.")
            continue

        sub = df.iloc[sel_idx].copy()

        # Extract FEM values
        try:
            fem_xx = sub["SigxxE"].to_numpy(dtype=float, copy=False)
            fem_yy = sub["SigyyE"].to_numpy(dtype=float, copy=False)
            fem_xy = sub["Sigxy"].to_numpy(dtype=float, copy=False)
        except KeyError as e:
            raise KeyError(f"Column {e} missing in {fp.name}; SigxxE, SigyyE, Sigxy required.")

        for model_name in MODEL_SPECS.keys():
            preds = {}
            ok = True
            for tgt in TARGETS:
                model, sX, sY, mtype = load_model_cached(model_name, tgt)
                if model is None:
                    ok = False
                    break
                preds[tgt] = predict_unified(model, sub, mtype, sX, sY)
            if not ok:
                print(f"  [Skip] {model_name}: Incomplete target models.")
                continue

            out = pd.DataFrame({
                "distance_m": dist,
                "X": sub["X"].to_numpy(),
                "Y": sub["Y"].to_numpy(),
                "SigxxE_fem": fem_xx,
                "SigyyE_fem": fem_yy,
                "SigxyE_fem": fem_xy,
                "SigxxE_ml": preds["SigxxE"],
                "SigyyE_ml": preds["SigyyE"],
                "SigxyE_ml": preds["Sigxy"],
            })
            per_model_rows[model_name].append(out)


            # Accumulate MAE
            def acc(t_name, fem, pred):
                ae = np.abs(pred - fem)
                mask = np.isfinite(ae)
                if mask.any():
                    sums[t_name][model_name][dist] += float(ae[mask].sum())
                    counts[t_name][model_name][dist] += int(mask.sum())


            acc("SigxxE", fem_xx, preds["SigxxE"])
            acc("SigyyE", fem_yy, preds["SigyyE"])
            acc("Sigxy", fem_xy, preds["Sigxy"])

    # ---- Save Raw Predictions ----
    wrote_predictions = False
    try:
        import openpyxl

        with pd.ExcelWriter(PRED_XLSX, engine="openpyxl") as w:
            for model_name, parts in per_model_rows.items():
                if not parts:
                    continue
                dfm = pd.concat(parts, ignore_index=True)
                cols = [
                    "distance_m", "X", "Y",
                    "SigxxE_fem", "SigyyE_fem", "SigxyE_fem",
                    "SigxxE_ml", "SigyyE_ml", "SigxyE_ml",
                ]
                dfm = dfm[cols]
                dfm.to_excel(w, index=False, sheet_name=model_name)
        wrote_predictions = True
        print(f"[OK] Predictions Saved: {PRED_XLSX}")
    except Exception as e:
        print(f"[Warn] Failed to write XLSX ({e}), falling back to CSV.")

    if not wrote_predictions:
        out_dir = PRED_XLSX.with_suffix("")
        out_dir = Path(str(out_dir) + "_csv")
        out_dir.mkdir(parents=True, exist_ok=True)
        for model_name, parts in per_model_rows.items():
            if not parts:
                continue
            dfm = pd.concat(parts, ignore_index=True)
            cols = [
                "distance_m", "X", "Y",
                "SigxxE_fem", "SigyyE_fem", "SigxyE_fem",
                "SigxxE_ml", "SigyyE_ml", "SigxyE_ml",
            ]
            dfm = dfm[cols]
            dfm.to_csv(out_dir / f"{model_name}.csv", index=False)
        print(f"[OK] Predictions CSV Saved: {out_dir}")


    # ---- MAE Summary ----
    def build_component_table(comp: str) -> pd.DataFrame:
        dists = sorted({d for mdl in sums[comp].values() for d in mdl.keys()})
        tab = pd.DataFrame({"distance_m": dists})
        for model_name in MODEL_SPECS.keys():
            mae_vals = []
            for d in dists:
                c = counts[comp][model_name].get(d, 0)
                s = sums[comp][model_name].get(d, 0.0)
                mae_vals.append((s / c) if c > 0 else np.nan)
            tab[model_name] = mae_vals
        return tab


    sigxx_tab = build_component_table("SigxxE")
    sigyy_tab = build_component_table("SigyyE")
    sigxy_tab = build_component_table("Sigxy")

    wrote_errors = False
    try:
        import openpyxl

        with pd.ExcelWriter(ERR_XLSX, engine="openpyxl") as w:
            sigxx_tab.to_excel(w, index=False, sheet_name="SigxxE")
            sigyy_tab.to_excel(w, index=False, sheet_name="SigyyE")
            sigxy_tab.to_excel(w, index=False, sheet_name="Sigxy")
        wrote_errors = True
        print(f"[OK] Error Summary Saved: {ERR_XLSX}")
    except Exception as e:
        print(f"[Warn] Failed to write XLSX ({e}), falling back to CSV.")

    if not wrote_errors:
        out_dir = ERR_XLSX.with_suffix("")
        out_dir = Path(str(out_dir) + "_csv")
        out_dir.mkdir(parents=True, exist_ok=True)
        sigxx_tab.to_csv(out_dir / "SigxxE.csv", index=False)
        sigyy_tab.to_csv(out_dir / "SigyyE.csv", index=False)
        sigxy_tab.to_csv(out_dir / "Sigxy.csv", index=False)
        print(f"[OK] Error Summary CSV Saved: {out_dir}")


    # Console Summary
    def _print_overall(label, tab: pd.DataFrame):
        print(f"\n== {label} ==")
        for model_name in MODEL_SPECS.keys():
            v = pd.to_numeric(tab.get(model_name), errors="coerce")
            if v is None or not v.notna().any():
                mean_mae = float("nan")
            else:
                mean_mae = float(np.nanmean(v.to_numpy()))
            print(f"  {model_name:12s}: {mean_mae:.6g}")


    _print_overall("SigxxE (sigma_xx)", sigxx_tab)
    _print_overall("SigyyE (sigma_yy)", sigyy_tab)
    _print_overall("Sigxy (tau_xy)", sigxy_tab)

    print("\nProcess Completed.")