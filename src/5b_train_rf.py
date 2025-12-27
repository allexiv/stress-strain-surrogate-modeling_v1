"""
Random Forest Training Pipeline.

This script executes the training process for Random Forest models across
cumulative training datasets. It performs the following steps:
1. Loads training and test datasets.
2. Iteratively trains models on increasing dataset sizes (k files).
3. Evaluates performance (R2, RMSE, MAE) on the test set.
4. Benchmarks training and inference time.
5. Saves the best model found for each target variable based on R2 score.
"""

import os
import json
import time
import math
import re
import logging
import warnings
import random
import joblib
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import config

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
ROOT_DIR = Path(getattr(config, "ROOT_DIR", "."))
TRAIN_DIR = Path(config.train_data_dir)
TEST_DIR = Path(config.test_data_dir)
VAL_CSV = TEST_DIR / "validation.csv"
TEST_CSV = TEST_DIR / "test.csv"

RAW_DIR = Path(config.raw_metrics_rf_dir)
RAW_DIR.mkdir(parents=True, exist_ok=True)

MODELS_ROOT = Path(getattr(config, "trained_models_rf_dir", str(ROOT_DIR / "results/02_trained_models/rf")))
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

FEATURES_DIR = Path(getattr(config, "features_dir", str(ROOT_DIR / "data/02_features")))

PREDICTED_PARAMETERS = list(getattr(config, "PREDICTED_PARAMETERS", []))
INPUT_FEATURES = list(getattr(config, "INPUT_FEATURES", ['X', 'Y']))
SEED = int(getattr(config, "RANDOM_STATE", 42))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RAW_DIR / "train_rf_log.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("RF_TRAIN")


def _list_train_files_sorted() -> List[Tuple[int, Path]]:
    """Lists training files sorted by the number of source files (k)."""
    pairs = []
    for p in TRAIN_DIR.glob("train_*.csv"):
        m = re.fullmatch(r"train_(\d+)\.csv", p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _size_mb(path: Path) -> float:
    """Calculates file size in Megabytes."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception:
        return float("nan")


def _load_best_hyperparams(target: str) -> Dict[str, Any]:
    """Loads optimal hyperparameters for the specified target variable."""
    hp_dir = Path(config.hyperparameters_rf_dir)
    for p in [*hp_dir.glob(f"*{target}*.json"), *hp_dir.glob("*.json")]:
        try:
            obj = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue

        if isinstance(obj, dict):
            if target in obj and isinstance(obj[target], dict):
                return dict(obj[target])
            if "best_hyperparameters" in obj and isinstance(obj["best_hyperparameters"], dict):
                return dict(obj["best_hyperparameters"])
            if {"n_estimators", "max_depth", "min_samples_leaf"} & set(obj.keys()):
                return dict(obj)

    raise FileNotFoundError(f"Random Forest hyperparameters not found for target: {target} in {hp_dir}")


def measure_inference_time_three_files_avg(model, feature_cols, features_dir: Path, seed: int = 42):
    """Benchmarks inference time by averaging predictions over three random feature files."""
    files = sorted(p for p in Path(features_dir).glob("*.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No feature files found in {features_dir}")

    rnd = random.Random(seed)
    k = min(3, len(files))
    chosen = rnd.sample(files, k=k)
    times = []
    used = []

    for p in chosen:
        df = pd.read_csv(p)
        X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        t0 = time.perf_counter()
        _ = model.predict(X)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        used.append(str(p))

    return float(sum(times) / len(times)), used


def _atomic_replace(src: Path, dst: Path):
    """Atomically replaces the destination file with the source file."""
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass
    os.replace(src, dst)


def _purge_old_best(dir_path: Path, target: str):
    """Removes previous best model files for the specific target."""
    for p in dir_path.glob(f"RF_*_{target}.pkl"):
        try:
            p.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    script_t0 = time.perf_counter()

    if not TEST_CSV.exists():
        raise SystemExit(f"Test dataset missing: {TEST_CSV}")

    train_list = _list_train_files_sorted()
    if not train_list:
        raise SystemExit(f"No training files (train_*.csv) found in {TRAIN_DIR}")

    df_test = pd.read_csv(TEST_CSV)
    for c in INPUT_FEATURES:
        if c not in df_test.columns:
            raise SystemExit(f"Feature '{c}' missing in test dataset.")

    X_test = df_test[INPUT_FEATURES].to_numpy(dtype=np.float32, copy=False)
    RF_THREADS = int(os.environ.get("RF_N_JOBS", str(os.cpu_count() or 4)))

    for target in PREDICTED_PARAMETERS:
        if target not in df_test.columns:
            logger.warning(f"Target [{target}] missing in test dataset. Skipping.")
            continue

        y_test = df_test[target].to_numpy(dtype=np.float32, copy=False)

        # Load hyperparameters
        hp = _load_best_hyperparams(target)
        hp.setdefault("random_state", SEED)
        hp.setdefault("n_jobs", RF_THREADS)

        metrics_by_size = {}
        best_info = dict(r2=-1e9, k=None, rmse=None, mae=None, train_time=None,
                         pred_time=None, model_size=None, train_rows=None, model_path=None)

        for k, train_path in train_list:
            df_tr = pd.read_csv(train_path)
            miss = [c for c in INPUT_FEATURES if c not in df_tr.columns]

            if miss or target not in df_tr.columns:
                logger.warning(f"Target [{target}] skipping train_{k}: missing features {miss} or target.")
                continue

            X_tr = df_tr[INPUT_FEATURES].to_numpy(dtype=np.float32, copy=False)
            y_tr = df_tr[target].to_numpy(dtype=np.float32, copy=False)

            model = RandomForestRegressor(**hp)

            # Training
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            train_time = time.perf_counter() - t0

            # Evaluation
            y_pred = model.predict(X_test)
            r2_test = float(r2_score(y_test, y_pred))
            rmse_test = float(math.sqrt(mean_squared_error(y_test, y_pred)))
            mae_test = float(mean_absolute_error(y_test, y_pred))

            # Measure Model Size
            tmp_model_path = RAW_DIR / f"_tmp_rf_{target}_k{k}.pkl"
            joblib.dump(model, tmp_model_path)
            size_mb = _size_mb(tmp_model_path)
            try:
                tmp_model_path.unlink()
            except Exception:
                pass

            # Benchmark Inference
            try:
                avg_pred_time, _ = measure_inference_time_three_files_avg(model, INPUT_FEATURES, FEATURES_DIR, SEED)
            except Exception as e:
                logger.warning(f"[{target}|k={k}] Inference benchmark failed: {e}. Using test set for timing.")
                t2 = time.perf_counter()
                _ = model.predict(X_test)
                avg_pred_time = time.perf_counter() - t2

            metrics_by_size[str(k)] = {
                "r2_mean": r2_test,
                "rmse_mean": rmse_test,
                "mae_mean": mae_test,
                "training_time_mean": float(train_time),
                "predict_time_mean": float(avg_pred_time),
                "model_size_mb_mean": float(size_mb),
            }

            logger.info(
                f"[RF][target={target}][k={k}] "
                f"R2={r2_test:.6f} RMSE={rmse_test:.6f} MAE={mae_test:.6f} | "
                f"Train Time={train_time:.2f}s Pred Time={avg_pred_time:.4f}s Size={size_mb:.2f}MB | Device=CPU Threads={RF_THREADS}"
            )

            # Update Best Model
            if r2_test > best_info["r2"]:
                _purge_old_best(MODELS_ROOT, target)
                best_filename = f"RF_{k}_{target}.pkl"
                tmp_save = MODELS_ROOT / f".tmp_{best_filename}"
                final_best_path = MODELS_ROOT / best_filename

                joblib.dump(model, tmp_save)
                _atomic_replace(tmp_save, final_best_path)

                best_info.update(dict(
                    r2=r2_test, k=int(k), rmse=rmse_test, mae=mae_test,
                    train_time=float(train_time), pred_time=float(avg_pred_time),
                    model_size=float(size_mb), train_rows=int(len(X_tr)),
                    model_path=str(final_best_path)
                ))

            # Cleanup
            try:
                del model
            except:
                pass
            gc.collect()

        # Save Metrics
        out = {
            "target": target,
            "metrics_by_size": metrics_by_size,
            "best_model_info": {
                "target": target,
                "k": int(best_info["k"]) if best_info["k"] is not None else None,
                "train_rows": int(best_info["train_rows"]) if best_info["train_rows"] is not None else None,
                "r2": float(best_info["r2"]) if best_info["r2"] is not None else None,
                "rmse": float(best_info["rmse"]) if best_info["rmse"] is not None else None,
                "mae": float(best_info["mae"]) if best_info["mae"] is not None else None,
                "train_time_sec": float(best_info["train_time"]) if best_info["train_time"] is not None else None,
                "predict_time_sec": float(best_info["pred_time"]) if best_info["pred_time"] is not None else None,
                "model_size_mb": float(best_info["model_size"]) if best_info["model_size"] is not None else None,
                "model_path": best_info["model_path"],
            },
            "totals": {
                "script_wall_time_sec": float(time.perf_counter() - script_t0),
                "target_wall_time_sec": float(sum(m["training_time_mean"] for m in metrics_by_size.values()))
            },
            "environment": {"device": "cpu", "n_threads": RF_THREADS}
        }
        out_path = RAW_DIR / f"raw_metrics_{target}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"[{target}] Metrics saved to JSON: {out_path}")