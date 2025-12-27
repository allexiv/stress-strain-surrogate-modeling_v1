"""
MLP Training Pipeline (PyTorch).

Executes training on cumulative training sets and evaluates metrics on the test set.
Functionality includes:
1. Training with full-batch gradient descent (or robust chunking if OOM).
2. Data standardization using Scikit-Learn.
3. Mixed Precision Training (AMP) support.
4. Automatic OOM handling (interrupts current target loop if full-batch fails).
5. Saving only the best model per target.
"""

import os
import json
import time
import math
import re
import logging
import warnings
import random
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim

import config

# --- CONFIGURATION ---
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(getattr(config, "ROOT_DIR", "."))
TRAIN_DIR = Path(config.train_data_dir)
TEST_DIR = Path(config.test_data_dir)
VAL_CSV = TEST_DIR / "validation.csv"
TEST_CSV = TEST_DIR / "test.csv"

RAW_DIR = Path(config.raw_metrics_mpl_dir)
RAW_DIR.mkdir(parents=True, exist_ok=True)

MODELS_ROOT = Path(getattr(config, "trained_models_mpl_dir", str(ROOT_DIR / "results/02_trained_models/mpl")))
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

FEATURES_DIR = Path(getattr(config, "features_dir", str(ROOT_DIR / "data/02_features")))

PREDICTED_PARAMETERS = list(getattr(config, "PREDICTED_PARAMETERS", []))
INPUT_FEATURES = list(getattr(config, "INPUT_FEATURES", ['X', 'Y']))
SEED = int(getattr(config, "RANDOM_STATE", 42))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Parameters
MAX_EPOCHS = int(os.environ.get("MLP_MAX_EPOCHS", "300"))
PATIENCE = int(os.environ.get("MLP_PATIENCE", "10"))
EVAL_EVERY = int(os.environ.get("MLP_EVAL_EVERY", "3"))

# Seed Initialization
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# Logging
log_path = RAW_DIR / "train_mlp_log.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]
)
logger = logging.getLogger("MLP_TRAIN")


class MLP(nn.Module):
    """Multi-Layer Perceptron architecture."""

    def __init__(self, in_dim: int, hp: Dict[str, Any]):
        super().__init__()
        act_name = hp.get("activation", "ReLU")
        num_layers = int(hp.get("num_hidden_layers", 2))
        dropout_p = float(hp.get("dropout", 0.0))
        Act = getattr(nn, act_name, nn.ReLU)
        layers, cur = [], in_dim
        for i in range(1, num_layers + 1):
            units = int(hp.get(f"layer_{i}_units", max(16, 2 ** (8 - i))))
            layers += [nn.Linear(cur, units), Act()]
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            cur = units
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _build_optimizer(model: nn.Module, hp: Dict[str, Any]):
    """Creates the optimizer based on hyperparameters."""
    lr = float(hp.get("learning_rate", 1e-3))
    wd = float(hp.get("weight_decay", 0.0))
    opt_name = hp.get("optimizer", "AdamW")
    if opt_name == "AdamW" and DEVICE.type == "cuda":
        try:
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=True)
        except TypeError:
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return getattr(optim, opt_name, optim.Adam)(model.parameters(), lr=lr, weight_decay=wd)


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
    """Loads optimal hyperparameters for the specified target."""
    hp_dir = Path(config.hyperparameters_mpl_dir)

    def try_parse(p: Path):
        try:
            obj = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            return None
        typical = {"learning_rate", "weight_decay", "num_hidden_layers", "activation", "dropout",
                   "layer_1_units", "layer_2_units", "layer_3_units", "optimizer"}
        if isinstance(obj, dict) and (typical & set(obj.keys())):
            return dict(obj)
        if isinstance(obj, dict) and target in obj and isinstance(obj[target], dict):
            blk = obj[target]
            if isinstance(blk.get("params"), dict): return dict(blk["params"])
            if isinstance(blk.get("best_hyperparameters"), dict): return dict(blk["best_hyperparameters"])
            if typical & set(blk.keys()): return dict(blk)
        if isinstance(obj, dict) and isinstance(obj.get("best_hyperparameters"), dict):
            return dict(obj["best_hyperparameters"])
        return None

    for p in [*hp_dir.glob(f"*{target}*.json"), *hp_dir.glob("*.json")]:
        hp = try_parse(p)
        if hp: return hp
    raise FileNotFoundError(f"Hyperparameters not found for '{target}' in {hp_dir}")


def _amp_dtype():
    """Determines the appropriate data type for Mixed Precision Training."""
    if os.environ.get("AMP_FORCE_FP16", "0") == "1":
        return torch.float16
    if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def measure_inference_time_three_files_avg(model, feature_cols, features_dir: Path, seed: int = 42,
                                           warmup: bool = True):
    """Benchmarks inference time by averaging predictions over three random feature files."""
    files = sorted(p for p in Path(features_dir).glob("*.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No feature files found in {features_dir}")

    rnd = random.Random(seed)
    k = min(3, len(files))
    chosen = rnd.sample(files, k=k)

    @torch.no_grad()
    def _predict_df(df: pd.DataFrame):
        X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        b = max(1024, 4096)
        i = 0
        while i < len(X):
            xb = torch.from_numpy(X[i:i + b]).to(DEVICE, non_blocking=True)
            try:
                ctx = torch.amp.autocast("cuda", dtype=_amp_dtype(), enabled=(DEVICE.type == "cuda"))
            except:
                ctx = torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda"))
            with ctx:
                _ = model(xb)
            i += b
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

    times, used = [], []
    for p in chosen:
        df = pd.read_csv(p)
        if warmup:
            _predict_df(df)
        t0 = time.perf_counter()
        _predict_df(df)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        used.append(str(p))
    return float(sum(times) / len(times)), used


@torch.no_grad()
def _predict_in_chunks_robust(model: nn.Module, X_np: np.ndarray, start_batch: int = 4096) -> np.ndarray:
    """Performs prediction in chunks to manage memory usage."""
    model.eval()
    if len(X_np) == 0:
        return np.empty((0,), dtype=np.float32)
    b = max(1024, int(start_batch))
    outs = []
    i = 0
    while i < len(X_np):
        try:
            xb = torch.from_numpy(X_np[i:i + b]).to(DEVICE, non_blocking=True)
            try:
                ctx = torch.amp.autocast("cuda", dtype=_amp_dtype(), enabled=(DEVICE.type == "cuda"))
            except:
                ctx = torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda"))
            with ctx:
                yb = model(xb)
            outs.append(yb.detach().to(dtype=torch.float32).cpu().numpy())
            i += b
        except torch.cuda.OutOfMemoryError:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            if b <= 1024:
                raise
            b //= 2
    return np.concatenate(outs, axis=0).ravel()


def train_full_batch(model, optimizer, scaler, Xtr_t, ytr_t, Xval_s, yval_s, max_epochs, patience, autocast_ctx,
                     criterion_mean):
    """Executes full-batch training loop with early stopping."""
    best_val, no_improve = float("inf"), 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            pred = model(Xtr_t)
            loss = criterion_mean(pred, ytr_t)

        if not torch.isfinite(loss):
            raise RuntimeError("Loss is NaN/Inf in full-batch training.")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % EVAL_EVERY == 0:
            val_pred_s = _predict_in_chunks_robust(model, Xval_s, start_batch=4096)
            val_mse = float(np.mean((val_pred_s - yval_s) ** 2))

            if val_mse + 1e-12 < best_val:
                best_val, no_improve = val_mse, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
    return best_state, best_val


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
    for p in dir_path.glob(f"MLP_*_{target}.pt"):
        try:
            p.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    script_t0 = time.perf_counter()

    if not VAL_CSV.exists() or not TEST_CSV.exists():
        raise SystemExit(f"Input datasets missing: {VAL_CSV} / {TEST_CSV}")

    train_list = _list_train_files_sorted()
    if not train_list:
        raise SystemExit(f"No train_*.csv files found in '{TRAIN_DIR}'")

    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)
    for name_df, df in (("val", df_val), ("test", df_test)):
        miss = [c for c in INPUT_FEATURES if c not in df.columns]
        if miss:
            raise SystemExit(f"Missing features in {name_df} data: {miss}")

    X_val_np = df_val[INPUT_FEATURES].to_numpy(dtype=np.float32, copy=False)
    X_test_np = df_test[INPUT_FEATURES].to_numpy(dtype=np.float32, copy=False)

    for target in PREDICTED_PARAMETERS:
        if target not in df_val.columns or target not in df_test.columns:
            logger.warning(f"Target [{target}] missing in val/test. Skipping.")
            continue

        y_val_np = df_val[target].to_numpy(dtype=np.float32, copy=False)
        y_test_np = df_test[target].to_numpy(dtype=np.float32, copy=False)

        hp = _load_best_hyperparams(target)

        metrics_by_size = {}
        best_info = dict(r2=-1e9, k=None, rmse=None, mae=None,
                         train_time=None, pred_time=None,
                         mode="full", mb=None, model_size=None,
                         train_rows=None, model_path=None)

        stop_on_oom = False

        for k, train_path in train_list:
            if stop_on_oom:
                break

            df_tr = pd.read_csv(train_path)
            miss = [c for c in INPUT_FEATURES if c not in df_tr.columns]
            if miss or target not in df_tr.columns:
                logger.warning(f"Target [{target}] skipping train_{k}: missing features {miss} or target.")
                continue

            X_tr_np = df_tr[INPUT_FEATURES].to_numpy(dtype=np.float32, copy=False)
            y_tr_np = df_tr[target].to_numpy(dtype=np.float32, copy=False)

            # Standardization
            sx = StandardScaler().fit(X_tr_np)
            sy = StandardScaler().fit(y_tr_np.reshape(-1, 1))
            Xtr_s = sx.transform(X_tr_np).astype(np.float32, copy=False)
            Xval_s = sx.transform(X_val_np).astype(np.float32, copy=False)
            Xtest_s = sx.transform(X_test_np).astype(np.float32, copy=False)
            ytr_s = sy.transform(y_tr_np.reshape(-1, 1)).astype(np.float32, copy=False).ravel()
            yval_s = sy.transform(y_val_np.reshape(-1, 1)).astype(np.float32, copy=False).ravel()
            ytest_s = sy.transform(y_test_np.reshape(-1, 1)).astype(np.float32, copy=False).ravel()

            model = MLP(in_dim=Xtr_s.shape[1], hp=hp).to(DEVICE)
            optimizer = _build_optimizer(model, hp)

            try:
                scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))


                def _autocast_ctx():
                    return torch.amp.autocast("cuda", dtype=_amp_dtype(), enabled=(DEVICE.type == "cuda"))
            except Exception:
                scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))


                def _autocast_ctx():
                    return torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda"))

            criterion_mean = nn.MSELoss(reduction="mean")

            try:
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                Xtr_t = torch.from_numpy(Xtr_s).to(DEVICE, non_blocking=True)
                ytr_t = torch.from_numpy(ytr_s.reshape(-1, 1)).to(DEVICE, non_blocking=True)

                best_state, best_val = train_full_batch(
                    model, optimizer, scaler, Xtr_t, ytr_t, Xval_s, yval_s,
                    MAX_EPOCHS, PATIENCE, _autocast_ctx, criterion_mean
                )

                del Xtr_t, ytr_t
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                train_time = time.perf_counter() - t0

                if not best_state:
                    logger.error(f"[MLP][{target}][k={k}] TRAIN FAIL (no best_state)")
                    del model, optimizer, scaler
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                model.load_state_dict(best_state)
                model.eval()

                # Inference & Warmup
                _ = _predict_in_chunks_robust(model, Xtest_s, start_batch=4096)
                y_test_pred_s = _predict_in_chunks_robust(model, Xtest_s, start_batch=4096)
                y_test_pred = sy.inverse_transform(y_test_pred_s.reshape(-1, 1)).ravel()
                y_test_orig = sy.inverse_transform(ytest_s.reshape(-1, 1)).ravel()

                r2_test = float(r2_score(y_test_orig, y_test_pred))
                rmse_test = float(math.sqrt(mean_squared_error(y_test_orig, y_test_pred)))
                mae_test = float(mean_absolute_error(y_test_orig, y_test_pred))

                tmp_model_path_sz = RAW_DIR / f"_tmp_mlp_{target}_k{k}.pt"
                torch.save(model.state_dict(), tmp_model_path_sz)
                size_mb = _size_mb(tmp_model_path_sz)
                try:
                    tmp_model_path_sz.unlink(missing_ok=True)
                except Exception:
                    pass

                # Benchmarking Inference Time
                try:
                    avg_pred_time, _ = measure_inference_time_three_files_avg(
                        model=model, feature_cols=INPUT_FEATURES,
                        features_dir=FEATURES_DIR, seed=SEED, warmup=True
                    )
                except Exception as e:
                    logger.warning(f"[MLP][{target}][k={k}] Benchmark failed ({e}); falling back to test set timing.")
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    _ = _predict_in_chunks_robust(model, Xtest_s, start_batch=4096)
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    avg_pred_time = time.perf_counter() - t3

                logger.info(
                    f"[MLP][{target}][k={k}] Rows={len(X_tr_np)} "
                    f"R2={r2_test:.6f} RMSE={rmse_test:.6f} MAE={mae_test:.6f} "
                    f"Train Time={train_time:.3f}s Pred Time={avg_pred_time:.6f}s Size={size_mb:.3f}MB"
                )

                metrics_by_size[str(k)] = {
                    "r2_mean": r2_test,
                    "rmse_mean": rmse_test,
                    "mae_mean": mae_test,
                    "training_time_mean": float(train_time),
                    "predict_time_mean": float(avg_pred_time),
                    "model_size_mb_mean": float(size_mb)
                }

                if r2_test > best_info["r2"]:
                    _purge_old_best(MODELS_ROOT, target)
                    best_filename = f"MLP_{k}_{target}.pt"
                    tmp_save = MODELS_ROOT / f".tmp_{best_filename}"
                    final_best_path = MODELS_ROOT / best_filename

                    torch.save(model.state_dict(), tmp_save)
                    _atomic_replace(tmp_save, final_best_path)

                    best_info.update(dict(
                        r2=r2_test, k=int(k), rmse=rmse_test, mae=mae_test,
                        train_time=float(train_time), pred_time=float(avg_pred_time),
                        mode="full", mb=None,
                        model_size=float(size_mb), train_rows=int(len(X_tr_np)),
                        model_path=str(final_best_path)
                    ))

            except torch.cuda.OutOfMemoryError:
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                logger.warning(
                    f"[MLP][{target}] OOM encountered at k={k} (full-batch). Stopping training for this target.")
                stop_on_oom = True
            finally:
                try:
                    del model, optimizer, scaler
                except Exception:
                    pass
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        script_wall = time.perf_counter() - script_t0
        env = {
            "device": str(DEVICE),
            "gpu_name": (
                torch.cuda.get_device_name(0) if (DEVICE.type == "cuda" and torch.cuda.is_available()) else None),
        }
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
                "mode": best_info["mode"],
                "microbatch": None,
                "model_size_mb": float(best_info["model_size"]) if best_info["model_size"] is not None else None,
                "model_path": best_info["model_path"]
            },
            "totals": {
                "script_wall_time_sec": float(script_wall),
                "target_wall_time_sec": float(sum(m["training_time_mean"] for m in metrics_by_size.values()))
            },
            "environment": env
        }
        out_path = RAW_DIR / f"raw_metrics_{target}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"[{target}] JSON saved: {out_path}")