"""
Metrics Aggregation Module.

Collects raw training metrics from JSON files for different models (GB, MLP, RF, CB)
and aggregates them into a summary Excel file.
The output file contains separate sheets for:
    - R2_vs_Size
    - Training_Time_vs_Size
    - Prediction_Time_vs_Size
    - Model_Size_vs_Size
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

# ---------- LOGGING ----------
LOG_FILE = Path(config.ROOT_DIR) / "summary_data.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- SETTINGS ----------
OUTPUT_EXCEL_FILE = Path(config.ROOT_DIR) / "summary_data.xlsx"

MODEL_TYPES = ["lightgbm", "mlp", "random_forest", "catboost"]
MODEL_DISPLAY = {
    "lightgbm": "LightGBM",
    "mlp": "MLP",
    "random_forest": "RandomForest",
    "catboost": "CatBoost",
}
MODEL_TYPE_MAP = {
    "lightgbm": "gb",
    "mlp": "mpl",
    "random_forest": "rf",
    "catboost": "cb",
}

PREDICTED_PARAMETERS = config.PREDICTED_PARAMETERS


def col_key(model_type: str, target: str) -> str:
    """Generates column key: <Model>_<Target>."""
    return f"{MODEL_DISPLAY[model_type]}_{target}"


if __name__ == "__main__":
    logger.info("--- Starting Metrics Aggregation ---")

    # Storage: {size: {Model_Target: value}}
    r2_data = {}
    training_time_data = {}
    predict_time_data = {}
    model_size_data = {}

    # --------- READ RAW METRICS ---------
    for target in PREDICTED_PARAMETERS:
        for model_type in MODEL_TYPES:
            key = col_key(model_type, target)
            dir_attr = f"raw_metrics_{MODEL_TYPE_MAP[model_type]}_dir"

            if not hasattr(config, dir_attr):
                logger.warning(f"Config attribute '{dir_attr}' missing for {key}")
                continue

            metrics_dir = Path(getattr(config, dir_attr))
            metrics_path = metrics_dir / f"raw_metrics_{target}.json"

            if not metrics_path.exists():
                logger.warning(f"Metrics file not found: {metrics_path}")
                continue

            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception as e:
                logger.exception(f"Error reading {metrics_path}: {e}")
                continue

            metrics_by_size = metrics.get("metrics_by_size")
            if not metrics_by_size:
                logger.warning(f"'metrics_by_size' missing in {metrics_path}")
                continue

            for size_str, values in metrics_by_size.items():
                try:
                    size = int(size_str)
                except ValueError:
                    logger.warning(
                        f"Invalid size key '{size_str}' in {metrics_path}"
                    )
                    continue

                r2 = values.get("r2_mean", np.nan)
                train_time = values.get("training_time_mean", np.nan)
                pred_time = values.get("predict_time_mean", np.nan)
                model_size_mb = values.get("model_size_mb_mean", np.nan)

                # Populate data structures
                r2_data.setdefault(size, {})[key] = r2
                training_time_data.setdefault(size, {})[key] = train_time
                predict_time_data.setdefault(size, {})[key] = pred_time
                model_size_data.setdefault(size, {})[key] = model_size_mb

            logger.info(f"Processed: {key}")

    # --------- SAVE TO EXCEL ---------
    try:
        with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine="openpyxl") as writer:
            datasets = {
                "R2_vs_Size": r2_data,
                "Training_Time_vs_Size": training_time_data,
                "Prediction_Time_vs_Size": predict_time_data,
                "Model_Size_vs_Size": model_size_data,
            }

            for sheet_name, data_dict in datasets.items():
                if not data_dict:
                    logger.warning(f"No data for sheet '{sheet_name}', skipping.")
                    continue

                df = (
                    pd.DataFrame.from_dict(data_dict, orient="index")
                    .sort_index()
                )
                df.index.name = "SampleSize_%"
                df.to_excel(writer, sheet_name=sheet_name)
                logger.info(f"Sheet '{sheet_name}' saved.")

        logger.info(f"Aggregation complete: {OUTPUT_EXCEL_FILE}")
    except Exception as e:
        logger.exception(f"Excel write error: {e}")
        raise