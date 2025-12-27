"""
Data Splitting Module.

Generates:
1. Cumulative training subsets based on a predefined schedule of file counts.
2. Fixed validation and test sets.

Output Structure:
  data/04_train/ -> train_{k}.csv (where k is number of source files)
  data/05_test/  -> validation.csv, test.csv
"""

import math
import random
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import config

# ===== CONFIGURATION =====
USE_THINNED_DATA = False  # True -> use config.thinned_data_dir, False -> use config.features_dir
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Predefined schedule for cumulative training set sizes (number of files)
EXACT_FILE_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 24, 32, 40, 56, 72, 80, 160, 240, 320, 400, 480, 560, 640, 720,
                     800]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_concat(files: list, desc: str) -> pd.DataFrame:
    """Concatenates a list of CSV files without shuffling rows."""
    dfs = []
    for f in tqdm(files, desc=desc):
        dfs.append(pd.read_csv(f))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    # 1. Define Data Source
    SOURCE_DIR = Path(config.thinned_data_dir) if USE_THINNED_DATA else Path(config.features_dir)
    OUT_TRAIN_DIR = Path(config.train_data_dir)
    OUT_TEST_DIR = Path(getattr(config, "test_data_dir", "data/05_test"))

    OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Validate Ratios
    if abs((TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("Sum of TRAIN, VALIDATION, and TEST ratios must equal 1.0")

    files = sorted(p for p in SOURCE_DIR.glob("*.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found in directory: {SOURCE_DIR}")

    logging.info(f"Source: {SOURCE_DIR} | Total files found: {len(files)}")

    # 2. Deterministic File Shuffling
    seed = int(getattr(config, "RANDOM_STATE", 42))
    random.seed(seed)
    random.shuffle(files)

    # 3. Split Files
    n_total = len(files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VALIDATION_RATIO)
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    logging.info(f"Split distribution: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # 4. Generate Validation and Test Sets
    if val_files:
        df_val = _read_concat(val_files, "Processing Validation Set")
        (OUT_TEST_DIR / "validation.csv").parent.mkdir(parents=True, exist_ok=True)
        df_val.to_csv(OUT_TEST_DIR / "validation.csv", index=False)
        logging.info(f"Validation set saved to {OUT_TEST_DIR} (Rows: {len(df_val)})")
    else:
        logging.warning("Validation file list is empty.")

    if test_files:
        df_test = _read_concat(test_files, "Processing Test Set")
        df_test.to_csv(OUT_TEST_DIR / "test.csv", index=False)
        logging.info(f"Test set saved to {OUT_TEST_DIR} (Rows: {len(df_test)})")
    else:
        logging.warning("Test file list is empty.")

    # 5. Generate Cumulative Training Sets
    if not train_files:
        logging.warning("Training file list is empty; cumulative sets not created.")
    else:
        # Select steps from schedule that fit within the available training files
        schedule = [k for k in EXACT_FILE_COUNTS if k <= len(train_files)]

        # Ensure the full training set is included
        if not schedule or schedule[-1] != len(train_files):
            schedule.append(len(train_files))

        logging.info(f"Cumulative schedule (file counts): {schedule}")

        for k in schedule:
            subset_files = train_files[:k]
            df_k = _read_concat(subset_files, f"Cumulative Train (first {k} files)")
            out_path = OUT_TRAIN_DIR / f"train_{k}.csv"
            df_k.to_csv(out_path, index=False)
            logging.info(f"Saved: {out_path.name} (Files: {k}, Rows: {len(df_k)})")

    logging.info("Data splitting process completed.")