import os
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent)

# Directory Structure
FOLDER_PATHS = {
    "raw_data":     "data/01_raw",
    "features":     "data/02_features",
    "train_data":   "data/03_train",
    "test_data":    "data/04_test",

    "hyperparameters": "results/01_hyperparameters",
    "trained_models":  "results/02_trained_models",
    "raw_metrics":     "results/03_metrics_raw",
    "figures":         "results/04_figures",

    # Paths mapped to your specific folder names
    "CatBoost":      "CatBoost",
    "LightGBM":      "LightGBM",
    "MLP":           "MLP",
    "RandomForest":  "RandomForest"
}

# Absolute paths for data
raw_data_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["raw_data"])
features_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["features"])
train_data_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["train_data"])
test_data_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["test_data"])

# Hyperparameters paths
hyperparameters_cb_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["hyperparameters"], "CatBoost")
hyperparameters_gb_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["hyperparameters"], "LightGBM")
hyperparameters_mpl_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["hyperparameters"], "MLP")
hyperparameters_rf_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["hyperparameters"], "RandomForest")

# Trained models paths
trained_models_cb_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["trained_models"], "CatBoost")
trained_models_gb_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["trained_models"], "LightGBM")
trained_models_mpl_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["trained_models"], "MLP")
trained_models_rf_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["trained_models"], "RandomForest")

# Metrics paths
raw_metrics_cb_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["raw_metrics"], "CatBoost")
raw_metrics_gb_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["raw_metrics"], "LightGBM")
raw_metrics_mpl_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["raw_metrics"], "MLP")
raw_metrics_rf_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["raw_metrics"], "RandomForest")

figures_dir = os.path.join(ROOT_DIR, FOLDER_PATHS["figures"])

# General settings
INPUT_PORT = 10000
OUTPUT_PORT = 10001
PASSWORD = 'wwPxyM/#tERaR1+A'
RESULT_TYPES = ["X", "Y", "NodeID", "SigxxE", "SigyyE", "Sigxy", "Epsxx", "Epsyy", "Gamxy", "Utot"]
PREDICTED_PARAMETERS = ["SigxxE", "SigyyE", "Sigxy", "Epsxx", "Epsyy", "Gamxy", "Utot"]

INPUT_FEATURES = [
    "X", "Y", "mean_width", "mean_height", "aspect1", "aspect2",
    "dist_norm", "shift_norm", "area_ratio",
    "Vertical_Projection", "Signed_Dist_Norm", "Curvature",
    "Density_Excavated_Distances", "Overlap_Index",
]

RANDOM_STATE = 42
VALIDATION_SIZE_RATIO = 0.2