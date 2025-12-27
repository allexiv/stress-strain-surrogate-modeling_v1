# Data-Driven Prediction of Stress–Strain Fields Around Interacting Mining Excavations in Jointed Rock: A Comparative Study of Surrogate Models

[![Preprint](https://img.shields.io/badge/Preprint-Available-blue)](https://doi.org/10.20944/preprints202512.0880.v1)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains the official source code and supplementary materials for the research paper **"Data-Driven Prediction of Stress–Strain Fields Around Interacting Mining Excavations in Jointed Rock"**.

## Abstract

Assessing the stress–strain state (SSS) around interacting mining excavations using the Finite Element Method (FEM) is computationally expensive for parametric studies. This project implements a high-speed surrogate modeling pipeline to replace repetitive FEM simulations.

Using a dataset of 1000 parametric FEM simulations (Hoek-Brown constitutive model), we train and evaluate four machine learning models: **Random Forest**, **LightGBM**, **CatBoost**, and **MLP**. The models predict full stress–strain fields based on engineered geometric features, achieving $R^2 > 0.96$ while offering speed-ups of 15–70x (Tree-based) to 700x (MLP) compared to direct numerical simulation.

## Project Structure

### Data Pipeline (`/data`)
*   **01_raw**: Raw FEM results exported from numerical simulation software (external download required).
*   **02_features**: Calculated geomechanical features (Curvature, Density, Overlap Index, etc.).
*   **03_train**: Cumulative training subsets used for learning curve analysis.
*   **04_test**: Fixed validation and test sets for unbiased benchmarking.
*   **evaluation_extrapolation**: Specialized datasets for testing model performance on unseen geometries (extrapolation study).

### Results (`/results`)
*   **01_hyperparameters**: Optimized model parameters found via Optuna.
*   **02_trained_models**: Directory for serialized model weights (not included in repo, see Data Availability).
*   **03_metrics_raw**: Performance logs ($R^2$, RMSE, Inference time).
*   **04_figures**: Visualization output including triptych plots and feature importance heatmaps.

### Source Code (`/src`)
*   **Data Generation**: Scripts 1-2 (requires compatible FEM software API).
*   **Preprocessing**: Scripts 3-4 (Feature engineering and splitting).
*   **Modeling**: Scripts 5-8 (Tuning and training for RF, LGBM, CatBoost, MLP).
*   **Analysis**: Scripts 10-12 (Metrics aggregation and plotting).
*   **Extrapolation Study**: Scripts prefixed with `study_extrapolation_`.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/allexiv/stress-strain-surrogate-modeling.git
   cd stress-strain-surrogate-modeling
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: To run the FEM data generation scripts (`1_generate_params.py`, `2b_run_fem_simulations.py`), you need the specific geotechnical FEM software and its Python scripting wrapper installed locally.*

## Execution Order

### 1. Main Pipeline (Training & Evaluation)

1.  **Data Generation (Optional):**
    If you have the required FEM software installed, run `2b_run_fem_simulations.py`. Otherwise, download the raw datasets (see **Data Availability**).

2.  **Feature Engineering:**
    Run `3_calc_features.py` to calculate contextual features (Vertical Projection, Curvature, etc.) and `4_split_data.py` to partition the dataset into train/val/test.

3.  **Hyperparameter Optimization:**
    Run the "a" scripts (e.g., `6a_tune_lgbm.py`) to find optimal parameters via Optuna.

4.  **Model Training:**
    Run the "b" scripts (e.g., `6b_train_lgbm.py`) to train models on cumulative subsets (learning curves).

5.  **Analysis:**
    *   `10_collect_metrics.py`: Aggregates metrics into an Excel summary.
    *   `11_plot_prediction.py`: Generates triptych plots (FEM vs Model vs Error).
    *   `12_plot_importance.py`: Calculates and plots Permutation Feature Importance.

### 2. Extrapolation Study (Section 3 of Paper)

To reproduce the analysis of model performance on unseen geometries (variable distances):

1.  **Generate Data:** Run `study_extrapolation_fem_2_plaxis_interaction.py` (requires FEM solver).
2.  **Process Features:** Run `study_extrapolation_fem_3_feature_engineering.py`.
3.  **Evaluate:** Run `study_extrapolation_fem_4_predict_and_analyze_V2.py` to generate error metrics and comparison tables.

## Data Availability

**Datasets:** Due to the large size, the raw and processed datasets are hosted externally.
*   **Raw Data**: [Link to Yandex Disk](https://disk.yandex.ru/d/EF5Qi17MxUe8FA) (Place in `data/01_raw`)
*   **Features**: [Link to Yandex Disk](https://disk.yandex.ru/d/QvyREM4ULvAWwg) (Place in `data/02_features`)
*   **Training Sets**: [Link to Yandex Disk](https://disk.yandex.ru/d/OnGQtplhkxdMxg) (Place in `data/03_train`)
*   **Test Sets**: [Link to Yandex Disk](https://disk.yandex.ru/d/_Gnk0j_6f9XhJQ) (Place in `data/04_test`)

**Pre-trained Models:** The specific pre-trained model weights are not included in this repository due to size. Researchers interested in accessing these artifacts may request them by contacting the corresponding author.

## Citation

If you use this code or data in your research, please cite the following preprint:

```bibtex
@article{protosenya2025data,
  title={Data-Driven Prediction of Stress–Strain Fields Around Interacting Mining Excavations in Jointed Rock: A Comparative Study of Surrogate Models},
  author={Protosenya, Anatoly and Ivanov, Alexey},
  journal={Preprints},
  year={2025},
  doi={10.20944/preprints202512.0880.v1}
}
```