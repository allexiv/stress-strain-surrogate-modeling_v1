\# Stress-Strain Surrogate Modeling for Twin Tunnels



This project implements a complete machine learning pipeline for predicting the stress-strain state of soil during twin tunnel excavation using surrogate modeling. It replaces computationally expensive Finite Element Method (FEM) simulations with high-speed ML models.



\## Project Structure



\### Data Pipeline (`/data`)

\*   `01\_raw/`: Raw FEM results from PLAXIS 2D (external link in README).

\*   `02\_features/`: Calculated geomechanical features (Curvature, Density, Overlap Index, etc.).

\*   `03\_train/`: Cumulative training subsets for learning curve analysis ($k$ files).

\*   `04\_test/`: Fixed validation and test sets for unbiased benchmarking.



\### Results (`/results`)

\*   `01\_hyperparameters/`: Optimized parameters found via Optuna (CatBoost, LightGBM, MLP, RandomForest).

\*   `02\_trained\_models/`: Pre-trained model weights (external link in README).

\*   `03\_metrics\_raw/`: Performance logs (R², RMSE, Inference time).

\*   `04\_figures/`: Visualization output (Triptychs and Feature Importance).



\## Execution Order



1\.  \*\*Data Generation:\*\* `1\_generate\_params.py` -> `2b\_run\_fem\_simulations.py` (Requires PLAXIS 2D).

2\.  \*\*Processing:\*\* `3\_calc\_features.py` -> `4\_split\_data.py`.

3\.  \*\*Optimization:\*\* `5a...8a\_tune\_\*.py` scripts to find best hyperparameters.

4\.  \*\*Training:\*\* `5b...8b\_train\_\*.py` to train models on cumulative subsets.

5\.  \*\*Analysis:\*\* `10\_collect\_metrics.py` for summary table and `11\_plot\_prediction.py` for heatmaps.



\## External Resources

Large files (raw data and trained models) are available on Yandex Disk. See README files in the respective folders for links.

