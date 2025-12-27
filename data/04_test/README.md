TEST AND VALIDATION DATASETS

This directory contains fixed datasets used for hyperparameter tuning and final model benchmarking.

FILES

- validation.csv: Used during the optimization phase (Optuna trials) to select the best hyperparameters.
- test.csv: The hold-out dataset used for the final unbiased evaluation of the trained models.

PURPOSE

Separating these files ensures that all models (Random Forest, LightGBM, CatBoost, MLP) are evaluated on exactly the same data points, allowing for a fair comparison of their predictive capabilities.

DOWNLOAD INFORMATION

- Link: https://disk.yandex.ru/d/_Gnk0j_6f9XhJQ