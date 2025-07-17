"""Configuration management for the Titanic Survival Predictor."""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "titanic"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Data file paths
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
SUBMISSION_TEMPLATE_PATH = DATA_DIR / "gender_submission.csv"

# Model configuration
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
    "min_accuracy_threshold": 0.80,
    "algorithms": {
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"],
            "max_iter": [1000]
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "age_imputation_method": "median_by_class_sex",
    "fare_imputation_method": "median_by_class_embarked",
    "embarked_imputation_method": "mode",
    "categorical_encoding": "onehot",
    "numerical_scaling": "standard",
    "derived_features": [
        "FamilySize",
        "IsAlone", 
        "Title",
        "Deck",
        "FarePerPerson"
    ]
}

# Output configuration
OUTPUT_CONFIG = {
    "submission_filename": "titanic_predictions.csv",
    "model_filename": "best_model.pkl",
    "feature_importance_filename": "feature_importance.csv",
    "performance_metrics_filename": "model_metrics.json"
}

def ensure_directories():
    """Create output directories if they don't exist."""
    for directory in [MODELS_DIR, PREDICTIONS_DIR, VISUALIZATIONS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)