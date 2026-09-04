"""Cross-Validation Utilities"""

import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline


def evaluate_with_cross_validation(pipeline: Pipeline, features, target, n_split = 5):
    """Evaluate A Pipeline Using Five-Fold Cross-Validation."""
    cv = KFold(n_splits=n_split, shuffle=True, random_state=42)
    results = cross_validate(pipeline, features, target, cv=cv, scoring={"mae": "neg_mean_absolute_error", "rmse":"neg_root_mean_squared_error", "r2":"r2"}, n_jobs=-1, return_train_score=True)
    test_mae = -results["test_mae"]
    test_rmse = -results["test_rmse"]
    test_r2 = results["test_r2"]

    return {
        "mae_mean": np.mean(test_mae),
        "mae_std": np.std(test_mae),
        "rmse_mean": np.mean(test_rmse),
        "rmse_std": np.std(test_rmse),
        "r2_mean": np.mean(test_r2),
        "r2_std": np.std(test_r2),
    }