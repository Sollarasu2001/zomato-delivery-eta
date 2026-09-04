"""Hyperparameter tuning utilities."""

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def tune_model(pipeline:Pipeline, param_grid, features, target, cv):
    """Tune A Pipeline Using Cross-Validation"""
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )

    search.fit(features, target)

    return search
