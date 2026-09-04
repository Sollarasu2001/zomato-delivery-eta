"""Regression evaluation metrics."""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def calculate_metrics(actual, predicted):
    """Calculate core regression metrics."""

    mae = mean_absolute_error(actual,predicted)
    mse = mean_squared_error(actual,predicted)
    rmse = mse ** 0.5
    r2 = r2_score(actual,predicted)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }