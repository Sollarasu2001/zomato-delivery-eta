"""Baseline Regression Models."""

import numpy as np

class MeanRegressor:
    """Predict The Training-Set Mean For Every Observation."""

    def __init__(self):
        self.mean_ = None

    def fit(self, target):
        """Fit The Mean Baseline"""
        self.mean_ = float(np.mean(target))
        return self

    def predict(self, size):
        """Predict The Training Mean."""
        if self.mean_ is None:
            raise RuntimeError("The Model Must Be Fitted Before Prediction.")
        return np.full(shape=size, fill_value=self.mean_)

    