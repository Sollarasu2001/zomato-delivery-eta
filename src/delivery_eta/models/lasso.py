"""Lasso Regression model factory."""

from sklearn.linear_model import Lasso

def build_lasso(alpha = 0.01):
    """Create A Lasso Regression Estimator."""
    return Lasso(alpha = alpha, max_iter = 10000)

