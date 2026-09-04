"""Ridge Regression model factory."""

from sklearn.linear_model import Ridge

def build_ridge(alpha=1.0):
    """Create A Ridge Regression estimator."""
    return Ridge(
        alpha = alpha
    )