"""Polynomial Regression Model Components."""

from sklearn.preprocessing import PolynomialFeatures

def build_polynomial_features(degree=2):
    """Create A Polynomial Features Transformer."""
    return PolynomialFeatures(degree=degree, include_bias=False)


