"""ElasticNet Regression Model Factory"""

from sklearn.linear_model import ElasticNet

def build_elastic_net(alpha = 0.01, l1_ratio = 0.5):
    """Create An ElasticNet Regression Estimator."""
    return ElasticNet(alpha = alpha, l1_ratio = l1_ratio, max_iter = 50000)