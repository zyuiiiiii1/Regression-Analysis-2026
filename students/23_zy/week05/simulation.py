import numpy as np
from sklearn.linear_model import LinearRegression

def run_simulation(X, true_beta, sigma, n_simulations=1000, random_state=42):
    np.random.seed(random_state)
    n = X.shape[0]
    result = []
    for _ in range(n_simulations):
        eps = np.random.normal(0, sigma, n)
        y = X @ true_beta + eps
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        result.append(model.coef_)
    return np.array(result)