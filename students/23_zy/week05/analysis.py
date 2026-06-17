import numpy as np

def theoretical_cov(X, sigma):
    X_feat = X[:, 1:]
    xtx = X_feat.T @ X_feat
    xtx_inv = np.linalg.inv(xtx)
    return sigma ** 2 * xtx_inv