import numpy as np
from utils.metrics import mean_squared_error


def k_fold_cross_validation(model_class, X, y, k=5, lr=0.01, n_iters=1000):
    fold_size = len(X) // k
    errors = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size

        X_val = X[start:end]
        y_val = y[start:end]

        X_train = np.concatenate((X[:start], X[end:]))
        y_train = np.concatenate((y[:start], y[end:]))

        model = model_class(lr=lr, n_iters=n_iters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        error = mean_squared_error(y_val, y_pred)
        errors.append(error)

    return np.mean(errors)