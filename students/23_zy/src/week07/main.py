import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from utils.linear_model import LinearRegressionGD
from utils.model_selection import k_fold_cross_validation


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std[std == 0] = 1

    return (X - mean) / std


# =========================
# 1. 生成模拟数据
# =========================
np.random.seed(42)

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(100)

# 加一列 1，表示截距项
X = np.c_[np.ones(X.shape[0]), X]


# =========================
# 2. 基础实验
# =========================
print("===== 基础实验 =====")

error = k_fold_cross_validation(
    LinearRegressionGD,
    X,
    y,
    k=5,
    lr=0.01,
    n_iters=1000
)

print("Cross Validation MSE:", error)


# =========================
# 3. 不同学习率对比
# =========================
print("\n===== 不同学习率对比 =====")

for lr in [0.1, 0.01, 0.001]:
    error = k_fold_cross_validation(
        LinearRegressionGD,
        X,
        y,
        k=5,
        lr=lr,
        n_iters=1000
    )

    print(f"lr={lr}, MSE={error}")


# =========================
# 4. 不同迭代次数对比
# =========================
print("\n===== 不同迭代次数对比 =====")

for n_iters in [100, 500, 1000, 2000]:
    error = k_fold_cross_validation(
        LinearRegressionGD,
        X,
        y,
        k=5,
        lr=0.01,
        n_iters=n_iters
    )

    print(f"n_iters={n_iters}, MSE={error}")


# =========================
# 5. 标准化对比
# =========================
print("\n===== 是否标准化对比 =====")

error_no_std = k_fold_cross_validation(
    LinearRegressionGD,
    X,
    y,
    k=5,
    lr=0.01,
    n_iters=1000
)

X_std = X.copy()
X_std[:, 1:] = standardize(X_std[:, 1:])

error_std = k_fold_cross_validation(
    LinearRegressionGD,
    X_std,
    y,
    k=5,
    lr=0.01,
    n_iters=1000
)

print("No standardization MSE:", error_no_std)
print("With standardization MSE:", error_std)


# =========================
# 6. 画 Loss 曲线
# =========================
print("\n===== 绘制 Loss 曲线 =====")

model = LinearRegressionGD(lr=0.01, n_iters=1000)
model.fit(X, y)

plt.plot(model.optimizer.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Curve of Gradient Descent")

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

plt.savefig(os.path.join(result_dir, "loss_curve.png"))
plt.show()

print("Loss 曲线已保存到 results/loss_curve.png")