"""
Week05 多重共线性蒙特卡洛模拟
学号姓名：23_zy
"""

import numpy as np
from data_generator import generate_data
from simulation import run_simulation
from analysis import theoretical_cov
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=== 开始运行 week05 作业 ===")

    n_samples = 1000
    true_beta = np.array([0.0, 5.0, 3.0])
    sigma = 2.0
    n_simulations = 1000

    # 实验 A
    XA = generate_data(n_samples, 0.0)
    bA = run_simulation(XA, true_beta, sigma, n_simulations)
    bA = bA[:, 1:]

    # 实验 B
    XB = generate_data(n_samples, 0.99)
    bB = run_simulation(XB, true_beta, sigma, n_simulations)
    bB = bB[:, 1:]

    print("\n===== 实验 B 经验协方差矩阵 =====")
    print(np.cov(bB.T))

    print("\n===== 实验 B 理论协方差矩阵 =====")
    print(theoretical_cov(XB, sigma))

    # 画图
    plt.figure(figsize=(8,8))
    plt.scatter(bA[:,0], bA[:,1], alpha=0.4, label='A: rho=0')
    plt.scatter(bB[:,0], bB[:,1], alpha=0.4, label='B: rho=0.99')
    plt.scatter(5,3,c='black',s=200,marker='*',label='true beta')
    plt.legend()
    plt.savefig('beta_scatter.png',dpi=300)
    print("\n运行完成！图片已保存")

if __name__ == "__main__":
    main()