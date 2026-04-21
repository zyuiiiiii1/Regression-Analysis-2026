import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models import CustomOLS
from utils import evaluate_model, setup_results_dir, generate_synthetic_data


def scenario_A_synthetic(results_dir: Path):
    """场景A：合成数据白盒测试"""
    print("=== 场景A：合成数据测试 ===")
    X, y = generate_synthetic_data(n_samples=1000)
    
    # 划分训练测试集
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 初始化模型
    model_custom = CustomOLS()
    from sklearn.linear_model import LinearRegression
    model_sklearn = LinearRegression()
    
    # 模型对比
    result_custom = evaluate_model(model_custom, X_train, y_train, X_test, y_test, "CustomOLS")
    result_sklearn = evaluate_model(model_sklearn, X_train, y_train, X_test, y_test, "sklearn LinearRegression")
    
    # 验证结果一致性
    assert abs(model_custom.score(X_test, y_test) - model_sklearn.score(X_test, y_test)) < 0.01, "模型结果不一致"
    print("✅ 合成数据测试通过")
    
    # 生成报告
    with open(results_dir / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(f"""# 场景A报告\n## 模型对比\n{result_custom}{result_sklearn}""")


def scenario_B_real_world(results_dir: Path):
    """场景B：真实数据与多实例验证"""
    print("\n=== 场景B：真实数据测试 ===")
    # 读取数据（路径根据实际调整）
    data_path = Path(__file__).parent.parent.parent / "homework" / "week06" / "data" / "q3_marketing.csv"
    df = pd.read_csv(data_path)
    
    # 划分市场（按数据结构调整）
    split_idx = len(df) // 2
    df_na, df_eu = df.iloc[:split_idx], df.iloc[split_idx:]
    X_na, y_na = df_na[["TV_Ad", "Radio_Ad"]].values, df_na["Sales"].values
    X_eu, y_eu = df_eu[["TV_Ad", "Radio_Ad"]].values, df_eu["Sales"].values
    
    # 训练两个独立模型
    model_na = CustomOLS().fit(X_na, y_na)
    model_eu = CustomOLS().fit(X_eu, y_eu)
    
    # F检验
    C = np.array([[1, 1]])
    d = np.array([1.0])
    na_test = model_na.f_test(C, d)
    eu_test = model_eu.f_test(C, d)
    
    # 生成对比图
    plt.figure(figsize=(10, 6))
    plt.bar(["北美", "欧洲"], [model_na.score(X_na, y_na), model_eu.score(X_eu, y_eu)])
    plt.title("市场广告效果对比")
    plt.savefig(results_dir / "market_comparison.png")
    plt.close()
    
    # 生成报告
    with open(results_dir / "real_world_report.md", "w", encoding="utf-8") as f:
        f.write(f"""# 场景B报告\n- 北美市场p值：{na_test['p_value']:.4f}\n- 欧洲市场p值：{eu_test['p_value']:.4f}""")


if __name__ == "__main__":
    # 初始化结果目录
    results_dir = setup_results_dir()
    print(f"📂 结果目录：{results_dir}")
    
    # 运行两个场景
    scenario_A_synthetic(results_dir)
    scenario_B_real_world(results_dir)
    print("\n✅ 所有场景运行完成，报告已生成！")
