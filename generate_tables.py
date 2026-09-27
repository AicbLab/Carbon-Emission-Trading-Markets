"""快速生成LaTeX表格 - 使用已保存的实验结果"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "calibration")
TABLE_DIR = os.path.join(os.path.dirname(__file__), "paper", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)

TARGET = {"mean_price": 70.47, "annual_vol": 0.21, "max_drawdown": -0.269, "var_95": -0.0214}

# 从实验结果读取
best_params = {"lam": 1.5, "alpha": 0.1, "beta": 0.8, "gamma": 0.3}
best_metrics = {"mean_price": 90.95, "annual_vol": 0.2162, "max_drawdown": -0.0603, "var_95": -0.0221}

# 蒙特卡洛结果
mc_results = {
    "means": [96.9], "vols": [0.221], "mdds": [-0.056], "vars": [-0.0214], "trades": [251],
    "means_std": 3.5, "vols_std": 0.011, "mdds_std": 0.009, "vars_std": 0.0014, "trades_std": 64
}

# 消融实验
ablation = [
    {"scenario": "Full biases", "mean_price": 95.9, "annual_vol": 0.219, "max_drawdown": -0.058, "total_trades": 239},
    {"scenario": "No loss aversion ($\\lambda$=1)", "mean_price": 95.9, "annual_vol": 0.219, "max_drawdown": -0.058, "total_trades": 239},
    {"scenario": "No anchoring ($\\alpha$=0)", "mean_price": 95.9, "annual_vol": 0.224, "max_drawdown": -0.059, "total_trades": 239},
    {"scenario": "No herding ($\\beta$=0)", "mean_price": 95.8, "annual_vol": 0.166, "max_drawdown": -0.048, "total_trades": 244},
    {"scenario": "No overconfidence ($\\gamma$=0)", "mean_price": 95.9, "annual_vol": 0.212, "max_drawdown": -0.056, "total_trades": 240},
    {"scenario": "Fully rational", "mean_price": 95.3, "annual_vol": 0.155, "max_drawdown": -0.045, "total_trades": 258},
]

# 压力测试
stress = [
    {"scenario": "Baseline", "mean_price": 91.0, "final_price": 90.0, "annual_vol": 0.216, "trades": 144},
    {"scenario": "Speculator -50\\%", "mean_price": 91.0, "final_price": 89.0, "annual_vol": 0.231, "trades": 139},
    {"scenario": "Penalty 5x", "mean_price": 91.0, "final_price": 90.0, "annual_vol": 0.216, "trades": 144},
    {"scenario": "Demand +20\\%", "mean_price": 96.5, "final_price": 95.0, "annual_vol": 0.213, "trades": 228},
]

# Table 1: 校准对比
with open(os.path.join(TABLE_DIR, "calibration_comparison.tex"), 'w') as f:
    f.write(r"\begin{table}[H]" + "\n\\centering\n")
    f.write(r"\caption{Model calibration: Calibrated ABM vs real China CEA market data}" + "\n")
    f.write(r"\label{tab:calibration}" + "\n")
    f.write(r"\begin{tabular}{lccc}" + "\n\\toprule\n")
    f.write(r"\textbf{Metric} & \textbf{Real CEA} & \textbf{Calibrated ABM} & \textbf{Error (\%)} \\" + "\n\\midrule\n")
    rows = [
        ("Mean price (CNY/ton)", TARGET["mean_price"], best_metrics["mean_price"]),
        ("Annualized volatility", TARGET["annual_vol"], best_metrics["annual_vol"]),
        ("Maximum drawdown", TARGET["max_drawdown"], best_metrics["max_drawdown"]),
        ("VaR (95\\%)", TARGET["var_95"], best_metrics["var_95"]),
    ]
    for name, target, actual in rows:
        err = abs(actual - target) / abs(target) * 100
        f.write(f"{name} & {target:.4f} & {actual:.4f} & {err:.1f}\\% \\\\\n")
    f.write(r"\bottomrule" + "\n\\end{tabular}\n\\end{table}\n")

# Table 2: 最优参数
with open(os.path.join(TABLE_DIR, "optimal_params.tex"), 'w') as f:
    f.write(r"\begin{table}[H]" + "\n\\centering\n")
    f.write(r"\caption{Calibrated cognitive bias parameters}" + "\n")
    f.write(r"\label{tab:cognitive_params}" + "\n")
    f.write(r"\begin{tabular}{llcc}" + "\n\\toprule\n")
    f.write(r"\textbf{Parameter} & \textbf{Symbol} & \textbf{Calibrated Value} & \textbf{Literature Range} \\" + "\n\\midrule\n")
    f.write(f"Loss aversion & $\\lambda$ & {best_params['lam']:.2f} & [1.5, 3.0] \\\\\n")
    f.write(f"Anchoring bias & $\\alpha$ & {best_params['alpha']:.2f} & [0.1, 0.5] \\\\\n")
    f.write(f"Herding tendency & $\\beta$ & {best_params['beta']:.2f} & [0.2, 0.8] \\\\\n")
    f.write(f"Overconfidence & $\\gamma$ & {best_params['gamma']:.2f} & [0.1, 0.3] \\\\\n")
    f.write(r"\bottomrule" + "\n\\end{tabular}\n\\end{table}\n")

# Table 3: 消融实验
with open(os.path.join(TABLE_DIR, "ablation_study.tex"), 'w') as f:
    f.write(r"\begin{table}[H]" + "\n\\centering\n")
    f.write(r"\caption{Ablation study: Impact of individual cognitive biases}" + "\n")
    f.write(r"\label{tab:ablation}" + "\n")
    f.write(r"\begin{tabular}{lcccc}" + "\n\\toprule\n")
    f.write(r"\textbf{Scenario} & \textbf{Mean Price} & \textbf{Ann. Vol} & \textbf{Max DD} & \textbf{Trades} \\" + "\n\\midrule\n")
    for r in ablation:
        f.write(f"{r['scenario']} & {r['mean_price']:.1f} & {r['annual_vol']*100:.1f}\\% & {r['max_drawdown']*100:.1f}\\% & {r['total_trades']:.0f} \\\\\n")
    f.write(r"\bottomrule" + "\n\\end{tabular}\n\\end{table}\n")

# Table 4: 压力测试
with open(os.path.join(TABLE_DIR, "stress_test.tex"), 'w') as f:
    f.write(r"\begin{table}[H]" + "\n\\centering\n")
    f.write(r"\caption{Stress test results}" + "\n")
    f.write(r"\label{tab:stress}" + "\n")
    f.write(r"\begin{tabular}{lcccc}" + "\n\\toprule\n")
    f.write(r"\textbf{Scenario} & \textbf{Mean Price} & \textbf{Final Price} & \textbf{Ann. Vol} & \textbf{Trades} \\" + "\n\\midrule\n")
    for r in stress:
        f.write(f"{r['scenario']} & {r['mean_price']:.1f} & {r['final_price']:.1f} & {r['annual_vol']*100:.1f}\\% & {r['trades']} \\\\\n")
    f.write(r"\bottomrule" + "\n\\end{tabular}\n\\end{table}\n")

print("LaTeX表格已生成到 paper/tables/")
print("  - calibration_comparison.tex")
print("  - optimal_params.tex")
print("  - ablation_study.tex")
print("  - stress_test.tex")
