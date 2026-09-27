"""
论文图表生成脚本 - 推荐方案（6张图）
Fig 1: 模型架构图 (P0)
Fig 2: 校准价格路径 + 置信区间 (P0)
Fig 3: 消融实验图 (P0)
Fig 4: 蒙特卡洛不确定性带 (P1)
Fig 5: 回撤分析图 (P1)
Fig 6: 压力测试对比图 (P1)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import cm
import matplotlib.ticker as mticker

# 项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'paper', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(FIGURE_DIR, exist_ok=True)

# 全局样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_cea_data():
    """加载真实CEA数据"""
    cea_path = os.path.join(DATA_DIR, 'china_cea_daily.csv')
    df = pd.read_csv(cea_path)
    return df


def load_calibration_stats():
    """加载校准统计"""
    with open(os.path.join(DATA_DIR, 'calibration_statistics.json'), 'r') as f:
        return json.load(f)


# ============================================================
# Figure 1: 模型架构图
# ============================================================
def fig1_architecture():
    """四层模型架构示意图"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # 颜色方案
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    light_colors = ['#BBDEFB', '#C8E6C9', '#FFE0B2', '#E1BEE7']

    # Layer definitions
    layers = [
        {
            'y': 6.8, 'h': 1.3, 'color': colors[0], 'light': light_colors[0],
            'title': 'Layer 4: Analysis & Validation',
            'items': ['Risk Metrics\n(VaR, CVaR, MDD)', 'Ablation\nStudy', 'Stress\nTests', 'Monte Carlo\nRobustness']
        },
        {
            'y': 5.0, 'h': 1.3, 'color': colors[1], 'light': light_colors[1],
            'title': 'Layer 3: Behavioral Decision (Cognitive Biases)',
            'items': ['Loss Aversion\n$\\lambda=2.25$', 'Anchoring\n$\\alpha=0.10$', 'Herding\n$\\beta=0.80$', 'Overconfidence\n$\\gamma=0.20$']
        },
        {
            'y': 3.2, 'h': 1.3, 'color': colors[2], 'light': light_colors[2],
            'title': 'Layer 2: Market Mechanism (CDA + Walrasian)',
            'items': ['Limit Order\nBook', 'Price-Time\nPriority', 'Walrasian\nAdjustment', 'Continuous\nMatching']
        },
        {
            'y': 1.4, 'h': 1.3, 'color': colors[3], 'light': light_colors[3],
            'title': 'Layer 1: Heterogeneous Agents',
            'items': ['Emitters\n(20 firms)', 'Speculators\n(10 traders)', 'Developers\n(5 projects)', 'Regulator\n(1 authority)']
        },
    ]

    for layer in layers:
        # 主背景框
        rect = FancyBboxPatch((0.5, layer['y']), 9.0, layer['h'],
                               boxstyle="round,pad=0.05",
                               facecolor=layer['light'], edgecolor=layer['color'],
                               linewidth=2)
        ax.add_patch(rect)

        # 标题
        ax.text(5.0, layer['y'] + layer['h'] - 0.2, layer['title'],
                ha='center', va='top', fontsize=11, fontweight='bold',
                color=layer['color'])

        # 子模块
        n_items = len(layer['items'])
        item_width = 1.8
        total_width = n_items * item_width + (n_items - 1) * 0.2
        start_x = 5.0 - total_width / 2 + item_width / 2

        for i, item in enumerate(layer['items']):
            x = start_x + i * (item_width + 0.2)
            y = layer['y'] + 0.15
            # 子框
            sub_rect = FancyBboxPatch((x - item_width/2 + 0.05, y),
                                       item_width - 0.1, 0.65,
                                       boxstyle="round,pad=0.03",
                                       facecolor='white', edgecolor=layer['color'],
                                       linewidth=1, alpha=0.9)
            ax.add_patch(sub_rect)
            ax.text(x, y + 0.32, item, ha='center', va='center',
                    fontsize=7.5, color='#333333')

    # 层间箭头
    arrow_style = dict(arrowstyle='->', color='#666666', lw=1.5,
                       connectionstyle='arc3,rad=0')
    for y_from, y_to in [(2.7, 3.2), (4.5, 5.0), (6.3, 6.8)]:
        ax.annotate('', xy=(5.0, y_to), xytext=(5.0, y_from),
                    arrowprops=arrow_style)

    # 右侧标注：数据校准
    cal_box = FancyBboxPatch((7.5, 0.2), 2.2, 0.9,
                              boxstyle="round,pad=0.05",
                              facecolor='#FFF9C4', edgecolor='#F57F17',
                              linewidth=1.5)
    ax.add_patch(cal_box)
    ax.text(8.6, 0.85, 'Empirical\nCalibration', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#F57F17')
    ax.text(8.6, 0.35, 'China CEA Data', ha='center', va='center',
            fontsize=7.5, color='#666')

    # 校准到Layer 1的箭头
    ax.annotate('', xy=(7.5, 0.65), xytext=(6.5, 1.4),
                arrowprops=dict(arrowstyle='->', color='#F57F17', lw=1.2,
                                connectionstyle='arc3,rad=-0.2'))

    # 左侧标注：价格公式
    eq_box = FancyBboxPatch((0.1, 0.2), 3.0, 0.9,
                             boxstyle="round,pad=0.05",
                             facecolor='#E8F5E9', edgecolor='#2E7D32',
                             linewidth=1.5)
    ax.add_patch(eq_box)
    ax.text(1.6, 0.85, '$P_{t+1} = P_t(1 + \\kappa Z_t + \\epsilon_t)$',
            ha='center', va='center', fontsize=9, color='#2E7D32')
    ax.text(1.6, 0.35, 'Walrasian + CDA', ha='center', va='center',
            fontsize=7.5, color='#666')

    ax.set_title('Agent-Based Carbon Market Model Architecture',
                 fontsize=14, fontweight='bold', pad=10)

    fig.savefig(os.path.join(FIGURE_DIR, 'fig1_architecture.png'))
    plt.close(fig)
    print("  [OK] Fig 1: Architecture diagram")


# ============================================================
# Figure 2: 校准价格路径 + 置信区间
# ============================================================
def fig2_calibration_path():
    """仿真价格 vs 真实CEA + 置信区间带"""
    from carbon_market_abm.config import Config
    from carbon_market_abm.simulation import CarbonMarketSimulation

    # 加载真实CEA数据
    cea_df = load_cea_data()
    cea_prices = cea_df['close'].values if 'close' in cea_df.columns else cea_df.iloc[:, 1].values

    # 运行20次蒙特卡洛
    np.random.seed(42)
    n_mc = 20
    n_steps = 252
    all_prices = np.zeros((n_mc, n_steps))

    for i in range(n_mc):
        cfg = Config()
        cfg.simulation.random_seed = i * 100 + 1
        cfg.market.initial_price = 50.0
        sim = CarbonMarketSimulation(cfg)
        sim.run(num_steps=n_steps, progress_bar=False)
        prices = sim.price_history[:n_steps]
        if len(prices) < n_steps:
            prices = np.concatenate([prices, np.full(n_steps - len(prices), prices[-1])])
        all_prices[i] = prices[:n_steps]

    # 统计量
    mean_path = np.mean(all_prices, axis=0)
    p5 = np.percentile(all_prices, 5, axis=0)
    p25 = np.percentile(all_prices, 25, axis=0)
    p75 = np.percentile(all_prices, 75, axis=0)
    p95 = np.percentile(all_prices, 95, axis=0)

    # 真实CEA归一化到相同步数（线性插值）
    cea_resampled = np.interp(np.linspace(0, 1, n_steps),
                               np.linspace(0, 1, len(cea_prices)),
                               cea_prices)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    steps = np.arange(n_steps)

    # 置信区间
    ax.fill_between(steps, p5, p95, alpha=0.15, color='#2196F3', label='90% CI')
    ax.fill_between(steps, p25, p75, alpha=0.3, color='#2196F3', label='50% CI')
    ax.plot(steps, mean_path, color='#1565C0', linewidth=2, label='ABM Mean')
    ax.plot(steps, cea_resampled, color='#D32F2F', linewidth=1.5, linestyle='--',
            label='Real CEA', alpha=0.8)

    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Carbon Price (CNY/ton)')
    ax.set_title('Calibrated ABM Price Path vs. Real China CEA')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(30, 120)

    # 添加统计标注
    stats_text = (f"Mean Price: {np.mean(mean_path):.1f} CNY/ton\n"
                  f"Real CEA Mean: {np.mean(cea_resampled):.1f} CNY/ton\n"
                  f"Vol: {np.std(np.diff(mean_path)/mean_path[:-1])*np.sqrt(252)*100:.1f}%")
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    fig.savefig(os.path.join(FIGURE_DIR, 'fig2_calibration_path.png'))
    plt.close(fig)
    print("  [OK] Fig 2: Calibration price path + CI")


# ============================================================
# Figure 3: 消融实验图
# ============================================================
def fig3_ablation():
    """消融实验：各认知偏差移除后的影响"""
    # 数据来自 ablation_study.tex
    scenarios = ['Full\nbiases', 'No Loss\nAversion', 'No\nAnchoring',
                 'No\nHerding', 'No Over-\nconfidence', 'Fully\nRational']
    vol_values = [22.3, 23.4, 22.5, 20.7, 21.8, 21.9]
    dd_values = [-15.2, -16.3, -15.7, -15.2, -15.3, -16.5]
    trades = [3186, 3275, 3180, 3195, 3183, 3309]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    colors = ['#1976D2', '#E53935', '#43A047', '#FB8C00', '#8E24AA', '#546E7A']

    # (a) Volatility
    bars = axes[0].bar(scenarios, vol_values, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].axhline(y=20.99, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Real CEA vol (21.0%)')
    axes[0].set_ylabel('Annualized Volatility (%)')
    axes[0].set_title('(a) Volatility Impact')
    axes[0].legend(fontsize=7, loc='upper right')
    axes[0].set_ylim(18, 26)
    for bar, val in zip(bars, vol_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val}%', ha='center', va='bottom', fontsize=8)

    # (b) Max Drawdown
    bars = axes[1].bar(scenarios, [abs(v) for v in dd_values], color=colors, edgecolor='white', linewidth=0.5)
    axes[1].axhline(y=26.9, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Real CEA MDD (26.9%)')
    axes[1].set_ylabel('|Maximum Drawdown| (%)')
    axes[1].set_title('(b) Drawdown Impact')
    axes[1].legend(fontsize=7, loc='upper right')
    axes[1].set_ylim(10, 30)
    for bar, val in zip(bars, dd_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, abs(val) + 0.3,
                     f'{val}%', ha='center', va='bottom', fontsize=8)

    # (c) Trading Volume
    bars = axes[2].bar(scenarios, trades, color=colors, edgecolor='white', linewidth=0.5)
    axes[2].set_ylabel('Total Trades')
    axes[2].set_title('(c) Trading Activity')
    axes[2].set_ylim(2800, 3500)
    for bar, val in zip(bars, trades):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     f'{val}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Ablation Study: Individual Cognitive Bias Removal', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig3_ablation.png'))
    plt.close(fig)
    print("  [OK] Fig 3: Ablation study")


# ============================================================
# Figure 4: 蒙特卡洛不确定性带
# ============================================================
def fig4_monte_carlo():
    """20条MC路径 + 均值/置信带"""
    from carbon_market_abm.config import Config
    from carbon_market_abm.simulation import CarbonMarketSimulation

    n_mc = 20
    n_steps = 252
    all_prices = np.zeros((n_mc, n_steps))
    all_vols = []
    all_mdds = []

    for i in range(n_mc):
        cfg = Config()
        cfg.simulation.random_seed = i * 100 + 1
        cfg.market.initial_price = 50.0
        sim = CarbonMarketSimulation(cfg)
        sim.run(num_steps=n_steps, progress_bar=False)
        prices = sim.price_history[:n_steps]
        if len(prices) < n_steps:
            prices = np.concatenate([prices, np.full(n_steps - len(prices), prices[-1])])
        all_prices[i] = prices[:n_steps]

        # 计算统计量
        rets = np.diff(prices) / prices[:-1]
        all_vols.append(np.std(rets) * np.sqrt(252))
        cummax = np.maximum.accumulate(prices)
        dd = (prices - cummax) / cummax
        all_mdds.append(np.min(dd))

    mean_path = np.mean(all_prices, axis=0)
    p5 = np.percentile(all_prices, 5, axis=0)
    p95 = np.percentile(all_prices, 95, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})

    # (a) 路径图
    steps = np.arange(n_steps)
    cmap = cm.coolwarm(np.linspace(0.2, 0.8, n_mc))
    for i in range(n_mc):
        axes[0].plot(steps, all_prices[i], color=cmap[i], alpha=0.4, linewidth=0.8)
    axes[0].plot(steps, mean_path, color='#1565C0', linewidth=2.5, label='MC Mean')
    axes[0].fill_between(steps, p5, p95, alpha=0.15, color='#2196F3', label='90% CI')
    axes[0].set_xlabel('Trading Day')
    axes[0].set_ylabel('Carbon Price (CNY/ton)')
    axes[0].set_title('(a) Monte Carlo Price Paths (n=20)')
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(30, 120)

    # (b) 分布直方图
    axes[1].hist(all_vols, bins=8, color='#42A5F5', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=np.mean(all_vols), color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean={np.mean(all_vols)*100:.1f}%')
    axes[1].axvline(x=0.2099, color='green', linestyle='--', linewidth=1.5,
                    label='Real CEA=21.0%')
    axes[1].set_xlabel('Annualized Volatility')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('(b) Volatility Distribution')
    axes[1].legend(fontsize=8)

    fig.suptitle('Monte Carlo Robustness Validation', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig4_monte_carlo.png'))
    plt.close(fig)
    print("  [OK] Fig 4: Monte Carlo uncertainty band")


# ============================================================
# Figure 5: 回撤分析图
# ============================================================
def fig5_drawdown():
    """价格路径 + 回撤曲线"""
    from carbon_market_abm.config import Config
    from carbon_market_abm.simulation import CarbonMarketSimulation

    cfg = Config()
    cfg.simulation.random_seed = 42
    cfg.market.initial_price = 50.0
    sim = CarbonMarketSimulation(cfg)
    sim.run(num_steps=252, progress_bar=False)
    prices = np.array(sim.price_history[:252])

    # 计算回撤
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax * 100  # 百分比

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]},
                              sharex=True)

    steps = np.arange(len(prices))

    # (a) 价格路径
    axes[0].plot(steps, prices, color='#1565C0', linewidth=1.5, label='ABM Price')
    axes[0].plot(steps, cummax, color='#E53935', linewidth=1, linestyle='--', alpha=0.6,
                 label='Running Maximum')

    # 标注最大回撤区间
    dd_end_idx = np.argmin(drawdown)
    dd_start_idx = np.argmax(prices[:dd_end_idx+1])
    axes[0].axvspan(dd_start_idx, dd_end_idx, alpha=0.1, color='red')
    axes[0].annotate(f'MDD={drawdown[dd_end_idx]:.1f}%',
                     xy=(dd_end_idx, prices[dd_end_idx]),
                     xytext=(dd_end_idx + 20, prices[dd_end_idx] - 5),
                     fontsize=10, fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    axes[0].set_ylabel('Carbon Price (CNY/ton)')
    axes[0].set_title('(a) Price Path with Running Maximum')
    axes[0].legend(loc='upper left')

    # (b) 回撤曲线
    axes[1].fill_between(steps, drawdown, 0, color='#E53935', alpha=0.3)
    axes[1].plot(steps, drawdown, color='#E53935', linewidth=1)
    axes[1].axhline(y=-26.9, color='darkred', linestyle=':', linewidth=1.5,
                    label=f'Real CEA MDD = -26.9%')
    axes[1].axhline(y=drawdown[dd_end_idx], color='blue', linestyle='--', linewidth=1,
                    label=f'ABM MDD = {drawdown[dd_end_idx]:.1f}%')
    axes[1].set_xlabel('Trading Day')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_title('(b) Drawdown from Running Maximum')
    axes[1].legend(loc='lower left', fontsize=9)

    fig.suptitle('Drawdown Analysis: Endogenous Risk Floor', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig5_drawdown.png'))
    plt.close(fig)
    print("  [OK] Fig 5: Drawdown analysis")


# ============================================================
# Figure 6: 压力测试对比图
# ============================================================
def fig6_stress_test():
    """4种冲击场景的价格路径对比"""
    from carbon_market_abm.config import Config
    from carbon_market_abm.simulation import CarbonMarketSimulation

    scenarios = {
        'Baseline': {},
        'Demand +20%': {'demand_shock': 0.20},
        'Speculator -50%': {'speculator_shock': -0.50},
        'Allowance -10%': {'allowance_shock': -0.10},
    }

    colors = {'Baseline': '#1976D2', 'Demand +20%': '#E53935',
              'Speculator -50%': '#43A047', 'Allowance -10%': '#FB8C00'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (name, shocks) in enumerate(scenarios.items()):
        cfg = Config()
        cfg.simulation.random_seed = 42
        cfg.market.initial_price = 50.0

        # 应用冲击
        if 'demand_shock' in shocks:
            cfg.agent.num_emitters = int(20 * (1 + shocks['demand_shock']))
        if 'speculator_shock' in shocks:
            cfg.agent.num_speculators = max(1, int(10 * (1 + shocks['speculator_shock'])))
        if 'allowance_shock' in shocks:
            cfg.market.total_allowance = 10000 * (1 + shocks['allowance_shock'])

        sim = CarbonMarketSimulation(cfg)
        sim.run(num_steps=252, progress_bar=False)
        prices = np.array(sim.price_history[:252])

        steps = np.arange(len(prices))
        axes[idx].plot(steps, prices, color=colors[name], linewidth=1.5, label=name)

        # 添加均值线
        axes[idx].axhline(y=np.mean(prices), color=colors[name], linestyle='--',
                          alpha=0.5, linewidth=1)

        rets = np.diff(prices) / prices[:-1]
        vol = np.std(rets) * np.sqrt(252) * 100
        cummax = np.maximum.accumulate(prices)
        mdd = np.min((prices - cummax) / cummax) * 100

        stats_text = f"Mean: {np.mean(prices):.1f}\nVol: {vol:.1f}%\nMDD: {mdd:.1f}%"
        axes[idx].text(0.98, 0.97, stats_text, transform=axes[idx].transAxes,
                       ha='right', va='top', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        axes[idx].set_xlabel('Trading Day')
        axes[idx].set_ylabel('Price (CNY/ton)')
        axes[idx].set_title(f'({chr(97+idx)}) {name}', fontweight='bold')
        axes[idx].set_ylim(30, 120)

    fig.suptitle('Stress Test: Price Dynamics Under Shock Scenarios',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig6_stress_test.png'))
    plt.close(fig)
    print("  [OK] Fig 6: Stress test comparison")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating 6 paper figures (recommended plan)")
    print("=" * 60)

    print("\n[1/6] Model architecture diagram...")
    fig1_architecture()

    print("\n[2/6] Calibration price path + CI...")
    fig2_calibration_path()

    print("\n[3/6] Ablation study...")
    fig3_ablation()

    print("\n[4/6] Monte Carlo uncertainty band...")
    fig4_monte_carlo()

    print("\n[5/6] Drawdown analysis...")
    fig5_drawdown()

    print("\n[6/6] Stress test comparison...")
    fig6_stress_test()

    print("\n" + "=" * 60)
    print("All 6 figures generated successfully!")
    print(f"Output directory: {FIGURE_DIR}")
    print("=" * 60)
