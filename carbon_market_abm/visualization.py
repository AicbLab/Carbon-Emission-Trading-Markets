"""
可视化模块
绘制仿真结果图表
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """可视化器"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_price_path(
        self,
        price_history: List[float],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """绘制价格路径图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(price_history, linewidth=1.5, color='#2E86AB')
        ax.fill_between(range(len(price_history)), price_history, alpha=0.3, color='#2E86AB')
        
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('碳价 (元/吨)', fontsize=12)
        ax.set_title('碳市场价格路径', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        if price_history:
            mean_price = np.mean(price_history)
            ax.axhline(y=mean_price, color='r', linestyle='--', alpha=0.7, label=f'均价: {mean_price:.2f}')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"价格路径图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_volume_analysis(
        self,
        trades: List[Dict],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """绘制成交量分析图"""
        if not trades:
            print("无交易数据")
            return
        
        # 按时间步聚合成交量
        volume_by_step = {}
        for trade in trades:
            step = trade.get("timestamp", 0)
            quantity = trade.get("quantity", 0)
            volume_by_step[step] = volume_by_step.get(step, 0) + quantity
        
        steps = sorted(volume_by_step.keys())
        volumes = [volume_by_step[s] for s in steps]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 成交量时间序列
        ax1.bar(steps, volumes, width=1, color='#A23B72', alpha=0.7)
        ax1.set_xlabel('时间步', fontsize=12)
        ax1.set_ylabel('成交量 (吨)', fontsize=12)
        ax1.set_title('成交量时间序列', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 成交量分布
        ax2.hist(volumes, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('成交量 (吨)', fontsize=12)
        ax2.set_ylabel('频次', fontsize=12)
        ax2.set_title('成交量分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"成交量分析图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_agent_wealth_distribution(
        self,
        agent_states: List[Dict],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """绘制智能体财富分布图"""
        # 按类型分组
        wealth_by_type = {}
        for state in agent_states:
            agent_type = state.get("agent_type", "Unknown")
            cash = state.get("cash", 0)
            
            if agent_type not in wealth_by_type:
                wealth_by_type[agent_type] = []
            wealth_by_type[agent_type].append(cash)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 箱线图
        data = [wealth_by_type.get(t, [0]) for t in ['控排企业', '投机机构', '项目开发商']]
        labels = ['控排企业', '投机机构', '项目开发商']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('现金持有量 (元)', fontsize=12)
        ax.set_title('智能体财富分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"财富分布图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_compliance_status(
        self,
        emitters: List,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """绘制合规状态图"""
        # 统计合规状态
        status_count = {"compliant": 0, "non_compliant": 0, "pending": 0}
        deficits = []
        
        for emitter in emitters:
            status = emitter.compliance_status
            status_count[status] = status_count.get(status, 0) + 1
            
            if hasattr(emitter, 'get_allowance_deficit'):
                deficit = emitter.get_allowance_deficit()
                deficits.append(deficit)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 合规状态饼图
        labels = ['已合规', '未合规', '待履约']
        sizes = [status_count.get("compliant", 0), 
                 status_count.get("non_compliant", 0),
                 status_count.get("pending", 0)]
        colors = ['#2E86AB', '#C73E1D', '#F18F01']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('控排企业合规状态分布', fontsize=14, fontweight='bold')
        
        # 配额缺口分布
        if deficits:
            ax2.hist(deficits, bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('配额缺口 (吨)', fontsize=12)
            ax2.set_ylabel('企业数量', fontsize=12)
            ax2.set_title('配额缺口分布', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"合规状态图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_risk_metrics(
        self,
        prices: List[float],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """绘制风险指标图"""
        from risk_metrics import RiskMetrics
        
        returns = RiskMetrics.calculate_returns(prices)
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # 价格与回撤
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(prices, linewidth=1.5, color='#2E86AB', label='价格')
        ax1.set_ylabel('价格', fontsize=11)
        ax1.set_title('价格走势与风险指标', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 计算回撤
        prices_arr = np.array(prices)
        cumulative_max = np.maximum.accumulate(prices_arr)
        drawdowns = (prices_arr - cumulative_max) / cumulative_max
        
        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3, color='#C73E1D', label='回撤')
        ax1_twin.set_ylabel('回撤', fontsize=11, color='#C73E1D')
        ax1_twin.legend(loc='upper right')
        
        # 收益率分布
        ax2 = fig.add_subplot(gs[1, 0])
        if len(returns) > 0:
            ax2.hist(returns, bins=50, color='#A23B72', alpha=0.7, edgecolor='black', density=True)
            
            # 拟合正态分布
            mu, std = np.mean(returns), np.std(returns)
            x = np.linspace(min(returns), max(returns), 100)
            ax2.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='正态分布拟合')
            ax2.legend()
        
        ax2.set_xlabel('收益率', fontsize=11)
        ax2.set_ylabel('密度', fontsize=11)
        ax2.set_title('收益率分布', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 波动率
        ax3 = fig.add_subplot(gs[1, 1])
        if len(prices) >= 20:
            rolling_vol = []
            for i in range(20, len(prices)):
                vol = np.std(RiskMetrics.calculate_returns(prices[i-20:i])) * np.sqrt(252)
                rolling_vol.append(vol)
            ax3.plot(range(20, len(prices)), rolling_vol, linewidth=1.5, color='#F18F01')
            ax3.set_xlabel('时间步', fontsize=11)
            ax3.set_ylabel('年化波动率', fontsize=11)
            ax3.set_title('滚动波动率 (20步窗口)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # VaR CVaR
        ax4 = fig.add_subplot(gs[2, :])
        if len(returns) >= 100:
            # 滚动VaR
            var_95_history = []
            var_99_history = []
            for i in range(100, len(returns)):
                window_returns = returns[i-100:i]
                var_95 = RiskMetrics.var_historical(window_returns, 0.95)
                var_99 = RiskMetrics.var_historical(window_returns, 0.99)
                var_95_history.append(var_95)
                var_99_history.append(var_99)
            
            ax4.plot(range(100, len(returns)), var_95_history, linewidth=1.5, color='#2E86AB', label='VaR 95%')
            ax4.plot(range(100, len(returns)), var_99_history, linewidth=1.5, color='#C73E1D', label='VaR 99%')
            ax4.set_xlabel('时间步', fontsize=11)
            ax4.set_ylabel('VaR', fontsize=11)
            ax4.set_title('滚动风险价值 (VaR)', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"风险指标图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_monte_carlo_results(
        self,
        mc_results: Dict,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """绘制蒙特卡洛仿真结果"""
        if not mc_results or "final_price" not in mc_results:
            print("无蒙特卡洛结果")
            return
        
        raw_results = mc_results.get("raw_results", [])
        if not raw_results:
            return
        
        final_prices = [r["final_price"] for r in raw_results]
        price_changes = [r["price_change"] for r in raw_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 最终价格分布
        ax1.hist(final_prices, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(mc_results["final_price"]["mean"], color='r', linestyle='--', linewidth=2, label='均值')
        ax1.axvline(mc_results["final_price"][" VaR_95"], color='orange', linestyle='--', linewidth=2, label='VaR 95%')
        ax1.set_xlabel('最终价格', fontsize=11)
        ax1.set_ylabel('频次', fontsize=11)
        ax1.set_title('最终价格分布', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 价格变化分布
        ax2.hist(price_changes, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('价格变化 (%)', fontsize=11)
        ax2.set_ylabel('频次', fontsize=11)
        ax2.set_title('价格变化分布', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 价格路径（抽样）
        ax3.set_title('价格路径样本', fontsize=12, fontweight='bold')
        ax3.set_xlabel('时间步', fontsize=11)
        ax3.set_ylabel('价格', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 散点图：波动率 vs 最终价格
        volatilities = [r.get("price_volatility", 0) for r in raw_results]
        ax4.scatter(volatilities, final_prices, alpha=0.5, color='#F18F01')
        ax4.set_xlabel('价格波动率', fontsize=11)
        ax4.set_ylabel('最终价格', fontsize=11)
        ax4.set_title('波动率与最终价格关系', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"蒙特卡洛结果图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_dashboard(
        self,
        simulation,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """创建综合仪表板"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # 价格路径
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(simulation.price_history, linewidth=1.5, color='#2E86AB')
        ax1.set_title('碳价路径', fontsize=12, fontweight='bold')
        ax1.set_ylabel('价格 (元/吨)')
        ax1.grid(True, alpha=0.3)
        
        # 统计信息
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        if simulation.price_history:
            stats_text = f"""
            仿真统计:
            
            初始价格: {simulation.price_history[0]:.2f}
            最终价格: {simulation.price_history[-1]:.2f}
            最高价格: {max(simulation.price_history):.2f}
            最低价格: {min(simulation.price_history):.2f}
            平均价格: {np.mean(simulation.price_history):.2f}
            价格变化: {((simulation.price_history[-1] - simulation.price_history[0]) / simulation.price_history[0] * 100):.2f}%
            
            总交易数: {len(simulation.market.order_book.trades)}
            智能体数量: {len(simulation.agents)}
            """
            ax2.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 各类主体持仓
        ax3 = fig.add_subplot(gs[1, 0])
        emitter_positions = [e.state.allowance for e in simulation.emitters]
        speculator_positions = [s.state.allowance for s in simulation.speculators]
        
        ax3.bar(['控排企业', '投机机构'], 
                [np.mean(emitter_positions) if emitter_positions else 0,
                 np.mean(speculator_positions) if speculator_positions else 0],
                color=['#2E86AB', '#A23B72'], alpha=0.7)
        ax3.set_title('平均持仓量', fontsize=12, fontweight='bold')
        ax3.set_ylabel('持仓 (吨)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 成交量
        ax4 = fig.add_subplot(gs[1, 1])
        if simulation.market.order_book.trades:
            volume_by_step = {}
            for trade in simulation.market.order_book.trades:
                step = trade.timestamp
                volume_by_step[step] = volume_by_step.get(step, 0) + trade.quantity
            
            steps = sorted(volume_by_step.keys())
            volumes = [volume_by_step[s] for s in steps]
            ax4.bar(steps, volumes, width=1, color='#F18F01', alpha=0.7)
        ax4.set_title('成交量', fontsize=12, fontweight='bold')
        ax4.set_ylabel('成交量 (吨)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 合规状态
        ax5 = fig.add_subplot(gs[1, 2])
        compliant = sum(1 for e in simulation.emitters if e.compliance_status == "compliant")
        non_compliant = sum(1 for e in simulation.emitters if e.compliance_status == "non_compliant")
        pending = len(simulation.emitters) - compliant - non_compliant
        
        ax5.pie([compliant, non_compliant, pending], 
                labels=['已合规', '未合规', '待履约'],
                colors=['#2E86AB', '#C73E1D', '#F18F01'],
                autopct='%1.1f%%')
        ax5.set_title('合规状态', fontsize=12, fontweight='bold')
        
        # 智能体财富分布
        ax6 = fig.add_subplot(gs[2, 0])
        wealth_data = []
        labels = []
        for agent_type, agents in [
            ('控排企业', simulation.emitters),
            ('投机机构', simulation.speculators),
            ('项目开发商', simulation.developers)
        ]:
            if agents:
                wealth_data.append([a.state.cash for a in agents])
                labels.append(agent_type)
        
        if wealth_data:
            bp = ax6.boxplot(wealth_data, labels=labels, patch_artist=True)
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax6.set_title('财富分布', fontsize=12, fontweight='bold')
        ax6.set_ylabel('现金 (元)')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 风险指标
        ax7 = fig.add_subplot(gs[2, 1:])
        from risk_metrics import RiskMetrics
        returns = RiskMetrics.calculate_returns(simulation.price_history)
        if len(returns) >= 20:
            rolling_vol = []
            for i in range(20, len(simulation.price_history)):
                vol = np.std(RiskMetrics.calculate_returns(simulation.price_history[i-20:i])) * np.sqrt(252)
                rolling_vol.append(vol)
            ax7.plot(range(20, len(simulation.price_history)), rolling_vol, 
                    linewidth=1.5, color='#C73E1D', label='波动率')
            ax7.set_title('风险指标', fontsize=12, fontweight='bold')
            ax7.set_xlabel('时间步')
            ax7.set_ylabel('年化波动率')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        plt.suptitle('碳金融市场ABM仿真仪表板', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"仪表板已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


# 导入scipy.stats用于可视化模块
from scipy import stats
