"""
碳金融市场ABM多智能体仿真 - 主程序入口
支持交互式配置LLM API Key
"""

import os
import sys
import argparse
import json
from typing import Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, Config
from llm_client import setup_llm_interactive, get_llm_client
from simulation import CarbonMarketSimulation, MonteCarloSimulation
from visualization import Visualizer
from risk_metrics import RiskMetrics, StressTest


def print_banner():
    """打印程序横幅"""
    print("=" * 70)
    print("  碳金融市场ABM多智能体仿真系统")
    print("  LLM驱动多智能体仿真研究")
    print("=" * 70)
    print()


def interactive_setup():
    """交互式配置"""
    print_banner()
    
    print("欢迎使用碳金融市场ABM仿真系统!")
    print()
    
    # 1. 配置LLM API
    print("-" * 70)
    print("第一步: 配置大语言模型API")
    print("-" * 70)
    llm_configured = setup_llm_interactive()
    
    if not llm_configured:
        print()
        print("⚠ 警告: LLM未配置，将使用规则决策模式")
        print("  如需启用LLM决策，请在环境变量中设置 DASHSCOPE_API_KEY")
        print("  或在运行时通过交互界面输入API Key")
        config.simulation.use_llm = False
    else:
        config.simulation.use_llm = True
    
    print()
    
    # 2. 配置仿真参数
    print("-" * 70)
    print("第二步: 配置仿真参数")
    print("-" * 70)
    
    print()
    print("请选择仿真模式:")
    print("  1. 单次仿真 (快速查看结果)")
    print("  2. 蒙特卡洛仿真 (统计分析)")
    print("  3. 压力测试 (政策冲击情景)")
    
    while True:
        mode = input("\n请选择 (1/2/3): ").strip()
        if mode in ['1', '2', '3']:
            break
        print("无效选择，请重新输入")
    
    print()
    
    # 配置步数
    default_steps = config.simulation.max_steps
    steps_input = input(f"仿真步数 (默认 {default_steps}, 直接回车使用默认值): ").strip()
    if steps_input:
        try:
            config.simulation.max_steps = int(steps_input)
        except ValueError:
            print(f"无效输入，使用默认值 {default_steps}")
    
    print()
    
    # 配置智能体数量
    print("智能体数量配置:")
    
    num_emitters = input(f"  控排企业数量 (默认 {config.agent.num_emitters}): ").strip()
    if num_emitters:
        config.agent.num_emitters = int(num_emitters)
    
    num_speculators = input(f"  投机机构数量 (默认 {config.agent.num_speculators}): ").strip()
    if num_speculators:
        config.agent.num_speculators = int(num_speculators)
    
    num_developers = input(f"  项目开发商数量 (默认 {config.agent.num_developers}): ").strip()
    if num_developers:
        config.agent.num_developers = int(num_developers)
    
    print()
    
    # 配置市场参数
    print("市场参数配置:")
    
    initial_price = input(f"  初始碳价 (默认 {config.market.initial_price} 元/吨): ").strip()
    if initial_price:
        config.market.initial_price = float(initial_price)
    
    print()
    
    return mode


def run_single_simulation():
    """运行单次仿真"""
    print()
    print("=" * 70)
    print("开始单次仿真")
    print("=" * 70)
    print()
    
    # 创建仿真
    llm_client = get_llm_client()
    sim = CarbonMarketSimulation(config, llm_client)
    
    # 运行仿真
    results = sim.run(progress_bar=True)
    
    print()
    print("仿真完成!")
    print()
    
    # 显示结果摘要
    print("结果摘要:")
    print(f"  总步数: {sim.current_step}")
    print(f"  初始价格: {sim.price_history[0]:.2f} 元/吨")
    print(f"  最终价格: {sim.price_history[-1]:.2f} 元/吨")
    print(f"  价格变化: {((sim.price_history[-1] - sim.price_history[0]) / sim.price_history[0] * 100):.2f}%")
    print(f"  总交易数: {len(sim.market.order_book.trades)}")
    
    # 计算风险指标
    print()
    print("风险指标:")
    metrics = RiskMetrics.calculate_all_metrics(
        sim.price_history,
        sim.market.order_book.trades,
        [agent.get_state_dict() for agent in sim.agents]
    )
    print(f"  日均波动率: {metrics['volatility']['daily']:.4f}")
    print(f"  年化波动率: {metrics['volatility']['annualized']:.4f}")
    print(f"  VaR 95%: {metrics['var']['var_95']:.4f}")
    print(f"  CVaR 95%: {metrics['var']['cvar_95']:.4f}")
    print(f"  最大回撤: {metrics['drawdown']['max_drawdown']:.2%}")
    
    # 保存结果
    print()
    output_dir = sim.save_results()
    
    # 生成可视化
    print()
    print("生成可视化图表...")
    viz = Visualizer(output_dir)
    
    viz.plot_price_path(sim.price_history, save_path=os.path.join(output_dir, "price_path.png"))
    viz.plot_volume_analysis(sim.market.order_book.trades, save_path=os.path.join(output_dir, "volume_analysis.png"))
    viz.plot_agent_wealth_distribution([agent.get_state_dict() for agent in sim.agents], 
                                        save_path=os.path.join(output_dir, "wealth_distribution.png"))
    viz.plot_compliance_status(sim.emitters, save_path=os.path.join(output_dir, "compliance_status.png"))
    viz.plot_risk_metrics(sim.price_history, save_path=os.path.join(output_dir, "risk_metrics.png"))
    viz.create_dashboard(sim, save_path=os.path.join(output_dir, "dashboard.png"))
    
    print()
    print(f"所有结果已保存到: {output_dir}")
    
    return sim


def run_monte_carlo_simulation():
    """运行蒙特卡洛仿真"""
    print()
    print("=" * 70)
    print("开始蒙特卡洛仿真")
    print("=" * 70)
    print()
    
    # 配置仿真次数
    default_sims = config.simulation.num_simulations
    num_sims = input(f"仿真次数 (默认 {default_sims}): ").strip()
    if num_sims:
        config.simulation.num_simulations = int(num_sims)
    
    print()
    
    # 运行蒙特卡洛仿真
    mc = MonteCarloSimulation(config)
    results = mc.run(progress_bar=True)
    
    print()
    print("蒙特卡洛仿真完成!")
    print()
    
    # 显示统计结果
    print("统计结果:")
    print(f"  仿真次数: {results['num_simulations']}")
    print()
    print("  最终价格统计:")
    print(f"    均值: {results['final_price']['mean']:.2f}")
    print(f"    标准差: {results['final_price']['std']:.2f}")
    print(f"    最小值: {results['final_price']['min']:.2f}")
    print(f"    最大值: {results['final_price']['max']:.2f}")
    print(f"    中位数: {results['final_price']['median']:.2f}")
    print(f"    VaR 95%: {results['final_price'][' VaR_95']:.2f}")
    print(f"    VaR 99%: {results['final_price'][' VaR_99']:.2f}")
    print()
    print("  价格变化统计:")
    print(f"    均值: {results['price_change']['mean']:.2f}%")
    print(f"    标准差: {results['price_change']['std']:.2f}%")
    
    # 保存结果
    output_dir = config.simulation.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"monte_carlo_results_{timestamp}.json")
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_path}")
    
    # 生成可视化
    print()
    print("生成可视化图表...")
    viz = Visualizer(output_dir)
    viz.plot_monte_carlo_results(results, save_path=os.path.join(output_dir, f"monte_carlo_{timestamp}.png"))
    
    return results


def run_stress_test():
    """运行压力测试"""
    print()
    print("=" * 70)
    print("开始压力测试")
    print("=" * 70)
    print()
    
    # 首先运行基准仿真
    print("运行基准仿真...")
    llm_client = get_llm_client()
    base_sim = CarbonMarketSimulation(config, llm_client)
    base_sim.run(progress_bar=True)
    
    print()
    print("基准仿真完成")
    print(f"  基准最终价格: {base_sim.price_history[-1]:.2f} 元/吨")
    print()
    
    # 选择压力测试情景
    print("选择压力测试情景:")
    print("  1. 配额紧缩情景 (配额缩减率增加)")
    print("  2. 罚款增加情景 (罚款倍数提高)")
    print("  3. 价格下限情景 (设置价格下限)")
    
    while True:
        scenario = input("\n请选择 (1/2/3): ").strip()
        if scenario in ['1', '2', '3']:
            break
        print("无效选择，请重新输入")
    
    print()
    
    # 配置冲击参数
    if scenario == '1':
        shock_type = "allowance_reduction"
        default_magnitude = 0.05
        magnitude = input(f"配额年缩减率 (默认 {default_magnitude:.1%}): ").strip()
        shock_magnitude = float(magnitude) if magnitude else default_magnitude
    elif scenario == '2':
        shock_type = "penalty_increase"
        default_magnitude = 3.0
        magnitude = input(f"罚款倍数 (默认 {default_magnitude:.1f}x): ").strip()
        shock_magnitude = float(magnitude) if magnitude else default_magnitude
    else:
        shock_type = "price_floor"
        default_magnitude = 40.0
        magnitude = input(f"价格下限 (默认 {default_magnitude:.0f} 元/吨): ").strip()
        shock_magnitude = float(magnitude) if magnitude else default_magnitude
    
    print()
    print(f"运行压力测试: {shock_type}, 冲击幅度: {shock_magnitude}")
    print()
    
    # 运行压力测试
    stress_result = StressTest.run_policy_shock_scenario(
        base_sim,
        shock_type,
        shock_magnitude
    )
    
    print()
    print("压力测试完成!")
    print()
    print("结果对比:")
    print(f"  基准最终价格: {stress_result['base_final_price']:.2f} 元/吨")
    print(f"  冲击后价格: {stress_result['shocked_final_price']:.2f} 元/吨")
    print(f"  价格变化: {stress_result['price_change_pct']:.2f}%")
    
    # 保存结果
    output_dir = config.simulation.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"stress_test_{shock_type}_{timestamp}.json")
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(stress_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_path}")
    
    return stress_result


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='碳金融市场ABM多智能体仿真')
    parser.add_argument('--mode', type=str, choices=['single', 'mc', 'stress', 'interactive'],
                       default='interactive', help='运行模式')
    parser.add_argument('--api-key', type=str, default='', help='LLM API Key')
    parser.add_argument('--base-url', type=str, default='', help='LLM API Base URL')
    parser.add_argument('--model', type=str, default='', help='LLM模型名称')
    parser.add_argument('--steps', type=int, default=0, help='仿真步数')
    parser.add_argument('--output', type=str, default='', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置LLM配置
    if args.api_key:
        config.setup_llm(api_key=args.api_key, base_url=args.base_url, model=args.model)
    
    if args.steps > 0:
        config.simulation.max_steps = args.steps
    
    if args.output:
        config.simulation.output_dir = args.output
    
    # 运行模式
    if args.mode == 'interactive':
        mode = interactive_setup()
    else:
        mode = {'single': '1', 'mc': '2', 'stress': '3'}[args.mode]
    
    # 执行仿真
    if mode == '1':
        run_single_simulation()
    elif mode == '2':
        run_monte_carlo_simulation()
    elif mode == '3':
        run_stress_test()
    
    print()
    print("=" * 70)
    print("仿真程序结束")
    print("=" * 70)


if __name__ == "__main__":
    try:
        import pandas as pd
        from tqdm import tqdm
    except ImportError:
        print("正在安装必要的依赖...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "tqdm", "matplotlib", "seaborn", "scipy", "numpy"])
        print("依赖安装完成，请重新运行程序")
        sys.exit(0)
    
    main()
