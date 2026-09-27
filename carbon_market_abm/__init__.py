"""
碳金融市场ABM多智能体仿真系统

基于前景理论的认知偏差建模 + 连续双边拍卖(CDA)市场撮合。
纯数学建模版本，无LLM依赖。

四类主体：控排企业、投机机构、项目开发商、监管机构。

主要功能：
- 连续双边拍卖(CDA)市场撮合
- 前景理论认知偏差建模（损失厌恶/锚定/羊群/过度自信）
- 蒙特卡洛仿真
- 风险指标计算(VaR/CVaR等)
- 压力测试
- 真实数据校准

使用方法:
    from carbon_market_abm import CarbonMarketSimulation, Config
    
    config = Config()
    sim = CarbonMarketSimulation(config)
    sim.run()
"""

__version__ = "2.0.0"
__author__ = "Carbon Market ABM Research Team"

from .config import Config, config
from .simulation import CarbonMarketSimulation, MonteCarloSimulation
from .market import CarbonMarket, OrderBook
from .agents import (
    BaseAgent,
    EmitterAgent,
    SpeculatorAgent,
    DeveloperAgent,
    RegulatorAgent,
)
from .risk_metrics import RiskMetrics, StressTest
from .visualization import Visualizer

__all__ = [
    'Config', 'config',
    'CarbonMarketSimulation', 'MonteCarloSimulation',
    'CarbonMarket', 'OrderBook',
    'BaseAgent', 'EmitterAgent', 'SpeculatorAgent', 'DeveloperAgent', 'RegulatorAgent',
    'RiskMetrics', 'StressTest',
    'Visualizer',
]
