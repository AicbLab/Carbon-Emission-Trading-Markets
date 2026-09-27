"""
碳金融市场ABM仿真配置文件
纯数学建模版本 - 无LLM依赖
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class MarketConfig:
    """市场配置"""
    initial_price: float = 50.0
    price_floor: float = 0.0
    price_ceiling: float = 200.0
    tick_size: float = 0.1
    total_allowance: float = 10000.0
    compliance_period: int = 252
    transaction_fee_rate: float = 0.001


@dataclass
class AgentConfig:
    """智能体配置"""
    num_emitters: int = 20
    num_speculators: int = 10
    num_developers: int = 5
    num_regulators: int = 1
    
    emitter_initial_capital: float = 1000000.0
    speculator_initial_capital: float = 5000000.0
    developer_initial_capital: float = 2000000.0
    
    # 认知偏差参数 θ（数学建模参数，非 prompt 装饰）
    # 基于 Kahneman & Tversky (1979) 前景理论
    cognitive_bias_params: Dict[str, float] = field(default_factory=lambda: {
        "loss_aversion": 2.25,      # λ: 损失厌恶系数 (Tversky & Kahneman, 1992 估计值)
        "anchoring_bias": 0.3,      # α: 锚定偏差强度 (0=完全跟随市场, 1=完全锚定历史)
        "herding_tendency": 0.5,    # β: 羊群效应倾向 (0=独立决策, 1=完全跟随市场)
        "overconfidence": 0.2,      # γ: 过度自信程度 (0=理性, 1=严重过度自信)
    })


@dataclass
class SimulationConfig:
    """仿真配置"""
    max_steps: int = 252
    random_seed: int = 42
    num_simulations: int = 100
    record_interval: int = 1
    output_dir: str = "./output"
    
    # 校准参数（基于真实中国碳市场数据）
    calibration_target: Dict[str, float] = field(default_factory=lambda: {
        "mean_price": 70.0,           # 中国CEA均价 ~70 CNY/吨
        "annualized_volatility": 0.21, # 年化波动率 ~21%
        "max_drawdown": -0.27,         # 最大回撤 ~27%
        "var_95": -0.021,              # 95% VaR
    })


@dataclass
class PolicyConfig:
    """政策情景配置"""
    allowance_reduction_rate: float = 0.0
    inspection_frequency: int = 30
    penalty_multiplier: float = 2.0
    allocation_method: str = "free"
    auction_reserve_price: float = 30.0


class Config:
    """全局配置类"""
    
    def __init__(self):
        self.market = MarketConfig()
        self.agent = AgentConfig()
        self.simulation = SimulationConfig()
        self.policy = PolicyConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market": self.market.__dict__,
            "agent": self.agent.__dict__,
            "simulation": self.simulation.__dict__,
            "policy": self.policy.__dict__,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        config = cls()
        if "market" in config_dict:
            config.market = MarketConfig(**config_dict["market"])
        if "agent" in config_dict:
            config.agent = AgentConfig(**config_dict["agent"])
        if "simulation" in config_dict:
            config.simulation = SimulationConfig(**config_dict["simulation"])
        if "policy" in config_dict:
            config.policy = PolicyConfig(**config_dict["policy"])
        return config
    
    def load_calibration(self, calibration_path: str):
        """从校准数据文件加载目标参数"""
        if os.path.exists(calibration_path):
            with open(calibration_path, 'r', encoding='utf-8') as f:
                cal_data = json.load(f)
            
            if "china_cea" in cal_data:
                cea = cal_data["china_cea"]
                self.simulation.calibration_target = {
                    "mean_price": cea.get("price_mean", 70.0),
                    "annualized_volatility": cea.get("annualized_volatility", 0.21),
                    "max_drawdown": cea.get("max_drawdown", -0.27),
                    "var_95": cea.get("var_95", -0.021),
                }


# 全局配置实例
config = Config()
