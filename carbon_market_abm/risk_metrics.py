"""
风险指标计算模块
计算VaR、CVaR、波动率等风险指标
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats


class RiskMetrics:
    """风险指标计算器"""
    
    @staticmethod
    def calculate_returns(prices: List[float]) -> np.ndarray:
        """计算收益率序列"""
        prices = np.array(prices)
        if len(prices) < 2:
            return np.array([])
        returns = np.diff(prices) / prices[:-1]
        return returns
    
    @staticmethod
    def var_historical(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        历史模拟法计算VaR
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
        
        Returns:
            VaR值（负数表示损失）
        """
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def var_parametric(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        参数法计算VaR（假设正态分布）
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
        
        Returns:
            VaR值
        """
        if len(returns) == 0:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence_level)
        return mean + z_score * std
    
    @staticmethod
    def cvar_historical(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        历史模拟法计算CVaR（条件风险价值）
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
        
        Returns:
            CVaR值
        """
        if len(returns) == 0:
            return 0.0
        var = RiskMetrics.var_historical(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def calculate_volatility(prices: List[float], window: int = 20, annualized: bool = True) -> float:
        """
        计算波动率
        
        Args:
            prices: 价格序列
            window: 计算窗口
            annualized: 是否年化
        
        Returns:
            波动率
        """
        if len(prices) < window + 1:
            return 0.0
        
        returns = RiskMetrics.calculate_returns(prices[-window:])
        volatility = np.std(returns)
        
        if annualized:
            # 假设252个交易日
            volatility *= np.sqrt(252)
        
        return volatility
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
        
        Returns:
            夏普比率
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = np.mean(returns) - risk_free_rate / 252  # 日度化
        return excess_returns / np.std(returns) * np.sqrt(252)  # 年化
    
    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> Tuple[float, int, int]:
        """
        计算最大回撤
        
        Args:
            prices: 价格序列
        
        Returns:
            (最大回撤比例, 峰值位置, 谷值位置)
        """
        if len(prices) < 2:
            return 0.0, 0, 0
        
        prices = np.array(prices)
        cumulative_max = np.maximum.accumulate(prices)
        drawdowns = (prices - cumulative_max) / cumulative_max
        
        max_drawdown = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)
        
        # 找到峰值位置
        peak_idx = np.argmax(prices[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
        
        return max_drawdown, peak_idx, max_dd_idx
    
    @staticmethod
    def calculate_liquidity_metrics(trades: List[Dict]) -> Dict:
        """
        计算流动性指标
        
        Args:
            trades: 交易列表
        
        Returns:
            流动性指标字典
        """
        if not trades:
            return {
                "total_volume": 0,
                "avg_trade_size": 0,
                "trade_frequency": 0,
            }
        
        # 处理Trade对象或字典
        volumes = []
        for t in trades:
            if hasattr(t, 'quantity'):
                volumes.append(t.quantity)
            elif hasattr(t, 'get'):
                volumes.append(t.get("quantity", 0))
            elif isinstance(t, dict):
                volumes.append(t.get("quantity", 0))
            else:
                volumes.append(0)
        
        return {
            "total_volume": sum(volumes),
            "avg_trade_size": np.mean(volumes),
            "trade_frequency": len(trades),
        }
    
    @staticmethod
    def calculate_market_impact(
        trades: List[Dict],
        prices: List[float],
        window: int = 5
    ) -> float:
        """
        计算市场冲击（简化版）
        
        Args:
            trades: 交易列表
            prices: 价格序列
            window: 冲击计算窗口
        
        Returns:
            市场冲击指标
        """
        if not trades or len(prices) < window:
            return 0.0
        
        impacts = []
        for trade in trades:
            # 简化的市场冲击计算
            quantity = trade.get("quantity", 0)
            # 假设冲击与交易量成正比
            impact = quantity / 1000  # 归一化
            impacts.append(impact)
        
        return np.mean(impacts) if impacts else 0.0
    
    @staticmethod
    def calculate_systemic_risk_indicators(
        agent_states: List[Dict],
        market_data: Dict
    ) -> Dict:
        """
        计算系统性风险指标
        
        Args:
            agent_states: 智能体状态列表
            market_data: 市场数据
        
        Returns:
            系统性风险指标
        """
        # 计算各类主体的财务压力
        emitter_stress = []
        speculator_stress = []
        
        for state in agent_states:
            agent_type = state.get("agent_type", "")
            
            if agent_type == "控排企业":
                # 控排企业压力 = 配额缺口 / 总排放
                deficit = state.get("allowance_deficit", 0)
                emissions = state.get("total_emissions", 1)
                stress = deficit / emissions if emissions > 0 else 0
                emitter_stress.append(stress)
            
            elif agent_type == "投机机构":
                # 投机者压力 = 最大回撤
                drawdown = state.get("max_drawdown", 0)
                speculator_stress.append(drawdown)
        
        return {
            "avg_emitter_stress": np.mean(emitter_stress) if emitter_stress else 0,
            "max_emitter_stress": np.max(emitter_stress) if emitter_stress else 0,
            "avg_speculator_stress": np.mean(speculator_stress) if speculator_stress else 0,
            "systemic_risk_score": np.mean(emitter_stress + speculator_stress) if (emitter_stress + speculator_stress) else 0,
        }
    
    @staticmethod
    def calculate_all_metrics(
        prices: List[float],
        trades: List[Dict],
        agent_states: List[Dict]
    ) -> Dict:
        """
        计算所有风险指标
        
        Args:
            prices: 价格序列
            trades: 交易列表
            agent_states: 智能体状态列表
        
        Returns:
            所有风险指标
        """
        returns = RiskMetrics.calculate_returns(prices)
        
        metrics = {
            "price_metrics": {
                "mean": np.mean(prices) if prices else 0,
                "std": np.std(prices) if prices else 0,
                "min": np.min(prices) if prices else 0,
                "max": np.max(prices) if prices else 0,
            },
            "return_metrics": {
                "mean": np.mean(returns) if len(returns) > 0 else 0,
                "std": np.std(returns) if len(returns) > 0 else 0,
                "skewness": stats.skew(returns) if len(returns) > 0 else 0,
                "kurtosis": stats.kurtosis(returns) if len(returns) > 0 else 0,
            },
            "var": {
                "var_95": RiskMetrics.var_historical(returns, 0.95),
                "var_99": RiskMetrics.var_historical(returns, 0.99),
                "cvar_95": RiskMetrics.cvar_historical(returns, 0.95),
                "cvar_99": RiskMetrics.cvar_historical(returns, 0.99),
            },
            "volatility": {
                "daily": RiskMetrics.calculate_volatility(prices, annualized=False),
                "annualized": RiskMetrics.calculate_volatility(prices, annualized=True),
            },
            "drawdown": {
                "max_drawdown": RiskMetrics.calculate_max_drawdown(prices)[0],
            },
            "liquidity": RiskMetrics.calculate_liquidity_metrics(trades),
            "systemic_risk": RiskMetrics.calculate_systemic_risk_indicators(agent_states, {}),
        }
        
        return metrics


class StressTest:
    """压力测试"""
    
    @staticmethod
    def run_policy_shock_scenario(
        base_simulation,
        shock_type: str,
        shock_magnitude: float
    ) -> Dict:
        """
        运行政策冲击情景
        
        Args:
            base_simulation: 基础仿真对象
            shock_type: 冲击类型（"allowance_reduction", "penalty_increase", "price_floor"）
            shock_magnitude: 冲击幅度
        
        Returns:
            压力测试结果
        """
        # 复制配置
        from .config import Config
        shocked_config = Config.from_dict(base_simulation.config.to_dict())
        
        # 应用冲击
        if shock_type == "allowance_reduction":
            shocked_config.policy.allowance_reduction_rate = shock_magnitude
        elif shock_type == "penalty_increase":
            shocked_config.policy.penalty_multiplier = shock_magnitude
        elif shock_type == "price_floor":
            shocked_config.market.price_floor = shock_magnitude
        
        # 运行受冲击的仿真
        from .simulation import CarbonMarketSimulation
        shocked_sim = CarbonMarketSimulation(shocked_config)
        shocked_sim.run(progress_bar=False)
        
        # 对比结果
        base_price = base_simulation.price_history[-1] if base_simulation.price_history else 0
        shocked_price = shocked_sim.price_history[-1] if shocked_sim.price_history else 0
        
        return {
            "shock_type": shock_type,
            "shock_magnitude": shock_magnitude,
            "base_final_price": base_price,
            "shocked_final_price": shocked_price,
            "price_change_pct": ((shocked_price - base_price) / base_price * 100) if base_price > 0 else 0,
            "shocked_metrics": RiskMetrics.calculate_all_metrics(
                shocked_sim.price_history,
                shocked_sim.market.order_book.trades,
                [agent.get_state_dict() for agent in shocked_sim.agents]
            ),
        }
