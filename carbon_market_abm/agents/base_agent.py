"""
基础智能体类
定义所有智能体的通用属性和行为

核心改进：
- 移除 LLM 依赖，采用纯数学建模
- 认知偏差参数通过前景理论、锚定效应、羊群效应等数学公式直接作用于决策
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import json

from ..config import config


@dataclass
class AgentState:
    """智能体状态"""
    cash: float = 0.0
    allowance: float = 0.0
    position_limit: float = float('inf')
    
    trade_history: List[Dict] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)
    
    @property
    def total_value(self, current_price: float = 0.0) -> float:
        """总资产价值"""
        return self.cash + self.allowance * current_price
    
    @property
    def position_ratio(self) -> float:
        """持仓比例"""
        if self.position_limit == float('inf'):
            return 0.0
        return self.allowance / self.position_limit


class BaseAgent(ABC):
    """
    基础智能体类
    
    认知偏差的数学实现：
    - 损失厌恶 λ: 前景理论价值函数 U(x) = x^α (gain), -λ(-x)^β (loss)
    - 锚定效应 α: P_ref = α·P_anchor + (1-α)·P_market
    - 羊群效应 β: D_effective = (1-β)·D_private + β·D_market_consensus
    - 过度自信 γ: signal_perceived = signal_true · (1 + γ·noise)
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        initial_cash: float = 1000000.0,
        initial_allowance: float = 0.0,
        cognitive_bias_params: Optional[Dict[str, float]] = None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        
        self.state = AgentState(
            cash=initial_cash,
            allowance=initial_allowance,
        )
        
        # 认知偏差参数 θ（数学建模，非 prompt 装饰）
        self.cognitive_bias_params = cognitive_bias_params or config.agent.cognitive_bias_params.copy()
        
        # 历史决策缓存
        self.decision_history: List[Dict] = []
        self.max_history = 10
        
        self.current_step = 0
        
        # 锚定价格（用于锚定效应的数学建模）
        self.anchor_price: Optional[float] = None
        self.reference_prices: List[float] = []
    
    # ==================== 认知偏差数学建模 ====================
    
    def prospect_theory_value(self, x: float) -> float:
        """
        前景理论价值函数 (Kahneman & Tversky, 1979; Tversky & Kahneman, 1992)
        
        U(x) = x^α           if x >= 0 (gain)
        U(x) = -λ·(-x)^β     if x < 0  (loss)
        
        其中:
            λ = loss_aversion (损失厌恶系数, 典型值 2.25)
            α = β = 0.88 (曲率参数, 反映风险态度)
        
        Args:
            x: 收益/损失值（可以是金额或收益率）
        
        Returns:
            前景理论主观价值
        """
        lam = self.cognitive_bias_params.get("loss_aversion", 2.25)
        alpha_curve = 0.88  # 曲率参数
        
        if x >= 0:
            return x ** alpha_curve
        else:
            return -lam * ((-x) ** alpha_curve)
    
    def prospect_theory_utility(self, wealth_change: float) -> float:
        """
        基于前景理论计算财富变化的主观效用
        
        用于评估交易决策的吸引力：
        - 预期收益被凹函数压缩（风险厌恶域）
        - 预期损失被凸函数放大且乘以λ（损失厌恶）
        
        Args:
            wealth_change: 财富变化量（元）
        
        Returns:
            主观效用值
        """
        return self.prospect_theory_value(wealth_change)
    
    def get_anchored_price(self, current_price: float, window: int = 20) -> float:
        """
        锚定效应数学模型
        
        参考价格 = α·P_anchor + (1-α)·P_historical_mean
        
        其中:
            α = anchoring_bias (锚定偏差强度, 0~1)
            P_anchor = 历史锚定价格（指数加权）
            P_historical_mean = 近 window 期均价
        
        高锚定偏差 → 参考价格更依赖初始锚定 → 对价格变化反应迟钝
        低锚定偏差 → 参考价格更跟随市场 → 更快适应新价格
        
        Args:
            current_price: 当前市场价格
            window: 历史均价窗口
        
        Returns:
            经锚定偏差调整后的参考价格
        """
        alpha = self.cognitive_bias_params.get("anchoring_bias", 0.3)
        
        # 历史均价
        if len(self.reference_prices) > 0:
            hist_mean = np.mean(self.reference_prices[-window:])
        else:
            hist_mean = current_price
        
        # 锚定价格
        if self.anchor_price is not None:
            anchored = alpha * self.anchor_price + (1 - alpha) * hist_mean
        else:
            anchored = hist_mean
        
        return anchored
    
    def get_herding_adjusted_signal(self, private_signal: float, market_consensus: float) -> float:
        """
        羊群效应数学模型
        
        D_effective = (1-β)·D_private + β·D_market_consensus
        
        其中:
            β = herding_tendency (羊群效应倾向, 0~1)
            D_private = 基于自身分析的私有信号 (-1到+1)
            D_market_consensus = 市场共识方向 (-1到+1)
        
        高羊群效应 → 更多跟随市场方向 → 可能加剧市场波动
        低羊群效应 → 更多依赖私有信号 → 更独立
        
        Args:
            private_signal: 私有信号强度 [-1, 1]
            market_consensus: 市场共识方向 [-1, 1]
        
        Returns:
            经羊群效应调整后的决策信号
        """
        beta = self.cognitive_bias_params.get("herding_tendency", 0.5)
        
        # 确保信号在合理范围内
        private_signal = np.clip(private_signal, -1, 1)
        market_consensus = np.clip(market_consensus, -1, 1)
        
        effective = (1 - beta) * private_signal + beta * market_consensus
        return effective
    
    def overconfident_signal(self, true_signal: float, noise_std: float = 0.1) -> float:
        """
        过度自信数学模型
        
        signal_perceived = true_signal · (1 + γ·ε),  ε ~ N(0, noise_std)
        
        其中:
            γ = overconfidence (过度自信程度, 0~1)
            ε = 随机噪声
        
        过度自信的投资者高估自己的信息精度，导致：
        - 交易频率增加
        - 仓位偏大
        - 对噪声信号过度反应
        
        Args:
            true_signal: 真实信号
            noise_std: 噪声标准差
        
        Returns:
            经过度自信偏差放大后的感知信号
        """
        gamma = self.cognitive_bias_params.get("overconfidence", 0.2)
        noise = np.random.normal(0, noise_std)
        perceived = true_signal * (1 + gamma * noise)
        return perceived
    
    # ==================== 基础方法 ====================
    
    def update_anchor(self, price: float):
        """更新锚定价格（指数加权）"""
        if self.anchor_price is None:
            self.anchor_price = price
        else:
            anchoring_bias = self.cognitive_bias_params.get("anchoring_bias", 0.3)
            self.anchor_price = anchoring_bias * self.anchor_price + (1 - anchoring_bias) * price
    
    def get_reference_price(self, window: int = 20) -> float:
        """获取参考价格"""
        if len(self.reference_prices) == 0:
            return config.market.initial_price
        prices = self.reference_prices[-window:]
        return np.mean(prices)
    
    def get_market_consensus(self, context: Dict) -> float:
        """
        计算市场共识方向
        
        Returns:
            [-1, 1] 范围的市场共识信号
            -1 = 强烈看空, 0 = 中性, +1 = 强烈看多
        """
        buying_pressure = context.get("buying_pressure", 0.5)
        selling_pressure = context.get("selling_pressure", 0.5)
        total = buying_pressure + selling_pressure
        
        if total > 0:
            consensus = (buying_pressure - selling_pressure) / total
        else:
            consensus = 0.0
        
        # 价格趋势也贡献共识
        price_trend = context.get("price_trend", 0)
        trend_signal = np.clip(price_trend * 10, -1, 1)
        
        # 综合共识（买卖压力60% + 价格趋势40%）
        return 0.6 * consensus + 0.4 * trend_signal
    
    def record_decision(self, decision: Dict):
        """记录决策"""
        self.decision_history.append({
            "step": self.current_step,
            **decision
        })
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
    
    def record_trade(self, trade: Dict):
        """记录交易"""
        self.state.trade_history.append({
            "step": self.current_step,
            **trade
        })
    
    def calculate_pnl(self, current_price: float) -> float:
        """计算盈亏（FIFO）"""
        if len(self.state.trade_history) < 2:
            return 0.0
        
        realized_pnl = 0.0
        buy_queue = []
        
        for trade in self.state.trade_history:
            if trade.get("side") == "buy":
                buy_queue.append((trade["price"], trade["quantity"]))
            elif trade.get("side") == "sell":
                sell_price = trade["price"]
                sell_qty = trade["quantity"]
                while sell_qty > 0 and buy_queue:
                    buy_price, buy_qty = buy_queue[0]
                    qty = min(sell_qty, buy_qty)
                    realized_pnl += (sell_price - buy_price) * qty
                    sell_qty -= qty
                    if qty >= buy_qty:
                        buy_queue.pop(0)
                    else:
                        buy_queue[0] = (buy_price, buy_qty - qty)
        
        unrealized_pnl = self.state.allowance * current_price
        total_pnl = realized_pnl + unrealized_pnl
        self.state.pnl_history.append(total_pnl)
        return total_pnl
    
    @abstractmethod
    def rule_based_decision(self, market_info: Dict, context: Dict) -> Dict:
        """
        基于规则的决策（核心决策方法）
        子类必须实现此方法，并在其中数学化地使用认知偏差参数
        """
        pass
    
    def make_decision(self, market_info: Dict, context: Dict, llm_client: Optional[Any] = None) -> Dict:
        """
        做出决策（纯数学建模，不使用LLM）
        """
        # 更新参考价格和锚定
        if market_info.get('current_price'):
            self.reference_prices.append(market_info['current_price'])
            self.update_anchor(market_info['current_price'])
        
        # 使用规则决策（纯数学建模）
        decision = self.rule_based_decision(market_info, context)
        
        # 记录决策
        self.record_decision(decision)
        
        return decision
    
    def execute_decision(self, decision: Dict, market: Any) -> List[Any]:
        """执行决策"""
        action = decision.get("action", "hold")
        quantity = decision.get("quantity", 0)
        price = decision.get("price", 0)
        
        trades = []
        
        if action == "buy" and quantity > 0:
            cost = quantity * price * (1 + config.market.transaction_fee_rate)
            if cost <= self.state.cash:
                order_id, trades = market.submit_buy_order(
                    agent_id=self.agent_id,
                    quantity=quantity,
                    price=price,
                )
                total_cost = sum(t.price * t.quantity for t in trades) * (1 + config.market.transaction_fee_rate)
                total_qty = sum(t.quantity for t in trades)
                self.state.cash -= total_cost
                self.state.allowance += total_qty
                
                for t in trades:
                    self.record_trade({
                        "side": "buy",
                        "price": t.price,
                        "quantity": t.quantity,
                    })
                    
        elif action == "sell" and quantity > 0:
            if quantity <= self.state.allowance:
                order_id, trades = market.submit_sell_order(
                    agent_id=self.agent_id,
                    quantity=quantity,
                    price=price,
                )
                total_revenue = sum(t.price * t.quantity for t in trades) * (1 - config.market.transaction_fee_rate)
                total_qty = sum(t.quantity for t in trades)
                self.state.cash += total_revenue
                self.state.allowance -= total_qty
                
                for t in trades:
                    self.record_trade({
                        "side": "sell",
                        "price": t.price,
                        "quantity": t.quantity,
                    })
        
        return trades
    
    def step(self, market: Any, context: Dict, llm_client: Optional[Any] = None):
        """智能体步进"""
        self.current_step += 1
        market_info = market.get_market_info()
        decision = self.make_decision(market_info, context, llm_client)
        trades = self.execute_decision(decision, market)
        return decision, trades
    
    def get_state_dict(self) -> Dict:
        """获取状态字典"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "cash": self.state.cash,
            "allowance": self.state.allowance,
            "total_trades": len(self.state.trade_history),
            "cognitive_bias_params": self.cognitive_bias_params,
        }
    
    @abstractmethod
    def get_role_description(self) -> str:
        """获取角色描述"""
        pass
