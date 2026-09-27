"""
投机机构智能体

核心改进：认知偏差通过数学公式直接作用于交易策略
- 损失厌恶 → 前景理论效用影响止损/止盈阈值和仓位大小
- 锚定效应 → 参考价格影响趋势判断
- 羊群效应 → 跟随市场方向调整持仓
- 过度自信 → 放大技术信号，增加交易频率
"""

from typing import Dict, Any, Optional
import numpy as np

from .base_agent import BaseAgent
from ..config import config


class SpeculatorAgent(BaseAgent):
    """
    投机机构智能体
    
    策略类型：趋势跟踪 / 均值回归 / 动量
    认知偏差通过数学公式影响每个策略的信号强度和仓位大小
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 5000000.0,
        initial_allowance: float = 0.0,
        strategy_type: str = "trend_following",
        risk_tolerance: float = 0.5,
        leverage_limit: float = 2.0,
        cognitive_bias_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="投机机构",
            initial_cash=initial_cash,
            initial_allowance=initial_allowance,
            cognitive_bias_params=cognitive_bias_params,
        )
        
        self.strategy_type = strategy_type
        self.risk_tolerance = risk_tolerance
        self.leverage_limit = leverage_limit
        
        self.price_history_window: list = []
        self.position_history: list = []
        
        self.trend_direction = 0
        self.entry_price: Optional[float] = None
        self.stop_loss_price: Optional[float] = None
        self.take_profit_price: Optional[float] = None
        
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = initial_cash
    
    def get_role_description(self) -> str:
        return f"投机机构 - 使用{self.strategy_type}策略追求短期盈利"
    
    def calculate_technical_indicators(self) -> Dict[str, Any]:
        """计算技术指标"""
        if len(self.price_history_window) < 20:
            return {"sma_short": None, "sma_long": None, "rsi": None, "volatility": 0}
        
        prices = np.array(self.price_history_window)
        sma_short = np.mean(prices[-5:])
        sma_long = np.mean(prices[-20:])
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        if len(gains) >= 14:
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
        else:
            rsi = 50
        
        volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) if len(prices) >= 20 else 0
        
        return {
            "sma_short": sma_short,
            "sma_long": sma_long,
            "rsi": rsi,
            "volatility": volatility,
        }
    
    def update_trend(self, current_price: float):
        """更新趋势判断"""
        self.price_history_window.append(current_price)
        if len(self.price_history_window) > 50:
            self.price_history_window.pop(0)
        
        indicators = self.calculate_technical_indicators()
        sma_short = indicators.get("sma_short")
        sma_long = indicators.get("sma_long")
        
        if sma_short is not None and sma_long is not None:
            if sma_short > sma_long * 1.02:
                self.trend_direction = 1
            elif sma_short < sma_long * 0.98:
                self.trend_direction = -1
            else:
                self.trend_direction = 0
    
    def _loss_aversion_adjusted_thresholds(self, current_price: float):
        """
        损失厌恶对止损/止盈阈值的调整
        
        前景理论预测：
        - 损失厌恶使投资者过早实现收益（止盈过近）
        - 同时过晚止损（因为实现损失很痛苦）
        
        数学模型:
            stop_loss_distance = base_distance / λ^0.5
            take_profit_distance = base_distance / λ^0.3
        
        λ 越高 → 止损越近（恐惧损失）→ 止盈也越近（急于锁定收益）
        """
        lam = self.cognitive_bias_params.get("loss_aversion", 2.25)
        
        base_stop = 0.05  # 5% 基础止损
        base_tp = 0.15    # 15% 基础止盈
        
        # 损失厌恶调整
        stop_distance = base_stop / (lam ** 0.3)   # λ高 → 止损更紧
        tp_distance = base_tp / (lam ** 0.2)       # λ高 → 止盈更近
        
        stop_loss = current_price * (1 - stop_distance)
        take_profit = current_price * (1 + tp_distance)
        
        return stop_loss, take_profit
    
    def _anchored_trend_signal(self, current_price: float, indicators: Dict) -> float:
        """
        锚定效应对趋势判断的影响
        
        高锚定偏差 → 更依赖历史价格 → 趋势判断更保守
        低锚定偏差 → 更跟随当前价格 → 趋势判断更灵敏
        
        数学模型:
            adjusted_trend = (1-α)·raw_trend + α·trend_from_anchor
        """
        alpha = self.cognitive_bias_params.get("anchoring_bias", 0.3)
        
        # 原始技术信号
        sma_short = indicators.get("sma_short")
        sma_long = indicators.get("sma_long")
        
        if sma_short is not None and sma_long is not None and sma_long > 0:
            raw_trend = (sma_short - sma_long) / sma_long  # 归一化
        else:
            raw_trend = 0
        
        # 基于锚定价格的趋势信号
        anchored_price = self.get_anchored_price(current_price)
        if anchored_price > 0:
            anchor_trend = (current_price - anchored_price) / anchored_price
        else:
            anchor_trend = 0
        
        # 锚定效应混合
        adjusted_trend = (1 - alpha) * raw_trend + alpha * anchor_trend
        
        return adjusted_trend
    
    def _overconfident_position_size(self, base_size: float, confidence: float) -> float:
        """
        过度自信对仓位大小的影响
        
        过度自信的投资者高估信号精度 → 仓位偏大
        
        数学模型:
            perceived_confidence = confidence · (1 + γ·ε)
            adjusted_size = base_size · perceived_confidence / confidence
        
        γ 越高 → 仓位越大 → 风险越高
        """
        gamma = self.cognitive_bias_params.get("overconfidence", 0.2)
        noise = np.random.normal(0, 0.1)
        perceived_conf = confidence * (1 + gamma * noise)
        perceived_conf = max(0.1, min(1.5, perceived_conf))
        
        adjusted_size = base_size * perceived_conf / max(confidence, 0.01)
        return adjusted_size
    
    def calculate_position_size(self, current_price: float, confidence: float) -> float:
        """计算仓位大小（含认知偏差调整）"""
        max_position_value = self.state.cash * self.risk_tolerance * self.leverage_limit
        
        indicators = self.calculate_technical_indicators()
        volatility = indicators.get("volatility", 0.1)
        volatility_factor = 1 / (1 + volatility * 10)
        
        base_size = max_position_value * confidence * volatility_factor / current_price if current_price > 0 else 0
        
        # 过度自信调整
        adjusted_size = self._overconfident_position_size(base_size, confidence)
        
        return max(0, adjusted_size)
    
    def rule_based_decision(self, market_info: Dict, context: Dict) -> Dict:
        """
        投机机构规则决策 - 认知偏差数学化版本
        
        决策流程:
        1. 更新趋势和技术指标
        2. 计算锚定效应调整的趋势信号
        3. 羊群效应混合市场共识
        4. 前景理论评估交易效用
        5. 损失厌恶调整止损/止盈
        6. 过度自信调整仓位
        """
        current_price = market_info.get('current_price', config.market.initial_price)
        
        # 更新趋势
        self.update_trend(current_price)
        
        # ========== 技术指标 ==========
        indicators = self.calculate_technical_indicators()
        rsi = indicators.get("rsi", 50) or 50
        
        # ========== 锚定效应调整的趋势信号 ==========
        trend_signal = self._anchored_trend_signal(current_price, indicators)
        
        # ========== 羊群效应 ==========
        market_consensus = self.get_market_consensus(context)
        private_signal = np.clip(trend_signal * 10, -1, 1)
        herding_signal = self.get_herding_adjusted_signal(private_signal, market_consensus)
        
        # ========== 损失厌恶调整止损/止盈 ==========
        if self.entry_price:
            stop_loss, take_profit = self._loss_aversion_adjusted_thresholds(self.entry_price)
        else:
            stop_loss, take_profit = self._loss_aversion_adjusted_thresholds(current_price)
        
        self.stop_loss_price = stop_loss
        self.take_profit_price = take_profit
        
        # ========== 止损检查 ==========
        if self.state.allowance > 0 and self.stop_loss_price is not None:
            if current_price <= self.stop_loss_price:
                return {
                    "action": "sell",
                    "quantity": self.state.allowance,
                    "price": current_price * 0.99,
                    "stop_loss": None,
                    "take_profit": None,
                    "reasoning": f"触发止损({self.stop_loss_price:.2f})，损失厌恶驱动平仓",
                    "confidence": 0.95,
                }
        
        # ========== 止盈检查 ==========
        if self.state.allowance > 0 and self.take_profit_price is not None:
            if current_price >= self.take_profit_price:
                return {
                    "action": "sell",
                    "quantity": self.state.allowance * 0.5,
                    "price": current_price * 0.99,
                    "stop_loss": self.stop_loss_price,
                    "take_profit": self.take_profit_price * 1.05,
                    "reasoning": f"触发止盈({self.take_profit_price:.2f})，前景理论锁定收益",
                    "confidence": 0.85,
                }
        
        # ========== 策略信号 ==========
        # 综合信号 = 技术趋势(40%) + 羊群效应(30%) + 价格吸引力(30%)
        price_attractiveness = self._perceived_price_attractiveness(current_price)
        composite_signal = 0.4 * herding_signal + 0.3 * market_consensus + 0.3 * price_attractiveness
        
        # 过度自信增加交易概率
        gamma = self.cognitive_bias_params.get("overconfidence", 0.2)
        trade_threshold = 0.08 / (1 + gamma)  # 降低阈值 → 更多交易
        
        # ========== 买入决策 ==========
        if composite_signal > trade_threshold and self.state.cash > current_price * 5:
            confidence = min(0.8, 0.5 + abs(composite_signal))
            position_size = self.calculate_position_size(current_price, confidence)
            
            if position_size > 0.1:
                self.entry_price = current_price
                self.stop_loss_price, self.take_profit_price = \
                    self._loss_aversion_adjusted_thresholds(current_price)
                
                return {
                    "action": "buy",
                    "quantity": position_size,
                    "price": current_price * 1.005,
                    "stop_loss": self.stop_loss_price,
                    "take_profit": self.take_profit_price,
                    "reasoning": f"综合信号{composite_signal:.2f}={0.4*herding_signal:.2f}(趋势)+"
                                f"{0.3*market_consensus:.2f}(羊群)+{0.3*price_attractiveness:.2f}(锚定)",
                    "confidence": confidence,
                }
        
        # ========== 卖出决策 ==========
        if composite_signal < -trade_threshold and self.state.allowance > 0:
            quantity = min(self.state.allowance * 0.5, 50)
            if quantity > 0.1:
                return {
                    "action": "sell",
                    "quantity": quantity,
                    "price": current_price * 0.995,
                    "stop_loss": None,
                    "take_profit": None,
                    "reasoning": f"综合信号{composite_signal:.2f}，趋势偏空",
                    "confidence": min(0.8, 0.5 + abs(composite_signal)),
                }
        
        # ========== 流动性提供（基础交易活动） ==========
        # 即使没有强信号，也提供流动性（真实碳市场中投机者活跃提供流动性）
        # 降低频率：真实碳市场投机者也不是每步都交易
        if np.random.random() < 0.25:  # 25%概率交易（从55%降低）
            if self.state.allowance > 0 and np.random.random() < 0.5:
                quantity = min(self.state.allowance * 0.08, 20)  # 减少单次卖出量
                if quantity > 0.1:
                    return {
                        "action": "sell",
                        "quantity": quantity,
                        "price": current_price * 0.997,
                        "reasoning": "提供流动性",
                        "confidence": 0.4,
                    }
            elif self.state.cash > current_price * 5:
                quantity = min(self.state.cash / current_price * 0.05, 15)  # 减少单次买入量
                if quantity > 0.1:
                    return {
                        "action": "buy",
                        "quantity": quantity,
                        "price": current_price * 1.003,
                        "reasoning": "提供流动性",
                        "confidence": 0.4,
                    }
        
        # ========== 末期清算：投机者平仓 ==========
        # 临近合规截止日，投机者平仓卖出持有的配额
        current_step = context.get('current_step', 0)
        deadline = context.get('compliance_deadline', 252)
        steps_to_dead = deadline - current_step
        if self.state.allowance > 5 and steps_to_dead <= 30:
            # 最后30步：渐进式平仓（配额即将过期）
            urgency = 1.0 + 4.0 * (1.0 - steps_to_dead / 30.0)  # 1→5
            sell_qty = min(self.state.allowance * 0.15 * urgency, 40 * urgency)
            if sell_qty > 0.1:
                return {
                    "action": "sell",
                    "quantity": sell_qty,
                    "price": current_price * 0.995,
                    "reasoning": f"末期平仓(剩余{steps_to_dead}步)",
                    "confidence": 0.7,
                }
        
        return {
            "action": "hold",
            "quantity": 0,
            "price": 0,
            "reasoning": f"信号不足: composite={composite_signal:.3f}, threshold={trade_threshold:.3f}",
            "confidence": 0.5,
        }
    
    def _perceived_price_attractiveness(self, current_price: float) -> float:
        """价格吸引力评估（锚定+过度自信）"""
        anchored = self.get_anchored_price(current_price)
        if anchored > 0:
            price_gap = (anchored - current_price) / anchored
        else:
            price_gap = 0
        perceived_gap = self.overconfident_signal(price_gap, noise_std=0.12)
        return float(np.clip(np.tanh(perceived_gap * 5), -1, 1))
    
    def record_trade(self, trade: Dict):
        """记录交易并更新业绩"""
        super().record_trade(trade)
        self.total_trades += 1
        
        if trade.get("side") == "sell" and self.entry_price:
            pnl = (trade["price"] - self.entry_price) * trade["quantity"]
            self.total_pnl += pnl
            if pnl > 0:
                self.winning_trades += 1
        
        current_value = self.state.cash + self.state.allowance * trade.get("price", 0)
        if current_value > self.peak_value:
            self.peak_value = current_value
        drawdown = (self.peak_value - current_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def get_state_dict(self) -> Dict:
        state = super().get_state_dict()
        state.update({
            "strategy_type": self.strategy_type,
            "risk_tolerance": self.risk_tolerance,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown,
            "trend_direction": self.trend_direction,
        })
        return state
