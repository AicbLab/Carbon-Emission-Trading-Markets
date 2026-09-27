"""
控排企业智能体

核心改进：认知偏差通过数学公式直接作用于决策
- 损失厌恶 → 前景理论效用函数影响买入/卖出阈值
- 锚定效应 → 参考价格影响履约成本评估
- 羊群效应 → 跟随其他企业的购买行为
- 过度自信 → 对未来价格的判断偏差
"""

from typing import Dict, Any, Optional
import numpy as np

from .base_agent import BaseAgent
from ..config import config


class EmitterAgent(BaseAgent):
    """
    控排企业智能体
    
    决策逻辑：
    1. 计算履约压力（基于排放缺口和时间）
    2. 用前景理论评估买入/持有的主观效用
    3. 用锚定价格评估当前价格是否"便宜"
    4. 用羊群效应调整购买量
    5. 用过度自信调整对价格走势的判断
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 1000000.0,
        initial_allowance: float = 100.0,
        emission_rate: float = 10.0,
        abatement_cost: float = 80.0,
        compliance_deadline: int = 252,
        cognitive_bias_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="控排企业",
            initial_cash=initial_cash,
            initial_allowance=initial_allowance,
            cognitive_bias_params=cognitive_bias_params,
        )
        
        self.emission_rate = emission_rate
        self.abatement_cost = abatement_cost
        self.compliance_deadline = compliance_deadline
        
        self.total_emissions = 0.0
        self.total_abatement = 0.0
        self.total_sold = 0.0  # 累计卖出量（用于限制正常期卖出）
        self.initial_forward_excess = max(0, initial_allowance - emission_rate * compliance_deadline)  # 初始超额
        self.compliance_status = "pending"
        self.compliance_cost = 0.0
        self.abatement_investment = 0.0
        self.emission_history: list = []
    
    def get_role_description(self) -> str:
        return "控排企业 - 需要购买碳配额以满足排放合规要求"
    
    def calculate_emissions(self, step: int) -> float:
        """计算当期排放"""
        base_emission = self.emission_rate
        noise = np.random.normal(0, base_emission * 0.1)
        emission = max(0, base_emission + noise)
        
        if self.abatement_investment > 0:
            reduction = min(emission * 0.1, self.abatement_investment / self.abatement_cost)
            emission -= reduction
            self.total_abatement += reduction
        
        self.total_emissions += emission
        self.emission_history.append(emission)
        return emission
    
    def get_forward_looking_deficit(self) -> float:
        """前瞻性配额缺口：基于预期未来排放
        
        真实企业中购买配额是基于对未来排放的预估，而非已发生的排放。
        expected_deficit = expected_total_emissions - current_allowance
        """
        remaining_steps = max(1, self.compliance_deadline - self.current_step)
        expected_remaining_emissions = self.emission_rate * remaining_steps
        # 预期总缺口 = 已排放 + 预期剩余排放 - 当前配额
        expected_total_need = self.total_emissions + expected_remaining_emissions
        return max(0, expected_total_need - self.state.allowance)
    
    def get_forward_looking_excess(self) -> float:
        """前瞻性配额盈余：配额 vs 预期全周期排放
        
        企业卖出决策基于全周期视角：如果配额超过预期总排放，多余部分可卖出
        """
        expected_total_need = self.emission_rate * self.compliance_deadline
        return max(0, self.state.allowance - expected_total_need)
    
    def get_allowance_deficit(self) -> float:
        """配额缺口"""
        return max(0, self.total_emissions - self.state.allowance)
    
    def get_compliance_pressure(self) -> float:
        """履约压力 (0-1)"""
        if self.current_step >= self.compliance_deadline:
            return 1.0 if self.get_allowance_deficit() > 0 else 0.0
        time_pressure = self.current_step / self.compliance_deadline
        deficit_ratio = self.get_allowance_deficit() / max(self.total_emissions, 1)
        return min(1.0, (time_pressure + deficit_ratio) / 2)
    
    def _perceived_price_attractiveness(self, current_price: float) -> float:
        """
        基于锚定效应和过度自信的价格吸引力评估
        
        返回 [-1, 1]:
            >0 表示价格偏低（适合买入）
            <0 表示价格偏高（不适合买入）
        
        数学模型:
            P_ref = anchored_price (受锚定偏差影响)
            perceived_gap = (P_ref - P_current) / P_ref * (1 + γ·ε)
            attractiveness = tanh(perceived_gap · sensitivity)
        """
        # 锚定参考价格
        anchored = self.get_anchored_price(current_price)
        
        # 价格差距（当前价格 vs 参考价格）
        if anchored > 0:
            price_gap = (anchored - current_price) / anchored
        else:
            price_gap = 0
        
        # 过度自信放大感知到的价格差距
        perceived_gap = self.overconfident_signal(price_gap, noise_std=0.15)
        
        # 用 tanh 映射到 [-1, 1]
        attractiveness = np.tanh(perceived_gap * 5)
        
        return attractiveness
    
    def _prospect_theory_buy_decision(self, current_price: float, deficit: float, pressure: float) -> Dict:
        """
        基于前景理论的买入决策
        
        核心思想：
        - 买入配额 = 确定性损失（花钱）换取确定性收益（避免罚款）
        - 损失厌恶使企业更倾向于避免罚款（损失域）而非追求省钱（收益域）
        - 当罚款的"损失效用" > 购买成本的"损失效用"时，选择买入
        
        数学模型:
            U(buy) = U(-cost) + λ·U(penalty_avoided)
            U(hold) = U(-expected_penalty)
            当 U(buy) > U(hold) 时买入
        """
        lam = self.cognitive_bias_params.get("loss_aversion", 2.25)
        
        # 预期购买成本
        buy_cost = deficit * current_price
        
        # 预期罚款（如果不买）
        expected_penalty = deficit * current_price * config.policy.penalty_multiplier
        
        # 前景理论效用
        u_buy_cost = self.prospect_theory_value(-buy_cost)  # 花钱的痛苦
        u_penalty_avoided = self.prospect_theory_value(expected_penalty)  # 避免罚款的价值
        u_expected_penalty = self.prospect_theory_value(-expected_penalty)  # 承受罚款的痛苦
        
        # 买入的净效用 = 避免罚款的收益 - 花钱的成本
        # 损失厌恶使避免罚款的价值被放大 λ 倍
        net_utility_buy = lam * abs(u_penalty_avoided) + u_buy_cost
        
        # 不买的效用 = 承受罚款
        net_utility_hold = u_expected_penalty
        
        # 决策：买入更优？
        buy_preferred = net_utility_buy > net_utility_hold
        
        return {
            "buy_preferred": buy_preferred,
            "net_utility": net_utility_buy - net_utility_hold,
            "buy_cost": buy_cost,
            "expected_penalty": expected_penalty,
        }
    
    def rule_based_decision(self, market_info: Dict, context: Dict) -> Dict:
        """
        控排企业规则决策 - 认知偏差数学化版本
        
        决策流程：
        1. 计算履约压力和配额缺口
        2. 前景理论评估买入 vs 等待的效用
        3. 锚定效应评估价格吸引力
        4. 羊群效应调整购买量
        5. 综合决策
        
        关键设计：每次交易量很小（2-4%的缺口），确保买卖流量平衡
        真实碳市场中企业每次只交易少量配额，全年持续交易
        """
        current_price = market_info.get('current_price', config.market.initial_price)
        deficit = self.get_forward_looking_deficit()  # 前瞻性缺口
        pressure = self.get_compliance_pressure()  # 0~1
        
        # ========== 步骤1: 前景理论评估 ==========
        pt_eval = self._prospect_theory_buy_decision(current_price, deficit, pressure)
        
        # ========== 步骤2: 锚定效应评估价格吸引力 ==========
        price_attractiveness = self._perceived_price_attractiveness(current_price)
        # attractiveness > 0: 价格低于参考 → 适合买入
        # attractiveness < 0: 价格高于参考 → 不宜买入
        
        # ========== 步骤3: 羊群效应 ==========
        market_consensus = self.get_market_consensus(context)
        # 私有信号：基于前景理论和价格吸引力
        private_signal = 0.6 * (1.0 if pt_eval["buy_preferred"] else -0.5) + \
                         0.4 * price_attractiveness
        # 经羊群效应调整
        herding_signal = self.get_herding_adjusted_signal(private_signal, market_consensus)
        
        # ========== 步骤4: 综合决策 ==========
        # 每次交易量很小（2-4%的缺口），模拟真实碳市场的分批小额交易
        
        # 情况A: 履约压力极高 + 有缺口 → 必须买入（损失厌恶强化）
        if pressure > 0.7 and deficit > 0:
            # 损失厌恶使企业更恐惧罚款 → 加速买入
            lam = self.cognitive_bias_params.get("loss_aversion", 2.25)
            urgency_factor = 1.0 + (lam - 1) * 0.5 * pressure  # λ越高越急
            base_quantity = deficit * 0.05  # 每次买入量小（流量平衡）
            quantity = min(base_quantity * urgency_factor, self.state.cash / current_price * 0.9)
            if quantity > 0.1:
                return {
                    "action": "buy",
                    "quantity": quantity,
                    "price": current_price * 1.002,
                    "reasoning": f"履约压力{pressure:.1%}，前景理论驱动紧急买入",
                    "confidence": 0.9,
                }
        
        # 情况B: 有缺口 + 前景理论支持买入 + 价格有吸引力
        if deficit > 0 and pt_eval["buy_preferred"] and herding_signal > -0.5:
            # 购买量受羊群效应和损失厌恶调节
            # λ越高 → 越恐惧罚款 → 每次买入更积极
            lam = self.cognitive_bias_params.get("loss_aversion", 2.25)
            lam_buy_factor = 0.8 + (lam - 1) * 0.3  # λ=1.5→0.95, λ=3→1.4
            base_quantity = deficit * 0.03 * lam_buy_factor  # 每次小量
            herding_multiplier = 1.0 + 0.5 * market_consensus  # 市场看多时多买
            quantity = min(base_quantity * herding_multiplier,
                          self.state.cash / current_price * 0.5)
            if quantity > 0.1:
                return {
                    "action": "buy",
                    "quantity": quantity,
                    "price": current_price * 1.001,
                    "reasoning": f"缺口{deficit:.1f}，PT效用支持买入，价格吸引力{price_attractiveness:.2f}",
                    "confidence": 0.7,
                }
        
        # 情况C: 有缺口但价格过高（锚定效应认为贵）
        if deficit > 0 and price_attractiveness < -0.3:
            # 少量买入补仓
            quantity = min(deficit * 0.015, self.state.cash / current_price * 0.05)
            if quantity > 0.1:
                return {
                    "action": "buy",
                    "quantity": quantity,
                    "price": current_price * 1.001,
                    "reasoning": f"价格偏高(attract={price_attractiveness:.2f})，少量补仓",
                    "confidence": 0.5,
                }
        
        # 情况D: 前瞻性超额配额 → 卖出（基于全周期视角，而非已累积排放）
        forward_excess = self.get_forward_looking_excess()
        # 实际盈余 = 当前配额 - 已累积排放（末期看实际剩余，而非预期）
        actual_surplus = max(0, self.state.allowance - self.total_emissions)
        steps_to_deadline = self.compliance_deadline - self.current_step
        
        # 损失厌恶使企业不愿"损失"配额（禀赋效应）
        lam = self.cognitive_bias_params.get("loss_aversion", 2.25)
        sell_threshold = -0.2 - (lam - 1) * 0.15  # λ=1.5→-0.275, λ=3→-0.35
        
        # ===== 末期清算优先 =====
        # 真实碳市场：临近合规截止日，持有富余配额的企业加速卖出
        # 因为配额即将过期/价值归零，企业倾向于在截止日前变现
        is_end_of_period = steps_to_deadline <= 30
        
        if is_end_of_period:
            # ===== 末期：基于实际盈余，激进卖出 =====
            if actual_surplus > 1:
                urgency = 1.0 + 9.0 * (1.0 - steps_to_deadline / 30.0)  # 1→10
                if price_attractiveness < 0:  # 价格偏高 → 好价格多卖
                    sell_qty = min(actual_surplus * 0.15 * urgency, 100 * urgency)
                else:  # 价格偏低也要卖（配额要过期，不得不卖）
                    sell_qty = min(actual_surplus * 0.12 * urgency, 80 * urgency)
                sell_qty = max(sell_qty, 0)
                if sell_qty > 0.1:
                    return {
                        "action": "sell",
                        "quantity": sell_qty,
                        "price": current_price * 0.995,  # 末期略降价确保成交
                        "reasoning": f"末期清算: 实际盈余{actual_surplus:.0f}吨，urgency={urgency:.1f}，剩余{steps_to_deadline}步",
                        "confidence": 0.7,
                    }
        else:
            # ===== 正常期：保守卖出，保留大部分超额配额到末期 =====
            # 累计卖出上限：只卖出初始超额的40%，保留60%到末期清算
            max_normal_sell = self.initial_forward_excess * 0.4
            remaining_sellable = max(0, max_normal_sell - self.total_sold)
            
            if remaining_sellable > 5 and actual_surplus > 50:  # 实际盈余也要足够
                if price_attractiveness < sell_threshold:  # 价格高于参考 → 好价格多卖
                    sell_qty = min(remaining_sellable * 0.01, 10)
                elif price_attractiveness < 0:
                    sell_qty = min(remaining_sellable * 0.005, 5)
                else:
                    sell_qty = min(remaining_sellable * 0.003, 3)
                sell_qty = max(sell_qty, 0)
                if sell_qty > 0.1:
                    return {
                        "action": "sell",
                        "quantity": sell_qty,
                        "price": current_price * 0.998,
                        "reasoning": f"超额{forward_excess:.0f}吨(剩{remaining_sellable:.0f}可卖)，价格吸引力{price_attractiveness:.2f}",
                        "confidence": 0.5,
                    }
        
        # 情况E: 考虑减排投资
        if current_price > self.abatement_cost * 1.2 and self.state.cash > 10000:
            investment = min(self.state.cash * 0.05, 5000)
            return {
                "action": "abate",
                "quantity": 0,
                "price": 0,
                "abatement_investment": investment,
                "reasoning": f"碳价{current_price:.1f}>减排成本{self.abatement_cost:.1f}，投资减排",
                "confidence": 0.65,
            }
        
        return {
            "action": "hold",
            "quantity": 0,
            "price": 0,
            "reasoning": f"观望: deficit={deficit:.1f}, pressure={pressure:.2f}, "
                        f"attract={price_attractiveness:.2f}, herding={herding_signal:.2f}",
            "confidence": 0.5,
        }
    
    def execute_decision(self, decision: Dict, market: Any) -> list:
        """执行决策（支持减排投资 + 追踪累计卖出）"""
        action = decision.get("action", "hold")
        
        if action == "abate":
            investment = decision.get("abatement_investment", 0)
            if investment > 0 and investment <= self.state.cash:
                self.abatement_investment += investment
                self.state.cash -= investment
                return []
        
        trades = super().execute_decision(decision, market)
        
        # 追踪累计卖出量
        if action == "sell" and trades:
            total_qty = sum(t.get('quantity', 0) if isinstance(t, dict) else t.quantity for t in trades)
            self.total_sold += total_qty
        
        return trades
    
    def step(self, market: Any, context: Dict, llm_client: Optional[Any] = None):
        """控排企业步进"""
        emission = self.calculate_emissions(self.current_step)
        decision, trades = super().step(market, context, llm_client)
        
        if self.current_step >= self.compliance_deadline:
            deficit = self.get_allowance_deficit()
            if deficit > 0:
                self.compliance_status = "non_compliant"
                penalty = deficit * market.get_current_price() * config.policy.penalty_multiplier
                self.compliance_cost += penalty
                self.state.cash -= penalty
            else:
                self.compliance_status = "compliant"
        
        return decision, trades
    
    def get_state_dict(self) -> Dict:
        state = super().get_state_dict()
        state.update({
            "total_emissions": self.total_emissions,
            "total_abatement": self.total_abatement,
            "allowance_deficit": self.get_allowance_deficit(),
            "compliance_status": self.compliance_status,
            "compliance_pressure": self.get_compliance_pressure(),
            "abatement_cost": self.abatement_cost,
        })
        return state
