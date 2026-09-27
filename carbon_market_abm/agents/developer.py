"""
项目开发商智能体
开发和销售CCER（国家核证自愿减排量）项目

认知偏差数学化：
- 损失厌恶 → 影响项目投资的风险评估
- 锚定效应 → 影响CCER价格预期
- 过度自信 → 影响项目开发决策
"""

from typing import Dict, Any, Optional, List
import numpy as np

from .base_agent import BaseAgent
from ..config import config


class DeveloperAgent(BaseAgent):
    """项目开发商智能体"""
    
    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 2000000.0,
        project_types: Optional[List[str]] = None,
        cognitive_bias_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="项目开发商",
            initial_cash=initial_cash,
            initial_allowance=0.0,
            cognitive_bias_params=cognitive_bias_params,
        )
        
        self.project_types = project_types or ["forestry", "renewable", "methane"]
        self.projects: List[Dict] = []
        self.project_costs = {
            "forestry": {"setup": 500000, "annual": 50000, "yield": 1000},
            "renewable": {"setup": 800000, "annual": 30000, "yield": 1500},
            "methane": {"setup": 300000, "annual": 40000, "yield": 800},
        }
        
        self.ccer_inventory = 0.0
        self.ccer_production_history: list = []
        self.development_queue: List[Dict] = []
        self.total_ccer_produced = 0.0
        self.total_ccer_sold = 0.0
        self.total_project_cost = 0.0
    
    def get_role_description(self) -> str:
        return "项目开发商 - 开发CCER项目并出售核证减排量"
    
    def develop_project(self, project_type: str) -> bool:
        if project_type not in self.project_costs:
            return False
        cost = self.project_costs[project_type]["setup"]
        if self.state.cash < cost:
            return False
        
        self.state.cash -= cost
        self.total_project_cost += cost
        
        project = {
            "id": f"{self.agent_id}_proj_{len(self.projects)}",
            "type": project_type,
            "setup_step": self.current_step,
            "operational": False,
            "operational_step": self.current_step + 21,
            "annual_yield": self.project_costs[project_type]["yield"],
            "annual_cost": self.project_costs[project_type]["annual"],
            "total_produced": 0.0,
        }
        self.projects.append(project)
        return True
    
    def operate_projects(self):
        for project in self.projects:
            if not project["operational"]:
                if self.current_step >= project["operational_step"]:
                    project["operational"] = True
                else:
                    continue
            
            if self.state.cash >= project["annual_cost"]:
                self.state.cash -= project["annual_cost"]
                base_yield = project["annual_yield"] / 252
                noise = np.random.normal(0, base_yield * 0.1)
                production = max(0, base_yield + noise)
                self.ccer_inventory += production
                project["total_produced"] += production
                self.total_ccer_produced += production
        
        self.ccer_production_history.append(self.ccer_inventory)
    
    def get_ccer_discount(self, allowance_price: float) -> float:
        base_discount = 0.1
        inventory_pressure = min(self.ccer_inventory / 1000, 0.1)
        return base_discount + inventory_pressure
    
    def get_ccer_offer_price(self, allowance_price: float) -> float:
        discount = self.get_ccer_discount(allowance_price)
        return allowance_price * (1 - discount)
    
    def _prospect_theory_investment_eval(self, expected_roi: float) -> float:
        """
        损失厌恶对项目投资评估的影响
        
        前景理论预测：投资者对投资损失的痛苦 > 对等额收益的快乐
        因此，只有当预期ROI足够高时，才会决定投资
        
        数学模型:
            U(invest) = p·U(gain) + (1-p)·U(-loss)
            其中 p = 成功概率（基于ROI估算）
        """
        lam = self.cognitive_bias_params.get("loss_aversion", 2.25)
        
        # 估算成功概率（ROI越高概率越大）
        p_success = min(0.9, 0.5 + expected_roi * 2)
        
        # 前景理论效用
        u_gain = self.prospect_theory_value(expected_roi)
        u_loss = self.prospect_theory_value(-0.5)  # 失败损失50%投资
        
        subjective_utility = p_success * u_gain + (1 - p_success) * lam * u_loss
        return subjective_utility
    
    def rule_based_decision(self, market_info: Dict, context: Dict) -> Dict:
        """项目开发商规则决策 - 认知偏差数学化"""
        current_price = market_info.get('current_price', config.market.initial_price)
        ccer_price = self.get_ccer_offer_price(current_price)
        
        # 锚定效应影响价格预期
        anchored_price = self.get_anchored_price(current_price)
        price_expectation = self.overconfident_signal(
            (current_price - anchored_price) / max(anchored_price, 1)
        )
        
        # 出售CCER
        if self.ccer_inventory > 100:
            buying_pressure = context.get('emitter_buying_pressure', 0.5)
            
            # 羊群效应：跟随市场方向
            market_consensus = self.get_market_consensus(context)
            herding_adj = self.get_herding_adjusted_signal(0.3, market_consensus)
            
            if buying_pressure > 0.5 or self.ccer_inventory > 500:
                sell_quantity = min(self.ccer_inventory * 0.3, 200)
                return {
                    "action": "sell",
                    "quantity": sell_quantity,
                    "price": ccer_price,
                    "reasoning": f"CCER库存{self.ccer_inventory:.0f}吨，"
                                f"锚定价格{anchored_price:.1f}，羊群信号{herding_adj:.2f}",
                    "confidence": 0.65,
                }
        
        # 开发新项目 - 用前景理论评估
        if self.state.cash > 1000000 and len(self.projects) < 5:
            best_roi = -float('inf')
            best_type = None
            
            for ptype in self.project_types:
                cost = self.project_costs[ptype]
                annual_revenue = cost["yield"] * ccer_price
                annual_profit = annual_revenue - cost["annual"]
                roi = annual_profit / cost["setup"]
                
                # 前景理论评估
                pt_utility = self._prospect_theory_investment_eval(roi)
                
                if pt_utility > 0 and self.state.cash >= cost["setup"]:
                    if roi > best_roi:
                        best_roi = roi
                        best_type = ptype
            
            if best_type and best_roi > 0.10:
                return {
                    "action": "develop",
                    "project_type": best_type,
                    "quantity": 0,
                    "price": 0,
                    "reasoning": f"{best_type}项目ROI={best_roi:.1%}，PT效用={self._prospect_theory_investment_eval(best_roi):.3f}>0",
                    "confidence": 0.6,
                }
        
        # 持有等待
        if self.ccer_inventory > 200 and price_expectation > 0:
            return {
                "action": "hold",
                "quantity": 0,
                "price": 0,
                "reasoning": f"预期价格上涨({price_expectation:.2f})，持有CCER",
                "confidence": 0.55,
            }
        
        return {
            "action": "hold",
            "quantity": 0,
            "price": 0,
            "reasoning": "维持现状",
            "confidence": 0.5,
        }
    
    def execute_decision(self, decision: Dict, market: Any) -> list:
        action = decision.get("action", "hold")
        
        if action == "develop":
            project_type = decision.get("project_type", "forestry")
            success = self.develop_project(project_type)
            if success:
                return []
        elif action == "sell":
            quantity = decision.get("quantity", 0)
            price = decision.get("price", 0)
            if quantity > 0 and quantity <= self.ccer_inventory:
                self.ccer_inventory -= quantity
                self.total_ccer_sold += quantity
                revenue = quantity * price * (1 - config.market.transaction_fee_rate)
                self.state.cash += revenue
                return [{
                    "trade_id": f"CCER_{self.current_step}",
                    "seller_id": self.agent_id,
                    "price": price,
                    "quantity": quantity,
                    "type": "ccer"
                }]
        return []
    
    def step(self, market: Any, context: Dict, llm_client: Optional[Any] = None):
        self.operate_projects()
        return super().step(market, context, llm_client)
    
    def get_state_dict(self) -> Dict:
        state = super().get_state_dict()
        state.update({
            "ccer_inventory": self.ccer_inventory,
            "total_ccer_produced": self.total_ccer_produced,
            "total_ccer_sold": self.total_ccer_sold,
            "num_projects": len(self.projects),
            "active_projects": sum(1 for p in self.projects if p["operational"]),
            "total_project_cost": self.total_project_cost,
        })
        return state
