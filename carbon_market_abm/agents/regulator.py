"""
监管机构智能体
负责市场监管、政策执行和风险控制
"""

from typing import Dict, Any, Optional, List
import numpy as np

from .base_agent import BaseAgent
from ..config import config


class RegulatorAgent(BaseAgent):
    """监管机构智能体"""
    
    def __init__(
        self,
        agent_id: str,
        inspection_frequency: int = 30,
        penalty_multiplier: float = 2.0,
        price_intervention_threshold: float = 0.3,
        cognitive_bias_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="监管机构",
            initial_cash=0.0,
            initial_allowance=0.0,
            cognitive_bias_params=cognitive_bias_params,
        )
        
        self.inspection_frequency = inspection_frequency
        self.penalty_multiplier = penalty_multiplier
        self.price_intervention_threshold = price_intervention_threshold
        
        self.price_history_reg: list = []
        self.volume_history: list = []
        self.agent_activities: Dict[str, List[Dict]] = {}
        self.alerts: List[Dict] = []
        self.interventions: List[Dict] = []
        self.total_inspections = 0
        self.violations_found = 0
        self.penalties_issued = 0.0
        self.current_policy = {
            "allowance_reduction_rate": config.policy.allowance_reduction_rate,
            "inspection_frequency": inspection_frequency,
            "penalty_multiplier": penalty_multiplier,
        }
    
    def get_role_description(self) -> str:
        return "监管机构 - 负责市场监管、政策执行和风险控制"
    
    def monitor_market(self, market_info: Dict, agents: List[Any]):
        current_price = market_info.get('current_price', 0)
        self.price_history_reg.append(current_price)
        volume = market_info.get('volume_24h', 0)
        self.volume_history.append(volume)
        
        if len(self.price_history_reg) >= 5:
            recent = self.price_history_reg[-5:]
            vol = np.std(recent) / np.mean(recent) if np.mean(recent) > 0 else 0
            if vol > self.price_intervention_threshold:
                self.alerts.append({
                    "step": self.current_step,
                    "type": "price_volatility",
                    "severity": "high" if vol > 0.5 else "medium",
                    "value": vol,
                    "message": f"价格波动率异常: {vol:.2%}",
                })
        
        if len(self.volume_history) >= 5:
            avg_vol = np.mean(self.volume_history[-20:]) if len(self.volume_history) >= 20 else np.mean(self.volume_history)
            if volume > avg_vol * 3 and avg_vol > 0:
                self.alerts.append({
                    "step": self.current_step,
                    "type": "volume_spike",
                    "severity": "medium",
                    "value": volume / avg_vol,
                    "message": f"交易量异常放大: {volume:.0f}",
                })
        
        for agent in agents:
            if agent.agent_id not in self.agent_activities:
                self.agent_activities[agent.agent_id] = []
            self.agent_activities[agent.agent_id].append({
                "step": self.current_step,
                "cash": agent.state.cash,
                "allowance": agent.state.allowance,
                "trades": len(agent.state.trade_history),
            })
    
    def conduct_inspection(self, agents: List[Any]) -> List[Dict]:
        self.total_inspections += 1
        results = []
        for agent in agents:
            if agent.agent_type == "控排企业":
                if hasattr(agent, 'get_allowance_deficit'):
                    deficit = agent.get_allowance_deficit()
                    if deficit > 0:
                        self.violations_found += 1
                        current_price = self.price_history_reg[-1] if self.price_history_reg else config.market.initial_price
                        penalty = deficit * current_price * self.penalty_multiplier
                        agent.state.cash -= penalty
                        self.penalties_issued += penalty
                        results.append({
                            "agent_id": agent.agent_id,
                            "type": "non_compliance",
                            "deficit": deficit,
                            "penalty": penalty,
                        })
        return results
    
    def adjust_policy(self, market_conditions: Dict) -> Dict:
        adjustments = {}
        avg_price = np.mean(self.price_history_reg[-30:]) if len(self.price_history_reg) >= 30 else config.market.initial_price
        
        if avg_price < config.market.initial_price * 0.7:
            adjustments["allowance_reduction_rate"] = min(
                self.current_policy["allowance_reduction_rate"] + 0.02, 0.10)
        elif avg_price > config.market.initial_price * 1.5:
            adjustments["allowance_reduction_rate"] = max(
                self.current_policy["allowance_reduction_rate"] - 0.01, 0.0)
        
        if self.violations_found > self.total_inspections * 0.3:
            adjustments["inspection_frequency"] = max(
                self.current_policy["inspection_frequency"] - 5, 7)
        
        self.current_policy.update(adjustments)
        return adjustments
    
    def rule_based_decision(self, market_info: Dict, context: Dict) -> Dict:
        if self.current_step % self.inspection_frequency == 0:
            return {
                "action": "inspect", "target": "all_emitters",
                "parameters": {"inspection_scope": "all"},
                "reasoning": f"第{self.total_inspections + 1}次合规检查",
                "confidence": 0.9,
            }
        
        if self.alerts and self.alerts[-1]["step"] == self.current_step:
            latest = self.alerts[-1]
            if latest["severity"] == "high":
                return {
                    "action": "intervene", "target": "market",
                    "parameters": {
                        "intervention_type": "price_stabilization" if latest["type"] == "price_volatility" else "liquidity_support",
                    },
                    "reasoning": latest["message"],
                    "confidence": 0.8,
                }
        
        if self.current_step % 60 == 0:
            adjustments = self.adjust_policy(market_info)
            if adjustments:
                return {
                    "action": "adjust_policy", "target": "policy_parameters",
                    "parameters": {"policy_adjustment": adjustments},
                    "reasoning": f"政策调整: {adjustments}",
                    "confidence": 0.7,
                }
        
        return {
            "action": "monitor", "target": "market",
            "parameters": {},
            "reasoning": "市场运行正常",
            "confidence": 0.6,
        }
    
    def step(self, market: Any, context: Dict, llm_client: Optional[Any] = None):
        self.current_step += 1
        market_info = market.get_market_info()
        agents = context.get('agents', [])
        self.monitor_market(market_info, agents)
        decision = self.make_decision(market_info, context, llm_client)
        
        if decision["action"] == "inspect":
            decision["inspection_results"] = self.conduct_inspection(agents)
        
        self.record_decision(decision)
        return decision, []
    
    def get_state_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "total_inspections": self.total_inspections,
            "violations_found": self.violations_found,
            "penalties_issued": self.penalties_issued,
            "num_alerts": len(self.alerts),
            "num_interventions": len(self.interventions),
            "current_policy": self.current_policy,
        }
