"""
仿真主模块
实现ABM仿真循环和数据收集
纯数学建模版本 - 无LLM依赖
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm

from .config import config, Config
from .market import CarbonMarket
from .agents.emitter import EmitterAgent
from .agents.speculator import SpeculatorAgent
from .agents.developer import DeveloperAgent
from .agents.regulator import RegulatorAgent


class CarbonMarketSimulation:
    """碳金融市场仿真类"""
    
    def __init__(self, simulation_config: Optional[Config] = None):
        """
        初始化仿真
        
        Args:
            simulation_config: 仿真配置
        """
        self.config = simulation_config or config
        
        # 设置随机种子
        random.seed(self.config.simulation.random_seed)
        np.random.seed(self.config.simulation.random_seed)
        
        # 初始化市场
        self.market = CarbonMarket(self.config)
        
        # 初始化智能体
        self.agents: List[Any] = []
        self.emitters: List[EmitterAgent] = []
        self.speculators: List[SpeculatorAgent] = []
        self.developers: List[DeveloperAgent] = []
        self.regulators: List[RegulatorAgent] = []
        
        self._create_agents()
        
        # 数据收集
        self.data_collector = DataCollector()
        
        # 仿真状态
        self.current_step = 0
        self.is_running = False
        
        # 历史记录
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.agent_states_history: List[Dict] = []
    
    def _create_agents(self):
        """创建智能体
        
        认知偏差参数以 config 值为均值，个体在 ±30% 范围内浮动，
        保持异质性同时确保网格搜索可校准。
        """
        agent_config = self.config.agent
        cb = agent_config.cognitive_bias_params  # 配置中的认知偏差参数
        
        lam_base = cb.get("loss_aversion", 2.25)
        alp_base = cb.get("anchoring_bias", 0.3)
        bet_base = cb.get("herding_tendency", 0.5)
        gam_base = cb.get("overconfidence", 0.2)
        
        # 配额分配比例：真实碳市场中企业配额分布接近均衡
        # 约50%企业配额有富余（净卖方），50%配额偏紧（净买方）
        allowance_frac = getattr(self.config, '_allowance_ratio', 0.5)
        
        # 创建控排企业
        # 异质化配额分配：50%企业配额充足（卖方），50%企业配额偏紧（买方）
        # 模拟真实碳市场：低排放企业有富余配额，高排放企业配额不足
        split_point = int(agent_config.num_emitters * 0.6)  # 60%卖方/40%买方 → 更多卖出流动性
        for i in range(agent_config.num_emitters):
            emission_rate = np.random.uniform(5, 20)
            abatement_cost = np.random.uniform(60, 120)
            total_need = emission_rate * self.config.market.compliance_period
            
            # 异质化配额：60%企业配额充足（卖方），40%企业配额偏紧（买方）
            if i < split_point:
                # 配额充足企业（低排放或历史配额多）
                initial_allowance = total_need * np.random.uniform(1.2, 1.8)
            else:
                # 配额偏紧企业（净买方）
                initial_allowance = total_need * np.random.uniform(0.3, 0.7)
            
            # 认知偏差：以配置值为均值，±30%浮动
            cognitive_bias = {
                "loss_aversion": np.random.uniform(max(1.0, lam_base * 0.7), lam_base * 1.3),
                "anchoring_bias": np.clip(np.random.uniform(alp_base * 0.7, alp_base * 1.3), 0.01, 0.99),
                "herding_tendency": np.clip(np.random.uniform(bet_base * 0.7, bet_base * 1.3), 0.01, 0.99),
                "overconfidence": np.clip(np.random.uniform(max(0.01, gam_base * 0.7), gam_base * 1.3), 0.01, 0.99),
            }
            
            emitter = EmitterAgent(
                agent_id=f"EMITTER_{i:03d}",
                initial_cash=agent_config.emitter_initial_capital * np.random.uniform(0.8, 1.2),
                initial_allowance=initial_allowance,
                emission_rate=emission_rate,
                abatement_cost=abatement_cost,
                compliance_deadline=self.config.market.compliance_period,
                cognitive_bias_params=cognitive_bias,
            )
            self.emitters.append(emitter)
            self.agents.append(emitter)
        
        # 创建投机机构
        # 投机者初始持有部分净缺口配额（提供市场卖出流动性，但不过度）
        total_emitter_deficit = sum(
            max(0, e.emission_rate * self.config.market.compliance_period - e.state.allowance)
            for e in self.emitters
        )
        total_emitter_surplus = sum(
            max(0, e.state.allowance - e.emission_rate * self.config.market.compliance_period)
            for e in self.emitters
        )
        net_deficit = max(0, total_emitter_deficit - total_emitter_surplus)
        # 投机者持有净缺口的30%（避免过度卖出冲击）
        speculator_allowance_each = max(0, net_deficit * 0.3) / max(agent_config.num_speculators, 1)
        
        strategies = ["trend_following", "mean_reversion", "momentum"]
        for i in range(agent_config.num_speculators):
            strategy = strategies[i % len(strategies)]
            
            cognitive_bias = {
                "loss_aversion": np.random.uniform(max(1.0, lam_base * 0.7), lam_base * 1.3),
                "anchoring_bias": np.clip(np.random.uniform(alp_base * 0.7, alp_base * 1.3), 0.01, 0.99),
                "herding_tendency": np.clip(np.random.uniform(bet_base * 0.7, bet_base * 1.3), 0.01, 0.99),
                "overconfidence": np.clip(np.random.uniform(max(0.01, gam_base * 0.7), gam_base * 1.3), 0.01, 0.99),
            }
            
            speculator = SpeculatorAgent(
                agent_id=f"SPEC_{i:03d}",
                initial_cash=agent_config.speculator_initial_capital * np.random.uniform(0.8, 1.2),
                initial_allowance=speculator_allowance_each * np.random.uniform(0.8, 1.2),
                strategy_type=strategy,
                risk_tolerance=np.random.uniform(0.3, 0.7),
                leverage_limit=np.random.uniform(1.5, 3.0),
                cognitive_bias_params=cognitive_bias,
            )
            self.speculators.append(speculator)
            self.agents.append(speculator)
        
        # 创建项目开发商
        for i in range(agent_config.num_developers):
            cognitive_bias = {
                "loss_aversion": np.random.uniform(max(1.0, lam_base * 0.7), lam_base * 1.3),
                "anchoring_bias": np.clip(np.random.uniform(alp_base * 0.7, alp_base * 1.3), 0.01, 0.99),
                "herding_tendency": np.clip(np.random.uniform(bet_base * 0.7, bet_base * 1.3), 0.01, 0.99),
                "overconfidence": np.clip(np.random.uniform(max(0.01, gam_base * 0.7), gam_base * 1.3), 0.01, 0.99),
            }
            
            developer = DeveloperAgent(
                agent_id=f"DEV_{i:03d}",
                initial_cash=agent_config.developer_initial_capital * np.random.uniform(0.8, 1.2),
                cognitive_bias_params=cognitive_bias,
            )
            self.developers.append(developer)
            self.agents.append(developer)
        
        # 创建监管机构
        for i in range(agent_config.num_regulators):
            regulator = RegulatorAgent(
                agent_id=f"REG_{i:03d}",
                inspection_frequency=self.config.policy.inspection_frequency,
                penalty_multiplier=self.config.policy.penalty_multiplier,
                cognitive_bias_params={"loss_aversion": lam_base, "anchoring_bias": alp_base * 0.5, "herding_tendency": bet_base * 0.2, "overconfidence": gam_base * 0.5},
            )
            self.regulators.append(regulator)
            self.agents.append(regulator)
    
    def _prepare_context(self) -> Dict:
        """准备上下文信息"""
        # 计算市场压力指标
        buying_pressure = 0
        selling_pressure = 0
        
        for agent in self.agents:
            if hasattr(agent, 'decision_history') and agent.decision_history:
                last_decision = agent.decision_history[-1]
                if last_decision.get('action') == 'buy':
                    buying_pressure += 1
                elif last_decision.get('action') == 'sell':
                    selling_pressure += 1
        
        total_pressure = buying_pressure + selling_pressure
        if total_pressure > 0:
            buying_pressure /= total_pressure
            selling_pressure /= total_pressure
        
        # 价格趋势
        price_trend = 0
        if len(self.price_history) >= 5:
            price_trend = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5] if self.price_history[-5] > 0 else 0
        
        # 合规率
        compliant_emitters = sum(1 for e in self.emitters if e.compliance_status == "compliant")
        compliance_rate = compliant_emitters / len(self.emitters) if self.emitters else 1.0
        
        return {
            "agents": self.agents,
            "buying_pressure": buying_pressure,
            "selling_pressure": selling_pressure,
            "price_trend": price_trend,
            "num_emitters": len(self.emitters),
            "num_speculators": len(self.speculators),
            "num_developers": len(self.developers),
            "compliance_rate": compliance_rate,
            "current_step": self.current_step,
            "compliance_deadline": self.config.market.compliance_period,
            "emitter_buying_pressure": sum(1 for e in self.emitters if e.decision_history and e.decision_history[-1].get('action') == 'buy') / len(self.emitters) if self.emitters else 0,
            "ccer_liquidity": sum(d.ccer_inventory for d in self.developers),
        }
    
    def step(self) -> Dict:
        """执行单步仿真 - Walrasian价格调整 + CDA撮合"""
        self.current_step += 1
        context = self._prepare_context()
        
        # 第一步：所有智能体做决策
        decisions = []
        for agent in self.agents:
            try:
                # 排放者先累积当期排放（真实排放是物理过程，独立于决策）
                if hasattr(agent, 'calculate_emissions'):
                    agent.calculate_emissions(self.current_step)
                # 同步智能体的内部时钟（仿真步 → 智能体步）
                # 因为仿真直接调用make_decision而非agent.step()，需要手动递增
                if hasattr(agent, 'current_step'):
                    agent.current_step = self.current_step
                market_info = self.market.get_market_info()
                decision = agent.make_decision(market_info, context)
                decisions.append((agent, decision))
            except Exception as e:
                pass
        
        # 第二步：收集供需量（用于Walrasian价格调整）
        total_buy_vol = sum(d.get('quantity', 0) for _, d in decisions if d.get('action') == 'buy')
        total_sell_vol = sum(d.get('quantity', 0) for _, d in decisions if d.get('action') == 'sell')
        
        # 第三步：执行卖单
        step_results = []
        for agent, decision in decisions:
            if decision.get('action') == 'sell':
                try:
                    trades = agent.execute_decision(decision, self.market)
                    step_results.append({"agent_id": agent.agent_id, "trades": len(trades)})
                except Exception:
                    pass
        
        # 第四步：执行买单
        for agent, decision in decisions:
            if decision.get('action') == 'buy':
                try:
                    trades = agent.execute_decision(decision, self.market)
                    step_results.append({"agent_id": agent.agent_id, "trades": len(trades)})
                except Exception:
                    pass
        
        # 第五步：执行其他操作
        for agent, decision in decisions:
            if decision.get('action') not in ['buy', 'sell']:
                try:
                    agent.execute_decision(decision, self.market)
                except Exception:
                    pass
        
        self.market.step()
        
        # 第六步：Walrasian价格调整 + 实证校准的噪声结构
        # 校准方法：三层校准策略 (Three-tier Calibration)
        #   Tier 1: 实证锚定 — σ, df 由真实碳市场数据确定
        #   Tier 2: 网格搜索 — 认知偏差参数 λ, α, β, γ 最小化仿真与数据偏差
        #   Tier 3: 制度参数 — 智能体数量基于中国碳市场制度设计
        # 详见 calibrate_v2.py, analyze_all_markets.py
        base_price = self.market.get_current_price()
        total_vol = total_buy_vol + total_sell_vol
        if total_vol > 0:
            excess_demand_ratio = (total_buy_vol - total_sell_vol) / total_vol
        else:
            excess_demand_ratio = 0
        
        kappa = 0.10  # Walrasian价格调整速度 — 保持较高值以产生末期清算的价格冲击
        
        # ── Tier 1: 实证锚定参数 ──
        sigma = 0.005  # 由方差分解法确定（考虑末期清算的价格冲击后校准）
        
        cb = self.config.agent.cognitive_bias_params
        herding = cb.get("herding_tendency", 0.5)
        anchoring = cb.get("anchoring_bias", 0.3)
        overconfidence = cb.get("overconfidence", 0.2)
        # 认知偏差影响波动率乘数：羊群+过度自信放大，锚定抑制
        # vol_multiplier = 1.0 + β×0.2 + γ×0.15 - α×0.15
        # 默认值: 1.0 + 0.16 + 0.045 - 0.015 = 1.19
        vol_multiplier = 1.0 + herding * 0.2 + overconfidence * 0.15 - anchoring * 0.15
        adjusted_sigma = sigma * vol_multiplier
        
        # 锚定效应降低价格调整速度：高锚定→企业更保守→超额需求对价格影响更小
        # κ_eff = κ × (1 - α×0.3), 默认: 0.15 × (1 - 0.09) = 0.1365
        kappa_eff = kappa * (1.0 - anchoring * 0.3)
        
        # 肥尾噪声：t分布 df=11
        # 依据: CEA收益率t分布拟合 df=11.1 (取整为11)
        #   跨市场对比: CEA=11.1, KRBN=4.0, CEFD=2.8, GRN=3.6
        #   CEA接近正态(Shapiro-Wilk不拒绝), 使用CEA特定df
        # 无跳跃组件: CEA超额峰度=0.70(近正态), 不需要额外跳跃
        #   跨市场跳跃参数(P≈5%, E≈4.7%)会导致150%目标方差
        noise = np.random.standard_t(df=11) * adjusted_sigma
        
        new_price = base_price * (1 + kappa_eff * excess_demand_ratio + noise)
        new_price = max(self.config.market.price_floor, min(self.config.market.price_ceiling, new_price))
        self.price_history.append(new_price)
        
        self.data_collector.collect_step(step=self.current_step, market=self.market, agents=self.agents)
        
        return {"step": self.current_step, "price": new_price, "excess_demand": excess_demand_ratio, "results": step_results}
    
    def run(self, num_steps: Optional[int] = None, progress_bar: bool = True) -> Dict:
        """
        运行仿真
        
        Args:
            num_steps: 仿真步数，默认使用配置值
            progress_bar: 是否显示进度条
        
        Returns:
            仿真结果
        """
        num_steps = num_steps or self.config.simulation.max_steps
        self.is_running = True
        
        iterator = tqdm(range(num_steps), desc="仿真进度") if progress_bar else range(num_steps)
        
        for _ in iterator:
            if not self.is_running:
                break
            self.step()
        
        self.is_running = False
        
        return self.get_results()
    
    def stop(self):
        """停止仿真"""
        self.is_running = False
    
    def get_results(self) -> Dict:
        """获取仿真结果"""
        return {
            "config": self.config.to_dict(),
            "price_history": self.price_history,
            "market_data": self.data_collector.market_data,
            "agent_data": self.data_collector.agent_data,
            "final_agent_states": [agent.get_state_dict() for agent in self.agents],
        }
    
    def save_results(self, output_dir: Optional[str] = None):
        """保存仿真结果"""
        output_dir = output_dir or self.config.simulation.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存配置
        config_path = os.path.join(output_dir, f"config_{timestamp}.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 保存市场数据
        market_df = pd.DataFrame(self.data_collector.market_data)
        market_path = os.path.join(output_dir, f"market_data_{timestamp}.csv")
        market_df.to_csv(market_path, index=False)
        
        # 保存智能体数据
        agent_df = pd.DataFrame(self.data_collector.agent_data)
        agent_path = os.path.join(output_dir, f"agent_data_{timestamp}.csv")
        agent_df.to_csv(agent_path, index=False)
        
        # 保存结果摘要
        results = self.get_results()
        summary = {
            "timestamp": timestamp,
            "total_steps": self.current_step,
            "final_price": self.price_history[-1] if self.price_history else None,
            "price_change": ((self.price_history[-1] - self.price_history[0]) / self.price_history[0] * 100) if len(self.price_history) > 1 else 0,
            "avg_price": np.mean(self.price_history) if self.price_history else 0,
            "price_volatility": np.std(self.price_history) / np.mean(self.price_history) if self.price_history else 0,
            "total_trades": len(self.market.order_book.trades),
        }
        
        summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {output_dir}")
        return output_dir


class DataCollector:
    """数据收集器"""
    
    def __init__(self):
        self.market_data: List[Dict] = []
        self.agent_data: List[Dict] = []
    
    def collect_step(self, step: int, market: CarbonMarket, agents: List[Any]):
        """收集单步数据"""
        # 市场数据
        market_info = market.get_market_info()
        market_record = {
            "step": step,
            **market_info,
        }
        self.market_data.append(market_record)
        
        # 智能体数据
        for agent in agents:
            agent_record = {
                "step": step,
                **agent.get_state_dict(),
            }
            self.agent_data.append(agent_record)


class MonteCarloSimulation:
    """蒙特卡洛仿真"""
    
    def __init__(self, base_config: Optional[Config] = None):
        self.base_config = base_config or config
        self.results: List[Dict] = []
    
    def run(self, num_simulations: Optional[int] = None, progress_bar: bool = True) -> Dict:
        """
        运行蒙特卡洛仿真
        
        Args:
            num_simulations: 仿真次数
            progress_bar: 是否显示进度条
        
        Returns:
            统计结果
        """
        num_simulations = num_simulations or self.base_config.simulation.num_simulations
        
        iterator = tqdm(range(num_simulations), desc="蒙特卡洛仿真") if progress_bar else range(num_simulations)
        
        for i in iterator:
            # 为每次仿真设置不同的随机种子
            sim_config = Config.from_dict(self.base_config.to_dict())
            sim_config.simulation.random_seed = self.base_config.simulation.random_seed + i
            
            # 运行单次仿真
            sim = CarbonMarketSimulation(sim_config)
            sim.run(progress_bar=False)
            
            # 收集结果
            results = sim.get_results()
            self.results.append({
                "simulation_id": i,
                "final_price": results["price_history"][-1] if results["price_history"] else 0,
                "price_change": ((results["price_history"][-1] - results["price_history"][0]) / results["price_history"][0] * 100) if len(results["price_history"]) > 1 else 0,
                "avg_price": np.mean(results["price_history"]) if results["price_history"] else 0,
                "price_volatility": np.std(results["price_history"]) / np.mean(results["price_history"]) if results["price_history"] else 0,
                "total_trades": len(sim.market.order_book.trades),
            })
        
        return self._calculate_statistics()
    
    def _calculate_statistics(self) -> Dict:
        """计算统计结果"""
        if not self.results:
            return {}
        
        final_prices = [r["final_price"] for r in self.results]
        price_changes = [r["price_change"] for r in self.results]
        volatilities = [r["price_volatility"] for r in self.results]
        
        return {
            "num_simulations": len(self.results),
            "final_price": {
                "mean": np.mean(final_prices),
                "std": np.std(final_prices),
                "min": np.min(final_prices),
                "max": np.max(final_prices),
                "median": np.median(final_prices),
                " VaR_95": np.percentile(final_prices, 5),  # 95% VaR
                " VaR_99": np.percentile(final_prices, 1),  # 99% VaR
            },
            "price_change": {
                "mean": np.mean(price_changes),
                "std": np.std(price_changes),
                "min": np.min(price_changes),
                "max": np.max(price_changes),
            },
            "volatility": {
                "mean": np.mean(volatilities),
                "std": np.std(volatilities),
            },
            "raw_results": self.results,
        }
