# Carbon Emission Trading Markets — Agent-Based Model

An open-source agent-based model (ABM) for carbon emission trading markets that integrates heterogeneous boundedly-rational agents with empirically calibrated cognitive biases.

## Overview

This software implements a four-layer carbon market simulation system:

1. **Heterogeneous Agent Layer** — Four participant types (emitters, speculators, developers, regulators) with boundedly-rational decision rules
2. **Market Mechanism Layer** — Continuous double auction (CDA) with Walrasian price adjustment
3. **Behavioral Decision Layer** — Prospect-theoretic utility functions modulated by four cognitive biases (loss aversion, anchoring, herding, overconfidence)
4. **Analysis Layer** — Risk metrics (VaR, CVaR, MDD), ablation studies, Monte Carlo validation, and stress tests

The model is calibrated against China's national carbon allowance (CEA) data using a three-tier strategy that decomposes the parameter space by empirical identifiability.

## Key Features

- **Empirical calibration**: Three-tier strategy (variance decomposition → grid search → institutional design)
- **Cognitive bias modeling**: Mathematically formalized loss aversion (λ), anchoring (α), herding (β), overconfidence (γ)
- **Continuous double auction**: Limit order book with price-time priority matching
- **Monte Carlo robustness**: Configurable multi-run uncertainty quantification
- **Ablation studies**: Isolate individual cognitive bias contributions
- **Stress testing**: Demand, supply, liquidity, and enforcement shock scenarios
- **Cross-market data**: Included calibration data for China CEA, KRBN, CEFD, and GRN markets

## Installation

```bash
git clone https://github.com/AicbLab/Carbon-Emission-Trading-Markets.git
cd Carbon-Emission-Trading-Markets
pip install -r requirements.txt
```

**Dependencies**: Python ≥ 3.8, NumPy, Pandas, SciPy, Matplotlib, Seaborn, tqdm

## Quick Start

```python
from carbon_market_abm import CarbonMarketSimulation, Config

# Create configuration
config = Config()
config.simulation.random_seed = 42
config.market.initial_price = 50.0

# Run simulation
sim = CarbonMarketSimulation(config)
sim.run(num_steps=252, progress_bar=True)

# Access results
print(f"Final price: {sim.price_history[-1]:.2f}")
print(f"Total trades: {sim.total_trades}")
```

## Usage

### Run a full experiment

```bash
python run_simulation.py
```

### Run all experiments (calibration, Monte Carlo, ablation, stress tests)

```bash
python run_full_experiments.py
```

### Generate paper figures

```bash
python generate_figures.py
```

### Custom simulation

```python
from carbon_market_abm import Config, CarbonMarketSimulation

config = Config()

# Adjust agent populations
config.agent.num_emitters = 30
config.agent.num_speculators = 15

# Adjust cognitive bias parameters
config.cognitive_bias.loss_aversion = 2.25
config.cognitive_bias.anchoring = 0.10
config.cognitive_bias.herding = 0.80
config.cognitive_bias.overconfidence = 0.20

# Adjust market parameters
config.market.initial_price = 60.0
config.market.total_allowance = 10000

sim = CarbonMarketSimulation(config)
sim.run(num_steps=252)
```

### Stress testing

```python
from carbon_market_abm import Config, CarbonMarketSimulation, StressTest

config = Config()
sim = CarbonMarketSimulation(config)

stress = StressTest()
results = stress.run_all_scenarios(sim)
```

## Project Structure

```
├── carbon_market_abm/          # Core package
│   ├── agents/                 # Agent implementations
│   │   ├── base_agent.py       # Base agent class
│   │   ├── emitter.py          # Regulated firms (emitters)
│   │   ├── speculator.py       # Financial intermediaries
│   │   ├── developer.py        # CCER project developers
│   │   └── regulator.py        # Market regulator
│   ├── config.py               # Configuration management
│   ├── market.py               # CDA market mechanism
│   ├── simulation.py           # Simulation engine
│   ├── risk_metrics.py         # VaR, CVaR, MDD, stress tests
│   ├── visualization.py        # Plotting utilities
│   ├── llm_client.py           # Optional LLM integration
│   └── main.py                 # CLI entry point
├── data/                       # Calibration data & results
│   ├── china_cea_daily.csv     # China CEA daily prices
│   ├── KRBN_daily.csv          # KRBN (US carbon) prices
│   ├── CEFD_daily.csv          # CEFD prices
│   ├── GRN_daily.csv           # GRN (European) prices
│   └── *.json                  # Calibration statistics
├── run_simulation.py           # Quick start script
├── run_full_experiments.py     # Full experiment suite
├── generate_figures.py         # Paper figure generation
├── generate_tables.py          # Paper table generation
└── requirements.txt            # Python dependencies
```

## Calibrated Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Loss aversion | λ | 2.25 | Prospect-theoretic loss aversion |
| Anchoring bias | α | 0.10 | Weight on historical price anchor |
| Herding tendency | β | 0.80 | Weight on market consensus |
| Overconfidence | γ | 0.20 | Private signal overestimation |
| Noise level | σ₀ | 0.005 | Base volatility |
| Tail thickness | df | 11 | Student-t degrees of freedom |
| Walrasian speed | κ | 0.10 | Price adjustment speed |

## Calibration Accuracy

| Metric | Real CEA | Calibrated ABM | Error |
|--------|----------|----------------|-------|
| Mean price (CNY/ton) | 70.47 | 75.74 | 7.5% |
| Annualized volatility | 21.0% | 22.3% | 6.2% |
| Maximum drawdown | -26.9% | -14.3% | 47.0% |
| VaR (95%) | -2.14% | -1.39% | 35.0% |

## Citation

If you use this software in your research, please cite:

> Yin, M. & Li, Z. (2026). Cognitive Biases and Price Formation in Carbon Emission Trading Markets: An Empirically Calibrated Agent-Based Model.

## License

This project is released for academic and research use.
