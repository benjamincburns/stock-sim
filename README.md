# Portfolio Stress Test Simulation

A high-performance Monte Carlo simulation tool for portfolio stress testing with advanced statistical modeling and goal-seeking capabilities.

> [!WARNING]
> I wouldn't recommend using this without reading the code very carefully. It's something I knocked together to compare the relative performance of a few different long-term investment strategies that I've been considering. It was largely written by AI, as may be evident from the verbose comments.

This tool stress-tests simple ETF portfolios with configurable rebalancing strategies by simulating a market crash, followed by a market recovery, and then a period of stable growth. It does not consider taxes or trade commissions, but it does produce estimated counts of sell trades (see rebalancing count in output), which works as a proxy for these drags on simple passive portfolios.

## Why?

This tool was built to compare the performance of two alternative rebalancing strategies.

Rebalancing is the process of trading the existing assets in a portfolio to make the proportion of each asset align with a desired target allocation.

For example, suppose you decide you want 70% stocks and 30% bonds. Over time, the stocks grow more quickly, and your portfolio drifts to 80% stocks and 20% bonds. Rebalancing is how you correct this drift.

A multitude of rebalancing strategies exist, however this simulator was intended to test two particular strategies. You can select the strategy to be used for your simulation by passing command line flags at runtime (see output of `python sim.py --help`).

### Rebalancing Strategy 1: Tight rebalancing

In this strategy, the target allocation for each asset is banded by an absolute 5%. For example, a 10% target would be "in-band" if the asset represents between 5% and 15% of the total portfolio value. If an asset's allocation falls outside this band, rebalancing occurs as follows:
- The monthly contribution is used to buy assets that are "underweight," prioritizing those furthest below their target.
- If any assets are still out of band after contributions, "overweight" assets are sold, and the proceeds are used to buy underweight assets to bring them back to their target.

This strategy is symmetric, as it can trigger sell orders regardless of whether an asset is below or above its band.

### Rebalancing Strategy 2: Loose rebalancing (default)

In this "asymmetric" strategy, the rebalancing band is relative to the target allocation. Specifically, the lower bound is 50% of the target, and the upper bound is 200% of the target. For example, if an asset's target is 10%, its allocation is considered in-band between 5% and 20%. For a 20% target, the band would be 10% to 40%. The rebalancing rules are as follows:

- Weight buy orders from monthly contributions based on how far each asset is "underweight" (below its target percentage).
- If an asset's proportion rises above its upper band, sell enough of it to bring it back into band.
- If an asset's proportion falls below its lower band, do nothing. We rely on future buy orders to bring the asset back into band over time.

This strategy is asymmetric because it only triggers sell orders when assets are significantly overvalued relative to the portfolio. It is more cost-optimized, triggering fewer total orders and taxable events. For this strategy to work effectively, you must be making regular portfolio contributions.

## Configuration

See `config.yaml` for all configurable parameters.

Price movements are produced by sampling a random probability distribution. The distribution for each asset is generated using static volatility and correlation matrices that are configurable for each phase of the test (crash, recovery, stable growth). The defaults simulate common historical trends, such as the high correlation of stocks during a market crash.

## 🚀 Features

- **📊 Comprehensive Risk Analysis:** VaR, CVaR, Sharpe ratio, and drawdown analysis.
- **🎯 Goal-Seeking Mode:** Find the time required to reach financial targets.
- **🔬 Advanced Statistical Modeling:** Simulates market crashes and recoveries with proper correlation structures.
- **⚙️ Fully Configurable:** A command-line interface for all parameters.
- **📈 Multiple Scenarios:** Test portfolios against severe, moderate, rapid, and no-crash scenarios.
- **⚡ High Performance:** Numba-optimized for simulations that are 10-50x faster than pure Python.

## 📋 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with defaults (15 years, $100K initial, $1K/month)
python sim.py

# Custom parameters
python sim.py --years 20 --monthly-contribution 2000

# Find time to reach $500K (VaR 95% confidence)
python sim.py --goal-p5 --goal-target 500000

# Find time to reach $500K (median outcome)
python sim.py --goal-median --goal-target 500000
```

### View All Options

```bash
python sim.py --help
```

## 🎯 Common Use Cases

### 1. Retirement Planning

Find how long until you can retire with $750K (95% confidence):

```bash
python sim.py --goal-p5 \
              --goal-target 750000 \
              --initial-investment 100000 \
              --monthly-contribution 2000
```

### 2. Compare Different Savings Rates

```bash
# Conservative
python sim.py --monthly-contribution 1000 > results_1k.txt

# Moderate
python sim.py --monthly-contribution 1500 > results_1.5k.txt

# Aggressive
python sim.py --monthly-contribution 2500 > results_2.5k.txt
```

### 3. High-Precision Analysis

```bash
python sim.py --num-simulations 10000
```

### 4. Quick Testing

```bash
python sim.py --num-simulations 100 --years 10
```

## 📊 Sample Output

The simulation provides two types of analysis:

### Standard Mode Output
- Portfolio compositions
- Monte Carlo results across 4 scenarios
- Performance metrics: returns, volatility, Sharpe ratio
- Risk metrics: VaR, CVaR, drawdowns
- Rebalancing statistics

### Goal-Seeking Mode Output
- Time to reach financial target for each portfolio
- Results by scenario (crash timing sensitivity)
- Final values and safety cushions at goal

## 🔬 Methodology & Performance

### Simulation Engine
- **Monte Carlo Method:** Runs 1,000 simulations per scenario (configurable) to model a wide range of outcomes.
- **Return Generation:** Uses a multivariate normal distribution to generate monthly returns, preserving the specified correlation structure for each market phase.
- **Optimization:** Numba JIT compilation provides a 10-50x speedup over pure Python. The engine also relies on vectorized NumPy operations and multiprocessing for parallel scenario execution.

### Scenarios Tested
1.  **Severe Crash:** 18-month crash, 48-month recovery
2.  **Moderate Crash:** 12-month crash, 36-month recovery
3.  **Rapid Crash:** 6-month crash, 18-month recovery (V-shape)
4.  **No Crash:** Steady growth over the entire period

### Risk Metrics
- **VaR (Value at Risk):** 95th percentile confidence level (p5 value in report).
- **CVaR (Conditional VaR):** Average of the worst 5% of outcomes.
- **Sharpe Ratio:** A measure of risk-adjusted return.
- **Maximum Drawdown:** The largest peak-to-trough decline in portfolio value.

### Performance Benchmarks
*(on an 8-core M1 Pro CPU)*
- **Standard Mode:** 4,000 simulations in ~6 seconds
- **Goal-Seeking Mode:** ~30-60 seconds per portfolio/scenario
- **High Precision (10K sims):** ~60 seconds

## 🛠️ Development

### Running Analyses
You can run quick analyses with fewer simulations for faster feedback.

```bash
# Quick run with 100 simulations
python sim.py --num-simulations 100

# Run goal-seeking with 100 simulations
python sim.py --goal-p5 --goal-target 500000 --num-simulations 100
```

### Running Tests
The project includes a test suite that can be run with `pytest`.

```bash
# Run the full test suite
pytest
```

## 🎓 Educational Use

This tool is excellent for learning about:
- Monte Carlo simulation techniques
- Portfolio risk analysis
- Correlation effects in crashes
- Rebalancing strategies
- Risk-adjusted returns (Sharpe ratio)
- Tail risk (VaR/CVaR)
- Sequence-of-returns risk

## ⚠️ Disclaimer

This tool is for educational and planning purposes only. It is NOT financial advice. Consult a qualified financial advisor before making investment decisions.

**Model Limitations:**
- Simplified return distributions (assumes normality)
- Fixed correlation structures
- No transaction costs or taxes
- No inflation adjustment
- Assumed crash/recovery patterns

## 📄 License

This project is unlicensed, but you are free to use it for educational and personal purposes. For broader use, consider adding a standard open-source license like the [MIT License](https://opensource.org/licenses/MIT).

## 🙏 Acknowledgments

Built with:
- **NumPy** - Numerical computing
- **Pandas** - Data structures
- **Numba** - JIT compilation
- **pyxirr** - IRR calculations
- **PyYAML** - Configuration management
- **pytest** - Testing

