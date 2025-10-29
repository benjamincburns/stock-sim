# Portfolio Stress Test Simulation

A Monte Carlo simulation tool for comparative stress testing of long term passive ETF portfolios.

This tool was primarily built to compare the relative performance of various rebalancing strategies under varying market shock scenarios. It does not and can not accurately predict actual returns, price movements, risk, return timelines, or any other performance metrics. It is only useful when comparing the performance outcomes of one strategy to the outcomes of another, and even then it's only likely to be informative in the case where the magnitude of those differences is substantial.

> [!WARNING]
> This tool is for educational and planning purposes only. It is NOT financial advice. The default statistical assumptions encoded in this tool and its configuration may vary widely from actual market dynamics. Use at your own risk.

> [!NOTE]
> This project was largely written by AI. While I have supervised carefully and tested manually to ensure that it conforms to a range of verifiable assumptions, I haven't exhaustively checked for bugs or validated the coverage or quality of the AI-written unit tests. Some portions of this document were also AI-generated, but it is mostly human-written, and the AI-generated portions have been closely scrutinized.

## Simulation Dynamics

The simulation engine uses the Monte Carlo method, which means it runs a configurable number (default 1,000) of independent simulations, using a different random seed during each simulation run. This allows the model to explore a wide range of potential outcomes and produce a statistical distribution of the final portfolio performance and risk metrics.

For each simulation, monthly price movements are generated for each asset by sampling a multivariate normal distribution. The shape of this distribution is defined by configurable volatility metrics for each asset and a correlation matrix that describes how the prices of assets tend to move relative to one another. These parameters are configurable for each scenario phase in the timeline (crash, recovery, stable growth). The default values of these parameters are intended to simulate common historical trends. For example, the defaults assume that investors will race to safety during a crash, causing stocks to correlate more strongly with eachother than usual, while high-quality government bonds become less correlated with stocks.

Each simulation then progresses month by month over the specified time horizon using the generated price data. The simulation begins with the initial investment allocated according to the target weights. Then for each month, the simulation performs actions in a specific order: first, the monthly contribution is added and allocated to purchase assets according to the selected rebalancing strategy. Second, the portfolio is checked against its target allocations to determine if a rebalancing trade is necessary. If it is, a rebalancing event is recorded. Finally, the pre-generated market returns for that month are applied to the new holdings to determine the portfolio's value at the end of the month.

To add realism, the simulation includes an optional "stochastic recovery" feature that is enabled by default. When enabled, the recovery target for each asset is randomized within a configurable range, preventing the recovery phase from being perfectly predictable. The model also uses geometric mean calculations to derive the average monthly returns for the crash and recovery phases, ensuring that the total returns for those periods align with the scenario's targets. To avoid extreme, unrealistic single-month price movements, all generated monthly returns are clipped to a [-30%, +30%] range.

## Rebalancing Strategies

Rebalancing is the process of trading the existing assets in a portfolio to make the proportion of each asset align with a desired target allocation.

For example, suppose you decide you want 70% stocks and 30% bonds. Over time, the stocks grow more quickly, and your portfolio drifts to 80% stocks and 20% bonds. Rebalancing is how you correct this drift.

A multitude of rebalancing strategies exist, however this simulator implements two particular strategies. You can select the strategy to be used for your simulation by passing command line flags at runtime (see `--rebalancing-strategy` in output of `python sim.py --help`).

### Tight, Symmetric Rebalancing

In this strategy, the target allocation for each asset is banded by an absolute 5%. For example, a 10% target would be "in-band" if the asset represents between 5% and 15% of the total portfolio value. If an asset's allocation falls outside this band, rebalancing occurs as follows:
- The monthly contribution is used to buy assets that are "underweight," prioritizing those furthest below their target.
- If any assets are still out of band after contributions, "overweight" assets are sold, and the proceeds are used to buy underweight assets to bring them back to their target.

This strategy is symmetric, as it can trigger sell orders regardless of whether an asset is below or above its band.

### Loose, Asymmetric Rebalancing (default)

In this "asymmetric" strategy, the rebalancing band is relative to the target allocation. Specifically, the lower bound is 50% of the target, and the upper bound is 200% of the target. For example, if an asset's target is 10%, its allocation is considered in-band between 5% and 20%. For a 20% target, the band would be 10% to 40%. The rebalancing rules are as follows:

- Weight buy orders from monthly contributions based on how far each asset is "underweight" (below its target percentage).
- If an asset's proportion rises above its upper band, sell enough of it to bring it back into band.
- If an asset's proportion falls below its lower band, do nothing. We rely on future buy orders to bring the asset back into band over time.

This strategy is asymmetric because it only triggers sell orders in order to prevent excessive exposure to assets or asset classes that are likely overvalued.

Compared to the tight, symmetric strategy above, it is more cost-optimized, as it triggers fewer sells, and fewer total orders overall. It also tends to capture more upside during bull market trends, and limits drawdowns during bear market trends.

This strategy introduces an important behavioral dependency that comes with a hidden risk: it is only effective if the investor consistently makes contributions that are large enough to correct portfolio drift. The simulation assumes this behavior occurs without fail and does not model the risk of portfolio drift resulting from missed contributions.

## Configuration

See `config.yaml` for all configurable parameters.

## Quick Start

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

# Find time to reach $500K (VaR 95% confidence aka 5th percentile expected returns)
python sim.py --goal-p5 --goal-target 500000

# Find time to reach $500K (median outcome, 50% confidence)
python sim.py --goal-median --goal-target 500000
```

### View All Options

```bash
python sim.py --help
```

## 🛠️ Development

### Running Tests
The project includes a test suite that can be run with `pytest`.

```bash
# Run the full test suite
pytest
```

## 🎓 Educational Use

This tool can be used for learning about:
- Monte Carlo simulation techniques
- Portfolio risk analysis
- Correlation effects in crashes
- Rebalancing strategies
- Risk-adjusted returns (Sharpe ratio)
- Tail risk (VaR/CVaR)
- Sequence-of-returns risk

**Model Limitations:**
- Simplified return distributions (assumes normality)
- Fixed correlation structures
- No transaction costs or taxes
- No inflation adjustment
- Assumed crash/recovery patterns

## 📄 License

This project is MIT licensed. See [LICENSE](LICENSE.md) for license text.

## 🙏 Acknowledgments

Built with:
- **NumPy** - Numerical computing
- **Pandas** - Data structures
- **Numba** - JIT compilation
- **pyxirr** - IRR calculations
- **PyYAML** - Configuration management
- **pytest** - Testing

