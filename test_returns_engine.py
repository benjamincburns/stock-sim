"""
Unit tests for returns_engine module.

These tests verify that return generation and metrics calculation work correctly
with explicit configuration (no globals).
"""
import pytest
import numpy as np
import pandas as pd
from config import (
    SimulationConfig, ScenarioConfig, AssetConfig,
    create_asset_config, load_config_from_yaml
)
from returns_engine import (
    calculate_geometric_monthly_return,
    generate_correlated_returns,
    generate_scenario_returns_array,
    _generate_crash_returns,
    _generate_recovery_returns,
    _generate_growth_returns
)
from simulation_engine import calculate_portfolio_metrics


@pytest.fixture(scope="class")
def asset_config():
    """Fixture to provide a default AssetConfig for tests."""
    app_config = load_config_from_yaml()
    return create_asset_config(app_config)


class TestGeometricMonthlyReturn:
    """Test geometric monthly return calculation."""
    
    def test_positive_return(self):
        """Test normal positive return calculation."""
        # 100% total return over 12 months should be ~5.95% per month
        monthly_ret = calculate_geometric_monthly_return(np.array([1.0]), 12)[0]
        assert 0.059 < monthly_ret < 0.060
        
        # Verify it compounds back to ~100%
        total = (1 + monthly_ret)**12 - 1
        assert 0.99 < total < 1.01
    
    def test_negative_return(self):
        """Test negative return calculation."""
        # -50% total return over 12 months
        monthly_ret = calculate_geometric_monthly_return(np.array([-0.5]), 12)[0]
        assert monthly_ret < 0
        
        # Verify it compounds back to ~-50%
        total = (1 + monthly_ret)**12 - 1
        assert -0.51 < total < -0.49
    
    def test_extreme_loss_capped(self):
        """Test that extreme losses are capped."""
        # -150% total return (impossible but test safety)
        # Should be clipped at the function level now, not just capped
        monthly_ret = calculate_geometric_monthly_return(np.array([-1.5]), 12)[0]
        assert np.isclose(monthly_ret, -0.30) # Clipped to -30%
    
    def test_zero_duration(self):
        """Test that zero duration returns 0."""
        monthly_ret = calculate_geometric_monthly_return(np.array([0.5]), 0)[0]
        assert monthly_ret == 0.0
    
    def test_large_positive_capped(self):
        """Test that very large positive returns are capped at 30% per month."""
        # 1000% return in 1 month would be 1000%, but should be capped at 30%
        monthly_ret = calculate_geometric_monthly_return(np.array([10.0]), 1)[0]
        assert monthly_ret == 0.30


class TestCorrelatedReturns:
    """Test correlated return generation."""
    
    def test_shape(self):
        """Test that output has correct shape."""
        num_assets = 3
        num_periods = 120
        mean_returns = np.array([0.01, 0.02, 0.03])
        std_devs = np.array([0.05, 0.06, 0.07])
        corr_matrix = np.eye(num_assets)  # Identity = uncorrelated
        
        returns = generate_correlated_returns(mean_returns, std_devs, corr_matrix, num_periods)
        
        assert returns.shape == (num_periods, num_assets)
    
    def test_means_approximate(self):
        """Test that generated returns have approximately correct means."""
        np.random.seed(42)
        num_periods = 10000
        mean_returns = np.array([0.01, -0.005, 0.02])
        std_devs = np.array([0.05, 0.05, 0.05])
        corr_matrix = np.eye(3)
        
        returns = generate_correlated_returns(mean_returns, std_devs, corr_matrix, num_periods)
        
        # With 10000 samples, means should be close
        actual_means = returns.mean(axis=0)
        np.testing.assert_array_almost_equal(actual_means, mean_returns, decimal=2)
    
    def test_correlation_structure(self):
        """Test that correlation structure is preserved."""
        np.random.seed(42)
        num_periods = 10000
        mean_returns = np.zeros(2)
        std_devs = np.array([0.05, 0.05])
        # High positive correlation
        corr_matrix = np.array([[1.0, 0.9], [0.9, 1.0]])
        
        returns = generate_correlated_returns(mean_returns, std_devs, corr_matrix, num_periods)
        
        # Calculate actual correlation
        actual_corr = np.corrcoef(returns.T)
        
        # Should be close to target correlation
        assert abs(actual_corr[0, 1] - 0.9) < 0.05


@pytest.mark.usefixtures("asset_config")
class TestScenarioReturns:
    """Test scenario-based return generation."""
    
    def test_no_crash_scenario(self, asset_config):
        """Test that no-crash scenario generates only growth returns."""
        sim_config = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=10,
            months=120,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=100,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        scenario = ScenarioConfig("No Crash", 0, 0)
        
        returns_array, tickers = generate_scenario_returns_array(scenario, sim_config, asset_config, seed=42)
        
        # Should have exactly 120 months of returns
        assert returns_array.shape[0] == 120
        
        # Should have returns for all assets
        assert tickers == asset_config.asset_tickers_ordered
        
        # All values should be numeric
        assert not np.isnan(returns_array).any()
    
    def test_crash_scenario_phases(self, asset_config):
        """Test that crash scenario has correct number of periods."""
        # 18 month crash, 48 month recovery, rest growth
        scenario = ScenarioConfig("Severe Crash", 18, 48)

        sim_config = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=10,
            months=120,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=100,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        
        returns_array, tickers = generate_scenario_returns_array(scenario, sim_config, asset_config, seed=42)
        
        # Should have exactly 120 months total
        assert returns_array.shape[0] == 120
        
        # First 18 months should be crash (negative for stocks, positive for bonds)
        crash_period = returns_array[:18, :]
        voo_idx = tickers.index('VOO')
        govt_idx = tickers.index('GOVT')

        # VOO should decline during crash
        voo_crash_performance = np.prod(1 + crash_period[:, voo_idx])
        assert voo_crash_performance < 1.0  # Should lose money
        
        # GOVT should rise during crash
        govt_crash_performance = np.prod(1 + crash_period[:, govt_idx])
        assert govt_crash_performance > 1.0  # Should gain money
    
    def test_seed_reproducibility(self, asset_config):
        """Test that same seed produces same returns."""
        scenario = ScenarioConfig("Test", 6, 18)

        sim_config = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=5,
            months=60,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=100,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        
        returns1, _ = generate_scenario_returns_array(scenario, sim_config, asset_config, seed=123)
        returns2, _ = generate_scenario_returns_array(scenario, sim_config, asset_config, seed=123)
        
        np.testing.assert_array_equal(returns1, returns2)
    
    def test_different_seeds_different_returns(self, asset_config):
        """Test that different seeds produce different returns."""
        scenario = ScenarioConfig("Test", 6, 18)

        sim_config = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=5,
            months=60,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=100,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        
        returns1, _ = generate_scenario_returns_array(scenario, sim_config, asset_config, seed=123)
        returns2, _ = generate_scenario_returns_array(scenario, sim_config, asset_config, seed=456)
        
        # Should be different
        assert not np.array_equal(returns1, returns2)


class TestPortfolioMetrics:
    """Test portfolio metrics calculation."""
    
    def test_simple_growth(self):
        """Test metrics for simple steady growth."""
        # Portfolio that doubles in value
        portfolio_values = np.array([100000, 110000, 121000, 133100, 146410, 161051], dtype=float)
        
        sim_config = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=0,  # No contributions for simplicity
            years=0.5,
            months=5,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=100,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        
        metrics = calculate_portfolio_metrics(portfolio_values, sim_config, rebalance_count=0)
        
        # Final value should be last value
        assert metrics["Final Value"] == 161051
        
        # Should have positive return
        assert metrics["Annualized Return (%)"] > 0
        
        # Should have some volatility (though small with steady growth)
        assert metrics["Annualized Volatility (%)"] >= 0
        
        # Max drawdown should be 0 for steady growth
        assert metrics["Max Drawdown (%)"] == 0
    
    def test_with_drawdown(self):
        """Test that drawdown is correctly calculated."""
        # Portfolio that goes up then down
        portfolio_values = np.array([100000, 120000, 110000, 130000, 125000], dtype=float)
        
        sim_config = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=0,
            years=0.33,
            months=4,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=100,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        
        metrics = calculate_portfolio_metrics(portfolio_values, sim_config, rebalance_count=0)
        
        # Max drawdown should be from 120k to 110k = -8.33%
        # Or from 130k to 125k = -3.85%
        # So max should be around -8.33%
        assert -9 < metrics["Max Drawdown (%)"] < -8
    

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

