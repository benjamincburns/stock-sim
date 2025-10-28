"""
Tests for the refactored simulation engine.

These tests verify that the refactored code:
1. Works correctly with explicit parameter passing
2. Produces consistent results with given seeds
3. Doesn't suffer from the multiprocessing global variable bug
"""
import pytest
import numpy as np
import multiprocessing
import time

from config import (
    SimulationConfig, ScenarioConfig, AssetConfig,
    load_config_from_yaml, create_asset_config, create_scenarios
)
from simulation_engine import run_single_mc_iteration_refactored, run_single_mc_iteration_wrapper


# --- Test Setup ---
# Load base configuration from YAML for all tests
APP_CONFIG = load_config_from_yaml('config.yaml')
ASSET_CONFIG = create_asset_config(APP_CONFIG)
SCENARIOS = create_scenarios(APP_CONFIG)

# Test portfolios
TEST_PORTFOLIOS = {
    "Test_60_40": {"VOO": 0.60, "GOVT": 0.40},
    "Test_100": {"VOO": 1.0},
}


class TestSingleIteration:
    """Test single Monte Carlo iteration."""
    
    def test_basic_execution(self):
        """Test that a single iteration runs without error."""
        sim_config = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=10,
            months=120,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=10,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        scenario = SCENARIOS["No Crash (0m/0m)"]
        
        scenario_name, results = run_single_mc_iteration_refactored(
            run_index=0,
            seed=42,
            scenario=scenario,
            sim_config=sim_config,
            asset_config=ASSET_CONFIG,
            portfolios=TEST_PORTFOLIOS
        )
        
        # Should return scenario name
        assert scenario_name == "No Crash (0m/0m)"
        
        # Should have results for both portfolios
        assert "Test_60_40" in results
        assert "Test_100" in results
        
        # Results should have expected keys
        for portfolio_result in results.values():
            if portfolio_result is not None:
                assert "Final Value" in portfolio_result
                assert "Annualized Return (%)" in portfolio_result
                assert "Sharpe Ratio" in portfolio_result
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        sim_config = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=5,
            months=60,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=10,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        scenario = SCENARIOS["Rapid Crash (6m/18m)"]
        
        _, results1 = run_single_mc_iteration_refactored(
            0, 123, scenario, sim_config, ASSET_CONFIG, TEST_PORTFOLIOS
        )
        _, results2 = run_single_mc_iteration_refactored(
            0, 123, scenario, sim_config, ASSET_CONFIG, TEST_PORTFOLIOS
        )
        
        # Results should be identical
        for portfolio_name in TEST_PORTFOLIOS.keys():
            if results1[portfolio_name] is not None and results2[portfolio_name] is not None:
                assert results1[portfolio_name]["Final Value"] == results2[portfolio_name]["Final Value"]
    
    def test_crash_vs_no_crash(self):
        """Test that crash scenarios produce different results than no-crash."""
        sim_config = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=10,
            months=120,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=10,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        
        no_crash = SCENARIOS["No Crash (0m/0m)"]
        severe_crash = SCENARIOS["Severe Crash (18m/48m)"]
        
        _, results_no_crash = run_single_mc_iteration_refactored(
            0, 42, no_crash, sim_config, ASSET_CONFIG, TEST_PORTFOLIOS
        )
        _, results_crash = run_single_mc_iteration_refactored(
            0, 42, severe_crash, sim_config, ASSET_CONFIG, TEST_PORTFOLIOS
        )
        
        # With same seed, crash should generally produce lower final values
        # (though not guaranteed due to stochastic nature)
        # At minimum, they should be different
        assert results_no_crash["Test_100"]["Final Value"] != results_crash["Test_100"]["Final Value"]


class TestMultiprocessingBugFix:
    """Test that the refactored code fixes the multiprocessing global variable bug."""
    
    def test_parameter_propagation(self):
        """Test that parameters are correctly passed to worker processes."""
        # Create two different configs
        config_short = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=5,
            months=60,  # SHORT: 5 years
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=10,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        
        config_long = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=20,
            months=240,  # LONG: 20 years
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=10,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        
        scenario = SCENARIOS["No Crash (0m/0m)"]
        
        # Prepare arguments for worker processes - explicitly passing configs
        args_short = [
            (i, 42 + i, scenario, config_short, ASSET_CONFIG, TEST_PORTFOLIOS)
            for i in range(4)
        ]
        
        args_long = [
            (i, 42 + i, scenario, config_long, ASSET_CONFIG, TEST_PORTFOLIOS)
            for i in range(4)
        ]
        
        # Run in multiprocessing pool
        with multiprocessing.Pool(processes=2) as pool:
            results_short = pool.map(run_single_mc_iteration_wrapper, args_short)
            results_long = pool.map(run_single_mc_iteration_wrapper, args_long)
        
        # Extract final values
        short_values = [r[1]["Test_100"]["Final Value"] for r in results_short]
        long_values = [r[1]["Test_100"]["Final Value"] for r in results_long]
        
        # Long simulations should have significantly higher final values
        # (20 years vs 5 years with contributions and growth)
        avg_short = np.mean(short_values)
        avg_long = np.mean(long_values)
        
        print(f"Average final value (5 years): ${avg_short:,.0f}")
        print(f"Average final value (20 years): ${avg_long:,.0f}")
        
        # 20-year simulation should have much higher values than 5-year
        assert avg_long > avg_short * 2, \
            f"Long simulation should be much larger: {avg_long:.0f} vs {avg_short:.0f}"
        
        # Verify that long values are in reasonable range for 20 years
        # With $100k initial + $1k/month for 20 years, should have at least $340k invested
        # With growth, should be significantly more
        assert avg_long > 400000, \
            f"20-year simulation too low: ${avg_long:,.0f} (expected > $400k)"


class TestRealisticSimulation:
    """Test with realistic parameters to verify correctness."""
    
    def test_100_year_simulation_produces_millions(self):
        """
        Test that 100-year simulation produces values in millions, not thousands.
        
        This is the key test that would have caught the original bug!
        """
        sim_config = SimulationConfig(
            initial_investment=170000,
            monthly_contribution=2000,
            years=100,
            months=1200,  # 100 years
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=10,
            base_seed=42,
            use_stochastic_recovery=False,
            recovery_uncertainty=0.15,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        
        scenario = SCENARIOS["No Crash (0m/0m)"]
        
        _, results = run_single_mc_iteration_refactored(
            0, 42, scenario, sim_config, ASSET_CONFIG, TEST_PORTFOLIOS
        )
        
        final_value = results["Test_100"]["Final Value"]
        
        print(f"\n100-year simulation final value: ${final_value:,.0f}")
        print(f"Total invested: ${sim_config.total_invested:,.0f}")
        
        # With 9% annual returns over 100 years, $170k should grow to ~$900M
        # Even with monthly contributions, should be in hundreds of millions
        assert final_value > 1_000_000, \
            f"100-year simulation should be in millions, not thousands! Got: ${final_value:,.0f}"
        
        # Should be significantly more than total invested
        assert final_value > sim_config.total_invested * 10, \
            f"Should have much more than invested: ${final_value:,.0f} vs ${sim_config.total_invested:,.0f}"

    def test_asymmetric_rebalancing_produces_different_results(self):
        """Test that asymmetric rebalancing produces different results."""
        sim_config_standard = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=20,
            months=240,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=10,
            base_seed=42,
            use_stochastic_recovery=True,
            recovery_uncertainty=0.25,
            cap_individual_losses=True,
            asymmetric_rebalancing=False
        )
        
        sim_config_asymmetric = SimulationConfig(
            initial_investment=100000,
            monthly_contribution=1000,
            years=20,
            months=240,
            rebalance_threshold=0.05,
            risk_free_rate=0.02,
            num_simulations=10,
            base_seed=42,
            use_stochastic_recovery=True,
            recovery_uncertainty=0.25,
            cap_individual_losses=True,
            asymmetric_rebalancing=True
        )

        scenario = SCENARIOS["Severe Crash (18m/48m)"]
        
        _, results_standard = run_single_mc_iteration_refactored(
            0, 42, scenario, sim_config_standard, ASSET_CONFIG, TEST_PORTFOLIOS
        )
        _, results_asymmetric = run_single_mc_iteration_refactored(
            0, 42, scenario, sim_config_asymmetric, ASSET_CONFIG, TEST_PORTFOLIOS
        )
        
        rebalances_standard = results_standard["Test_60_40"]["Rebalances"]
        rebalances_asymmetric = results_asymmetric["Test_60_40"]["Rebalances"]
        
        final_value_standard = results_standard["Test_60_40"]["Final Value"]
        final_value_asymmetric = results_asymmetric["Test_60_40"]["Final Value"]

        print(f"Rebalances (Standard): {rebalances_standard}")
        print(f"Rebalances (Asymmetric): {rebalances_asymmetric}")
        
        assert rebalances_asymmetric < rebalances_standard, \
            "Asymmetric should have fewer rebalances"
            
        assert final_value_standard != final_value_asymmetric, \
            "Final values should differ between rebalancing strategies"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

