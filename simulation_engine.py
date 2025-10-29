"""
Refactored simulation engine that uses explicit parameter passing.

This module fixes the multiprocessing bug by passing all configuration
explicitly instead of relying on global variables.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from numba import njit
import pyxirr

from config import SimulationConfig, ScenarioConfig, AssetConfig
from returns_engine import generate_scenario_returns_array


@njit
def _allocate_contribution_asymmetric(holdings_value, target_weights, contrib):
    """
    Allocate contribution using asymmetric rebalancing strategy.
    """
    num_assets = len(target_weights)
    portfolio_value = np.sum(holdings_value)
    current_weights = holdings_value / portfolio_value if portfolio_value > 1e-9 else target_weights
    
    low_bands = target_weights / 2.0
    below_low = current_weights < low_bands
    below_target = (current_weights >= low_bands) & (current_weights < target_weights)
    
    allocation = np.zeros(num_assets)
    
    if np.any(below_low):
        deficits = np.maximum(low_bands - current_weights, 0.0)
        total_deficit = np.sum(deficits)
        
        if total_deficit > 1e-9:
            allocation = (deficits / total_deficit) * contrib
        else:
            num_below = np.sum(below_low)
            if num_below > 0:
                allocation[below_low] = contrib / num_below
    elif np.any(below_target):
        deficits = np.maximum(target_weights - current_weights, 0.0)
        total_deficit = np.sum(deficits)
        
        if total_deficit > 1e-9:
            allocation = (deficits / total_deficit) * contrib
        else:
            allocation = target_weights * contrib
    else:
        allocation = target_weights * contrib
    
    return allocation

@njit
def _check_asymmetric_rebalance_needed(holdings_value, target_weights):
    """Check if selling rebalance is needed (any asset above high band)."""
    portfolio_value = np.sum(holdings_value)
    if portfolio_value <= 1e-9:
        return False
    
    current_weights = holdings_value / portfolio_value
    high_bands = target_weights * 2.0
    
    return np.any(current_weights > high_bands)

@njit
def _simulate_portfolio_numba(target_weights, monthly_returns_array, contrib, 
                               initial_investment, rebalance_threshold, num_months,
                               initial_holdings=None, initial_rebalance_count=0,
                               rebalancing_strategy="asymmetric"):
    """
    Numba-optimized core simulation loop using pure NumPy arrays.
    """
    num_assets = len(target_weights)
    
    if initial_holdings is None:
        holdings_value = target_weights * initial_investment
        portfolio_value = initial_investment
        rebalance_count = 0
    else:
        holdings_value = initial_holdings.copy()
        portfolio_value = np.sum(holdings_value)
        rebalance_count = initial_rebalance_count

    portfolio_values = np.zeros(num_months + 1)
    portfolio_values[0] = portfolio_value

    for i in range(num_months):
        if portfolio_value <= 1e-9:
            break

        if contrib > 0:
            if rebalancing_strategy == "asymmetric":
                allocation_of_contribution = _allocate_contribution_asymmetric(
                    holdings_value, target_weights, contrib
                )
            else: # symmetric
                target_dollar_values_after_contrib = target_weights * (portfolio_value + contrib)
                amount_to_buy = np.maximum(target_dollar_values_after_contrib - holdings_value, 0)
                total_to_buy = np.sum(amount_to_buy)
                
                if total_to_buy > 1e-9:
                    allocation_of_contribution = (amount_to_buy / total_to_buy) * contrib
                else:
                    allocation_of_contribution = target_weights * contrib
            
            holdings_value += allocation_of_contribution
            portfolio_value = np.sum(holdings_value)

        if rebalancing_strategy == "asymmetric":
            needs_rebalance = _check_asymmetric_rebalance_needed(holdings_value, target_weights)
        else: # symmetric
            current_weights = holdings_value / portfolio_value
            deviation = np.abs(current_weights - target_weights)
            needs_rebalance = np.any(deviation > rebalance_threshold)
        
        if needs_rebalance:
            old_holdings = holdings_value.copy()
            holdings_value = target_weights * portfolio_value
            if np.any(holdings_value < old_holdings - 1e-6):
                rebalance_count += 1
            portfolio_value = np.sum(holdings_value)

        returns_this_month = monthly_returns_array[i, :]
        holdings_value = holdings_value * (1 + returns_this_month)
        holdings_value = np.maximum(holdings_value, 1e-9)
        portfolio_value = np.sum(holdings_value)
        
        if portfolio_value < 0:
            portfolio_value = 0.0
        
        portfolio_values[i + 1] = portfolio_value
    
    return portfolio_values, rebalance_count, holdings_value

def calculate_portfolio_metrics(portfolio_values: np.ndarray,
                                sim_config: SimulationConfig,
                                rebalance_count: int) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics for a single simulation run.
    """
    final_val = portfolio_values[-1]
    
    # Calculate monthly returns from the values array
    if len(portfolio_values) > 1:
        monthly_rets_array = np.diff(portfolio_values) / portfolio_values[:-1]
        monthly_rets_array = monthly_rets_array[np.isfinite(monthly_rets_array)]
    else:
        monthly_rets_array = np.array([])

    if monthly_rets_array.size == 0 or final_val <= 0:
        return {
            "Final Value": max(0, final_val),
            "Annualized Return (%)": -1.0 if final_val <= 0 else 0.0,
            "Annualized Volatility (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown (%)": -100.0 if portfolio_values[0] > 0 else 0.0,
            "Rebalances": rebalance_count
        }
    
    cash_flows = np.full(sim_config.months + 1, -sim_config.monthly_contribution, dtype=float)
    cash_flows[0] = -sim_config.initial_investment
    cash_flows[-1] += final_val
    
    try:
        if final_val > 0 and (sim_config.initial_investment > 0 or sim_config.monthly_contribution > 0):
            monthly_irr = pyxirr.irr(cash_flows)
            annualized_return = 0.0 if np.isnan(monthly_irr) else (1 + monthly_irr)**12 - 1
        else:
            annualized_return = -1.0 if final_val <= 0 and sim_config.initial_investment > 0 else 0.0
    except Exception:
        annualized_return = 0.0
    
    annualized_volatility = np.std(monthly_rets_array) * np.sqrt(12)
    sharpe_ratio = (annualized_return - sim_config.risk_free_rate) / annualized_volatility if annualized_volatility > 1e-9 else 0
    
    cumulative_max = np.maximum.accumulate(portfolio_values)
    cumulative_max = np.maximum(cumulative_max, 1e-9)
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdown)
    
    return {
        "Final Value": final_val,
        "Annualized Return (%)": annualized_return * 100,
        "Annualized Volatility (%)": annualized_volatility * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown * 100,
        "Rebalances": rebalance_count
    }

def run_single_mc_iteration_refactored(
    run_index: int,
    seed: int,
    scenario: ScenarioConfig,
    sim_config: SimulationConfig,
    asset_config: AssetConfig,
    portfolios: Dict[str, Dict[str, float]]
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Run one Monte Carlo iteration with explicit parameter passing (no globals).
    
    Args:
        run_index: Index of this run (for logging)
        seed: Random seed for reproducibility
        scenario: Scenario configuration
        sim_config: Simulation configuration
        asset_config: Asset configuration
        portfolios: Dictionary of portfolio allocations
        
    Returns:
        Tuple of (scenario_name, results_dict)
    """
    # Generate returns for this scenario
    returns_array, tickers = generate_scenario_returns_array(scenario, sim_config, asset_config, seed)
    
    if returns_array.shape[0] == 0:
        # No valid returns - return empty results for all portfolios
        run_summary = {}
        for name in portfolios.keys():
            run_summary[name] = {
                "Final Value": sim_config.initial_investment if sim_config.months == 0 else 0,
                "Annualized Return (%)": 0.0,
                "Annualized Volatility (%)": 0.0,
                "Sharpe Ratio": 0.0,
                "Max Drawdown (%)": 0.0,
                "Rebalances": 0
            }
        return (scenario.name, run_summary)
    
    # Run portfolio simulations
    run_summary = {}
    for portfolio_name, allocation in portfolios.items():
        sim_assets = list(allocation.keys())
        
        # Create a mapping from ticker to index once
        ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
        
        # Check that all assets are available
        if not all(asset in ticker_to_idx for asset in sim_assets):
            continue
            
        # Get indices for portfolio assets
        asset_indices = [ticker_to_idx[asset] for asset in sim_assets]

        try:
            # Prepare inputs for numba function
            target_weights = np.array([allocation[t] for t in sim_assets])
            monthly_returns_array = returns_array[:, asset_indices]
            num_months = len(monthly_returns_array)
            
            # Call the optimized numba function
            portfolio_values, rebalance_count, _ = _simulate_portfolio_numba(
                target_weights,
                monthly_returns_array,
                sim_config.monthly_contribution,
                sim_config.initial_investment,
                sim_config.rebalance_threshold,
                num_months,
                None,  # initial_holdings
                0,     # initial_rebalance_count
                sim_config.rebalancing_strategy
            )
            
            # Calculate metrics
            metrics = calculate_portfolio_metrics(portfolio_values, sim_config, rebalance_count)
            run_summary[portfolio_name] = metrics
            
        except Exception as e:
            print(f"Error during run {run_index+1}, portfolio {portfolio_name}: {e}", flush=True)
            run_summary[portfolio_name] = None
    
    return (scenario.name, run_summary)


def run_single_mc_iteration_wrapper(args_tuple):
    """
    Wrapper for multiprocessing that unpacks arguments.
    
    This is necessary because multiprocessing.Pool.map requires a single argument.
    All parameters are explicitly passed in the tuple - no globals!
    """
    (run_index, seed, scenario, sim_config, asset_config, portfolios) = args_tuple
    return run_single_mc_iteration_refactored(
        run_index, seed, scenario, sim_config, asset_config, portfolios
    )

