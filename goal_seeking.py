"""
Refactored goal-seeking functions that use explicit parameter passing.

These functions find the time it takes for a portfolio to reach a financial goal,
without relying on global variables.
"""
import numpy as np
import time
from typing import Dict, Tuple, Optional

from config import SimulationConfig, ScenarioConfig, AssetConfig, GoalSeekingConfig
from returns_engine import generate_scenario_returns_array
from simulation_engine import _simulate_portfolio_numba


def run_goal_seeking_simulation(
    allocation: Dict[str, float],
    scenario: ScenarioConfig,
    sim_config: SimulationConfig,
    goal_config: GoalSeekingConfig,
    asset_config: AssetConfig,
    base_seed_offset: int
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[np.ndarray]]:
    """
    Run simulations incrementally, stopping when the goal is reached.
    
    Args:
        allocation: Portfolio allocation dict
        scenario: Scenario configuration
        sim_config: Base simulation configuration
        goal_config: Goal-seeking parameters
        asset_config: Asset characteristics
        base_seed_offset: Starting seed for reproducibility
        
    Returns:
        Tuple of (values_array, months_stopped, rebalance_counts) where:
        - values_array: Portfolio values from month 0 to months_stopped
        - months_stopped: The month when goal was met (or max_months if not reached)
        - rebalance_counts: Array of rebalance counts for each simulation
    """
    max_months = int(round(goal_config.goal_search_max_years * 12))
    min_months = int(round(goal_config.goal_search_min_years * 12))
    
    # Create extended simulation config for goal seeking
    extended_sim_config = SimulationConfig(
        initial_investment=sim_config.initial_investment,
        monthly_contribution=sim_config.monthly_contribution,
        years=goal_config.goal_search_max_years,
        months=max_months,
        rebalance_threshold=sim_config.rebalance_threshold,
        risk_free_rate=sim_config.risk_free_rate,
        num_simulations=sim_config.num_simulations,
        base_seed=sim_config.base_seed,
        use_stochastic_recovery=sim_config.use_stochastic_recovery,
        recovery_uncertainty=sim_config.recovery_uncertainty,
        cap_individual_losses=sim_config.cap_individual_losses,
        asymmetric_rebalancing=sim_config.asymmetric_rebalancing
    )
    
    growth_dur = max_months - scenario.crash_duration - scenario.recovery_duration
    if growth_dur < 0:
        return None, None, None
    
    # Pre-generate ALL returns for all simulations
    sim_assets = list(allocation.keys())
    all_sim_returns = []
    
    # Get column indices once, assuming order is constant
    returns_array, tickers = generate_scenario_returns_array(scenario, extended_sim_config, asset_config, base_seed_offset)
    asset_indices = [tickers.index(asset) for asset in sim_assets]

    for sim_idx in range(sim_config.num_simulations):
        seed = base_seed_offset + sim_idx
        
        # Generate returns using refactored function
        returns_array, _ = generate_scenario_returns_array(scenario, extended_sim_config, asset_config, seed)
        
        if returns_array.shape[0] == 0:
            continue
        
        # Extract only needed columns as numpy array using advanced indexing
        portfolio_returns_array = returns_array[:, asset_indices]
        all_sim_returns.append(portfolio_returns_array)
    
    if not all_sim_returns:
        return None, None, None
    
    # Run simulations incrementally, stopping when goal is reached
    chunk_size_months = 1
    num_sims = len(all_sim_returns)
    
    # Pre-allocate array for all values (sims x months)
    values_array = np.zeros((num_sims, max_months + 1))
    
    # State for each simulation (holdings and rebalance count)
    sim_holdings = [None] * num_sims
    sim_rebalance_counts = np.zeros(num_sims, dtype=np.int32)
    
    # Get target weights for direct numba calls
    target_weights = np.array([allocation[asset] for asset in sim_assets])
    
    # Start with min_months
    for sim_idx, sim_returns_array in enumerate(all_sim_returns):
        try:
            chunk_returns = sim_returns_array[:min_months]
            values, rebalance_count, final_holdings = _simulate_portfolio_numba(
                target_weights, chunk_returns, sim_config.monthly_contribution,
                sim_config.initial_investment, sim_config.rebalance_threshold, min_months,
                None, 0, sim_config.asymmetric_rebalancing
            )
            values_array[sim_idx, :min_months+1] = values
            sim_holdings[sim_idx] = final_holdings
            sim_rebalance_counts[sim_idx] = rebalance_count
        except Exception:
            return None, None, None
    
    current_month = min_months
    
    # Check if goal is already met at min_months
    if _check_goal_reached(values_array[:, current_month], goal_config):
        return values_array[:, :current_month+1], current_month, sim_rebalance_counts
    
    # Continue simulating in chunks until goal is reached or max_months
    while current_month < max_months:
        next_month = min(current_month + chunk_size_months, max_months)
        chunk_months = next_month - current_month
        
        # Run next chunk for each simulation
        for sim_idx, sim_returns_array in enumerate(all_sim_returns):
            try:
                chunk_returns = sim_returns_array[current_month:next_month]
                values, new_rebalance_count, final_holdings = _simulate_portfolio_numba(
                    target_weights, chunk_returns, sim_config.monthly_contribution,
                    sim_config.initial_investment, sim_config.rebalance_threshold, chunk_months,
                    sim_holdings[sim_idx], sim_rebalance_counts[sim_idx], sim_config.asymmetric_rebalancing
                )
                
                # Store new values (skip first value since it's the ending value from previous chunk)
                values_array[sim_idx, current_month+1:next_month+1] = values[1:]
                sim_holdings[sim_idx] = final_holdings
                sim_rebalance_counts[sim_idx] = new_rebalance_count
            except Exception:
                return None, None, None
        
        current_month = next_month
        
        # Check if goal is reached
        if _check_goal_reached(values_array[:, current_month], goal_config):
            return values_array[:, :current_month+1], current_month, sim_rebalance_counts
    
    # Goal not reached by max_months
    return values_array[:, :max_months+1], max_months, sim_rebalance_counts


def _check_goal_reached(values_at_month: np.ndarray, goal_config: GoalSeekingConfig) -> bool:
    """
    Check if goal is reached at this month using an efficient selection algorithm.
    This avoids a full sort, which is a major performance bottleneck.
    """
    num_sims = len(values_at_month)
    if num_sims == 0:
        return False

    if goal_config.goal_metric == "Median_Final_Value":
        # Find the median value using partition, which is much faster than sorting.
        # For an even number of simulations, this finds the lower of the two median elements.
        median_idx = (num_sims - 1) // 2
        metric_value = np.partition(values_at_month, median_idx)[median_idx]
    elif goal_config.goal_metric == "P5_Final_Value":
        # Find the 5th percentile value using partition.
        p5_idx = max(0, int(num_sims * 0.05) - 1)
        metric_value = np.partition(values_at_month, p5_idx)[p5_idx]
    else:
        metric_value = 0
    
    return metric_value >= goal_config.goal_target_value - goal_config.goal_tolerance


def find_time_to_goal(
    portfolio_name: str,
    allocation: Dict[str, float],
    scenario: ScenarioConfig,
    sim_config: SimulationConfig,
    goal_config: GoalSeekingConfig,
    asset_config: AssetConfig
) -> Dict:
    """
    Find the first month where the goal is reached.
    
    Returns a dictionary with results including whether goal was reached,
    time taken, and various metrics.
    """
    print(f"  Finding time to reach {goal_config.goal_metric} = ${goal_config.goal_target_value:,.0f} for {portfolio_name}...", flush=True)
    
    base_seed_offset = int(time.time()) + hash(portfolio_name + scenario.name) % 10000
    
    # Run simulations, stopping early if goal is reached
    values_array, months_stopped, rebalance_counts = run_goal_seeking_simulation(
        allocation, scenario, sim_config, goal_config, asset_config, base_seed_offset
    )
    
    if values_array is None or months_stopped is None or rebalance_counts is None:
        return {
            'portfolio': portfolio_name,
            'scenario': scenario.name,
            'goal_reached': False,
            'years': None,
            'months': None,
            'Median_Final_Value': 0,
            'P5_Final_Value': 0,
            'message': "Simulation failed"
        }
    
    # Get values at the point where simulation stopped
    values_at_month = values_array[:, -1]  # Last column = values at months_stopped
    
    # Calculate the aggregate metric at this point
    if goal_config.goal_metric == "Median_Final_Value":
        metric_value = np.median(values_at_month)
    elif goal_config.goal_metric == "P5_Final_Value":
        metric_value = np.percentile(values_at_month, 5)
    else:
        metric_value = 0
    
    # Check if goal was actually reached (vs hitting max_months)
    goal_was_reached = (metric_value >= goal_config.goal_target_value - goal_config.goal_tolerance)
    
    years_reached = months_stopped / 12
    
    if goal_was_reached:
        # Goal reached! Calculate metrics
        median_val = np.median(values_at_month)
        p5_val = np.percentile(values_at_month, 5)
        avg_val = np.mean(values_at_month)
        p95_val = np.percentile(values_at_month, 95)
        cvar_5pct = np.mean(values_at_month[values_at_month <= p5_val])
        
        # Calculate annualized return for median
        if years_reached > 0:
            total_invested = sim_config.initial_investment + (months_stopped * sim_config.monthly_contribution)
            ann_return = ((median_val / total_invested) ** (1 / years_reached) - 1) * 100
            p5_ann_return = ((p5_val / total_invested) ** (1 / years_reached) - 1) * 100
        else:
            ann_return = 0
            p5_ann_return = 0
        
        # Vectorized calculation of volatility and Sharpe ratio
        if values_array.shape[1] > 1:
            # Calculate monthly returns for all simulations at once, handling division by zero
            safe_values = np.maximum(values_array[:, :-1], 1e-9)
            monthly_returns_all_sims = np.diff(values_array, axis=1) / safe_values
            
            # Flatten all returns into a single array to calculate overall volatility
            all_returns_flat = monthly_returns_all_sims.flatten()
            
            monthly_vol = np.std(all_returns_flat)
            ann_vol = monthly_vol * np.sqrt(12) * 100
            sharpe = (ann_return/100 - sim_config.risk_free_rate) / (ann_vol/100) if ann_vol > 0 else 0
        else:
            ann_vol = 0
            sharpe = 0
        
        # Vectorized calculation of max drawdown
        cummax = np.maximum.accumulate(values_array, axis=1)
        drawdowns = (values_array - cummax) / np.maximum(cummax, 1e-9)
        max_drawdowns = np.min(drawdowns, axis=1) * 100
        
        median_drawdown = np.median(max_drawdowns)
        worst_drawdown = np.min(max_drawdowns)
        
        # Rebalance statistics
        median_rebalances = np.median(rebalance_counts)
        p5_rebalances = np.percentile(rebalance_counts, 5)
        p95_rebalances = np.percentile(rebalance_counts, 95)
        
        return {
            'portfolio': portfolio_name,
            'scenario': scenario.name,
            'goal_reached': True,
            'years': years_reached,
            'months': months_stopped,
            goal_config.goal_metric: metric_value,
            'Median_Final_Value': median_val,
            'Avg_Final_Value': avg_val,
            'P95_Final_Value': p95_val,
            'P5_Final_Value': p5_val,
            'CVaR_5pct': cvar_5pct,
            'Ann_Return': ann_return,
            'P5_Ann_Return': p5_ann_return,
            'Ann_Volatility': ann_vol,
            'Sharpe_Ratio': sharpe,
            'Median_Drawdown': median_drawdown,
            'Worst_Drawdown': worst_drawdown,
            'Median_Rebalances': median_rebalances,
            'P5_Rebalances': p5_rebalances,
            'P95_Rebalances': p95_rebalances
        }
    
    # Goal not reached by max_years
    median_val = np.median(values_at_month)
    avg_val = np.mean(values_at_month)
    p5_val = np.percentile(values_at_month, 5)
    p95_val = np.percentile(values_at_month, 95)
    cvar_5pct = np.mean(values_at_month[values_at_month <= p5_val])
    
    # Calculate metrics at max_years
    if years_reached > 0:
        total_invested = sim_config.initial_investment + (months_stopped * sim_config.monthly_contribution)
        ann_return = ((median_val / total_invested) ** (1 / years_reached) - 1) * 100
        p5_ann_return = ((p5_val / total_invested) ** (1 / years_reached) - 1) * 100
    else:
        ann_return = 0
        p5_ann_return = 0
    
    # Vectorized calculation of volatility, Sharpe, and drawdown
    if values_array.shape[1] > 1:
        safe_values = np.maximum(values_array[:, :-1], 1e-9)
        monthly_returns_all_sims = np.diff(values_array, axis=1) / safe_values
        all_returns_flat = monthly_returns_all_sims.flatten()
        
        monthly_vol = np.std(all_returns_flat)
        ann_vol = monthly_vol * np.sqrt(12) * 100
        sharpe = (ann_return/100 - sim_config.risk_free_rate) / (ann_vol/100) if ann_vol > 0 else 0
    else:
        ann_vol = 0
        sharpe = 0
    
    cummax = np.maximum.accumulate(values_array, axis=1)
    drawdowns = (values_array - cummax) / np.maximum(cummax, 1e-9)
    max_drawdowns = np.min(drawdowns, axis=1) * 100
    
    median_drawdown = np.median(max_drawdowns)
    worst_drawdown = np.min(max_drawdowns)
    
    median_rebalances = np.median(rebalance_counts)
    p5_rebalances = np.percentile(rebalance_counts, 5)
    p95_rebalances = np.percentile(rebalance_counts, 95)
    
    return {
        'portfolio': portfolio_name,
        'scenario': scenario.name,
        'goal_reached': False,
        'years': years_reached,
        'months': months_stopped,
        'Median_Final_Value': median_val,
        'Avg_Final_Value': avg_val,
        'P95_Final_Value': p95_val,
        'P5_Final_Value': p5_val,
        'CVaR_5pct': cvar_5pct,
        'Ann_Return': ann_return,
        'P5_Ann_Return': p5_ann_return,
        'Ann_Volatility': ann_vol,
        'Sharpe_Ratio': sharpe,
        'Median_Drawdown': median_drawdown,
        'Worst_Drawdown': worst_drawdown,
        'Median_Rebalances': median_rebalances,
        'P5_Rebalances': p5_rebalances,
        'P95_Rebalances': p95_rebalances,
        'message': f"Goal not reached (shown: {goal_config.goal_search_max_years}y value)"
    }

