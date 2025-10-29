"""
Return generation and metrics calculation for portfolio simulation.

Pure functions that don't rely on global state, making them testable and
suitable for multiprocessing.
"""
import numpy as np
import pandas as pd
import pyxirr
from typing import Tuple, Dict

from config import SimulationConfig, ScenarioConfig, AssetConfig, AssetTargets, AssetVolatility


def calculate_geometric_monthly_return(total_return: np.ndarray, duration_months: int) -> np.ndarray:
    """
    Calculate geometric monthly return from total return over duration.
    Handles extreme losses properly by capping at -99% per month to avoid portfolio wipeout.
    Now supports vectorized operations on NumPy arrays.
    """
    if duration_months <= 0:
        return np.zeros_like(total_return)
    
    base = 1 + total_return
    
    # For extreme losses, cap monthly return at -99%
    base[base <= 0] = 0.01  # Equivalent to 1 + (-0.99)
    
    monthly_return = base**(1 / duration_months) - 1
    
    # Clip to a reasonable range
    return np.clip(monthly_return, -0.30, 0.30)


def generate_correlated_returns(mean_returns: np.ndarray, std_devs: np.ndarray, 
                                corr_matrix: np.ndarray, num_periods: int) -> np.ndarray:
    """
    Generate correlated asset returns using multivariate normal distribution.
    
    Args:
        mean_returns: Array of mean returns for each asset
        std_devs: Array of standard deviations for each asset
        corr_matrix: Correlation matrix between assets
        num_periods: Number of time periods to generate
        
    Returns:
        Array of shape (num_periods, num_assets) with correlated returns
    """
    mean_returns = np.array(mean_returns)
    std_devs = np.array(std_devs)
    cov_matrix = np.outer(std_devs, std_devs) * corr_matrix
    
    try:
        correlated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=num_periods)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Warning: Covariance matrix issue ({e}). Using uncorrelated returns.", flush=True)
        uncorrelated_samples = np.random.normal(0.0, 1.0, size=(num_periods, len(mean_returns)))
        correlated_returns = mean_returns[np.newaxis, :] + uncorrelated_samples * std_devs[np.newaxis, :]
    
    return correlated_returns


def generate_scenario_returns_array(scenario: ScenarioConfig, 
                                     sim_config: SimulationConfig,
                                     asset_config: AssetConfig,
                                     seed: int) -> Tuple[np.ndarray, list]:
    """
    Generate returns as a NumPy array for maximum performance.
    
    Returns a tuple of (returns_array, ordered_tickers).
    """
    np.random.seed(seed)
    
    total_months = sim_config.months
    crash_dur = scenario.crash_duration
    rec_dur = scenario.recovery_duration
    growth_dur = scenario.growth_duration(total_months)
    
    tickers = asset_config.asset_tickers_ordered
    
    if growth_dur < 0:
        return np.empty((0, len(tickers))), tickers
    
    all_returns_arrays = []
    
    # Phase 1: Crash
    if crash_dur > 0:
        crash_returns = _generate_crash_returns(crash_dur, asset_config)
        all_returns_arrays.append(crash_returns)
        last_returns_array = crash_returns
    else:
        last_returns_array = np.empty((0, asset_config.num_assets))
    
    # Phase 2: Recovery
    if rec_dur > 0:
        if crash_dur > 0:
            crash_end_values = np.prod(1 + last_returns_array, axis=0)
        else:
            crash_end_values = np.ones(asset_config.num_assets)
        
        recovery_returns = _generate_recovery_returns(
            rec_dur, crash_end_values, asset_config, sim_config
        )
        all_returns_arrays.append(recovery_returns)
    
    # Phase 3: Growth
    if growth_dur > 0:
        growth_returns = _generate_growth_returns(growth_dur, asset_config)
        all_returns_arrays.append(growth_returns)
    
    if not all_returns_arrays:
        return np.empty((0, len(tickers))), tickers
        
    combined_returns_array = np.vstack(all_returns_arrays)
    return combined_returns_array, tickers


def _generate_crash_returns(crash_dur: int, asset_config: AssetConfig) -> np.ndarray:
    """Generate returns for crash phase."""
    
    crash_targets = np.array([asset_config.asset_targets[ticker].crash_total_return for ticker in asset_config.asset_tickers_ordered])
    crash_stds = np.array([asset_config.asset_volatility[ticker].crash for ticker in asset_config.asset_tickers_ordered])
    
    crash_means = calculate_geometric_monthly_return(crash_targets, crash_dur)
    
    return generate_correlated_returns(
        crash_means, 
        crash_stds, 
        asset_config.corr_crash, 
        crash_dur
    )


def _generate_recovery_returns(rec_dur: int, 
                               crash_end_values: np.ndarray,
                               asset_config: AssetConfig,
                               sim_config: SimulationConfig) -> np.ndarray:
    """Generate returns for recovery phase."""
    
    base_recovery_targets = np.array([asset_config.asset_targets[ticker].recovery_target_value for ticker in asset_config.asset_tickers_ordered])
    recovery_stds = np.array([asset_config.asset_volatility[ticker].recovery for ticker in asset_config.asset_tickers_ordered])
    
    if sim_config.use_stochastic_recovery:
        uncertainty = np.random.uniform(
            1 - sim_config.recovery_uncertainty, 
            1 + sim_config.recovery_uncertainty,
            size=asset_config.num_assets
        )
        final_recovery_targets = base_recovery_targets * uncertainty
    else:
        final_recovery_targets = base_recovery_targets
        
    total_rec_return = np.divide(final_recovery_targets, crash_end_values, 
                                 out=np.zeros_like(final_recovery_targets), 
                                 where=crash_end_values!=0) - 1
    
    recovery_means = calculate_geometric_monthly_return(total_rec_return, rec_dur)
    
    return generate_correlated_returns(
        recovery_means,
        recovery_stds,
        asset_config.corr_rec_growth,
        rec_dur
    )


def _generate_growth_returns(growth_dur: int, asset_config: AssetConfig) -> np.ndarray:
    """Generate returns for growth phase."""
    
    annual_returns = np.array([asset_config.asset_targets[ticker].growth_annual_return for ticker in asset_config.asset_tickers_ordered])
    growth_stds = np.array([asset_config.asset_volatility[ticker].growth for ticker in asset_config.asset_tickers_ordered])
    
    growth_means = (1 + annual_returns)**(1/12) - 1
    
    return generate_correlated_returns(
        growth_means,
        growth_stds,
        asset_config.corr_rec_growth,
        growth_dur
    )

