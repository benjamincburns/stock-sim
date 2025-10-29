"""
Configuration loading for portfolio simulation.

This module reads configuration from a YAML file (e.g., config.yaml) and populates
the necessary configuration objects for the simulation engine.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Literal, NamedTuple
import numpy as np


class AssetTargets(NamedTuple):
    crash_total_return: float
    recovery_target_value: float
    growth_annual_return: float


class AssetVolatility(NamedTuple):
    crash: float
    recovery: float
    growth: float


@dataclass(frozen=True)
class SimulationConfig:
    """Main simulation parameters."""
    initial_investment: float
    monthly_contribution: float
    years: int
    months: int
    rebalance_threshold: float
    risk_free_rate: float
    num_simulations: int
    base_seed: int
    use_stochastic_recovery: bool
    recovery_uncertainty: float
    cap_individual_losses: bool
    rebalancing_strategy: Literal["symmetric", "asymmetric"] = "asymmetric"
    
    @property
    def total_invested(self) -> float:
        """Calculate total amount invested over the simulation period."""
        return self.initial_investment + (self.months * self.monthly_contribution)


@dataclass(frozen=True)
class GoalSeekingConfig:
    """Goal-seeking mode parameters."""
    enable_goal_seeking: bool
    goal_metric: Literal["P5_Final_Value", "Median_Final_Value"]
    goal_target_value: float
    goal_search_min_years: int
    goal_search_max_years: int
    goal_tolerance: float


@dataclass(frozen=True)
class AssetConfig:
    """Asset characteristics including targets, volatility, and correlations."""
    asset_targets: Dict[str, AssetTargets]
    
    asset_volatility: Dict[str, AssetVolatility]
    
    asset_tickers_ordered: list
    
    corr_crash: np.ndarray
    corr_rec_growth: np.ndarray
    
    @property
    def num_assets(self) -> int:
        """Number of assets in the universe."""
        return len(self.asset_tickers_ordered)


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a specific market scenario."""
    name: str
    crash_duration: int  # months
    recovery_duration: int  # months
    
    def growth_duration(self, total_months: int) -> int:
        """Calculate growth phase duration given total simulation months."""
        return total_months - self.crash_duration - self.recovery_duration


def load_config_from_yaml(path: str = 'config.yaml') -> Dict[str, Any]:
    """Load and validate configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path.resolve()}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate the loaded configuration
    import config_validator
    config_validator.validate_config(config)

    return config

def _adjust_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Ensure correlation matrix is positive semi-definite."""
    min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
    if min_eig < -1e-10:
        matrix = matrix - (min_eig - 1e-10) * np.eye(matrix.shape[0])
        diag_elements = np.diag(matrix)
        safe_diag = np.where(np.abs(diag_elements) < 1e-10, 1e-10, diag_elements)
        inv_sqrt_diag = np.diag(1.0 / np.sqrt(safe_diag))
        matrix = inv_sqrt_diag @ matrix @ inv_sqrt_diag
        np.clip(matrix, -1.0, 1.0, out=matrix)
    return matrix


def create_asset_config(config: Dict[str, Any]) -> AssetConfig:
    """Create the asset configuration from the loaded YAML data."""
    
    asset_targets_raw = config['asset_assumptions']['targets']
    asset_targets = {
        ticker: AssetTargets(
            crash_total_return=data['crash_total_return'], 
            recovery_target_value=data['recovery_target_value'], 
            growth_annual_return=data['growth_annual_return']
        )
        for ticker, data in asset_targets_raw.items()
    }
    
    asset_volatility_raw = config['asset_assumptions']['volatility']
    asset_volatility = {
        ticker: AssetVolatility(
            crash=data['crash'], 
            recovery=data['recovery'], 
            growth=data['growth']
        )
        for ticker, data in asset_volatility_raw.items()
    }
    
    asset_tickers_ordered = config['asset_assumptions']['tickers']
    
    corr_crash = np.array(config['asset_assumptions']['correlations']['crash'])
    corr_rec_growth = np.array(config['asset_assumptions']['correlations']['recovery_growth'])
    
    # Ensure correlation matrices are positive semi-definite
    corr_crash = _adjust_correlation_matrix(corr_crash)
    corr_rec_growth = _adjust_correlation_matrix(corr_rec_growth)
    
    return AssetConfig(
        asset_targets=asset_targets,
        asset_volatility=asset_volatility,
        asset_tickers_ordered=asset_tickers_ordered,
        corr_crash=corr_crash,
        corr_rec_growth=corr_rec_growth
    )


def create_scenarios(config: Dict[str, Any]) -> Dict[str, ScenarioConfig]:
    """Create the market scenarios from the loaded YAML data."""
    scenarios_raw = config['scenarios']
    return {
        name: ScenarioConfig(
            name=name,
            crash_duration=data['crash_duration'],
            recovery_duration=data['recovery_duration']
        )
        for name, data in scenarios_raw.items()
    }

