import numpy as np
from typing import Dict, Any, Literal

class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors."""
    pass

def validate_config(config: Dict[str, Any]):
    """Main function to validate the entire configuration dictionary."""
    if 'simulation_parameters' not in config:
        raise ConfigValidationError("Missing 'simulation_parameters' section in config.")
    _validate_simulation_parameters(config['simulation_parameters'])

    if 'advanced_modeling' in config:
        _validate_advanced_modeling(config['advanced_modeling'])

    if 'portfolios' not in config:
        raise ConfigValidationError("Missing 'portfolios' section in config.")
    _validate_portfolios(config['portfolios'])

    if 'asset_assumptions' not in config:
        raise ConfigValidationError("Missing 'asset_assumptions' section in config.")
    _validate_asset_assumptions(config['asset_assumptions'])

    if 'scenarios' not in config:
        raise ConfigValidationError("Missing 'scenarios' section in config.")
    _validate_scenarios(config['scenarios'])

    if 'goal_seeking' in config:
        _validate_goal_seeking(config['goal_seeking'])
    
    print("Configuration successfully validated.")

def _validate_simulation_parameters(params: Dict[str, Any]):
    """Validate the 'simulation_parameters' section."""
    _check_positive(params, 'initial_investment', can_be_zero=True)
    _check_positive(params, 'monthly_contribution', can_be_zero=True)
    _check_positive(params, 'years')
    _check_in_range(params, 'rebalance_threshold', 0, 1)
    _check_positive(params, 'num_simulations')

def _validate_advanced_modeling(params: Dict[str, Any]):
    """Validate the 'advanced_modeling' section."""
    if 'rebalancing_strategy' in params:
        strategy = params['rebalancing_strategy']
        if strategy not in ['symmetric', 'asymmetric']:
            raise ConfigValidationError(
                f"Invalid rebalancing_strategy '{strategy}'. Must be 'symmetric' or 'asymmetric'."
            )

def _validate_portfolios(portfolios: Dict[str, Dict[str, float]]):
    """Validate the 'portfolios' section."""
    if not portfolios:
        raise ConfigValidationError("Configuration must define at least one portfolio.")
    for name, allocation in portfolios.items():
        if not np.isclose(sum(allocation.values()), 1.0):
            raise ConfigValidationError(f"Portfolio '{name}' weights do not sum to 1.0 (sum is {sum(allocation.values())}).")

def _validate_asset_assumptions(assumptions: Dict[str, Any]):
    """Validate the 'asset_assumptions' section."""
    tickers = assumptions.get('tickers')
    if not tickers or not isinstance(tickers, list):
        raise ConfigValidationError("'asset_assumptions.tickers' must be a non-empty list.")
    
    num_tickers = len(tickers)
    section: Literal['targets', 'volatility']
    for section in ['targets', 'volatility']:
        if section not in assumptions:
            raise ConfigValidationError(f"Missing '{section}' in 'asset_assumptions'.")
        if set(assumptions[section].keys()) != set(tickers):
            raise ConfigValidationError(f"Tickers in '{section}' do not match the master ticker list.")

    for ticker, vol_data in assumptions['volatility'].items():
        if not all(v > 0 for v in vol_data.values()):
            raise ConfigValidationError(f"All volatility values for ticker '{ticker}' must be positive.")

    correlations = assumptions.get('correlations')
    if not correlations:
        raise ConfigValidationError("Missing 'correlations' section in 'asset_assumptions'.")
    
    _validate_correlation_matrix(correlations.get('crash'), 'crash', num_tickers)
    _validate_correlation_matrix(correlations.get('recovery_growth'), 'recovery_growth', num_tickers)

def _validate_correlation_matrix(matrix_data: list, name: Literal['crash', 'recovery_growth'], num_tickers: int):
    """Validate a single correlation matrix."""
    if matrix_data is None:
        raise ConfigValidationError(f"Correlation matrix '{name}' is missing.")
    
    try:
        matrix = np.array(matrix_data, dtype=float)
    except ValueError:
        raise ConfigValidationError(f"Correlation matrix '{name}' contains non-numeric values.")

    if matrix.shape != (num_tickers, num_tickers):
        raise ConfigValidationError(
            f"Correlation matrix '{name}' has incorrect dimensions. "
            f"Expected ({num_tickers}, {num_tickers}), but got {matrix.shape}."
        )

    if not np.allclose(np.diag(matrix), 1.0):
        raise ConfigValidationError(f"Diagonals of correlation matrix '{name}' are not all 1.0.")

    if not np.allclose(matrix, matrix.T):
        raise ConfigValidationError(f"Correlation matrix '{name}' is not symmetric.")
    
    if np.min(matrix) < -1.0 or np.max(matrix) > 1.0:
        raise ConfigValidationError(f"Values in correlation matrix '{name}' are outside the range [-1, 1].")

def _validate_scenarios(scenarios: Dict[str, Any]):
    """Validate the 'scenarios' section."""
    for name, params in scenarios.items():
        _check_positive(params, 'crash_duration', can_be_zero=True, context=f"scenario '{name}'")
        _check_positive(params, 'recovery_duration', can_be_zero=True, context=f"scenario '{name}'")

def _validate_goal_seeking(goal_params: Dict[str, Any]):
    """Validate the 'goal_seeking' section."""
    if goal_params.get('enable'):
        allowed_metrics = ["P5_Final_Value", "Median_Final_Value"]
        metric = goal_params.get('metric')
        if metric not in allowed_metrics:
            raise ConfigValidationError(
                f"Invalid goal_seeking.metric '{metric}'. Must be one of {allowed_metrics}."
            )
        _check_positive(goal_params, 'target_value')
        _check_positive(goal_params, 'search_min_years')
        _check_positive(goal_params, 'search_max_years')
        _check_positive(goal_params, 'tolerance', can_be_zero=True)
        if goal_params['search_max_years'] <= goal_params['search_min_years']:
            raise ConfigValidationError(
                "'goal_search_max_years' must be greater than 'goal_search_min_years'."
            )

def _check_positive(params: Dict, key: str, can_be_zero=False, context=""):
    """Helper to check if a parameter is positive."""
    full_context = f"in {context}" if context else ""
    if key not in params:
        raise ConfigValidationError(f"Missing required parameter '{key}' {full_context}.")
    
    value = params[key]
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(f"Parameter '{key}' {full_context} must be a number. Got {type(value).__name__}.")

    if can_be_zero and value < 0:
        raise ConfigValidationError(f"Parameter '{key}' {full_context} cannot be negative. Got {value}.")
    if not can_be_zero and value <= 0:
        raise ConfigValidationError(f"Parameter '{key}' {full_context} must be positive. Got {value}.")

def _check_in_range(params: Dict, key: str, min_val, max_val, context=""):
    """Helper to check if a parameter is within a [min, max] range."""
    full_context = f"in {context}" if context else ""
    if key not in params:
        raise ConfigValidationError(f"Missing required parameter '{key}' {full_context}.")
    
    value = params[key]
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(f"Parameter '{key}' {full_context} must be a number. Got {type(value).__name__}.")

    if not (min_val <= value <= max_val):
        raise ConfigValidationError(
            f"Parameter '{key}' {full_context} must be between {min_val} and {max_val}. Got {value}."
        )
