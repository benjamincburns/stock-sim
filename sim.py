"""
Portfolio Stress Test Simulation with Monte Carlo Analysis

Key Features:
- Numba-optimized for high performance (10-50x faster than pure Python)
- Stochastic recovery modeling (realistic market uncertainty)
- Tail risk metrics (VaR, CVaR) for worst-case analysis
- Proper handling of extreme losses and negative returns
- Multi-scenario stress testing with configurable parameters
"""

import pandas as pd
import numpy as np
import pyxirr
import sys
import time
import multiprocessing
import os
import argparse
from config import SimulationConfig, GoalSeekingConfig
from reporting import StandardSimulationResults, GoalSeekingResults, print_standard_results, print_goal_seeking_results

# --- Configuration Loading ---
# All configuration is now loaded from config.yaml via the config module.
# This section is intentionally left blank.

# --- Dataclasses for Runtime Configuration ---
# These dataclasses are populated at runtime from a combination of the config file
# and command-line arguments.
# (Moved to config.py)


# --- Command-Line Interface and Main Execution ---

def parse_command_line_arguments(app_config):
    """Parse command-line arguments and return configuration dictionary."""
    # Get default values from the loaded YAML config
    defaults = {
        **app_config['simulation_parameters'],
        **app_config['advanced_modeling'],
        **app_config['goal_seeking']
    }

    parser = argparse.ArgumentParser(
        description='Portfolio Stress Test Simulation with Monte Carlo Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults from config.yaml
  python sim.py
  
  # Run 20-year simulation with $2000/month contributions
  python sim.py --years 20 --monthly-contribution 2000
  
  # Find time to reach $500K (VaR conservative)
  python sim.py --goal-var --goal-target 500000
  
  # Find time to reach $500K (median outcome)
  python sim.py --goal-median --goal-target 500000
        """
    )
    
    # Simulation parameters
    parser.add_argument('--initial-investment', type=float, default=defaults['initial_investment'],
                        help=f"Initial investment amount (default: ${defaults['initial_investment']:,.0f})")
    parser.add_argument('--monthly-contribution', type=float, default=defaults['monthly_contribution'],
                        help=f"Monthly contribution amount (default: ${defaults['monthly_contribution']:,.0f})")
    parser.add_argument('--years', type=int, default=defaults['years'],
                        help=f"Investment time horizon in years (default: {defaults['years']})")
    parser.add_argument('--rebalance-threshold', type=float, default=defaults['rebalance_threshold'],
                        help=f"Rebalancing threshold as decimal (default: {defaults['rebalance_threshold']})")
    parser.add_argument('--risk-free-rate', type=float, default=defaults['risk_free_rate'],
                        help=f"Risk-free rate for Sharpe ratio (default: {defaults['risk_free_rate']})")
    parser.add_argument('--num-simulations', type=int, default=defaults['num_simulations'],
                        help=f"Number of Monte Carlo runs per scenario (default: {defaults['num_simulations']})")
    
    # Advanced modeling options
    parser.add_argument('--use-stochastic-recovery', type=lambda x: x.lower() == 'true', 
                        default=defaults['use_stochastic_recovery'],
                        help=f"Enable stochastic recovery uncertainty (default: {defaults['use_stochastic_recovery']})")
    parser.add_argument('--recovery-uncertainty', type=float, default=defaults['recovery_uncertainty'],
                        help=f"Recovery target uncertainty (default: {defaults['recovery_uncertainty']})")
    parser.add_argument('--cap-individual-losses', type=lambda x: x.lower() == 'true',
                        default=defaults['cap_individual_losses'],
                        help=f"Prevent individual holdings from going negative (default: {defaults['cap_individual_losses']})")
    parser.add_argument('--asymmetric-rebalancing', type=lambda x: x.lower() == 'true',
                        default=defaults.get('asymmetric_rebalancing', True),
                        help=f"Use asymmetric rebalancing strategy (default: {defaults.get('asymmetric_rebalancing', True)})")
    
    # Goal-seeking mode
    parser.add_argument('--enable-goal-seeking', type=lambda x: x.lower() == 'true',
                        default=None, nargs='?', const=True,
                        help='Enable goal-seeking mode to find time to reach target (auto-enabled if any --goal-* arg used)')
    
    goal_metric_group = parser.add_mutually_exclusive_group()
    goal_metric_group.add_argument('--goal-p5', action='store_true',
                        help='Target P5_Final_Value (conservative, 5th percentile) - default if neither specified')
    goal_metric_group.add_argument('--goal-median', action='store_true',
                        help='Target Median_Final_Value (typical outcome, 50th percentile)')
    
    parser.add_argument('--goal-target', '--goal-target-value', type=float, default=None,
                        dest='goal_target_value',
                        help=f"Target value for goal-seeking (default: ${defaults['target_value']:,.0f})")
    parser.add_argument('--goal-min-years', '--goal-search-min-years', type=int, 
                        default=None, dest='goal_search_min_years',
                        help=f"Minimum years to search in goal-seeking (default: {defaults['search_min_years']})")
    parser.add_argument('--goal-max-years', '--goal-search-max-years', type=int,
                        default=None, dest='goal_search_max_years',
                        help=f"Maximum years to search in goal-seeking (default: {defaults['search_max_years']})")
    parser.add_argument('--goal-tolerance', type=float, default=None,
                        help=f"Tolerance in dollars for goal-seeking (default: ${defaults['tolerance']:,.0f})")
    
    args = parser.parse_args()
    
    if args.goal_median:
        goal_metric_final = "Median_Final_Value"
    elif args.goal_p5:
        goal_metric_final = "P5_Final_Value"
    else:
        goal_metric_final = defaults['metric']
    
    goal_args_specified = any([
        args.goal_p5, args.goal_median, args.goal_target_value is not None,
        args.goal_search_min_years is not None, args.goal_search_max_years is not None,
        args.goal_tolerance is not None
    ])
    
    if args.enable_goal_seeking is not None:
        enable_goal_seeking_final = args.enable_goal_seeking
    elif goal_args_specified:
        enable_goal_seeking_final = True
    else:
        enable_goal_seeking_final = defaults['enable']
    
    return {
        'initial_investment': args.initial_investment,
        'monthly_contribution': args.monthly_contribution,
        'years': args.years,
        'months': args.years * 12,
        'rebalance_threshold': args.rebalance_threshold,
        'risk_free_rate': args.risk_free_rate,
        'num_simulations': args.num_simulations,
        'use_stochastic_recovery': args.use_stochastic_recovery,
        'recovery_uncertainty': args.recovery_uncertainty,
        'cap_individual_losses': args.cap_individual_losses,
        'asymmetric_rebalancing': args.asymmetric_rebalancing,
        'enable_goal_seeking': enable_goal_seeking_final,
        'goal_metric': goal_metric_final,
        'goal_target_value': args.goal_target_value if args.goal_target_value is not None else defaults['target_value'],
        'goal_search_min_years': args.goal_search_min_years if args.goal_search_min_years is not None else defaults['search_min_years'],
        'goal_search_max_years': args.goal_search_max_years if args.goal_search_max_years is not None else defaults['search_max_years'],
        'goal_tolerance': args.goal_tolerance if args.goal_tolerance is not None else defaults['tolerance'],
    }


def run_standard_simulation(config, portfolios, scenarios, asset_config, base_seed):
    """Run standard time-bounded Monte Carlo simulation."""
    from simulation_engine import run_single_mc_iteration_wrapper
    
    print(f"Starting Monte Carlo simulation for {len(scenarios)} scenarios.", flush=True)
    print(f"Each scenario runs {config['num_simulations']} times using {os.cpu_count()} CPU cores.", flush=True)
    
    sim_config = SimulationConfig(
        initial_investment=config['initial_investment'],
        monthly_contribution=config['monthly_contribution'],
        years=config['years'],
        months=config['months'],
        rebalance_threshold=config['rebalance_threshold'],
        risk_free_rate=config['risk_free_rate'],
        num_simulations=config['num_simulations'],
        base_seed=base_seed,
        use_stochastic_recovery=config['use_stochastic_recovery'],
        recovery_uncertainty=config['recovery_uncertainty'],
        cap_individual_losses=config['cap_individual_losses'],
        asymmetric_rebalancing=config['asymmetric_rebalancing']
    )
    
    total_runs_overall = len(scenarios) * config['num_simulations']
    print(f"Total runs: {total_runs_overall}", flush=True)
    print(f"Base seed for first scenario: {base_seed}", flush=True)

    combined_run_args = []
    for s_idx, (scenario_name, scenario) in enumerate(scenarios.items()):
        current_base_seed = base_seed + s_idx * config['num_simulations']
        scenario_args = [
            (i, current_base_seed + i, scenario, sim_config, asset_config, portfolios)
            for i in range(config['num_simulations'])
        ]
        combined_run_args.extend(scenario_args)
    
    raw_results_list = []
    chunk_size = 1
    runs_completed = 0
    progress_interval = max(1, total_runs_overall // 20)
    start_time = time.time()

    print(f"Distributing {total_runs_overall} runs with chunksize {chunk_size}...", flush=True)
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        for result in pool.imap_unordered(run_single_mc_iteration_wrapper, combined_run_args, chunksize=chunk_size):
            raw_results_list.append(result)
            runs_completed += 1
            if runs_completed % progress_interval == 0 or runs_completed == total_runs_overall:
                elapsed_time = time.time() - start_time
                print(f"  Completed {runs_completed}/{total_runs_overall} total runs... ({elapsed_time:.1f}s elapsed)", flush=True)

    end_time = time.time()
    print(f"\nCompleted all {total_runs_overall} runs across {len(scenarios)} scenarios in {end_time - start_time:.2f} seconds.", flush=True)

    all_scenario_results = {}
    grouped_raw_results = {}
    
    for scenario_name, run_summary in raw_results_list:
        if run_summary is not None:
            if scenario_name not in grouped_raw_results:
                grouped_raw_results[scenario_name] = []
            grouped_raw_results[scenario_name].append(run_summary)

    for scenario_name, list_of_run_summaries in grouped_raw_results.items():
        if not list_of_run_summaries:
             print(f"Warning: No valid results collected for scenario '{scenario_name}'.", flush=True)
             all_scenario_results[scenario_name] = pd.DataFrame()
             continue

        results_df = pd.DataFrame.from_dict({(run_idx, portfolio_name): data
                                             for run_idx, run_data in enumerate(list_of_run_summaries)
                                             for portfolio_name, data in run_data.items()
                                             if data is not None}, orient='index')

        if results_df.empty:
            print(f"Warning: DataFrame empty after processing results for scenario '{scenario_name}'.", flush=True)
            all_scenario_results[scenario_name] = pd.DataFrame()
            continue

        results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=['Run', 'Portfolio'])

        aggregated_summary = results_df.groupby('Portfolio').agg(
            Avg_Final_Value=('Final Value', 'mean'), Median_Final_Value=('Final Value', 'median'),
            P5_Final_Value=('Final Value', lambda x: x.quantile(0.05)), P95_Final_Value=('Final Value', lambda x: x.quantile(0.95)),
            CVaR_5pct=('Final Value', lambda x: x[x <= x.quantile(0.05)].mean()),
            Avg_Ann_Return=('Annualized Return (%)', 'mean'), Median_Ann_Return=('Annualized Return (%)', 'median'),
            P5_Ann_Return=('Annualized Return (%)', lambda x: x.quantile(0.05)),
            Avg_Ann_Vol=('Annualized Volatility (%)', 'mean'), Median_Ann_Vol=('Annualized Volatility (%)', 'median'),
            Avg_Sharpe=('Sharpe Ratio', 'mean'), Median_Sharpe=('Sharpe Ratio', 'median'),
            Avg_Max_Drawdown=('Max Drawdown (%)', 'mean'), Median_Max_Drawdown=('Max Drawdown (%)', 'median'),
            Worst_Drawdown=('Max Drawdown (%)', 'min'),
            Median_Rebalances=('Rebalances', 'median'), P5_Rebalances=('Rebalances', lambda x: x.quantile(0.05)),
            P95_Rebalances=('Rebalances', lambda x: x.quantile(0.95))
        ).round(2)

        aggregated_summary = aggregated_summary.sort_values(by=["Median_Sharpe", "Median_Ann_Return"], ascending=[False, False])
        all_scenario_results[scenario_name] = aggregated_summary
    
    return StandardSimulationResults(results_by_scenario=all_scenario_results)


def run_goal_seeking_analysis(config, portfolios, scenarios, asset_config, base_seed):
    """Run goal-seeking mode to find time to reach target."""
    from goal_seeking import find_time_to_goal
    
    sim_config = SimulationConfig(
        initial_investment=config['initial_investment'],
        monthly_contribution=config['monthly_contribution'],
        years=config['goal_search_max_years'],
        months=config['goal_search_max_years'] * 12,
        rebalance_threshold=config['rebalance_threshold'],
        risk_free_rate=config['risk_free_rate'],
        num_simulations=config['num_simulations'],
        base_seed=base_seed,
        use_stochastic_recovery=config['use_stochastic_recovery'],
        recovery_uncertainty=config['recovery_uncertainty'],
        cap_individual_losses=config['cap_individual_losses'],
        asymmetric_rebalancing=config['asymmetric_rebalancing']
    )
    
    goal_config = GoalSeekingConfig(
        enable_goal_seeking=True,
        goal_metric=config['goal_metric'],
        goal_target_value=config['goal_target_value'],
        goal_search_min_years=config['goal_search_min_years'],
        goal_search_max_years=config['goal_search_max_years'],
        goal_tolerance=config['goal_tolerance']
    )
    
    print("\n" + "="*80, flush=True)
    print(f"GOAL-SEEKING MODE: Finding time to reach {config['goal_metric']} = ${config['goal_target_value']:,.0f}", flush=True)
    print("="*80, flush=True)
    print(f"Search range: {config['goal_search_min_years']} to {config['goal_search_max_years']} years", flush=True)
    print(f"Running {config['num_simulations']} simulations per portfolio per scenario...\n", flush=True)
    
    goal_results_all = []
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n--- Goal-Seeking: {scenario_name} ---", flush=True)
        
        for portfolio_name, allocation in portfolios.items():
            result = find_time_to_goal(
                portfolio_name, allocation, scenario,
                sim_config, goal_config, asset_config
            )
            goal_results_all.append(result)
    
    return GoalSeekingResults(results=goal_results_all)


# --- Main Execution Block ---
if __name__ == '__main__':
    overall_start_time = time.time()
    
    # Load all static configuration from YAML file
    from config import load_config_from_yaml, create_asset_config, create_scenarios
    app_config = load_config_from_yaml('config.yaml')
    
    portfolios = app_config['portfolios']
    asset_config = create_asset_config(app_config)
    scenarios = create_scenarios(app_config)
    
    # Base seed for reproducibility if needed, changes each script run
    base_seed = int(time.time())
    
    # Parse command-line arguments, using loaded config for defaults
    runtime_config = parse_command_line_arguments(app_config)
    
    # Run appropriate mode based on runtime config
    if runtime_config['enable_goal_seeking']:
        goal_results = run_goal_seeking_analysis(runtime_config, portfolios, scenarios, asset_config, base_seed)
        print_goal_seeking_results(goal_results, runtime_config, scenarios)
    else:
        standard_results = run_standard_simulation(runtime_config, portfolios, scenarios, asset_config, base_seed)
        print_standard_results(standard_results, runtime_config, portfolios, scenarios)
    
    overall_end_time = time.time()
    print(f"\n\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds.", flush=True)

