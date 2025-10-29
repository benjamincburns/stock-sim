from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Any

@dataclass(frozen=True)
class StandardSimulationResults:
    """Holds the aggregated results of a standard time-bounded simulation."""
    results_by_scenario: Dict[str, pd.DataFrame]

@dataclass(frozen=True)
class GoalSeekingResults:
    """Holds the results of a goal-seeking analysis."""
    results: List[Dict[str, Any]]

def print_standard_results(results: StandardSimulationResults, config, portfolios, scenarios):
    """Renders the results of a standard simulation to the console."""
    print("\n--- Portfolio Compositions ---", flush=True)
    portfolio_order = [
        "Portfolio_3_50_50", "Portfolio_3_60_40", "Portfolio_3_70_30", "Portfolio_3_80_20",
        "Portfolio_1", "Portfolio_2", "Portfolio_4", "Portfolio_5", "VOO_100"
    ]
    valid_portfolio_order = [p for p in portfolio_order if p in portfolios]
    for name in valid_portfolio_order:
        allocation = portfolios[name]
        print(f"{name}:")
        for asset, weight in allocation.items():
            print(f"  {asset}: {weight*100:.0f}%")
    
    for name, allocation in portfolios.items():
        if name not in valid_portfolio_order:
            print(f"{name}:")
            for asset, weight in allocation.items():
                print(f"  {asset}: {weight*100:.0f}%")
    
    print(f"\n--- Initial Investment: ${config['initial_investment']:,.2f} ---", flush=True)
    total_invested = config['initial_investment'] + (config['months'] * config['monthly_contribution'])
    print(f"--- Monthly Contribution: ${config['monthly_contribution']:,.2f} ---", flush=True)
    print(f"--- Total Invested: ${total_invested:,.2f} ---", flush=True)

    for scenario_name, summary_table in results.results_by_scenario.items():
        print(f"\n\n--- Monte Carlo Simulation Summary: {scenario_name} ({config['num_simulations']} Runs) ---", flush=True)

        sc_crash_dur = scenarios[scenario_name].crash_duration
        sc_rec_dur = scenarios[scenario_name].recovery_duration
        sc_growth_dur = config['months'] - sc_crash_dur - sc_rec_dur

        print(f"Time Period: {config['years']} years ({config['months']} months).", flush=True)
        if sc_crash_dur > 0:
            print("Scenario: Assumes market CRASH + RECOVERY, WITH monthly contributions.", flush=True)
            print(f"Phase 1 (Months 1-{sc_crash_dur}): Market Crash")
            print(f" - US stocks (VOO) fall ~55%, Intl stocks fall ~47-48%.")
            print(f" - High-quality bonds (GOVT/IGOV) rise ~15%.")
            print(f" - High volatility and negative stock/bond correlation assumed.")
            if sc_rec_dur > 0:
                print(f"Phase 2 (Months {sc_crash_dur+1}-{sc_crash_dur+sc_rec_dur}): Recovery")
                print(f" - Stocks recover to their pre-crash value over {sc_rec_dur} months.")
                print(f" - Bonds give back their gains over the first 12 months of this phase.")
                print(f" - Moderate volatility and low stock/bond correlation assumed.")
            if sc_growth_dur > 0:
                print(f"Phase 3 (Remaining Months to {config['months']}): Stable Growth")
                print(f" - Assumes moderate annual growth (e.g., US Stocks +9%, Bonds +3%).")
                print(f" - Lower volatility and low stock/bond correlation assumed.")
        else:
            print("Scenario: Assumes NO market crash, WITH monthly contributions.", flush=True)
            print(f" - Applies stable growth assumptions for the entire period.")
            print(f" - Assumes moderate annual growth (e.g., US Stocks +9%, Bonds +3%).")
            print(f" - Assumes lower volatility and low stock/bond correlation.")

        pd.options.display.float_format = '{:,.2f}'.format
        pd.options.display.max_columns = None
        pd.options.display.width = None
        
        if not summary_table.empty:
            print("\n=== Main Performance Metrics ===", flush=True)
            print(summary_table[[
                'Median_Final_Value', 'Avg_Final_Value', 'P5_Final_Value', 'P95_Final_Value',
                'Median_Ann_Return', 'P5_Ann_Return', 'Median_Ann_Vol', 'Median_Sharpe'
            ]], flush=True)
            
            print("\n=== Risk & Tail Metrics ===", flush=True)
            print(summary_table[[
                'Median_Max_Drawdown', 'Worst_Drawdown', 'P5_Final_Value', 'CVaR_5pct',
                'Median_Rebalances', 'P5_Rebalances', 'P95_Rebalances'
            ]], flush=True)
        else:
            print("  (No valid results for this scenario)")


def print_goal_seeking_results(results: GoalSeekingResults, config, scenarios):
    """Renders the results of a goal-seeking analysis to the console."""
    print("\n" + "="*80, flush=True)
    print("GOAL-SEEKING RESULTS SUMMARY", flush=True)
    print("="*80, flush=True)
    
    for scenario_name in scenarios.keys():
        scenario_results = [r for r in results.results if r['scenario'] == scenario_name]
        
        print(f"\n\n--- {scenario_name} ---", flush=True)
        print(f"Goal: {config['goal_metric']} = ${config['goal_target_value']:,.0f}\n", flush=True)
        
        df_data = []
        for r in scenario_results:
            # Consolidate common data extraction to reduce duplication
            data = {
                'Portfolio': r.get('portfolio'),
                'Median_Final_Value': r.get('Median_Final_Value', 0),
                'Avg_Final_Value': r.get('Avg_Final_Value', 0),
                'P95_Final_Value': r.get('P95_Final_Value', 0),
                'P5_Final_Value': r.get('P5_Final_Value', 0),
                'CVaR_5pct': r.get('CVaR_5pct', 0),
                'Ann_Return_%': r.get('Ann_Return', 0),
                'P5_Ann_Return_%': r.get('P5_Ann_Return', 0),
                'Ann_Vol_%': r.get('Ann_Volatility', 0),
                'Sharpe': r.get('Sharpe_Ratio', 0),
                'Median_DD_%': r.get('Median_Drawdown', 0),
                'Worst_DD_%': r.get('Worst_Drawdown', 0),
                'Median_Rebalances': r.get('Median_Rebalances', 0),
                'P5_Rebalances': r.get('P5_Rebalances', 0),
                'P95_Rebalances': r.get('P95_Rebalances', 0),
            }

            # Handle the different logic for time-to-goal display
            if r.get('goal_reached'):
                years = r.get('years', 0)
                years_int = int(years)
                months_remaining = int((years - years_int) * 12)
                data['Time_to_Goal'] = f"{years_int}y {months_remaining}m"
                data['Years'] = years
            else:
                years = r.get('years')  # Can be None if sim failed
                if years is not None:
                    years_int = int(years)
                    months_remaining = int((years - years_int) * 12)
                    data['Time_to_Goal'] = f"{years_int}y {months_remaining}m*"
                else:
                    data['Time_to_Goal'] = "Not reached"
                data['Years'] = years if years is not None else 999
            
            df_data.append(data)
        
        if df_data:
            df_goal = pd.DataFrame(df_data).sort_values('Years').set_index('Portfolio')
            
            # Format columns into strings for aligned printing in a single table
            df_display = pd.DataFrame(index=df_goal.index)
            df_display['Time_to_Goal'] = df_goal['Time_to_Goal']
            df_display['Median_Final_Value'] = df_goal['Median_Final_Value'].apply('{:,.0f}'.format)
            df_display['Avg_Final_Value'] = df_goal['Avg_Final_Value'].apply('{:,.0f}'.format)
            df_display['P95_Final_Value'] = df_goal['P95_Final_Value'].apply('{:,.0f}'.format)
            df_display['P5_Final_Value'] = df_goal['P5_Final_Value'].apply('{:,.0f}'.format)
            df_display['CVaR_5pct'] = df_goal['CVaR_5pct'].apply('{:,.0f}'.format)
            df_display['Ann_Return_%'] = df_goal['Ann_Return_%'].apply('{:.2f}'.format)
            df_display['P5_Ann_Return_%'] = df_goal['P5_Ann_Return_%'].apply('{:.2f}'.format)
            df_display['Ann_Vol_%'] = df_goal['Ann_Vol_%'].apply('{:.2f}'.format)
            df_display['Sharpe'] = df_goal['Sharpe'].apply('{:.2f}'.format)
            df_display['Median_DD_%'] = df_goal['Median_DD_%'].apply('{:.2f}'.format)
            df_display['Worst_DD_%'] = df_goal['Worst_DD_%'].apply('{:.2f}'.format)
            df_display['Median_Rebalances'] = df_goal['Median_Rebalances'].apply('{:.0f}'.format)
            df_display['P5_Rebalances'] = df_goal['P5_Rebalances'].apply('{:.0f}'.format)
            df_display['P95_Rebalances'] = df_goal['P95_Rebalances'].apply('{:.0f}'.format)
            
            pd.options.display.max_columns = None
            pd.options.display.width = None

            print(df_display)
            
            if any(not r['goal_reached'] for r in scenario_results):
                print("\n* Goal not reached within search range - values shown at max years")
    
    print("\n" + "="*80, flush=True)
    print(f"Goal-seeking analysis complete!", flush=True)
    print("="*80 + "\n", flush=True)
