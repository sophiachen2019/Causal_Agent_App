import pandas as pd
import numpy as np

def simulate_bsts_demo_data():
    np.random.seed(42)
    regions = [f'Region_{i}' for i in range(1, 41)]
    total_days = 364
    start_date = pd.to_datetime('2023-01-01')
    date_range = pd.date_range(start=start_date, periods=total_days)
    data_list = []
    
    global_trend = np.linspace(100, 150, total_days)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(total_days) / 7)
    monthly_seasonality = 20 * np.sin(2 * np.pi * np.arange(total_days) / 30)
    
    for region in regions:
        base_offset = np.random.normal(0, 20)
        regional_trend = global_trend + base_offset
        noise = np.random.normal(0, 2, total_days)
        metric = regional_trend + weekly_seasonality + monthly_seasonality + noise
        
        intervention_day = 304
        if region == 'Region_1':
            lift = np.zeros(total_days)
            lift[intervention_day:] = 30 + np.cumsum(np.random.normal(0.5, 0.1, total_days - intervention_day))
            metric += lift
        
        region_df = pd.DataFrame({
            'Date': pd.to_datetime(date_range).floor('D'), 'Region': region, 'Daily_Revenue': metric,
            'Marketing_Spend': np.random.normal(50, 5, total_days),
            'App_Downloads': np.random.poisson(500, total_days) + (regional_trend).astype(int),
            'Website_Traffic': np.random.normal(5000, 500, total_days) + regional_trend * 10,
            'Social_Media_Mentions': np.random.poisson(100, total_days) + (regional_trend / 2).astype(int),
            'Is_Post_Intervention': (np.arange(total_days) >= intervention_day).astype(int),
            'Is_Treated_Region': 1 if region == 'Region_1' else 0
        })
        data_list.append(region_df)
    return pd.concat(data_list, ignore_index=True)

import causal_utils

def main():
    print("Simulating dataset...")
    df = simulate_bsts_demo_data()
    
    date_col = 'Date'
    geo_col = 'Region'
    kpi_col = 'Daily_Revenue'
    treated_geo = 'Region_1'
    covariates = ['Marketing_Spend', 'App_Downloads', 'Website_Traffic', 'Social_Media_Mentions']
    
    # Intervention date is day 304, from start 2023-01-01 -> 2023-11-01
    intervention_date_str = '2023-11-01'
    treatment_duration = 60
    
    results_list = []
    
    print("Running GeoLift without covariates...")
    res_gl_no = causal_utils.run_geolift_analysis(
        df, date_col, geo_col, treated_geo, kpi_col, intervention_date_str,
        treatment_duration=treatment_duration,
        model="none",
        confidence_intervals=True
    )
    if 'metrics' in res_gl_no:
        m = res_gl_no['metrics']
        rl_fmt = f"{m['perc_lift']:.2%} [{m['rel_lower']:.2%}, {m['rel_upper']:.2%}]" if m.get('rel_lower') else f"{m['perc_lift']:.2%}"
        results_list.append({
            "Method": "GeoLift", "Covariates": "No",
            "ATE": f"{m['avg_lift']:.2f} [{m['ate_lower']:.2f}, {m['ate_upper']:.2f}]",
            "Cum Lift": f"{m['cum_lift']:.2f} [{m['cum_lower']:.2f}, {m['cum_upper']:.2f}]",
            "Rel Lift": rl_fmt,
            "P-Value": round(m['p_val'], 3)
        })

    print("Running GeoLift with covariates...")
    res_gl_cov = causal_utils.run_geolift_analysis(
        df, date_col, geo_col, treated_geo, kpi_col, intervention_date_str,
        treatment_duration=treatment_duration,
        model="none", covariates=covariates,
        confidence_intervals=True
    )
    if 'metrics' in res_gl_cov:
        m = res_gl_cov['metrics']
        rl_fmt = f"{m['perc_lift']:.2%} [{m['rel_lower']:.2%}, {m['rel_upper']:.2%}]" if m.get('rel_lower') else f"{m['perc_lift']:.2%}"
        results_list.append({
            "Method": "GeoLift", "Covariates": "Yes",
            "ATE": f"{m['avg_lift']:.2f} [{m['ate_lower']:.2f}, {m['ate_upper']:.2f}]",
            "Cum Lift": f"{m['cum_lift']:.2f} [{m['cum_lower']:.2f}, {m['cum_upper']:.2f}]",
            "Rel Lift": rl_fmt,
            "P-Value": round(m['p_val'], 3)
        })

    print("Running CausalImpact without covariates...")
    res_ci_no = causal_utils.run_causal_impact_analysis(
        df, date_col, kpi_col, intervention_date_str,
        unit_col=geo_col, treated_unit=treated_geo, use_panel=True
    )
    if 'metrics' in res_ci_no:
        m = res_ci_no['metrics']
        results_list.append({
            "Method": "CausalImpact", "Covariates": "No",
            "ATE": f"{m['ate']:.2f} [{m['ate_lower']:.2f}, {m['ate_upper']:.2f}]",
            "Cum Lift": f"{m['cum_abs']:.2f} [{m['cum_lower']:.2f}, {m['cum_upper']:.2f}]",
            "Rel Lift": f"{m['rel_effect']:.2%} [{m['rel_lower']:.2%}, {m['rel_upper']:.2%}]",
            "P-Value": round(m['p_value'], 3)
        })

    print("Running CausalImpact with covariates...")
    res_ci_cov = causal_utils.run_causal_impact_analysis(
        df, date_col, kpi_col, intervention_date_str,
        unit_col=geo_col, treated_unit=treated_geo, use_panel=True, covariates=covariates
    )
    if 'metrics' in res_ci_cov:
        m = res_ci_cov['metrics']
        results_list.append({
            "Method": "CausalImpact", "Covariates": "Yes",
            "ATE": f"{m['ate']:.2f} [{m['ate_lower']:.2f}, {m['ate_upper']:.2f}]",
            "Cum Lift": f"{m['cum_abs']:.2f} [{m['cum_lower']:.2f}, {m['cum_upper']:.2f}]",
            "Rel Lift": f"{m['rel_effect']:.2%} [{m['rel_lower']:.2%}, {m['rel_upper']:.2%}]",
            "P-Value": round(m['p_value'], 3)
        })

    md_table = "| Method | Covariates | ATE (Average Effect) | Cumulative Lift | Relative Lift | P-Value |\n|---|---|---|---|---|---|\n"
    for row in results_list:
        md_table += f"| {row['Method']} | {row['Covariates']} | {row['ATE']} | {row['Cum Lift']} | {row['Rel Lift']} | {row['P-Value']} |\n"
    
    with open('/Users/sophiachen/.gemini/antigravity/brain/53df59a0-9d17-4102-b658-503c037e7d00/results_summary.md', 'w') as f:
        f.write("# Causal Inference Experiment Results\n\n")
        f.write(md_table)
    
    print("\nResults Table:")
    print(md_table)

if __name__ == '__main__':
    main()
