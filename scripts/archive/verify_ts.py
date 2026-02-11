
import pandas as pd
import numpy as np
import causal_utils

# Create dummy data
df = pd.DataFrame({
    'T': np.random.binomial(1, 0.5, 100),
    'Y': np.random.normal(100, 10, 100),
    'X': np.random.normal(0, 1, 100)
    # Effect depends on nothing, random.
})
df['Y'] += df['T'] * 10 # True effect = 10

try:
    ate, ci_lower, ci_upper = causal_utils.calculate_period_effect(
        df, 'T', 'Y', ['X'], "Linear Double Machine Learning (LinearDML)"
    )
    print(f"DML ATE: {ate}")
    
    # Test OLS
    ate_ols, _, _  = causal_utils.calculate_period_effect(
        df, 'T', 'Y', ['X'], "Linear/Logistic Regression (OLS/Logit)"
    )
    print(f"OLS ATE: {ate_ols}")
    
    # Test Script Gen
    script = causal_utils.generate_script(
        data_source="Simulated Data", treatment='T', outcome='Y', confounders=['X'], 
        time_period=None, estimation_method="Linear/Logistic Regression (OLS/Logit)",
        impute_enable=False, num_impute_method=None, num_custom_val=0, cat_impute_method=None, cat_custom_val=None,
        winsorize_enable=False, winsorize_cols=[], percentile=0.05, log_transform_cols=[], standardize_cols=[], n_iterations=0,
        ts_params={'enabled': True, 'date_col': 'Date', 'freq': 'Monthly'}
    )
    if "df['Date'] = pd.to_datetime(df['Date'])" in script:
        print("Script generation includes Time Series logic.")
    else:
        print("Script generation MISSING Time Series logic.")

except Exception as e:
    print(f"Error: {e}")
