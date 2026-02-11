import sys
import os
import pandas as pd
from causal_utils import generate_script

# Mock parameters for DiD with Multi-Feature HTE
params = {
    'data_source': "Simulated Data",
    'treatment': "Feature_Adoption",
    'outcome': "Account_Value",
    'confounders': "['Customer_Segment', 'Historical_Usage']",
    'time_period': "Quarter",
    'estimation_method': "Difference-in-Differences (DiD)",
    'impute_enable': False,
    'num_impute_method': "Mean",
    'num_custom_val': 0,
    'cat_impute_method': "Mode",
    'cat_custom_val': "Missing",
    'winsorize_enable': False,
    'winsorize_cols': [],
    'percentile': 0.05,
    'log_transform_cols': [],
    'standardize_cols': [],
    'n_iterations': 10,
    'rd_running_variable': None,
    'rd_cutoff': 0.0,
    'rd_bandwidth': 0.0,
    'control_val': None,
    'treat_val': None,
    'hte_features': "['Customer_Segment', 'Historical_Usage']" # Pass as string repr of list
}

print("Generating script for DiD with Multi-Feature HTE...")
generated_code = generate_script(**params)

print("Executing generated script...")
try:
    exec(generated_code)
    print("\nSUCCESS: Generated script executed successfully.")
except Exception as e:
    print(f"\nFAILURE: Generated script execution failed: {e}")
    # Print the code for debugging
    print("\n--- Generated Code ---")
    print(generated_code)
