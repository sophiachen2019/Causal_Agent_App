import sys
import os
import pandas as pd
from causal_utils import generate_script

# Mock parameters for S-Learner with HTE
params = {
    'data_source': "Simulated Data",
    'treatment': "Feature_Adoption",
    'outcome': "Account_Value",
    'confounders': "['Customer_Segment', 'Historical_Usage']",
    'time_period': None, # Not needed for S-Learner
    'estimation_method': "Meta-Learner: S-Learner",
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
    'control_val': 0,
    'treat_val': 1,
    'hte_features': "['Customer_Segment', 'Historical_Usage']"
}

print("Generating script for S-Learner with HTE...")
generated_code = generate_script(**params)

import traceback

print("Executing generated script...")
try:
    exec(generated_code)
    print("\nSUCCESS: Generated script executed successfully.")
except Exception:
    print("\nFAILURE: Generated script execution failed:")
    traceback.print_exc()
    # Print the code for debugging
    print("\n--- Generated Code ---")
    print(generated_code)
