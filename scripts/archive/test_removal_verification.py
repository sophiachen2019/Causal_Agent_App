import sys
import os
import pandas as pd
from causal_utils import generate_script

# Mock parameters for A/B Test (should work without RD args)
params = {
    'data_source': "Simulated Data",
    'treatment': "Feature_Adoption",
    'outcome': "Account_Value",
    'confounders': "['Customer_Segment', 'Historical_Usage']",
    'time_period': None,
    'estimation_method': "A/B Test (Difference in Means)",
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
    # RD args removed
    'control_val': 0,
    'treat_val': 1,
    'hte_features': None
}

print("Generating script for A/B Test (Verification)...")
try:
    generated_code = generate_script(**params)
    print("SUCCESS: Script generated successfully without RD arguments.")
except TypeError as e:
    print(f"FAILURE: TypeError during script generation (likely due to unexpected arguments): {e}")
except Exception as e:
    print(f"FAILURE: Unexpected error: {e}")

# Verify IV/RD code is NOT in the generated script
if "iv.instrumental_variable" in generated_code or "iv.regression_discontinuity" in generated_code:
    print("FAILURE: IV/RD code found in generated script!")
else:
    print("SUCCESS: IV/RD code correctly absent from generated script.")
