import pandas as pd
import numpy as np
from causal_utils import generate_script

# Mock Data
df = pd.DataFrame({
    'Segment': ['A', 'B', 'A', 'C', 'B'],
    'Value': [10, 20, 15, 30, 25]
})

# Mock Session State Logic
dummy_ops = []

# 1. Test Dummy Creation (Drop First = True)
col = 'Segment'
drop_first = True
dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
df_with_dummies = pd.concat([df, dummies], axis=1)

dummy_ops.append({
    "col": col,
    "drop_first": drop_first
})

print(f"Original Columns: {df.columns.tolist()}")
print(f"New Columns: {df_with_dummies.columns.tolist()}")
print(f"Added Columns: {[c for c in df_with_dummies.columns if c not in df.columns]}")

# 2. Test Script Generation
print("\n--- Generating Script ---")
script = generate_script(
    data_source="Simulated Data",
    treatment="Feature_Adoption",
    outcome="Account_Value",
    confounders=[],
    time_period=None,
    estimation_method="A/B Test (Difference in Means)",
    impute_enable=False,
    num_impute_method=None,
    num_custom_val=0,
    cat_impute_method=None,
    cat_custom_val="",
    winsorize_enable=False,
    winsorize_cols=[],
    percentile=0.05,
    log_transform_cols=[],
    standardize_cols=[],
    n_iterations=50,
    dummy_ops=dummy_ops
)

print("\n--- Verifying Script Content ---")
expected_code = "dummies = pd.get_dummies(df['Segment'], prefix='Segment', drop_first=True)"
expected_concat = "df = pd.concat([df, dummies], axis=1)"

if expected_code in script:
    print("✅ Dummy creation code found in script.")
else:
    print("❌ Dummy creation code NOT found in script.")

if expected_concat in script:
    print("✅ Concat code found in script.")
else:
    print("❌ Concat code NOT found in script.")

# Optional: Print snippet
print("\n--- Script Snippet ---")
start_idx = script.find("# Dummy Variable Creation")
print(script[start_idx:start_idx+300])
