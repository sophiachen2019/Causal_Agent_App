import pandas as pd
import numpy as np
from causal_utils import generate_script

# Mock Data
df = pd.DataFrame({
    'Age': np.random.randint(18, 80, 100),
    'Income': np.random.normal(50000, 15000, 100)
})

# Mock Session State Logic
bucketing_ops = []

# 1. Test Equal Width (cut)
new_col_cut = 'Age_Group'
df[new_col_cut] = pd.cut(df['Age'], bins=4, labels=False)
bucketing_ops.append({
    "col": 'Age',
    "n_bins": 4,
    "method": 'cut',
    "new_col": new_col_cut
})
print(f"Created {new_col_cut} with cut. Unique values: {df[new_col_cut].unique()}")

# 2. Test Equal Frequency (qcut)
new_col_qcut = 'Income_Tier'
df[new_col_qcut] = pd.qcut(df['Income'], q=3, labels=False, duplicates='drop')
bucketing_ops.append({
    "col": 'Income',
    "n_bins": 3,
    "method": 'qcut',
    "new_col": new_col_qcut
})
print(f"Created {new_col_qcut} with qcut. Unique values: {df[new_col_qcut].unique()}")

# 3. Test Script Generation
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
    bucketing_ops=bucketing_ops
)

print("\n--- Verifying Script Content ---")
expected_cut = "df['Age_Group'] = pd.cut(df['Age'], bins=4, labels=False)"
expected_qcut = "df['Income_Tier'] = pd.qcut(df['Income'], q=3, labels=False, duplicates='drop')"

if expected_cut in script:
    print("✅ Equal Width (cut) code found in script.")
else:
    print("❌ Equal Width (cut) code NOT found in script.")

if expected_qcut in script:
    print("✅ Equal Frequency (qcut) code found in script.")
else:
    print("❌ Equal Frequency (qcut) code NOT found in script.")

# Optional: Print snippet
print("\n--- Script Snippet ---")
start_idx = script.find("# Variable Bucketing")
print(script[start_idx:start_idx+300])
