import pandas as pd
import numpy as np
from causal_utils import generate_script

# Mock Data
df = pd.DataFrame({
    'Age': np.random.randint(18, 80, 100),
    'Segment': np.random.choice(['A', 'B', 'C'], 100),
    'Income': np.random.normal(50000, 15000, 100)
})

# Mock Session State Logic
filtering_ops = []

# 1. Test Numeric Filter (> 30)
col_num = 'Age'
val_num = 30
df_filtered_num = df[df[col_num] > val_num]
filtering_ops.append({
    "col": col_num,
    "op": ">",
    "val": val_num
})
print(f"Filtered Age > 30. Rows: {len(df)} -> {len(df_filtered_num)}")

# 2. Test String Filter (== 'A')
col_str = 'Segment'
val_str = 'A'
df_filtered_str = df_filtered_num[df_filtered_num[col_str] == val_str]
filtering_ops.append({
    "col": col_str,
    "op": "==",
    "val": val_str
})
print(f"Filtered Segment == 'A'. Rows: {len(df_filtered_num)} -> {len(df_filtered_str)}")

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
    filtering_ops=filtering_ops
)

print("\n--- Verifying Script Content ---")
expected_num = "df = df[df['Age'] > 30]"
expected_str = "df = df[df['Segment'] == 'A']"

if expected_num in script:
    print("✅ Numeric filter code found in script.")
else:
    print("❌ Numeric filter code NOT found in script.")

if expected_str in script:
    print("✅ String filter code found in script.")
else:
    print("❌ String filter code NOT found in script.")

# Optional: Print snippet
print("\n--- Script Snippet ---")
start_idx = script.find("# Data Filtering")
print(script[start_idx:start_idx+300])
