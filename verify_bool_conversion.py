import pandas as pd
import numpy as np
from causal_utils import generate_script

# 1. Test Logic
def convert_bool_to_int(df):
    # 1. Actual boolean types
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Converted boolean columns to integer: {', '.join(bool_cols)}")
    
    # 2. String "TRUE"/"FALSE" (case insensitive)
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        unique_vals = set(df[col].dropna().astype(str).str.upper().unique())
        if unique_vals.issubset({'TRUE', 'FALSE', 'T', 'F', '1', '0', '1.0', '0.0'}):
            try:
                mapping = {'TRUE': 1, 'FALSE': 0, 'T': 1, 'F': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
                df[col] = df[col].astype(str).str.upper().map(mapping).fillna(df[col])
                df[col] = pd.to_numeric(df[col], errors='ignore')
                print(f"Converted string boolean column to integer: {col}")
            except:
                pass
    return df

print("--- Testing Logic ---")
df = pd.DataFrame({
    'Real_Bool': [True, False, True],
    'String_Bool': ['TRUE', 'FALSE', 'True'],
    'Short_Bool': ['T', 'F', 'T'],
    'Mixed_Bool': ['1', '0', '1'],
    'Not_Bool': ['A', 'B', 'C']
})

print("Original Dtypes:")
print(df.dtypes)

df_converted = convert_bool_to_int(df.copy())

print("\nConverted Dtypes:")
print(df_converted.dtypes)

print("\nConverted Values:")
print(df_converted)

if df_converted['Real_Bool'].dtype == 'int64' or df_converted['Real_Bool'].dtype == 'int32':
    print("✅ Real_Bool converted to int.")
else:
    print("❌ Real_Bool NOT converted.")

if df_converted['String_Bool'].dtype == 'int64' or df_converted['String_Bool'].dtype == 'int32':
    print("✅ String_Bool converted to int.")
else:
    print("❌ String_Bool NOT converted.")

if df_converted['Not_Bool'].dtype == 'object':
    print("✅ Not_Bool kept as object.")
else:
    print("❌ Not_Bool incorrectly converted.")


# 2. Test Script Generation
print("\n--- Generating Script ---")
script = generate_script(
    data_source="Upload CSV", # To trigger the CSV path
    treatment="T",
    outcome="Y",
    confounders=[],
    time_period=None,
    estimation_method="A/B Test",
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
    n_iterations=50
)

print("\n--- Verifying Script Content ---")
if "def convert_bool_to_int(df):" in script:
    print("✅ convert_bool_to_int function found in script.")
else:
    print("❌ convert_bool_to_int function NOT found in script.")

if "df = convert_bool_to_int(df)" in script:
    print("✅ Function call found in script.")
else:
    print("❌ Function call NOT found in script.")
