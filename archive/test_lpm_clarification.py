from causal_utils import generate_script

# Verify Export Script Generation for A/B Test with Binary Outcome
print("\nGenerating script for A/B Test with Binary Outcome...")
script = generate_script(
    data_source="Simulated Data",
    treatment="Feature_Adoption",
    outcome="Conversion", # Binary
    confounders=["Customer_Segment"],
    time_period=None,
    estimation_method="A/B Test (Difference in Means)",
    impute_enable=False, num_impute_method="Mean", num_custom_val=0, cat_impute_method="Mode", cat_custom_val="Missing",
    winsorize_enable=False, winsorize_cols=[], percentile=0.05,
    log_transform_cols=[], standardize_cols=[], n_iterations=10
)

if "Binary Outcome: Using Linear Probability Model (LPM)" in script:
    print("SUCCESS: A/B Test script includes LPM comment.")
else:
    print("FAILURE: A/B Test script missing LPM comment.")
    print(script)
    exit(1)

# Verify Export Script Generation for DiD with Binary Outcome
print("\nGenerating script for DiD with Binary Outcome...")
script_did = generate_script(
    data_source="Simulated Data",
    treatment="Feature_Adoption",
    outcome="Conversion", # Binary
    confounders=["Customer_Segment"],
    time_period="Quarter",
    estimation_method="Difference-in-Differences (DiD)",
    impute_enable=False, num_impute_method="Mean", num_custom_val=0, cat_impute_method="Mode", cat_custom_val="Missing",
    winsorize_enable=False, winsorize_cols=[], percentile=0.05,
    log_transform_cols=[], standardize_cols=[], n_iterations=10
)

if "Binary Outcome: Using Linear Probability Model (LPM)" in script_did:
    print("SUCCESS: DiD script includes LPM comment.")
else:
    print("FAILURE: DiD script missing LPM comment.")
    exit(1)

print("\nAll verifications passed!")
