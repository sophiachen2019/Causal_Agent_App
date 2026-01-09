from causal_utils import generate_script

# Verify Export Script Generation for A/B Test with Logit
print("\nGenerating script for A/B Test with Logit...")
script = generate_script(
    data_source="Simulated Data",
    treatment="Feature_Adoption",
    outcome="Conversion", # Binary
    confounders=["Customer_Segment"],
    time_period=None,
    estimation_method="A/B Test (Difference in Means)",
    impute_enable=False, num_impute_method="Mean", num_custom_val=0, cat_impute_method="Mode", cat_custom_val="Missing",
    winsorize_enable=False, winsorize_cols=[], percentile=0.05,
    log_transform_cols=[], standardize_cols=[], n_iterations=10,
    use_logit=True # Enable Logit
)

if "sm.Logit(y_ab, X_ab)" in script and "Estimated Odds Ratio" in script:
    print("SUCCESS: A/B Test script uses Logit.")
else:
    print("FAILURE: A/B Test script does not use Logit.")
    print(script)
    exit(1)

# Verify Export Script Generation for DiD with Logit
print("\nGenerating script for DiD with Logit...")
script_did = generate_script(
    data_source="Simulated Data",
    treatment="Feature_Adoption",
    outcome="Conversion", # Binary
    confounders=["Customer_Segment"],
    time_period="Quarter",
    estimation_method="Difference-in-Differences (DiD)",
    impute_enable=False, num_impute_method="Mean", num_custom_val=0, cat_impute_method="Mode", cat_custom_val="Missing",
    winsorize_enable=False, winsorize_cols=[], percentile=0.05,
    log_transform_cols=[], standardize_cols=[], n_iterations=10,
    use_logit=True # Enable Logit
)

if "sm.Logit(y_did, X_did)" in script_did and "Estimated Odds Ratio (Interaction)" in script_did:
    print("SUCCESS: DiD script uses Logit.")
else:
    print("FAILURE: DiD script does not use Logit.")
    exit(1)

print("\nAll verifications passed!")
