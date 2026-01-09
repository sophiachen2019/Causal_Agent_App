import pandas as pd
import numpy as np
from causal_utils import generate_script

# 1. Verify Simulation Logic
def simulate_data(n_samples=100):
    np.random.seed(42)
    customer_segment = np.random.binomial(1, 0.3, n_samples)
    historical_usage = np.random.normal(50, 15, n_samples) + (customer_segment * 20)
    marketing_nudge = np.random.binomial(1, 0.5, n_samples)
    quarter = np.random.binomial(1, 0.5, n_samples)
    prob_adoption = 1 / (1 + np.exp(-( -2 + 0.5 * customer_segment + 0.05 * historical_usage + 1.5 * marketing_nudge)))
    feature_adoption = np.random.binomial(1, prob_adoption, n_samples)
    
    # Outcome: Conversion (Binary)
    prob_conversion = 1 / (1 + np.exp(-( -1 + 0.5 * customer_segment + 0.5 * feature_adoption)))
    conversion = np.random.binomial(1, prob_conversion, n_samples)
    
    df = pd.DataFrame({
        'Customer_Segment': customer_segment,
        'Historical_Usage': historical_usage,
        'Marketing_Nudge': marketing_nudge,
        'Quarter': quarter,
        'Feature_Adoption': feature_adoption,
        'Conversion': conversion
    })
    return df

df = simulate_data()
print(f"Simulated Data Columns: {df.columns.tolist()}")
print(f"Conversion Unique Values: {df['Conversion'].unique()}")

if 'Conversion' not in df.columns:
    print("FAILURE: Conversion column missing.")
    exit(1)

if not set(df['Conversion'].unique()).issubset({0, 1}):
    print("FAILURE: Conversion is not binary.")
    exit(1)

# 2. Verify Export Script Generation for Binary Outcome
print("\nGenerating script for Binary Outcome (LinearDML)...")
script = generate_script(
    data_source="Simulated Data",
    treatment="Feature_Adoption",
    outcome="Conversion",
    confounders=["Customer_Segment", "Historical_Usage"],
    time_period=None,
    estimation_method="Double Machine Learning (LinearDML)",
    impute_enable=False, num_impute_method="Mean", num_custom_val=0, cat_impute_method="Mode", cat_custom_val="Missing",
    winsorize_enable=False, winsorize_cols=[], percentile=0.05,
    log_transform_cols=[], standardize_cols=[], n_iterations=10
)

if "RandomForestClassifier" in script and "model_y = RandomForestClassifier" in script:
    print("SUCCESS: LinearDML uses Classifier for binary outcome.")
else:
    print("FAILURE: LinearDML does not use Classifier for binary outcome.")
    print(script)
    exit(1)

print("\nGenerating script for Binary Outcome (S-Learner)...")
script_s = generate_script(
    data_source="Simulated Data",
    treatment="Feature_Adoption",
    outcome="Conversion",
    confounders=["Customer_Segment", "Historical_Usage"],
    time_period=None,
    estimation_method="Meta-Learner: S-Learner",
    impute_enable=False, num_impute_method="Mean", num_custom_val=0, cat_impute_method="Mode", cat_custom_val="Missing",
    winsorize_enable=False, winsorize_cols=[], percentile=0.05,
    log_transform_cols=[], standardize_cols=[], n_iterations=10
)

if "overall_model = RandomForestClassifier" in script_s:
    print("SUCCESS: S-Learner uses Classifier for binary outcome.")
else:
    print("FAILURE: S-Learner does not use Classifier for binary outcome.")
    exit(1)

print("\nAll verifications passed!")
