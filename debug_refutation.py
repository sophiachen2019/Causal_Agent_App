import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import statsmodels.api as sm

# Simulate Data
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'Customer_Segment': np.random.binomial(1, 0.3, n_samples),
    'Historical_Usage': np.random.normal(50, 15, n_samples),
    'Marketing_Nudge': np.random.binomial(1, 0.5, n_samples),
    'Feature_Adoption': np.random.binomial(1, 0.5, n_samples), # Treatment
    'Conversion': np.random.binomial(1, 0.3, n_samples) # Binary Outcome
})

treatment = 'Feature_Adoption'
outcome = 'Conversion'
confounders = ['Customer_Segment', 'Historical_Usage']

# Define Causal Model
model = CausalModel(
    data=df,
    treatment=treatment,
    outcome=outcome,
    common_causes=confounders
)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

print("\n--- Testing A/B Logit Refutation ---")
try:
    # Manual A/B Logit
    X_ab_cols = [treatment] + confounders
    X_ab = df[X_ab_cols]
    X_ab = sm.add_constant(X_ab)
    y_ab = df[outcome]
    
    ab_model = sm.Logit(y_ab, X_ab).fit(disp=0)
    ab_coeff = ab_model.params[treatment]
    
    # Mimic the dummy object from the app
    # CASE 1: The one that likely failed (A/B Logit)
    estimate = type('obj', (object,), {'value': ab_coeff})
    
    print("Running Placebo Refutation...")
    refute = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=2
    )
    print("Refutation Success!")
except Exception as e:
    print(f"Refutation Failed: {e}")

print("\n--- Testing DiD Logit Refutation ---")
try:
    # Manual DiD Logit
    # ... (simplified for brevity, just testing the object structure)
    estimate_did = type('obj', (object,), {
        'value': 0.5, 
        'estimator': type('obj', (object,), {'__str__': lambda self: "Logistic Regression (DiD)"})()
    })
    
    print("Running Placebo Refutation...")
    refute = model.refute_estimate(
        identified_estimand,
        estimate_did,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=2
    )
    print("Refutation Success!")
except Exception as e:
    print(f"Refutation Failed: {e}")
