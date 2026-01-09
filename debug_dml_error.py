import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import LinearDML, CausalForestDML
from econml.metalearners import SLearner, TLearner

# Simulate Data with Binary Outcome
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
    common_causes=confounders,
    effect_modifiers=confounders # Add effect_modifiers for CausalForestDML
)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Test LinearDML with Classifier for model_y
print("\n--- Testing LinearDML with RandomForestRegressor (Fix) ---")
try:
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.dml.LinearDML",
        method_params={
            "init_params": {
                "model_y": RandomForestRegressor(random_state=42), # Changed to Regressor
                "model_t": RandomForestClassifier(random_state=42),
                "discrete_treatment": True,
                "linear_first_stages": False,
                "cv": 3,
                "random_state": 42
            },
            "fit_params": {}
        }
    )
    print("LinearDML Success!")
except Exception as e:
    print(f"LinearDML Failed: {e}")

# Test CausalForestDML with Classifier for model_y
print("\n--- Testing CausalForestDML with RandomForestRegressor (Fix) ---")
try:
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.dml.CausalForestDML",
        method_params={
            "init_params": {
                "model_y": RandomForestRegressor(random_state=42), # Changed to Regressor
                "model_t": RandomForestClassifier(random_state=42),
                "discrete_treatment": True,
                "cv": 3,
                "random_state": 42
            },
            "fit_params": {}
        }
    )
    print("CausalForestDML Success!")
except Exception as e:
    print(f"CausalForestDML Failed: {e}")

# Test S-Learner with Classifier
print("\n--- Testing S-Learner with RandomForestClassifier ---")
try:
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.metalearners.SLearner",
        method_params={
            "init_params": {
                "overall_model": RandomForestClassifier(random_state=42)
            },
            "fit_params": {}
        }
    )
    print("S-Learner Success!")
except Exception as e:
    print(f"S-Learner Failed: {e}")

# Test T-Learner with Classifier
print("\n--- Testing T-Learner with RandomForestClassifier ---")
try:
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.metalearners.TLearner",
        method_params={
            "init_params": {
                "models": RandomForestClassifier(random_state=42)
            },
            "fit_params": {}
        }
    )
    print("T-Learner Success!")
except Exception as e:
    print(f"T-Learner Failed: {e}")
