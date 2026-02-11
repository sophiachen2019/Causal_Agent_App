import dowhy
import econml
import pandas as pd
import numpy as np
from dowhy import CausalModel
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

print("Libraries imported successfully.")

# Quick simulation
df = pd.DataFrame({
    'T': np.random.binomial(1, 0.5, 100),
    'Y': np.random.normal(0, 1, 100),
    'X': np.random.normal(0, 1, 100)
})

# Quick model build
model = CausalModel(
    data=df,
    treatment='T',
    outcome='Y',
    common_causes=['X']
)
print("Model built successfully.")

# Quick identification
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("Identification successful.")

# Quick estimation
try:
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.dml.LinearDML",
        method_params={
            "init_params": {
                "model_y": RandomForestRegressor(),
                "model_t": RandomForestClassifier(),
                "discrete_treatment": True,
                "linear_first_stages": False,
                "cv": 3,
            },
            "fit_params": {}
        }
    )
    print("Estimation successful.")
except Exception as e:
    print(f"Estimation failed: {e}")

print("Verification complete.")
