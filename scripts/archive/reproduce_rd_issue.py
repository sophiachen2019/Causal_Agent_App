import pandas as pd
import numpy as np
from dowhy import CausalModel
import sys

def reproduce_rd_issue():
    print("Reproducing RD Issue...")
    
    # Simulate Data
    np.random.seed(42)
    n_samples = 1000
    running_variable = np.random.uniform(0, 100, n_samples)
    cutoff = 50.0
    
    # Treatment assignment based on cutoff (Sharp RD)
    treatment = (running_variable >= cutoff).astype(int)
    
    # Outcome
    outcome = 2 * running_variable + 10 * treatment + np.random.normal(0, 5, n_samples)
    
    df = pd.DataFrame({
        'RunningVar': running_variable,
        'Treatment': treatment,
        'Outcome': outcome,
        'Confounder': np.random.normal(0, 1, n_samples) # Just to have one
    })
    
    # RD Setup as in the app
    df['RD_Indicator'] = (df['RunningVar'] >= cutoff).astype(int)
    
    print("Data simulated.")
    
    # Causal Model
    model = CausalModel(
        data=df,
        treatment='Treatment',
        outcome='Outcome',
        common_causes=['Confounder'],
        instruments=['RD_Indicator'],
        effect_modifiers=['Confounder']
    )
    
    print("Model built.")
    
    try:
        # Attempt Identification
        # The app uses 'iv.regression_discontinuity' or 'iv.instrumental_variable'
        # Let's try what was likely failing or what we want to test
        
        # Attempt Identification
        # Try auto-identification or explicitly asking for IV
        print("Attempting identification (auto)...")
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print("Identification successful.")
        print(f"Estimand type: {identified_estimand.estimand_type}")
        
        # Check if IV estimand is present
        if identified_estimand.estimands.get("iv", None):
             print("IV Estimand found.")
        else:
             print("IV Estimand NOT found.")

        # Attempt Estimation
        print("Attempting estimation with 'iv.instrumental_variable'...")
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="iv.instrumental_variable"
        )
        print(f"Estimation successful. ATE: {estimate.value}")
        
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_rd_issue()
