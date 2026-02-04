
import pandas as pd
import numpy as np
import causal_utils
import warnings
warnings.filterwarnings('ignore')

def simulate_quasi_data():
    np.random.seed(42)
    n_samples = 1000
    
    # 1. DiD Data
    # Pre/Post and Control/Treatment
    time = np.random.binomial(1, 0.5, n_samples) # 0=Pre, 1=Post
    treatment_group = np.random.binomial(1, 0.5, n_samples) # 0=Control, 1=Treatment
    
    # Treatment is only active in Post period for Treatment Group
    treatment_active = time * treatment_group
    
    # Outcome
    # Baseline + 10*Group + 5*Time + 20*TreatmentEffect + noise
    outcome = 100 + 10*treatment_group + 5*time + 20*treatment_active + np.random.normal(0, 5, n_samples)
    
    df_did = pd.DataFrame({
        'Time': time,
        'Group': treatment_group,
        'Outcome': outcome,
        'Treatment_Active': treatment_active
    })
    
    # 2. CausalImpact Data (Time Series)
    # 100 days
    dates = pd.date_range(start='2023-01-01', periods=100)
    
    # Baseline trend
    y = np.linspace(10, 20, 100) + np.random.normal(0, 1, 100)
    
    # Intervention at day 70
    intervention_idx = 70
    effect = 5
    y[intervention_idx:] += effect
    
    df_ci = pd.DataFrame({
        'Date': dates,
        'Outcome': y
    })
    
    return df_did, df_ci

def test_did():
    print("Testing DiD Analysis...")
    df_did, _ = simulate_quasi_data()
    
    # run_did_analysis expects: df, treatment_col, outcome_col, time_col, confounders
    # Note: 'treatment_col' in run_did_analysis is the GROUP indicator (0/1), not the interaction.
    # The interaction is created internally as group * time.
    # Wait, my implementation of run_did_analysis:
    # df['DiD_Interaction'] = df[treatment_col] * df[time_col]
    # So yes, treatment_col should be the group assignment, and time_col the period.
    
    results = causal_utils.run_did_analysis(
        df_did, 
        treatment_col='Group', 
        outcome_col='Outcome', 
        time_col='Time', 
        confounders=[]
    )
    
    if 'error' in results:
        print(f"FAILED: {results['error']}")
    else:
        print("SUCCESS")
        print(f"Estimated Coefficient (True ~20): {results['coefficient']:.4f}")
        print(f"CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
        
        # Validation
        if 18 < results['coefficient'] < 22:
             print("CHECK: Estimate is within expected range.")
        else:
             print("CHECK WARNING: Estimate is off.")

def test_causal_impact():
    print("\nTesting CausalImpact Analysis...")
    _, df_ci = simulate_quasi_data()
    
    intervention_date = df_ci['Date'].iloc[70]
    
    results = causal_utils.run_causal_impact_analysis(
        df_ci,
        date_col='Date',
        outcome_col='Outcome',
        intervention_date=intervention_date
    )
    
    if 'error' in results:
        print(f"FAILED: {results['error']}")
    else:
        print("SUCCESS")
        print(f"Estimated ATE (True ~5): {results['ate']:.4f}")
        print(f"P-value: {results['p_value']:.4f}")
        
        if 4 < results['ate'] < 6:
            print("CHECK: Estimate is within expected range.")
        else:
            print("CHECK WARNING: Estimate is off.")

if __name__ == "__main__":
    test_did()
    test_causal_impact()
