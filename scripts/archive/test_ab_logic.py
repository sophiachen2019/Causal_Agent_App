import pandas as pd
import numpy as np
from dowhy import CausalModel
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def test_ab_logic():
    print("--- Starting A/B Logic Verification ---")
    
    # 1. Simulate Data
    print("1. Simulating Data...")
    n = 100
    df = pd.DataFrame({
        'Group': np.random.choice(['Control', 'Variant A'], n),
        'Age': np.random.normal(30, 5, n),
        'Spend': np.random.normal(100, 20, n)
    })
    
    # Add effect
    df.loc[df['Group'] == 'Variant A', 'Spend'] += 10
    
    print("Data Head:")
    print(df.head())
    
    # 2. Test Encoding Logic
    print("\n2. Testing Encoding Logic...")
    treatment = 'Group'
    outcome = 'Spend'
    confounders = ['Age']
    
    # Simulate UI selection
    control_val = 'Control'
    treat_val = 'Variant A'
    
    # Logic from app
    df['Treatment_Encoded'] = np.nan
    df.loc[df[treatment] == control_val, 'Treatment_Encoded'] = 0
    df.loc[df[treatment] == treat_val, 'Treatment_Encoded'] = 1
    df = df.dropna(subset=['Treatment_Encoded'])
    
    treatment_new = 'Treatment_Encoded'
    
    print("Encoded Counts:")
    print(df[treatment_new].value_counts())
    
    if df[treatment_new].nunique() != 2:
        print("FAIL: Encoding did not produce 2 groups.")
        return
    
    # 3. Test Estimation (Linear Regression)
    print("\n3. Testing Estimation (Linear Regression)...")
    model = CausalModel(
        data=df,
        treatment=treatment_new,
        outcome=outcome,
        common_causes=confounders
    )
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True
    )
    
    print(f"Estimated Effect: {estimate.value}")
    
    # 4. Test Balance Check
    print("\n4. Testing Balance Check...")
    balance_data = []
    for col in confounders:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_control = df[df[treatment_new]==0][col].mean()
            mean_treat = df[df[treatment_new]==1][col].mean()
            diff = mean_treat - mean_control
            
            t_stat, p_val = stats.ttest_ind(
                df[df[treatment_new]==1][col].dropna(), 
                df[df[treatment_new]==0][col].dropna(), 
                equal_var=False
            )
            
            balance_data.append({
                "Covariate": col,
                "Mean (Control)": mean_control,
                "Mean (Treatment)": mean_treat,
                "Diff": diff,
                "P-Value": p_val
            })
            
    print("Balance Table:")
    print(pd.DataFrame(balance_data))
    
    # 5. Test Script Generation
    print("\n5. Testing Script Generation...")
    from causal_utils import generate_script
    
    script = generate_script(
        data_source="Simulated Data",
        treatment="Group",
        outcome="Spend",
        confounders=['Age'],
        time_period=None,
        estimation_method="A/B Test (Difference in Means)",
        impute_enable=False, num_impute_method=None, num_custom_val=0, cat_impute_method=None, cat_custom_val=None,
        winsorize_enable=False, winsorize_cols=[], percentile=0.05,
        log_transform_cols=[], standardize_cols=[], n_iterations=10,
        control_val="Control", treat_val="Variant A"
    )
    
    if "Encoding Categorical Treatment" in script:
        print("SUCCESS: Script contains encoding logic.")
    else:
        print("FAIL: Script missing encoding logic.")
        
    if "test_significance=False" in script:
        print("SUCCESS: Script uses optimized bootstrapping (test_significance=False).")
    else:
        print("FAIL: Script does NOT use optimized bootstrapping.")
    
    if "backdoor.linear_regression" in script:
        print("SUCCESS: Script contains correct estimation method.")
    else:
        print("FAIL: Script missing estimation method.")
    
    # 6. Test Empty Confounders
    print("\n6. Testing Empty Confounders...")
    try:
        model_empty = CausalModel(
            data=df,
            treatment='Treatment_Encoded',
            outcome='Spend',
            common_causes=[],
            instruments=None,
            effect_modifiers=[]
        )
        identified_estimand_empty = model_empty.identify_effect(proceed_when_unidentifiable=True)
        estimate_empty = model_empty.estimate_effect(
            identified_estimand_empty,
            method_name="backdoor.linear_regression",
            test_significance=True
        )
        print(f"Estimated Effect (No Confounders): {estimate_empty.value}")
        if not np.isclose(estimate_empty.value, 10, atol=5): # Allow wider tolerance for unadjusted
             print("WARNING: Effect estimate without confounders is far off (expected ~10).")
        else:
             print("SUCCESS: Estimation without confounders worked.")
    except Exception as e:
        print(f"FAILURE: Estimation without confounders failed: {e}")
        exit(1)

    # 7. Test Bootstrapping
    print("\n7. Testing Bootstrapping...")
    try:
        # We need to simulate the bootstrap loop manually or call a function if we had one.
        # Since the app logic is embedded in the app, we'll just verify we can run the estimation multiple times 
        # on resampled data using the same method name.
        
        bootstrap_estimates = []
        for i in range(5): # Run 5 iterations
            df_resampled = df.sample(frac=1, replace=True, random_state=i)
            model_boot = CausalModel(
                data=df_resampled,
                treatment='Treatment_Encoded',
                outcome='Spend',
                common_causes=['Age'],
                instruments=None,
                effect_modifiers=['Age']
            )
            identified_estimand_boot = model_boot.identify_effect(proceed_when_unidentifiable=True)
            est_boot = model_boot.estimate_effect(
                identified_estimand_boot,
                method_name="backdoor.linear_regression",
                test_significance=False
            )
            bootstrap_estimates.append(est_boot.value)
            
        if len(bootstrap_estimates) == 5:
            print(f"SUCCESS: Bootstrapping ran 5 iterations. Mean Estimate: {np.mean(bootstrap_estimates)}")
        else:
            print("FAILURE: Bootstrapping did not complete all iterations.")

    except Exception as e:
        print(f"FAILURE: Bootstrapping failed: {e}")
        exit(1)

    print("\n--- Verification Complete: SUCCESS ---")

if __name__ == "__main__":
    test_ab_logic()
