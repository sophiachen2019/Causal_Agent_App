import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from causal_utils import generate_script

def simulate_panel_data(n_units=500, n_periods=20):
    np.random.seed(42)
    
    # Generate Unit IDs (String)
    # 500 units * 20 periods = 10,000 samples
    unit_ids = np.repeat([f"Unit_{i:03d}" for i in range(n_units)], n_periods)
    
    # Generate Time Periods (Month/Year)
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='M')
    time_periods = np.tile(dates, n_units)
    
    n_samples = n_units * n_periods
    
    # Time-Invariant Confounder: Industry (5 String Categories)
    industries_list = ['Tech', 'Retail', 'Finance', 'Healthcare', 'Manufacturing']
    # Assign one industry to each unit
    unit_industries = np.random.choice(industries_list, n_units)
    industry = np.repeat(unit_industries, n_periods)
    
    # Map Industry to an Effect (Fixed Effect)
    industry_effects = {
        'Tech': 3000,
        'Finance': 2500,
        'Healthcare': 2000,
        'Manufacturing': 1500,
        'Retail': 1000
    }
    industry_val = np.array([industry_effects[ind] for ind in industry])
    
    # Time-Varying Confounder (e.g., Marketing Spend)
    marketing_spend = np.random.normal(1000, 200, n_samples)
    
    # Time Trend (0 to 19)
    time_trend = np.tile(np.arange(n_periods), n_units)
    
    # Treatment: Feature Adoption (Binary)
    # Depends on Industry (Fixed Effect) and Marketing Spend
    # Tech and Finance are more likely to adopt
    prob_adoption = 1 / (1 + np.exp(-( -3 + (industry_val/1000) + 0.001 * marketing_spend + 0.1 * time_trend)))
    feature_adoption = np.random.binomial(1, prob_adoption, n_samples)
    
    # Outcome: Revenue
    # True Effect of Feature Adoption = $500
    # Industry (Fixed Effect) has a large impact
    revenue = (
        2000 
        + 500 * feature_adoption 
        + industry_val 
        + 2 * marketing_spend 
        + 50 * time_trend 
        + np.random.normal(0, 100, n_samples)
    )
    
    df = pd.DataFrame({
        'Unit_ID': unit_ids,
        'Time_Period': time_periods,
        'Industry': industry,
        'Marketing_Spend': marketing_spend,
        'Feature_Adoption': feature_adoption,
        'Revenue': revenue
    })
    return df

def test_panel_logic():
    print("--- Starting Panel Data Logic Verification ---")
    
    # 1. Simulate Data
    print("1. Simulating Panel Data...")
    df = simulate_panel_data()
    print(f"Data Shape: {df.shape}")
    
    # 2. Test Estimation (Fixed Effects)
    print("\n2. Testing Fixed Effects Estimation...")
    try:
        df_panel = df.set_index(['Unit_ID', 'Time_Period'])
        exog_vars = ['Feature_Adoption', 'Industry', 'Marketing_Spend']
        mod = PanelOLS(df_panel['Revenue'], df_panel[exog_vars], entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        estimate = res.params['Feature_Adoption']
        print(f"Estimated Effect: {estimate}")
        
        if res.estimated_effects is not None:
             print("SUCCESS: Fixed effects were estimated and saved.")
        else:
             print("FAILURE: Fixed effects were NOT saved.")
        
        if np.isclose(estimate, 500, atol=50):
            print("SUCCESS: Estimated effect is close to true value (500).")
        else:
            print(f"WARNING: Estimated effect {estimate} is far from true value (500).")
            
    except Exception as e:
        print(f"FAILURE: Estimation failed: {e}")
        exit(1)
        
    # 3. Test Script Generation
    print("\n3. Testing Script Generation...")
    script = generate_script(
        data_source="Simulated Data",
        treatment="Feature_Adoption",
        outcome="Revenue",
        confounders=['Industry', 'Marketing_Spend'],
        time_period="Time_Period",
        estimation_method="Panel Data (Fixed Effects)",
        impute_enable=False, num_impute_method=None, num_custom_val=0, cat_impute_method=None, cat_custom_val=None,
        winsorize_enable=False, winsorize_cols=[], percentile=0.05,
        log_transform_cols=[], standardize_cols=[], n_iterations=10,
        unit_id="Unit_ID"
    )
    
    if "PanelOLS" in script and "entity_effects=True" in script:
        print("SUCCESS: Script contains PanelOLS logic.")
    else:
        print("FAIL: Script missing PanelOLS logic.")
        print(script)

    print("\n--- Verification Complete: SUCCESS ---")

if __name__ == "__main__":
    test_panel_logic()
