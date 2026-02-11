import pandas as pd
import numpy as np
import statsmodels.api as sm

def simulate_data(n_samples=1000):
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
print("Data generated.")

# A/B Test Logic
treatment = 'Feature_Adoption'
outcome = 'Conversion'
confounders = ['Customer_Segment', 'Historical_Usage']

print("\n--- Testing A/B Logit ---")
try:
    X_ab_cols = [treatment]
    if confounders:
        X_ab_cols.extend(confounders)
    X_ab = df[X_ab_cols]
    X_ab = sm.add_constant(X_ab)
    y_ab = df[outcome]
    
    print(f"X shape: {X_ab.shape}")
    print(f"y unique: {y_ab.unique()}")
    
    ab_model = sm.Logit(y_ab, X_ab).fit(disp=0)
    print("Fit success!")
    print(ab_model.summary())
except Exception as e:
    print(f"A/B Logit Failed: {e}")

# DiD Logic
print("\n--- Testing DiD Logit ---")
time_period = 'Quarter'
try:
    df_did = df.copy()
    df_did['DiD_Interaction'] = df_did[treatment] * df_did[time_period]
    
    X_cols = [treatment, time_period, 'DiD_Interaction'] + confounders
    X_did = df_did[X_cols]
    X_did = sm.add_constant(X_did)
    y_did = df_did[outcome]
    
    print(f"X shape: {X_did.shape}")
    
    did_model = sm.Logit(y_did, X_did).fit(disp=0)
    print("Fit success!")
    print(did_model.summary())
    
    # Test CI Conversion Logic
    did_coeff = did_model.params['DiD_Interaction']
    odds_ratio = np.exp(did_coeff)
    did_conf_int = did_model.conf_int().loc['DiD_Interaction']
    
    baseline_risk = df_did[df_did[treatment] == 0][outcome].mean()
    print(f"Baseline Risk: {baseline_risk}")
    
    if 0 < baseline_risk < 1:
        def or_to_rd(or_val, p0):
            return (or_val * p0 / (1 - p0 + (or_val * p0))) - p0
        
        implied_risk_diff = or_to_rd(odds_ratio, baseline_risk)
        or_lower = np.exp(did_conf_int[0])
        or_upper = np.exp(did_conf_int[1])
        rd_lower = or_to_rd(or_lower, baseline_risk)
        rd_upper = or_to_rd(or_upper, baseline_risk)
        
        print(f"Implied RD: {implied_risk_diff}")
        print(f"Implied RD CI: [{rd_lower}, {rd_upper}]")
        
except Exception as e:
    print(f"DiD Logit Failed: {e}")
