
import pandas as pd
import numpy as np
import causal_utils
import warnings
warnings.filterwarnings('ignore')

def simulate_panel_data():
    np.random.seed(42)
    # 3 Units, 100 days
    dates = pd.date_range(start='2023-01-01', periods=100)
    units = ['Los Angeles', 'New York', 'Chicago']
    
    data = []
    
    # Base trends
    trend_la = np.linspace(10, 20, 100)
    trend_ny = np.linspace(12, 22, 100) # Correlated
    trend_chi = np.linspace(8, 15, 100) # Correlated
    
    trends = {'Los Angeles': trend_la, 'New York': trend_ny, 'Chicago': trend_chi}
    
    for i, date in enumerate(dates):
        for unit in units:
            val = trends[unit][i] + np.random.normal(0, 1)
            
            # Intervention on LA after day 70
            if unit == 'Los Angeles' and i >= 70:
                val += 5 # Effect
                
            data.append({
                'Date': date,
                'City': unit,
                'Sales': val
            })
            
    return pd.DataFrame(data)

def test_panel_pivot():
    print("Testing CausalImpact Panel Pivot...")
    df = simulate_panel_data()
    intervention_date = df['Date'].unique()[70]
    
    print("Simulated Data Shape:", df.shape)
    
    # Run CausalImpact with Unit ID
    results = causal_utils.run_causal_impact_analysis(
        df, 
        date_col='Date', 
        outcome_col='Sales', 
        intervention_date=intervention_date,
        unit_col='City',
        treated_unit='Los Angeles'
    )
    
    if 'error' in results:
        print(f"FAILED: {results['error']}")
    else:
        print("SUCCESS")
        print(f"Estimated ATE (True ~5): {results['ate']:.4f}")
        
        # Validation
        if 4 < results['ate'] < 6:
            print("CHECK: Estimate is within expected range.")
        else:
             print("CHECK WARNING: Estimate is off.")

if __name__ == "__main__":
    test_panel_pivot()
