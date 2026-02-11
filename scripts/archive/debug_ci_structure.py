
import pandas as pd
import numpy as np
from causalimpact import CausalImpact

# Monkey patch for compatibility (same as in causal_utils.py)
if not hasattr(pd.DataFrame, 'applymap'):
    pd.DataFrame.applymap = pd.DataFrame.map

def debug_ci():
    print("Running Debug CausalImpact...")
    data = pd.DataFrame(np.random.randn(100, 2), columns=['y', 'x'])
    data['y'] += 5
    
    # FIX: Reset Index and Column Names to standard Integers to avoid CausalImpact internal KeyError: 0
    data = data.reset_index(drop=True)
    data.columns = range(len(data.columns))
    
    pre_period = [0, 69]
    post_period = [70, 99]
    
    ci = CausalImpact(data, pre_period, post_period)
    
    print("\n--- Summary Data Columns ---")
    print(ci.summary_data.columns)
    
    print("\n--- Summary Data Index ---")
    print(ci.summary_data.index)
    
    print("\n--- Summary Data Head ---")
    print(ci.summary_data)
    
if __name__ == "__main__":
    debug_ci()
