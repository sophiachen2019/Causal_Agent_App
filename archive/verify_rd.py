import sys
import os
sys.path.append(os.getcwd())
from causal_utils import generate_script

def verify_rd_script_generation():
    print("Testing generate_script for Regression Discontinuity...")
    
    # Dummy arguments for RD
    data_source = "Simulated Data"
    treatment = "Treatment"
    outcome = "Outcome"
    confounders = ["Confounder1"]
    time_period = None
    estimation_method = "Regression Discontinuity"
    impute_enable = False
    num_impute_method = None
    num_custom_val = 0.0
    cat_impute_method = None
    cat_custom_val = None
    winsorize_enable = False
    winsorize_cols = []
    percentile = 0.05
    log_transform_cols = []
    standardize_cols = []
    n_iterations = 10
    
    # RD specific params
    rd_running_variable = "RunningVar"
    rd_cutoff = 50.0
    rd_bandwidth = 0.0

    try:
        script = generate_script(
            data_source, treatment, outcome, confounders, time_period, estimation_method,
            impute_enable, num_impute_method, num_custom_val, cat_impute_method, cat_custom_val,
            winsorize_enable, winsorize_cols, percentile,
            log_transform_cols, standardize_cols, n_iterations,
            rd_running_variable=rd_running_variable,
            rd_cutoff=rd_cutoff,
            rd_bandwidth=rd_bandwidth
        )
        
        print("RD Script generated successfully.")
        
        # Verify syntax
        compile(script, '<string>', 'exec')
        print("Generated RD script syntax is valid.")
        
        # Check for RD specific logic
        # Check for RD specific logic
        # We now use generic identification which finds the IV estimand
        if "identify_effect(proceed_when_unidentifiable=True)" in script:
            print("Generic identification method found (correct for RD auto-detection).")
        else:
            print("ERROR: Identification method NOT found.")
            sys.exit(1)
            
        if "RD_Indicator" in script:
            print("RD Indicator creation found.")
        else:
            print("ERROR: RD Indicator creation NOT found.")
            sys.exit(1)

    except Exception as e:
        print(f"Error verifying RD script: {e}")
        if isinstance(e, SyntaxError):
            print("\n--- Generated Script ---")
            print(script)
            print("------------------------")
        sys.exit(1)

if __name__ == "__main__":
    verify_rd_script_generation()
