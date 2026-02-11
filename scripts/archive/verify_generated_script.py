import sys
import os
sys.path.append(os.getcwd())
from causal_utils import generate_script

def verify_script_generation():
    print("Testing generate_script...")
    
    # Dummy arguments
    data_source = "Simulated Data"
    treatment = "Feature_Adoption"
    outcome = "Account_Value"
    confounders = ["Customer_Segment", "Historical_Usage"]
    time_period = "Quarter"
    estimation_method = "Double Machine Learning (LinearDML)"
    impute_enable = True
    num_impute_method = "Mean"
    num_custom_val = 0.0
    cat_impute_method = "Mode"
    cat_custom_val = "Missing"
    winsorize_enable = True
    winsorize_cols = ["Historical_Usage"]
    percentile = 0.05
    log_transform_cols = ["Account_Value"]
    standardize_cols = ["Historical_Usage"]
    n_iterations = 10

    try:
        script = generate_script(
            data_source, treatment, outcome, confounders, time_period, estimation_method,
            impute_enable, num_impute_method, num_custom_val, cat_impute_method, cat_custom_val,
            winsorize_enable, winsorize_cols, percentile,
            log_transform_cols, standardize_cols, n_iterations
        )
        
        print("Script generated successfully.")
        
        # Verify syntax
        compile(script, '<string>', 'exec')
        print("Generated script syntax is valid.")
        
        # Check for bootstrapping code
        if "Bootstrapping" in script and "bootstrap_estimates" in script:
            print("Bootstrapping logic found in script.")
        else:
            print("ERROR: Bootstrapping logic NOT found in script.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error verifying script: {e}")
        # Print the script for debugging if syntax error
        if isinstance(e, SyntaxError):
            print("\n--- Generated Script ---")
            print(script)
            print("------------------------")
        sys.exit(1)

if __name__ == "__main__":
    verify_script_generation()
