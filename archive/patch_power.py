import re

with open('causal_utils.py', 'r') as f:
    content = f.read()

power_func = """
def run_geolift_power(df, date_col, geo_col, kpi_col, effect_sizes=[0.1, 0.2, 0.3]):
    \"\"\"
    Runs GeoLift Power Analysis (Market Selection) via rpy2.
    \"\"\"
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    
    # Enable pandas conversion
    pandas2ri.activate()
    
    try:
        base = importr('base')
        utils = importr('utils')
        geolift = importr('GeoLift')
    except Exception as e:
        return {"error": f"Failed to load R GeoLift package. Details: {e}"}

    df_clean = df[[date_col, geo_col, kpi_col]].copy()
    df_clean = df_clean.dropna()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col]).dt.strftime('%Y-%m-%d')
    df_clean[kpi_col] = pd.to_numeric(df_clean[kpi_col])
    
    # Pass DataFrame to R
    r_df = pandas2ri.py2rpy(df_clean)
    robjects.globalenv['py_data'] = r_df
    robjects.globalenv['time_id'] = date_col
    robjects.globalenv['geo_id'] = geo_col
    robjects.globalenv['Y_id'] = kpi_col
    
    robjects.r(\"\"\"
    geo_data <- GeoDataRead(data = py_data,
                            time_id = time_id,
                            location_id = geo_id,
                            Y_id = Y_id,
                            format = "yyyy-mm-dd")
    \"\"\")
    
    # Run GeoLiftPower
    effect_sizes_str = ", ".join(map(str, effect_sizes))
    
    try:
         # Note: A real power analysis might take hours depending on market size.
         # For the UI, we'll try a very limited scope search or a direct GeoLiftPower call.
         # For Streamlit demo purposes, we will wrap this with a quick execution or just return some mock recommendations if it stalls.
         # Actually trying to run real GeoLiftPower:
         robjects.r(f\"\"\"
         # We limit the search to speed it up for the demo
         power_res <- GeoLiftPower(
             data = geo_data,
             locations = unique(geo_data$location), # Try all locations as potential single treatments
             effect_size = seq(0.1, 0.25, 0.05),
             time_id = "time",
             location_id = "location",
             Y_id = "Y"
         )
         \"\"\")
         
         # Convert results back
         # This is highly dependent on GeoLiftPower's returned structure.
         # Given the complexity of bridging the R list to Python, we'll use a simpler approach:
         # let's just use `summary()` and return it as text for the UI.
         
         robjects.r(\"\"\"
         pow_summary <- capture.output(print(power_res))
         \"\"\")
         
         pow_summary = "\\n".join(list(robjects.globalenv['pow_summary']))
         return {"summary": pow_summary}
         
    except Exception as e:
         return {"error": f"GeoLift Power Analysis failed: {e}"}
"""

if "def run_geolift_power" not in content:
    content += "\n" + power_func
    with open('causal_utils.py', 'w') as f:
         f.write(content)
    print("Added run_geolift_power to causal_utils.py")
else:
    print("run_geolift_power already exists")
