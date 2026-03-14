import re

with open('causal_utils.py', 'r') as f:
    content = f.read()

geolift_func = """
def run_geolift_analysis(df, date_col, geo_col, treated_geo, kpi_col, intervention_date):
    \"\"\"
    Runs GeoLift Analysis using rpy2 to bridge Python and Meta's GeoLift R package.
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
        return {"error": f"Failed to load R GeoLift package. Ensure it is installed via Rscript. Details: {e}"}

    df_clean = df[[date_col, geo_col, kpi_col]].copy()
    
    # Filter out missing values
    df_clean = df_clean.dropna()
    
    # Convert dates to string format expected by R GeoDataRead
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
    
    # Time periods
    intervention_dt = pd.to_datetime(intervention_date)
    pre_end = (intervention_dt - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    robjects.globalenv['treated_loc'] = treated_geo
    robjects.globalenv['treatment_start'] = intervention_dt.strftime('%Y-%m-%d')
    
    try:
        robjects.r(\"\"\"
        # Get start/end indices for time
        time_index <- unique(geo_data$time)
        treatment_start_time <- min(time_index[time_index >= treatment_start])
        treatment_start_idx <- which(time_index == treatment_start_time)
        
        gl_res <- GeoLift(Y_id = "Y",
                          time_id = "time",
                          location_id = "location",
                          data = geo_data,
                          locations = c(treated_loc),
                          treatment_start_time = treatment_start_idx,
                          treatment_end_time = max(geo_data$time))
                          
        summary_res <- summary(gl_res)
        avg_lift <- summary_res$AverageEstimatedTreatmentEffect
        cumulative_lift <- summary_res$CumulativeLift
        p_val <- summary_res$p_val
        \"\"\")
        
        avg_lift = robjects.globalenv['avg_lift'][0]
        cum_lift = robjects.globalenv['cumulative_lift'][0]
        p_val = robjects.globalenv['p_val'][0]
        
        significant = "Yes" if p_val < 0.05 else "No"
        
        # We can also generate the plot to a file and read it in Streamlit
        robjects.r(\"\"\"
        png("geolift_plot.png", width=800, height=600)
        plot(gl_res)
        dev.off()
        \"\"\")
        
        report = f\"\"\"
        ### GeoLift Analysis Results
        **Treated Geography**: {treated_geo}
        
        **Average Estimated Treatment Effect**: {avg_lift:.2f}
        **Cumulative Lift**: {cum_lift:.2f}
        **P-Value**: {p_val:.4f}
        
        **Statistically Significant (p < 0.05)?**: {significant}
        \"\"\"
        
        return {
            "summary": report,
            "plot_path": "geolift_plot.png"
        }
        
    except Exception as e:
         return {"error": f"GeoLift execution failed: {e}"}
"""

if "def run_geolift_analysis" not in content:
    content += "\n" + geolift_func

with open('causal_utils.py', 'w') as f:
    f.write(content)

print("Patched causal_utils.py with run_geolift_analysis")
