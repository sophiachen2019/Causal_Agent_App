import re

with open('causal_agent_app.py', 'r') as f:
    content = f.read()

# 1. Update dropdown options
old_dropdown = '["Difference-in-Differences (DiD)", "CausalImpact (Bayesian Time Series)", "GeoLift (Synthetic Control via R)", "GeoLift Power Analysis (Market Selection)"]'
new_dropdown = '["Difference-in-Differences (DiD)", "CausalImpact (Bayesian Time Series)", "GeoLift (Synthetic Control via R)"]'
if old_dropdown in content:
    content = content.replace(old_dropdown, new_dropdown)

# 2. Extract and replace the two separate blocks with a combined block
combined_block = """    # --- GeoLift Analysis ---
    elif quasi_method == "GeoLift (Synthetic Control via R)":
        st.subheader("Configuration: GeoLift Analysis")
        
        # Power Analysis vs Estimation Toggle
        geolift_mode = st.radio(
            "What do you want to do?",
            ["Power Analysis (Market Selection)", "Impact Estimation (Post-Test)"],
            horizontal=True,
            help="Use Power Analysis before a test to find the best markets. Use Impact Estimation after a test to measure the results."
        )
        
        st.write("---")
        
        if geolift_mode == "Power Analysis (Market Selection)":
            st.markdown(\"\"\"
            **Methodology (Power Analysis):**
            Before running an experiment, GeoLift can simulate historical data to find the best test markets (Treated Geographies) and the required duration to detect a specific Minimum Detectable Effect (MDE).
            
            This uses the `GeoLiftPower` function in R to iterate through potential treated units and evaluate their pre-treatment fit and power.
            \"\"\")
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                power_date = st.selectbox("Date Column", df.columns, index=get_index(df.columns, 'Date', 0), key="pow_date")
                power_geo = st.selectbox("Geography/Location Column", df.columns, index=get_index(df.columns, 'Region', 1), key="pow_geo")
            with col_p2:
                num_cols = df.select_dtypes(include=[np.number]).columns
                power_kpi = st.selectbox("KPI/Outcome Column", num_cols, index=get_index(num_cols, 'KPI', 0) if len(num_cols) > 0 else 0, key="pow_kpi")
                
            st.info("Power Analysis will search through all unique geographies in the dataset to find the best candidate for treatment.")
            
            if st.button("Run Power Analysis", type="primary"):
                st.write("---")
                with st.spinner("Running R GeoLiftPower (This may take a minute depending on data size)..."):
                    try:
                        results_power = causal_utils.run_geolift_power(df, power_date, power_geo, power_kpi)
                        
                        st.session_state.quasi_results = {
                            'method': 'GeoLift Power Analysis (Market Selection)',
                            'result': results_power
                        }
                        st.session_state.quasi_analysis_run = True
                        st.session_state.quasi_method_run = "GeoLift Power Analysis (Market Selection)"
                        st.success("Power Analysis Complete! Scroll down to see results.")
                    except Exception as e:
                        st.error(f"Power Analysis failed: {e}")
                        
        else: # Impact Estimation
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                geo_lift_date = st.selectbox("Date Column", df.columns, index=get_index(df.columns, 'Date', 0), key="gl_date")
                geo_lift_geo = st.selectbox("Geography/Location Column", df.columns, index=get_index(df.columns, 'Region', 1), key="gl_geo")
            with col_g2:
                num_cols = df.select_dtypes(include=[np.number]).columns
                geo_lift_kpi = st.selectbox("KPI/Outcome Column", num_cols, index=get_index(num_cols, 'KPI', 0) if len(num_cols) > 0 else 0, key="gl_kpi")
                
                treated_geo_options = []
                if geo_lift_geo in df.columns:
                    treated_geo_options = df[geo_lift_geo].dropna().astype(str).unique().tolist()
                geo_lift_treated = st.selectbox("Treated Geography", treated_geo_options, key="gl_treat")
            
            min_date = df[geo_lift_date].min()
            max_date = df[geo_lift_date].max()
            default_int = min_date + (max_date - min_date) / 2
            
            try:
                # Let's try to infer if they are using the demo dataset
                if 'Region' in df.columns and 'Date' in df.columns and 'Chicago' in treated_geo_options:
                     default_int_val = pd.to_datetime('2023-03-02')
                     if default_int_val >= min_date and default_int_val <= max_date:
                         default_int = default_int_val
            except:
                 pass
                 
            geo_lift_intervention_date = st.date_input("Intervention Start Date", value=default_int, min_value=min_date.date() if hasattr(min_date, 'date') else None, max_value=max_date.date() if hasattr(max_date, 'date') else None, key="gl_int")
    
            if st.button("Run GeoLift Analysis", type="primary"):
                st.write("---")
                with st.spinner("Initializing R Environment and Running GeoLift..."):
                    try:
                        results = causal_utils.run_geolift_analysis(df, geo_lift_date, geo_lift_geo, geo_lift_treated, geo_lift_kpi, str(geo_lift_intervention_date))
                        st.session_state.quasi_results = {
                            'method': 'GeoLift (Synthetic Control via R)',
                            'result': results,
                            'date_col': geo_lift_date,
                            'geo_col': geo_lift_geo,
                            'kpi_col': geo_lift_kpi,
                            'treated_geo': geo_lift_treated,
                            'intervention_date': str(geo_lift_intervention_date)
                        }
                        st.session_state.quasi_analysis_run = True
                        st.session_state.quasi_method_run = "GeoLift (Synthetic Control via R)"
                        st.success("GeoLift Analysis Complete! Scroll down to see results.")
                    except Exception as e:
                        st.error(f"GeoLift Analysis failed: {e}")"""


# This relies on regex to grab the two separate blocks and replace them.
# The blocks start at "    # --- GeoLift Analysis ---" and end before "    # --- SHARED RESULTS & EXPORT MODULE ---"
start_marker = "    # --- GeoLift Analysis ---"
end_marker = "    # --- SHARED RESULTS & EXPORT MODULE ---"

if start_marker in content and end_marker in content:
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    # Extract the portion to replace formatting issues
    old_blocks = content[start_idx:end_idx]
    content = content.replace(old_blocks, combined_block + "\n\n")

with open('causal_agent_app.py', 'w') as f:
    f.write(content)

print("Merged GeoLift UI options")
