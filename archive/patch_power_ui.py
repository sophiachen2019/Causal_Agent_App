import re

with open('causal_agent_app.py', 'r') as f:
    content = f.read()

# I want to add "GeoLift Power Analysis (Market Selection)" to the Tab options.
# Wait, let's see where an appropriate place for Power Analysis is.
# Maybe inside the Quasi-Experimental tab, as an additional Method? Yes.
old_options = '["Difference-in-Differences (DiD)", "CausalImpact (Bayesian Time Series)", "GeoLift (Synthetic Control via R)"]'
new_options = '["Difference-in-Differences (DiD)", "CausalImpact (Bayesian Time Series)", "GeoLift (Synthetic Control via R)", "GeoLift Power Analysis (Market Selection)"]'
content = content.replace(old_options, new_options)

power_ui_block = """
    # --- GeoLift Power Analysis ---
    elif quasi_method == "GeoLift Power Analysis (Market Selection)":
        st.subheader("Configuration: GeoLift Power Analysis")
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
"""

# I need to insert this right after the GeoLift analysis block...
insertion_target = "    # --- SHARED RESULTS & EXPORT MODULE ---"
if "elif quasi_method == \"GeoLift Power Analysis (Market Selection)\":" not in content:
    content = content.replace(insertion_target, power_ui_block + "\n\n" + insertion_target)
    
# Now the shared results section needs to handle the new method
old_results_handling = """        elif quasi_method_run == "GeoLift (Synthetic Control via R)":
            st.divider()"""
new_results_handling = """        elif quasi_method_run == "GeoLift Power Analysis (Market Selection)":
            st.divider()
            if "error" in results['result']:
                st.error(results['result']['error'])
            else:
                st.subheader("Results: GeoLift Power Analysis")
                st.markdown("Below is the output from R's `GeoLiftPower` function, showing the best test market candidates and expected power:")
                st.text(results['result']['summary'])
                
        elif quasi_method_run == "GeoLift (Synthetic Control via R)":
            st.divider()"""
if "elif quasi_method_run == \"GeoLift Power Analysis (Market Selection)\":" not in content:
    content = content.replace(old_results_handling, new_results_handling)

with open('causal_agent_app.py', 'w') as f:
    f.write(content)

print("Added Power Analysis UI")
