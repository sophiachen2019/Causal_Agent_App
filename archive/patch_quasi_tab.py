import re

with open('causal_agent_app.py', 'r') as f:
    content = f.read()

geolift_block = """
    # --- GeoLift Analysis ---
    elif quasi_method == "GeoLift (Synthetic Control via R)":
        st.subheader("Configuration: GeoLift Analysis")
        st.markdown(\"\"\"
        **Methodology (Augmented Synthetic Control):**
        GeoLift uses the Augmented Synthetic Control Method (ASCM) developed by Meta to estimate the causal effect of an intervention at the geographic level. It constructs a "synthetic" version of the treated location by finding a weighted combination of untreated locations that best matches the pre-treatment time series of the treated location.
        
        **Formula Definition:**
        Let $Y_{it}$ be the outcome for geography $i$ at time $t$. Let region $1$ be the treated region, and regions $2, \dots, N$ be the donor pool.
        
        The Synthetic Control estimator seeks weights $W = (w_2, \dots, w_N)$ such that the pre-intervention distance is minimized:
        $$ \min_W \sum_{t=1}^{T_0} (Y_{1t} - \sum_{j=2}^N w_j Y_{jt})^2 $$
        subject to $w_j \ge 0$ and $\sum w_j = 1$.
        
        In **Augmented Synthetic Control** (used by GeoLift), ridge regression is added to adjust for poor pre-treatment fit:
        $$ \hat{\\tau}_{1t} = Y_{1t} - (\sum_{j=2}^N \hat{w}_j Y_{jt} + (\mathbf{X}_{1t} - \sum_{j=2}^N \hat{w}_j \mathbf{X}_{jt})^T \hat{\\beta}) $$
        
        **Reference:**
        [GeoLift: Meta's open source solution for Geo-based Experimentation](https://github.com/facebookincubator/GeoLift)
        \"\"\")
        
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
                    st.error(f"GeoLift Analysis failed: {e}")

    # --- SHARED RESULTS & EXPORT MODULE ---"""

target_str = "                        st.success(\"Analysis Complete! Scroll down to see results.\")\n\n    # --- SHARED RESULTS & EXPORT MODULE ---"
if target_str in content and "elif quasi_method == \"GeoLift (Synthetic Control via R)\":" not in content:
    content = content.replace(target_str, "                        st.success(\"Analysis Complete! Scroll down to see results.\")\n\n" + geolift_block)
    with open('causal_agent_app.py', 'w') as f:
        f.write(content)
    print("Successfully patched GeoLift UI into quasi_method block")
else:
    print("Could not find target string or GeoLift is already present")

