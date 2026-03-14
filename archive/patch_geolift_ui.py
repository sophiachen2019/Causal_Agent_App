import re

with open('causal_agent_app.py', 'r') as f:
    content = f.read()

# 1. Update simulation options to include GeoLift
old_sim_options = 'sim_options = ["Standard (Cross-sectional/DiD)", "BSTS Demo (Multi-region Time Series)", "Switchback Demo (Region/Time Randomized)"]'
new_sim_options = 'sim_options = ["Standard (Cross-sectional/DiD)", "BSTS Demo (Multi-region Time Series)", "Switchback Demo (Region/Time Randomized)", "GeoLift Demo (Geographic Intervention)"]'
if old_sim_options in content:
    content = content.replace(old_sim_options, new_sim_options)

# 2. Add simulate_geolift_data function
geolift_sim_func = """
def simulate_geolift_data():
    \"\"\"Generates geographic time-series data suitable for GeoLift analysis.\"\"\"
    np.random.seed(42)
    # 10 regions
    regions = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    total_days = 90
    start_date = pd.to_datetime('2023-01-01')
    date_range = pd.date_range(start=start_date, periods=total_days)
    
    data = []
    treatment_region = 'Chicago'
    intervention_day = 60 # 60 days pre-test, 30 days post-test
    
    # Base trend
    global_trend = 100 + 0.5 * np.arange(total_days)
    
    for r in regions:
        # Region-specific baseline
        base = np.random.normal(50, 20)
        
        # Add noise and some region trend
        metric = global_trend + base + np.random.normal(0, 10, total_days)
        
        # Add intervention effect
        if r == treatment_region:
            lift = np.zeros(total_days)
            lift[intervention_day:] = 25 # True lift is 25
            metric += lift
            
        for d_idx, date in enumerate(date_range):
            data.append([date, r, metric[d_idx]])
            
    df = pd.DataFrame(data, columns=['Date', 'Region', 'KPI'])
    return df
"""
if "def simulate_geolift_data():" not in content:
    content = content.replace('def simulate_switchback_data():', geolift_sim_func + '\ndef simulate_switchback_data():')

# 3. Update simulate_data routing
old_simulate_data = """def simulate_data(n_samples=1000, type="Standard"):
    if type == "BSTS Demo":
        return simulate_bsts_demo_data()
    elif type == "Switchback Demo":
        return simulate_switchback_data()
    return simulate_standard_data(n_samples)"""
new_simulate_data = """def simulate_data(n_samples=1000, type="Standard"):
    if type == "BSTS Demo":
        return simulate_bsts_demo_data()
    elif type == "Switchback Demo":
        return simulate_switchback_data()
    elif type == "GeoLift Demo":
        return simulate_geolift_data()
    return simulate_standard_data(n_samples)"""
content = content.replace(old_simulate_data, new_simulate_data)

# 4. Update target_type logic
old_target = """        target_type = "Standard" if "Standard" in simulate_type else ("BSTS Demo" if "BSTS" in simulate_type else "Switchback Demo")"""
new_target = """        target_type = "Standard" if "Standard" in simulate_type else ("BSTS Demo" if "BSTS" in simulate_type else ("Switchback Demo" if "Switchback" in simulate_type else "GeoLift Demo"))"""
content = content.replace(old_target, new_target)

# 5. Add GeoLift to UI Dropdown
old_methods = '["Difference-in-Differences (DiD)", "Interrupted/Bayesian Time Series (ITS/BSTS)", "Switchback Experiment"]'
new_methods = '["Difference-in-Differences (DiD)", "Interrupted/Bayesian Time Series (ITS/BSTS)", "Switchback Experiment", "GeoLift (Synthetic Control via R)"]'
content = content.replace(old_methods, new_methods)

# 6. Add GeoLift UI Logic
geolift_ui = """
                    elif estimation_method == "GeoLift (Synthetic Control via R)":
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
                        
                        geo_lift_date = st.selectbox("Date Column", df.columns, index=get_index(df.columns, 'Date', 0))
                        geo_lift_geo = st.selectbox("Geography Column", df.columns, index=get_index(df.columns, 'Region', 1))
                        geo_lift_kpi = st.selectbox("KPI/Outcome Column", df.columns, index=get_index(df.columns, 'KPI', 2))
                        
                        treated_geo_options = df[geo_lift_geo].dropna().unique().tolist()
                        geo_lift_treated = st.selectbox("Treated Geography", treated_geo_options)
                        
                        min_date = df[geo_lift_date].min()
                        max_date = df[geo_lift_date].max()
                        default_int = min_date + (max_date - min_date) / 2
                        try:
                            default_int_val = pd.to_datetime('2023-03-02') # default intervention day for demo
                            if default_int_val >= min_date and default_int_val <= max_date:
                                default_int = default_int_val
                        except:
                            pass
                            
                        geo_lift_intervention_date = st.date_input("Intervention Start Date", value=default_int, min_value=min_date.date() if hasattr(min_date, 'date') else None, max_value=max_date.date() if hasattr(max_date, 'date') else None)
"""
if "GeoLift (Synthetic Control via R)" not in content:
    content = content.replace('                    elif estimation_method == "Switchback Experiment":', geolift_ui + '\n                    elif estimation_method == "Switchback Experiment":')


# 7. Add GeoLift run branch
geolift_run = """
                    elif estimation_method == "GeoLift (Synthetic Control via R)":
                        with st.status("Initializing R Environment and Running GeoLift...", expanded=True) as status:
                            st.write("Passing dataframe to R...")
                            try:
                                result = causal_utils.run_geolift_analysis(df, geo_lift_date, geo_lift_geo, geo_lift_treated, geo_lift_kpi, str(geo_lift_intervention_date))
                                st.session_state.quasi_results = {
                                    'method': 'GeoLift (Synthetic Control via R)',
                                    'result': result,
                                    'date_col': geo_lift_date,
                                    'geo_col': geo_lift_geo,
                                    'kpi_col': geo_lift_kpi,
                                    'treated_geo': geo_lift_treated,
                                    'intervention_date': str(geo_lift_intervention_date)
                                }
                                status.update(label="GeoLift Analysis Complete!", state="complete", expanded=False)
                                st.rerun()
                            except Exception as e:
                                st.error(f"GeoLift Analysis failed: {e}")
                                status.update(label="Error in GeoLift", state="error", expanded=True)
"""
if "result = causal_utils.run_geolift_analysis" not in content:
    content = content.replace('                    elif estimation_method == "Switchback Experiment":\n                        with st.status', geolift_run + '\n                    elif estimation_method == "Switchback Experiment":\n                        with st.status')


with open('causal_agent_app.py', 'w') as f:
    f.write(content)

print("Patched causal_agent_app.py")
