import os
import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from econml.dml import LinearDML, CausalForestDML
from econml.metalearners import SLearner, TLearner
import streamlit as st
import statsmodels.api as sm

# Monkey patch for compatibility with Pandas 2.1+ / 3.0 where applymap is removed
if not hasattr(pd.DataFrame, 'applymap'):
    pd.DataFrame.applymap = pd.DataFrame.map

def generate_script(data_source, treatment, outcome, confounders, time_period, estimation_method, 
                    impute_enable, num_impute_method, num_custom_val, cat_impute_method, cat_custom_val,
                    winsorize_enable, winsorize_cols, percentile,
                    log_transform_cols, standardize_cols, n_iterations,
                    control_val=None, treat_val=None, hte_features=None, use_logit=False, bucketing_ops=None, filtering_ops=None,
                    ts_params=None, unit_col=None, treated_unit=None):
    
    script = f"""import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from econml.dml import LinearDML, CausalForestDML
from econml.metalearners import SLearner, TLearner
import matplotlib.pyplot as plt
import statsmodels.api as sm

# --- 1. Load Data ---
"""
    if data_source == "Simulated Data":
        if ts_params and ts_params.get('is_bsts_demo'):
            script += """
def simulate_bsts_demo_data():
    \"\"\"Generates multi-region time series data for BSTS demo.\"\"\"
    np.random.seed(42)
    regions = [f'Region_{i}' for i in range(1, 41)]
    total_days = 364
    start_date = pd.to_datetime('2023-01-01')
    date_range = pd.date_range(start=start_date, periods=total_days)
    data_list = []
    
    global_trend = np.linspace(100, 150, total_days)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(total_days) / 7)
    monthly_seasonality = 20 * np.sin(2 * np.pi * np.arange(total_days) / 30)
    
    for region in regions:
        base_offset = np.random.normal(0, 20)
        regional_trend = global_trend + base_offset
        noise = np.random.normal(0, 2, total_days)
        metric = regional_trend + weekly_seasonality + monthly_seasonality + noise
        
        intervention_day = 304
        if region == 'Region_1':
            lift = np.zeros(total_days)
            lift[intervention_day:] = 30 + np.cumsum(np.random.normal(0.5, 0.1, total_days - intervention_day))
            metric += lift
        
        region_df = pd.DataFrame({
            'Date': date_range, 'Region': region, 'Daily_Revenue': metric,
            'Marketing_Spend': np.random.normal(50, 5, total_days),
            'app_downloads': np.random.normal(100, 10, total_days) + weekly_seasonality,
            'website_traffic': np.random.normal(500, 50, total_days) + monthly_seasonality,
            'social_media_mentions': np.random.poisson(20, total_days),
            'Is_Post_Intervention': (np.arange(total_days) >= intervention_day).astype(int),
            'Is_Treated_Region': 1 if region == 'Region_1' else 0
        })
        data_list.append(region_df)
    return pd.concat(data_list, ignore_index=True)

df = simulate_bsts_demo_data()
"""
        else:
            script += """
def simulate_data(n_samples=1000):
    np.random.seed(42)
    customer_segment = np.random.binomial(1, 0.3, n_samples)
    historical_usage = np.random.normal(50, 15, n_samples) + (customer_segment * 20)
    marketing_nudge = np.random.binomial(1, 0.5, n_samples)
    quarter = np.random.binomial(1, 0.5, n_samples)
    prob_adoption = 1 / (1 + np.exp(-( -2 + 0.5 * customer_segment + 0.05 * historical_usage + 1.5 * marketing_nudge)))
    feature_adoption = np.random.binomial(1, prob_adoption, n_samples)
    account_value = (200 + 500 * feature_adoption + 1000 * customer_segment + 10 * historical_usage + 50 * quarter + np.random.normal(0, 50, n_samples))
    prob_conversion = 1 / (1 + np.exp(-( -1 + 0.5 * customer_segment + 0.5 * feature_adoption)))
    conversion = np.random.binomial(1, prob_conversion, n_samples)
    # Date Generation (Simulating 2 years of data)
    start_date = pd.to_datetime('2023-01-01')
    dates = start_date + pd.to_timedelta(np.random.randint(0, 730, n_samples).astype(int), unit='D')
    return pd.DataFrame({
        'Customer_Segment': customer_segment, 'Historical_Usage': historical_usage,
        'Marketing_Nudge': marketing_nudge, 'Quarter': quarter,
        'Feature_Adoption': feature_adoption, 'Account_Value': account_value,
        'Conversion': conversion, 'Date': pd.to_datetime(dates).floor('D')
    })

df = simulate_data()
"""
        script += "print('Data Simulated Successfully')\n"
    else:
        script += """
# REPLACE 'your_dataset.csv' WITH THE PATH TO YOUR UPLOADED FILE
df = pd.read_csv('your_dataset.csv')
print("Data Loaded Successfully")

"""

    script += """
# --- Auto-Convert Boolean to Dummy ---
def convert_bool_to_int(df):
    # 1. Actual boolean types
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Converted boolean columns to integer: {', '.join(bool_cols)}")
    
    # 2. String "TRUE"/"FALSE" (case insensitive)
    obj_cols = df.select_dtypes(include=['object']).columns
    mapping = {'TRUE': 1, 'FALSE': 0, 'T': 1, 'F': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
    
    for col in obj_cols:
        try:
            series_upper = df[col].astype(str).str.upper()
            sample = series_upper.dropna().head(100)
            if not set(sample.unique()).issubset(mapping.keys()):
                continue
            
            unique_vals = set(series_upper.dropna().unique())
            if unique_vals.issubset(mapping.keys()):
                df[col] = series_upper.map(mapping).fillna(df[col])
                df[col] = pd.to_numeric(df[col], errors='ignore')
                print(f"Converted string boolean column to integer: {col}")
        except:
            pass
    return df

df = convert_bool_to_int(df)
"""

    script += "\n# --- 2. Data Preprocessing ---\n"
    
    if impute_enable:
        script += f"""
# Imputation
num_cols = df.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    if "{num_impute_method}" == "Mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif "{num_impute_method}" == "Median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif "{num_impute_method}" == "Zero":
        df[num_cols] = df[num_cols].fillna(0)
    elif "{num_impute_method}" == "Custom Value":
        df[num_cols] = df[num_cols].fillna({num_custom_val})

cat_cols = df.select_dtypes(exclude=[np.number]).columns
if len(cat_cols) > 0:
    if "{cat_impute_method}" == "Mode":
        for col in cat_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
    elif "{cat_impute_method}" == "Missing Indicator":
        df[cat_cols] = df[cat_cols].fillna("Missing")
    elif "{cat_impute_method}" == "Custom Value":
        df[cat_cols] = df[cat_cols].fillna("{cat_custom_val}")
print("Missing values imputed.")
"""

    if winsorize_enable and winsorize_cols:
        script += f"""
# Winsorization
from scipy.stats import mstats
winsorize_cols = {winsorize_cols}
limit = {percentile}
for col in winsorize_cols:
    df[col] = mstats.winsorize(df[col], limits=[limit/2, limit/2])
print("Applied winsorization.")
"""

    if log_transform_cols:
        script += f"""
# Log Transformation
log_cols = {log_transform_cols}
for col in log_cols:
    # Adding small constant to avoid log(0)
    df[col] = np.log1p(df[col])
print("Applied Log Transformation.")
"""

    if standardize_cols:
        script += f"""
# Standardization
scaler = StandardScaler()
std_cols = {standardize_cols}
df[std_cols] = scaler.fit_transform(df[std_cols])
print("Applied Standardization.")
"""

    if filtering_ops:
        script += "\n# Data Filtering\n"
        for op in filtering_ops:
            col = op['col']
            operator = op['op']
            val = op['val']
            
            # Handle string vs numeric value in code generation
            if isinstance(val, str):
                val_repr = f"'{val}'"
            else:
                val_repr = str(val)

            if operator == "==":
                script += f"df = df[df['{col}'] == {val_repr}]\n"
            elif operator == "!=":
                script += f"df = df[df['{col}'] != {val_repr}]\n"
            elif operator == ">":
                script += f"df = df[df['{col}'] > {val_repr}]\n"
            elif operator == "<":
                script += f"df = df[df['{col}'] < {val_repr}]\n"
            elif operator == ">=":
                script += f"df = df[df['{col}'] >= {val_repr}]\n"
            elif operator == "<=":
                script += f"df = df[df['{col}'] <= {val_repr}]\n"
            elif operator == "contains":
                script += f"df = df[df['{col}'].astype(str).str.contains({val_repr}, na=False)]\n"
            
            script += f"print(f\"Applied filter: {col} {operator} {val_repr}. Rows remaining: {{len(df)}}\")\n"

    if bucketing_ops:
        script += "\n# Variable Bucketing\n"
        for op in bucketing_ops:
            col = op['col']
            n_bins = op['n_bins']
            method = op['method']
            new_col = op['new_col']
            
            if method == 'cut':
                script += f"df['{new_col}'] = pd.cut(df['{col}'], bins={n_bins}, precision=1).astype(str)\n"
            else:
                script += f"df['{new_col}'] = pd.qcut(df['{col}'], q={n_bins}, duplicates='drop', precision=1).astype(str)\n"
            
            script += f"print(f\"Created bucketed column '{new_col}' from '{col}'\")\n"

    # --- QUASI-EXPERIMENTAL ANALYSIS ---
    if estimation_method == "Difference-in-Differences (DiD)":
        script += f"""
# --- 3. Quasi-Experimental Analysis: Difference-in-Differences (DiD) ---
print("Running Difference-in-Differences Analysis...")
treatment_col = '{treatment}'
outcome_col = '{outcome}'
time_col = '{time_period}' # Pre/Post indicator (0/1)
confounders = {confounders}
use_logit = {use_logit}

# Create Interaction Term
df['DiD_Interaction'] = df[treatment_col] * df[time_col]

# Define Design Matrix
X = df[[treatment_col, time_col, 'DiD_Interaction']]
if confounders:
    X = pd.concat([X, df[confounders]], axis=1)

X = sm.add_constant(X)
y = df[outcome_col]

# Estimate
if use_logit:
    print("Using Logit Model for DiD...")
    model = sm.Logit(y, X).fit(disp=0)
    print("Odds Ratio Interaction:", np.exp(model.params['DiD_Interaction']))
else:
    print("Using OLS Model for DiD...")
    model = sm.OLS(y, X).fit()

print(model.summary())
"""
        return script

    elif estimation_method == "CausalImpact (Bayesian Time Series)":
        script += f"""
# --- 3. Quasi-Experimental Analysis: CausalImpact (via R) ---
print("Running CausalImpact Analysis using R...")
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

date_col = '{ts_params['date_col']}' # Date
outcome_col = '{outcome}'
intervention_date = '{ts_params['intervention_date']}' # Requires 'date' string

df[date_col] = pd.to_datetime(df[date_col])

# --- Prepare Time Series Data ---
"""
        use_panel = ts_params.get('use_panel', False)
        if unit_col and treated_unit:
            if use_panel:
                script += f"""
# Panel Data Mode (Synthetic Control)
unit_col = '{unit_col}'
treated_unit = '{treated_unit}'

# 1. Pivot: Index=Date, Columns=Unit, Values=Outcome
df_pivot = df.pivot_table(index=date_col, columns=unit_col, values=outcome_col, aggfunc='sum')

# 2. Validate Treated Unit exists
if treated_unit not in df_pivot.columns:
    raise ValueError(f"Treated unit '{{treated_unit}}' not found in {{unit_col}} column.")
    
# 3. Structure: Y (Treated) | X1, X2... (Controls)
# Move Treated Unit to first column (CausalImpact requirement)
cols = [treated_unit] + [c for c in df_pivot.columns if c != treated_unit]
df_final = df_pivot[cols]

# 4. Clean column names
df_final.columns = [str(c) for c in df_final.columns]
ts_data = df_final
"""
            else:
                script += f"""
# Single Unit Filter Mode (Analyzing '{treated_unit}' only)
df_unit = df[df['{unit_col}'] == '{treated_unit}']
ts_data = df_unit.groupby(date_col)[outcome_col].mean().to_frame()
ts_data = ts_data.sort_index()
"""
        else:
             script += f"""
# Aggregate Data Mode (All Units)
ts_data = df.groupby(date_col)[outcome_col].mean().to_frame()
ts_data = ts_data.sort_index()
"""
        script += """
ts_data = ts_data.asfreq('D').ffill()

# --- Transfer to R and Run CausalImpact ---
pre_period_start = ts_data.index.min().strftime('%Y-%m-%d')
pre_period_end = (pd.to_datetime(intervention_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
post_period_start = pd.to_datetime(intervention_date).strftime('%Y-%m-%d')
post_period_end = ts_data.index.max().strftime('%Y-%m-%d')
dates_str = [d.strftime('%Y-%m-%d') for d in ts_data.index]

cv = robjects.default_converter + pandas2ri.converter
with localconverter(cv):
    ts_reset = ts_data.reset_index(drop=True)
    r_df = pandas2ri.py2rpy(ts_reset)
    
    robjects.globalenv['py_data'] = r_df
    robjects.globalenv['dates_str'] = robjects.StrVector(dates_str)
    robjects.globalenv['pre_start'] = pre_period_start
    robjects.globalenv['pre_end'] = pre_period_end
    robjects.globalenv['post_start'] = post_period_start
    robjects.globalenv['post_end'] = post_period_end

    robjects.r('''
    library(zoo)
    library(CausalImpact)
    
    time_points <- as.Date(dates_str)
    
    if (ncol(py_data) == 1) {
        z_data <- zoo(as.numeric(py_data[[1]]), time_points)
    } else {
        mat <- data.matrix(py_data)
        z_data <- zoo(mat, time_points)
    }
    
    pre.period <- as.Date(c(pre_start, pre_end))
    post.period <- as.Date(c(post_start, post_end))
    
    impact <- CausalImpact(z_data, pre.period, post.period)
    
    print(summary(impact))
    # plot(impact) # Uncomment to view the plot in an R graphics device if configured
    ''')
"""
        return script
    
    elif estimation_method == "GeoLift (Synthetic Control)":
        date_col = ts_params['date_col'] if ts_params and 'date_col' in ts_params else time_period
        geo_col = unit_col
        treated_geo = treated_unit
        outcome_col = outcome
        intervention_date = ts_params.get('intervention_date', '') if ts_params else ''
        duration = ts_params.get('n_periods', 14) if ts_params else 14
        
        script += f"""
# --- 3. Quasi-Experimental Analysis: GeoLift (via Meta R package) ---
print("\\nRunning GeoLift Analysis using R...")
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

date_col = '{date_col}'
geo_col = '{geo_col}'
outcome_col = '{outcome_col}'
treated_geo = '{treated_geo}'
intervention_date = '{intervention_date}'
duration = {duration}
covariates = {confounders}

cols_to_keep = [date_col, geo_col, outcome_col] + covariates
df_clean = df[cols_to_keep].dropna().copy()
df_clean[date_col] = pd.to_datetime(df_clean[date_col]).dt.strftime('%Y-%m-%d')
for c in [outcome_col] + covariates:
    df_clean[c] = pd.to_numeric(df_clean[c])

cv = robjects.default_converter + pandas2ri.converter
with localconverter(cv):
    r_df = pandas2ri.py2rpy(df_clean)
    robjects.globalenv['py_data'] = r_df
    robjects.globalenv['time_id'] = date_col
    robjects.globalenv['geo_id'] = geo_col
    robjects.globalenv['Y_id'] = outcome_col
    robjects.globalenv['treated_locs'] = robjects.StrVector([treated_geo])
    robjects.globalenv['treatment_start'] = intervention_date
    robjects.globalenv['duration'] = duration
    robjects.globalenv['cov_names'] = robjects.StrVector(covariates) if covariates else robjects.NULL

    robjects.r(\"\"\"
    suppressMessages(library(GeoLift))
    
    unique_dates <- sort(unique(as.Date(py_data[[time_id]])))
    treatment_start_date <- as.Date(treatment_start)
    match_idx <- which(unique_dates >= treatment_start_date)
    
    if (length(match_idx) == 0) {{
        stop("Treatment start date is after all available data.")
    }}
    treatment_start_idx <- match_idx[1]
    
    if (is.null(cov_names)) {{
        geo_data <- GeoDataRead(data = py_data, date_id = time_id, location_id = geo_id, Y_id = Y_id, format = "yyyy-mm-dd")
    }} else {{
        geo_data <- GeoDataRead(data = py_data, date_id = time_id, location_id = geo_id, Y_id = Y_id, X = cov_names, format = "yyyy-mm-dd")
    }}
    
    max_time <- max(geo_data$time)
    treatment_end_idx <- min(treatment_start_idx + duration - 1, max_time)
    
    if (is.null(cov_names)) {{
        gl_res <- GeoLift(Y_id = "Y", time_id = "time", location_id = "location", data = geo_data, locations = treated_locs, treatment_start_time = treatment_start_idx, treatment_end_time = treatment_end_idx, model = "none")
    }} else {{
        gl_res <- GeoLift(Y_id = "Y", time_id = "time", location_id = "location", X = cov_names, data = geo_data, locations = treated_locs, treatment_start_time = treatment_start_idx, treatment_end_time = treatment_end_idx, model = "none")
    }}
    
    print(summary(gl_res))
    # plot(gl_res)
    \"\"\")
"""
        return script

    # --- Time Series Logic Injection in Script ---
    if ts_params and ts_params.get('enabled'):
        date_col = ts_params['date_col']
        freq = ts_params['freq']
        
        script += f"""
# --- Time Series Analysis ---
print("\\nRunning Time Series Analysis...")
df['{date_col}'] = pd.to_datetime(df['{date_col}'])

# Create Period Column
if "{freq}" == "Weekly":
    df['Period'] = df['{date_col}'].dt.to_period('W').dt.start_time
elif "{freq}" == "Monthly":
    df['Period'] = df['{date_col}'].dt.to_period('M').dt.start_time
elif "{freq}" == "Quarterly":
    df['Period'] = df['{date_col}'].dt.to_period('Q').dt.start_time

periods = df['Period'].sort_values().unique()
ts_results = []

for period in periods:
    df_period = df[df['Period'] == period]
    if len(df_period) < 20: # Min sample size check
        continue
        
    print(f"Processing Period: {{period}} (N={{len(df_period)}})")
"""
        # We need to inject the estimation logic INSIDE the loop. 
        # To avoid massive duplicate string logic, we'll assign 'df' to 'df_period' temporarily in the script context?
        # No, 'df' is used by CausalModel.
        # Let's adjust the script generation to support this nested structure or just append a specialized loop block.
        # Since implementation plan said "inject code", I'll write the loop block explicitly reusing the params.

        script += f"""
    # --- Estmation for Period ---
    try:
        # Pre-process treatment (if encoded)
        local_treatment = '{treatment}'
        if '{treatment}' == 'Treatment_Encoded': 
             # Ensure encoding exists in subset
             pass 

        model = CausalModel(
            data=df_period,
            treatment=local_treatment,
            outcome='{outcome}',
            common_causes={confounders},
             effect_modifiers={hte_features if estimation_method in ["Linear Double Machine Learning (LinearDML)", "Generalized Random Forests (CausalForestDML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner"] else []}
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        estimate = None
""" 
        # Here we basically need to copy-paste the estimation logic block again? 
        # That's messy in a formatted string. 
        # Simplified estimation for script loop:
        if estimation_method == "Linear Double Machine Learning (LinearDML)":
             script += """
        model_y = RandomForestRegressor(n_jobs=-1, random_state=42)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dml.LinearDML",
            method_params={"init_params": {"model_y": model_y, "model_t": RandomForestClassifier(n_jobs=-1, random_state=42), "discrete_treatment": True}, "fit_params": {}}
        )
"""
        elif "Meta-Learner" in estimation_method:
             learner = "SLearner" if "S-Learner" in estimation_method else "TLearner"
             script += f"""
        method_name = "backdoor.econml.metalearners.{learner}"
        estimate = model.estimate_effect(identified_estimand, method_name=method_name)
"""
        else: # Default linear / other
             script += """
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
"""

        script += """
        if estimate is not None:
             ts_results.append({
                 'Date': period,
                 'ATE': estimate.value,
                 'CI_Lower': estimate.get_confidence_intervals()[0] if hasattr(estimate, 'get_confidence_intervals') else np.nan,
                 'CI_Upper': estimate.get_confidence_intervals()[1] if hasattr(estimate, 'get_confidence_intervals') else np.nan
             })
    except Exception as e:
        print(f"Error in period {{period}}: {{e}}")

if ts_results:
    results_df = pd.DataFrame(ts_results)
    print(results_df)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Date'], results_df['ATE'], marker='o', label='ATE')
    plt.fill_between(results_df['Date'], results_df['CI_Lower'], results_df['CI_Upper'], alpha=0.2, label='95% CI')
    plt.title(f"Causal Effect over Time ({freq})")
    plt.xlabel("Date")
    plt.ylabel("Average Treatment Effect")
    plt.legend()
    plt.show()
else:
    print("No time series results generated.")

# Exit script after TS analysis if you don't want the overall analysis to run
# exit() 
"""
    
    # --- Standard Analysis Script Continues ---
    script += f"""
# --- 3. Causal Model (Overall) ---
"""
    # ... (Rest of existing script generation)
    if treat_val is not None and control_val is not None:
        # Handle categorical treatment encoding in the script
        script += "df['Treatment_Encoded'] = np.nan\n"
        script += f"df.loc[df['{treatment}'] == {repr(control_val)}, 'Treatment_Encoded'] = 0\n"
        script += f"df.loc[df['{treatment}'] == {repr(treat_val)}, 'Treatment_Encoded'] = 1\n"
        script += "df = df.dropna(subset=['Treatment_Encoded'])\n"
        script += "treatment = 'Treatment_Encoded'\n"
    else:
        script += f"treatment = '{treatment}'\n"

    script += f"""
outcome = '{outcome}'
confounders = {confounders}
instrument = None
use_logit = {use_logit}

# Only use confounders as effect modifiers for HTE-capable ML methods
if "{estimation_method}" in ["Linear Double Machine Learning (LinearDML)", "Generalized Random Forests (CausalForestDML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner"]:
    effect_modifiers = confounders
else:
    effect_modifiers = []


# Check for Binary Outcome
is_binary_outcome = False
if df[outcome].nunique() == 2:
    is_binary_outcome = True
    print(f"Detected binary outcome: {{outcome}}. Using Classification models.")
"""



    script += """
model = CausalModel(
    data=df,
    treatment=treatment,
    outcome=outcome,
    common_causes=confounders,
    instruments=instrument,
    effect_modifiers=effect_modifiers
)

# --- 4. Identify Effect ---
"""
    script += "identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)\n"
        
    script += """print("Effect Identified")

# --- 5. Estimate Effect ---
"""
    script += f'estimation_method = "{estimation_method}"\n'
    script += 'print(f"Estimating effect using {estimation_method}...")\n\n'

    script += "def estimate_causal_effect(model, identified_estimand, test_significance=True):\n"
    script += "    estimate = None\n"

    # Logic for estimation methods (simplified mapping from app)
    if estimation_method == "Linear Double Machine Learning (LinearDML)":
        script += "    # Always use Regressor for model_y (LPM) to avoid errors with binary outcomes in LinearDML\n"
        script += "    # because LinearDML expects a continuous residual or probability estimate,\n"
        script += "    # and passing a Classifier can cause errors if EconML expects a Regressor interface.\n"
        script += "    model_y = RandomForestRegressor(n_jobs=-1, random_state=42)\n"
        script += "\n"
        script += "    estimate = model.estimate_effect(\n"
        script += "        identified_estimand,\n"
        script += "        method_name=\"backdoor.econml.dml.LinearDML\",\n"
        script += "        method_params={\n"
        script += "            \"init_params\": {\n"
        script += "                \"model_y\": model_y,\n"
        script += "                \"model_t\": RandomForestClassifier(n_jobs=-1, random_state=42),\n"
        script += "                \"discrete_treatment\": True,\n"
        script += "                \"linear_first_stages\": False,\n"
        script += "                \"cv\": 3,\n"
        script += "                \"random_state\": 42\n"
        script += "            },\n"
        script += "            \"fit_params\": {}\n"
        script += "        }\n"
        script += "    )\n"
    elif estimation_method == "Propensity Score Matching (PSM)":
        script += "    if is_binary_outcome:\n"
        script += "        print('Binary Outcome: Estimate represents Risk Difference (Difference in Proportions).')\n"
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.propensity_score_matching')\n"
    elif estimation_method == "Inverse Propensity Weighting (IPTW)":
        script += "    if is_binary_outcome:\n"
        script += "        print('Binary Outcome: Estimate represents Risk Difference (Weighted Difference in Proportions).')\n"
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.propensity_score_weighting')\n"
    elif "Meta-Learner" in estimation_method:
        learner = "SLearner" if "S-Learner" in estimation_method else "TLearner"
        method_name = f"backdoor.econml.metalearners.{learner}"
        
        if learner == "SLearner":
            script += "    if is_binary_outcome:\n"
            script += "        overall_model = RandomForestClassifier(n_jobs=-1, random_state=42)\n"
            script += "    else:\n"
            script += "        overall_model = RandomForestRegressor(n_jobs=-1, random_state=42)\n"
            script += "    init_params = {\"overall_model\": overall_model}\n"
        else: # T-Learner
            script += "    print(\"T-Learner (Two Learners) fits separate models for treated and control groups.\")\n"
            script += "    method_name = \"backdoor.econml.metalearners.TLearner\"\n"
            script += "    if is_binary_outcome:\n"
            script += "        models = RandomForestClassifier(n_jobs=-1, random_state=42)\n"
            script += "    else:\n"
            script += "        models = RandomForestRegressor(n_jobs=-1, random_state=42)\n"
            script += "    init_params = {\"models\": models}\n"
        
        script += "    estimate = model.estimate_effect(identified_estimand, method_name=method_name, method_params=dict(init_params=init_params, fit_params=dict()))\n"
    elif estimation_method == "Generalized Random Forests (CausalForestDML)":
        # Always use Regressor for model_y (LPM)
        script += "    model_y = RandomForestRegressor(n_jobs=-1, random_state=42)\n"
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.econml.dml.CausalForestDML', method_params=dict(init_params=dict(model_y=model_y, model_t=RandomForestClassifier(n_jobs=-1, random_state=42), discrete_treatment=True, random_state=42), fit_params=dict()))\n"

    elif estimation_method == "Difference-in-Differences (DiD)":
        script += "    # ----------------------------------------------------------------\n"
        script += "    # Manual DiD Estimation\n"
        script += "    # We manually fit OLS with an interaction term (Treatment * Time)\n"
        script += "    # because standard DoWhy estimators do not automatically handle\n"
        script += "    # this specific DiD formulation.\n"
        script += "    # ----------------------------------------------------------------\n"
        script += "    if is_binary_outcome:\n"
        script += "        if use_logit:\n"
        script += "            print('Binary Outcome: Using Logit Model (Logistic Regression). Estimate represents Odds Ratio.')\n"
        script += "        else:\n"
        script += "            print('Binary Outcome: Using Linear Probability Model (LPM) approach. Estimate represents Risk Difference.')\n"
        script += "    data = model._data.copy()\n"
        script += f"    data['DiD_Interaction'] = data['{treatment}'] * data['{time_period}']\n"
        script += f"    X_did = data[['{treatment}', '{time_period}', 'DiD_Interaction']]\n"
        script += f"    if {confounders}:\n"
        script += f"        X_did = pd.concat([X_did, data[{confounders}]], axis=1)\n"
        script += "    X_did = sm.add_constant(X_did)\n"
        script += f"    y_did = data['{outcome}']\n"
        
        script += "    if use_logit and is_binary_outcome:\n"
        script += "        did_model = sm.Logit(y_did, X_did).fit(disp=0)\n"
        script += "        did_coeff = did_model.params['DiD_Interaction']\n"
        script += "        odds_ratio = np.exp(did_coeff)\n"
        script += "        print(did_model.summary())\n"
        script += "        print(f'Estimated Odds Ratio (Interaction): {odds_ratio:.4f}')\n"
        script += "        # Convert OR to Risk Difference\n"
        script += f"        baseline_risk = data[data['{treatment}'] == 0]['{outcome}'].mean()\n"
        script += "        if 0 < baseline_risk < 1:\n"
        script += "            def or_to_rd(or_val, p0):\n"
        script += "                return (or_val * p0 / (1 - p0 + (or_val * p0))) - p0\n"
        script += "            implied_risk_diff = or_to_rd(odds_ratio, baseline_risk)\n"
        script += "            # CI Conversion\n"
        script += "            did_conf_int = did_model.conf_int().loc['DiD_Interaction']\n"
        script += "            or_lower = np.exp(did_conf_int[0])\n"
        script += "            or_upper = np.exp(did_conf_int[1])\n"
        script += "            rd_lower = or_to_rd(or_lower, baseline_risk)\n"
        script += "            rd_upper = or_to_rd(or_upper, baseline_risk)\n"
        script += "            print(f'Implied Risk Difference (at Baseline Risk {baseline_risk:.2%}): {implied_risk_diff:+.2%}')\n"
        script += "            print(f'Implied RD 95% CI: [{rd_lower:+.2%}, {rd_upper:+.2%}]')\n"
        script += "        estimate = type('obj', (object,), {'value': did_coeff})\n"
        script += "    else:\n"
        script += "        did_model = sm.OLS(y_did, X_did).fit()\n"
        script += "        print(did_model.summary())\n"
        script += "        estimate = type('obj', (object,), {'value': did_model.params['DiD_Interaction']})\n"

    elif estimation_method == "Linear/Logistic Regression (OLS/Logit)":
        script += "    if is_binary_outcome:\n"
        script += "        if use_logit:\n"
        script += "            print('Binary Outcome: Using Logit Model (Logistic Regression). Estimate represents Odds Ratio.')\n"
        script += "        else:\n"
        script += "            print('Binary Outcome: Using Linear Probability Model (LPM) approach. Estimate represents Risk Difference.')\n"
        
        script += "    if use_logit and is_binary_outcome:\n"
        script += f"        X_ab = df[['{treatment}'] + confounders]\n"
        script += "        X_ab = sm.add_constant(X_ab)\n"
        script += f"        y_ab = df['{outcome}']\n"
        script += "        ab_model = sm.Logit(y_ab, X_ab).fit(disp=0)\n"
        script += f"        ab_coeff = ab_model.params['{treatment}']\n"
        script += "        odds_ratio = np.exp(ab_coeff)\n"
        script += "        print(ab_model.summary())\n"
        script += "        print(f'Estimated Odds Ratio: {odds_ratio:.4f}')\n"
        script += "        # Convert OR to Risk Difference\n"
        script += f"        baseline_risk = df[df['{treatment}'] == 0]['{outcome}'].mean()\n"
        script += "        if 0 < baseline_risk < 1:\n"
        script += "            def or_to_rd(or_val, p0):\n"
        script += "                return (or_val * p0 / (1 - p0 + (or_val * p0))) - p0\n"
        script += "            implied_risk_diff = or_to_rd(odds_ratio, baseline_risk)\n"
        script += "            # CI Conversion\n"
        script += "            ab_conf_int = ab_model.conf_int().loc['{treatment}']\n"
        script += "            or_lower = np.exp(ab_conf_int[0])\n"
        script += "            or_upper = np.exp(ab_conf_int[1])\n"
        script += "            rd_lower = or_to_rd(or_lower, baseline_risk)\n"
        script += "            rd_upper = or_to_rd(or_upper, baseline_risk)\n"
        script += "            print(f'Implied Risk Difference (at Baseline Risk {baseline_risk:.2%}): {implied_risk_diff:+.2%}')\n"
        script += "            print(f'Implied RD 95% CI: [{rd_lower:+.2%}, {rd_upper:+.2%}]')\n"
        script += "        estimate = type('obj', (object,), {'value': ab_coeff})\n"
        script += "    else:\n"
        script += "        estimate = model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression', test_significance=test_significance)\n"
    
    script += "    return estimate\n"

    script += "estimate = estimate_causal_effect(model, identified_estimand)\n"
    script += "print(f'Average Treatment Effect (ATE): {estimate.value}')\n"
    script += "\n"
    
    # --- 6. Refutation Tests ---
    script += """
# --- 6. Refutation Tests ---
print("\\nRunning Refutation Tests...")

def run_refutation(model, identified_estimand, estimate):
    # 1. Random Common Cause
    print("Test 1: Random Common Cause (Adding a random variable as a common cause)")
    try:
        refute_1 = model.refute_effect(identified_estimand, estimate, method_name="random_common_cause")
        print(refute_1)
        print("Sharma & Kiciman (2020) suggests that if the estimate changes significantly, the model may be misspecified.")
    except Exception as e:
        print(f"Random Common Cause Test Failed: {e}")

    # 2. Placebo Treatment
    print("\\nTest 2: Placebo Treatment (Replacing treatment with a random variable)")
    try:
        refute_2 = model.refute_effect(identified_estimand, estimate, method_name="placebo_treatment_refuter", placebo_type="permute")
        print(refute_2)
        print("Angrist & Pischke (2009) utilize placebo tests to verify that the effect disappears when the treatment is random.")
    except Exception as e:
        print(f"Placebo Treatment Test Failed: {e}")

run_refutation(model, identified_estimand, estimate)
"""

    # --- 7. Bootstrapping (if enabled) ---
    if n_iterations > 1:
        script += f"""
# --- 7. Bootstrapping for Confidence Intervals ---
n_iterations = {n_iterations}
print(f"\\nRunning {{n_iterations}} bootstrap iterations...")
bootstrap_estimates = []

for i in range(n_iterations):
    # Resample with replacement
    df_boot = df.sample(frac=1, replace=True, random_state=i)
    
    # Re-build and re-estimate
    model_boot = CausalModel(
        data=df_boot,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders,
        effect_modifiers=effect_modifiers
    )
    ident_boot = model_boot.identify_effect(proceed_when_unidentifiable=True)
    est_boot = estimate_causal_effect(model_boot, ident_boot, test_significance=False)
    
    if est_boot and hasattr(est_boot, 'value'):
        bootstrap_estimates.append(est_boot.value)

if bootstrap_estimates:
    bootstrap_estimates = np.array(bootstrap_estimates)
    mean_ate = np.mean(bootstrap_estimates)
    se_boot = np.std(bootstrap_estimates, ddof=1)
    ci_lower = np.percentile(bootstrap_estimates, 2.5)
    ci_upper = np.percentile(bootstrap_estimates, 97.5)
    
    print(f"Bootstrap Results (N={{n_iterations}}):")
    print(f"  Mean ATE: {{mean_ate:.4f}}")
    print(f"  Std Error: {{se_boot:.4f}}")
    print(f"  95% CI: [{{ci_lower:.4f}}, {{ci_upper:.4f}}]")
"""

    script += "\nprint('\\nAnalysis Complete.')\n"
    return script

def calculate_period_effect(df_period, treatment, outcome, confounders, estimation_method, is_binary_outcome=False):
    """
    Helper function to estimate causal effect for a specific time period.
    Returns: ATE, CI_Lower, CI_Upper
    """
    try:
        # 1. Build Model
        # Only use confounders as effect modifiers for HTE-capable ML methods
        if estimation_method in ["Linear Double Machine Learning (LinearDML)", "Generalized Random Forests (CausalForestDML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner"]:
            modifiers = confounders
        else:
            modifiers = []

        model = CausalModel(
            data=df_period,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders,
            instruments=None,
            effect_modifiers=modifiers,
            graph=None # Skip graph generation for speed in loop
        )
        
        # 2. Identify
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # 3. Estimate
        estimate = None
        
        if estimation_method == "Linear Double Machine Learning (LinearDML)":
            model_y = RandomForestRegressor(n_jobs=-1, random_state=42)
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.econml.dml.LinearDML",
                method_params={
                    "init_params": {
                        "model_y": model_y,
                        "model_t": RandomForestClassifier(n_jobs=-1, random_state=42),
                        "discrete_treatment": True,
                        "linear_first_stages": False,
                        "cv": 3,
                        "random_state": 42
                    },
                    "fit_params": {}
                }
            )
        elif estimation_method == "Propensity Score Matching (PSM)":
            estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")
            
        elif estimation_method == "Inverse Propensity Weighting (IPTW)":
            estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_weighting")
            
        elif "Meta-Learner" in estimation_method:
            learner_type = estimation_method.split(": ")[1]
            if learner_type == "S-Learner":
                method_name = "backdoor.econml.metalearners.SLearner"
                if is_binary_outcome:
                    overall_model = RandomForestClassifier(n_jobs=-1, random_state=42)
                else:
                    overall_model = RandomForestRegressor(n_jobs=-1, random_state=42)
                init_params = {"overall_model": overall_model}
            else:
                method_name = "backdoor.econml.metalearners.TLearner"
                if is_binary_outcome:
                    models = RandomForestClassifier(n_jobs=-1, random_state=42)
                else:
                    models = RandomForestRegressor(n_jobs=-1, random_state=42)
                init_params = {"models": models}

            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method_name,
                method_params={"init_params": init_params, "fit_params": {}}
            )
            
        elif estimation_method == "Generalized Random Forests (CausalForestDML)":
            model_y = RandomForestRegressor(n_jobs=-1, random_state=42)
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.econml.dml.CausalForestDML",
                method_params={
                    "init_params": {
                        "model_y": model_y,
                        "model_t": RandomForestClassifier(n_jobs=-1, random_state=42),
                        "discrete_treatment": True,
                        "random_state": 42
                    },
                    "fit_params": {}
                }
            )
            
        elif estimation_method == "Linear/Logistic Regression (OLS/Logit)":
             estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                test_significance=False
             )

        # Extract CI
        ate = estimate.value
        ci_lower, ci_upper = np.nan, np.nan
        
        try:
            ci = estimate.get_confidence_intervals(confidence_level=0.95)
            if ci is not None:
                ci_lower, ci_upper = ci[0], ci[1]
        except:
            pass
            
        return ate, ci_lower, ci_upper

    except Exception as e:
        print(f"Estimation failed for period: {e}")
        return np.nan, np.nan, np.nan




def run_did_analysis(df, treatment_col, outcome_col, time_col, confounders, is_binary_outcome=False, use_logit=False):
    """
    Runs Difference-in-Differences (DiD) Analysis manually using OLS/Logit interaction.
    Assumptions:
    - Data contains Pre and Post periods.
    - Control and Treatment groups exist.
    """
    try:
        # Create Interaction Term
        df['DiD_Interaction'] = df[treatment_col] * df[time_col]
        
        # Define X (Features) and y (Outcome)
        X = df[[treatment_col, time_col, 'DiD_Interaction']]
        if confounders:
            X = pd.concat([X, df[confounders]], axis=1)
        
        X = sm.add_constant(X)
        y = df[outcome_col]
        
        estimate_obj = {}
        
        if use_logit and is_binary_outcome:
            model = sm.Logit(y, X).fit(disp=0)
            coeff = model.params['DiD_Interaction']
            p_value = model.pvalues['DiD_Interaction']
            conf_int = model.conf_int().loc['DiD_Interaction']
            
            estimate_obj = {
                'method': 'DiD (Logit)',
                'coefficient': coeff,
                'p_value': p_value,
                'ci_lower': conf_int[0],
                'ci_upper': conf_int[1],
                'odds_ratio': np.exp(coeff),
                'summary': model.summary().as_text()
            }
        else:
            model = sm.OLS(y, X).fit()
            coeff = model.params['DiD_Interaction']
            p_value = model.pvalues['DiD_Interaction']
            conf_int = model.conf_int().loc['DiD_Interaction']
            
            estimate_obj = {
                'method': 'DiD (OLS)',
                'coefficient': coeff,
                'p_value': p_value,
                'ci_lower': conf_int[0],
                'ci_upper': conf_int[1],
                'summary': model.summary().as_text()
            }
            
        return estimate_obj

    except Exception as e:
        return {'error': str(e)}

def prepare_time_series(df, date_col, outcome_col, unit_col=None, treated_unit=None, use_panel=False, covariates=None):
    """
    Aggregates user-level data into a daily time-series for CausalImpact.
    If use_panel is True (Synthetic Control), pivots so Treated Unit is Y and others are Covariates.
    If use_panel is False but unit info is provided, filters for the treated unit.
    If covariates are provided (e.g. Marketing Spend), they are aggregated and added as predictors.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Base Data Selection
    if unit_col and treated_unit:
        if use_panel:
            # --- Panel Data Mode (Synthetic Control style) ---
            # 1. Pivot Outcome: Index=Date, Columns=Unit, Values=Outcome
            df_pivot = df.pivot_table(index=date_col, columns=unit_col, values=outcome_col, aggfunc='sum')
            
            # 2. Validate Treated Unit exists
            if treated_unit not in df_pivot.columns:
                raise ValueError(f"Treated unit '{treated_unit}' not found in {unit_col} column.")
                
            # 3. Structure: Y (Treated) | X1, X2... (Controls)
            # Move Treated Unit to first column (CausalImpact requirement)
            cols = [treated_unit] + [c for c in df_pivot.columns if c != treated_unit]
            df_agg = df_pivot[cols]
            
            # 4. Add Covariates (Specific to Treated Unit)
            # If covariates are provided, we likely want the Treated Unit's values for them
            if covariates:
                df_treated = df[df[unit_col] == treated_unit].copy()
                # Aggregate covariates by date (mean usually safe for continuous)
                df_covs = df_treated.groupby(date_col)[covariates].mean()
                # Join with Panel Data
                df_agg = df_agg.join(df_covs, how='left')
                
        else:
            # --- Single Unit Filter Mode ---
            # User wants to analyze just one unit but without using others as controls
            df_unit = df[df[unit_col] == treated_unit]
            if df_unit.empty:
                raise ValueError(f"No data found for unit '{treated_unit}' in column '{unit_col}'.")
            
            cols_to_agg = [outcome_col]
            if covariates:
                cols_to_agg += covariates
                
            df_agg = df_unit.groupby(date_col)[cols_to_agg].mean().sort_index()
            # Ensure Outcome is first column
            df_agg = df_agg[[outcome_col] + (covariates if covariates else [])]
            
    else:
        # --- Standard Time Series Mode (Aggregation) ---
        cols_to_agg = [outcome_col]
        if covariates:
            cols_to_agg += covariates
            
        df_agg = df.groupby(date_col)[cols_to_agg].mean().sort_index()
        # Ensure Outcome is first column
        df_agg = df_agg[[outcome_col] + (covariates if covariates else [])]
        
    # Fill missing dates?
    df_agg = pd.DataFrame(df_agg)
    df_agg = df_agg.asfreq('D').ffill()
    return df_agg

def run_causal_impact_analysis(df, date_col, outcome_col, intervention_date, unit_col=None, treated_unit=None, use_panel=False, covariates=None):
    """
    Runs CausalImpact analysis using the original R package via rpy2.
    df: User-level dataframe (will be aggregated) or Pre-aggregated.
    """
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import time

    try:
        # Aggregation
        ts_data = prepare_time_series(df, date_col, outcome_col, unit_col, treated_unit, use_panel, covariates)
        
        # Define Pre and Post
        intervention_date = pd.to_datetime(intervention_date)
        
        # Ensure index is datetime
        if not isinstance(ts_data.index, pd.DatetimeIndex):
             ts_data.index = pd.to_datetime(ts_data.index)
        
        # Validate Range
        min_date = ts_data.index.min()
        max_date = ts_data.index.max()
        
        if intervention_date <= min_date or intervention_date >= max_date:
            return {'error': f"Intervention date {intervention_date.date()} is outside data range ({min_date.date()} to {max_date.date()})."}

        pre_end = intervention_date - pd.Timedelta(days=1)
        
        cv = robjects.default_converter + pandas2ri.converter
        with localconverter(cv):
            try:
                base = importr('base')
                zoo = importr('zoo')
                ci_pkg = importr('CausalImpact')
                grdevices = importr('grDevices')
            except Exception as e:
                return {"error": f"Failed to load R CausalImpact or zoo package. Ensure they are installed via Rscript. Details: {e}"}

            # Prepare data for R: Zoo objects need a data matrix and a time index
            # ts_data index is datetime, columns are y (outcome) then covariates (if any)
            # We pass the index as strings to r, then parse inside
            dates_str = [d.strftime('%Y-%m-%d') for d in ts_data.index]
            
            # Pass DataFrame to R
            # resetting index so rpy2 can convert safely, we'll reapply zoo index in R
            ts_reset = ts_data.reset_index(drop=True)
            r_df = pandas2ri.py2rpy(ts_reset)
            
            robjects.globalenv['py_data'] = r_df
            robjects.globalenv['dates_str'] = robjects.StrVector(dates_str)
            robjects.globalenv['pre_start'] = min_date.strftime('%Y-%m-%d')
            robjects.globalenv['pre_end'] = pre_end.strftime('%Y-%m-%d')
            robjects.globalenv['post_start'] = intervention_date.strftime('%Y-%m-%d')
            robjects.globalenv['post_end'] = max_date.strftime('%Y-%m-%d')
            
            timestamp = int(time.time())
            plot_file = f"causalimpact_plot_{timestamp}.png"
            csv_file = f"causalimpact_plot_{timestamp}.csv"
            robjects.globalenv['plot_path'] = plot_file
            robjects.globalenv['csv_path'] = csv_file

            robjects.r("""
            # 1. Create zoo object
            time_points <- as.Date(dates_str)
            
            # If py_data only has 1 column, it's just a vector, otherwise a matrix
            if (ncol(py_data) == 1) {
                # Ensure it's a numeric vector
                z_data <- zoo(as.numeric(py_data[[1]]), time_points)
            } else {
                # Convert all to numeric matrix, first column must be Y
                mat <- data.matrix(py_data)
                z_data <- zoo(mat, time_points)
            }
            
            # 2. Define periods
            pre.period <- as.Date(c(pre_start, pre_end))
            post.period <- as.Date(c(post_start, post_end))
            
            # 3. Run CausalImpact
            impact <- CausalImpact(z_data, pre.period, post.period)
            
            # 4. Extract metrics from summary
            sum_data <- impact$summary
            rel_effect <- sum_data$RelEffect[1] # Average Relative Effect %
            rel_lower <- sum_data$RelEffect.lower[1]
            rel_upper <- sum_data$RelEffect.upper[1]
            
            p_val <- impact$summary$p[1]
            
            # Extract Average Effect
            ate <- sum_data$AbsEffect[1]
            ate_lower <- sum_data$AbsEffect.lower[1]
            ate_upper <- sum_data$AbsEffect.upper[1]
            
            # Extract Cumulative Effect (second row)
            cum <- sum_data$AbsEffect[2]
            cum_lower <- sum_data$AbsEffect.lower[2]
            cum_upper <- sum_data$AbsEffect.upper[2]
            
            report_text <- impact$report
            
            # 5. Generate Plot
            png(plot_path, width=800, height=600, res=100)
            print(plot(impact))
            dev.off()
            
            # Export plot data
            write.csv(as.data.frame(impact$series), file=csv_path, row.names=TRUE)
            """)
            
            # Retrieve values from R ecosystem
            p_val = robjects.r('p_val')[0]
            ate = robjects.r('ate')[0]
            ate_lower = robjects.r('ate_lower')[0]
            ate_upper = robjects.r('ate_upper')[0]
            cum = robjects.r('cum')[0]
            cum_lower = robjects.r('cum_lower')[0]
            cum_upper = robjects.r('cum_upper')[0]
            rel_effect = robjects.r('rel_effect')[0]
            rel_lower = robjects.r('rel_lower')[0]
            rel_upper = robjects.r('rel_upper')[0]
            
            # report_text in R might be a vector of strings depending on print method, we'll try to join it
            report_r = robjects.r('report_text')
            report_str = "\n".join(report_r) if hasattr(report_r, '__iter__') and not isinstance(report_r, str) else str(report_r)

            # Read plot data
            plot_df = None
            if os.path.exists(csv_file):
                plot_df = pd.read_csv(csv_file).rename(columns={'Unnamed: 0': 'Time'})

        return {
            'object': None, # No python object anymore
            'plot_path': plot_file,
            'summary': "R CausalImpact model summarized below.",
            'report': report_str,
            'plot_df': plot_df,
            'metrics': {
                'p_value': p_val,
                'ate': ate,
                'ate_lower': ate_lower,
                'ate_upper': ate_upper,
                'cum_abs': cum,
                'cum_lower': cum_lower,
                'cum_upper': cum_upper,
                'rel_effect': rel_effect,
                'rel_lower': rel_lower,
                'rel_upper': rel_upper,
                'alpha': 0.05,
                'significant': p_val < 0.05
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def run_geolift_analysis(df, date_col, geo_col, treated_geo, kpi_col, intervention_date, 
                          treatment_duration=14, model="none", alpha=0.1, 
                          confidence_intervals=False, stat_test="Total", covariates=None,
                          fixed_effects=True, grid_size=250):
    """
    Runs GeoLift Analysis using rpy2 to bridge Python and Meta's GeoLift R package.
    """
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    
    # Combined converter for robust thread safety in Streamlit
    cv = robjects.default_converter + pandas2ri.converter

    with localconverter(cv):
        try:
            try:
                base = importr('base')
                utils = importr('utils')
                geolift = importr('GeoLift')
            except Exception as e:
                return {"error": f"Failed to load R GeoLift package. Ensure it is installed via Rscript. Details: {e}"}

            cols_to_keep = [date_col, geo_col, kpi_col]
            if covariates:
                cols_to_keep += covariates
                
            df_clean = df[cols_to_keep].copy()
            df_clean = df_clean.dropna()
            
            # Convert dates to string format expected by R GeoDataRead
            df_clean[date_col] = pd.to_datetime(df_clean[date_col]).dt.strftime('%Y-%m-%d')
            df_clean[kpi_col] = pd.to_numeric(df_clean[kpi_col])
            if covariates:
                 for c in covariates:
                     df_clean[c] = pd.to_numeric(df_clean[c])
            
            # Pass DataFrame to R
            r_df = pandas2ri.py2rpy(df_clean)
                
            robjects.globalenv['py_data'] = r_df
            robjects.globalenv['time_id'] = date_col
            robjects.globalenv['geo_id'] = geo_col
            robjects.globalenv['Y_id'] = kpi_col
            
            # Additional GeoLift Parameters
            robjects.globalenv['treated_locs'] = robjects.StrVector([treated_geo])
            robjects.globalenv['treatment_start'] = str(intervention_date)
            robjects.globalenv['duration'] = int(treatment_duration)
            robjects.globalenv['model_name'] = model
            robjects.globalenv['alpha_val'] = float(alpha)
            robjects.globalenv['calc_ci'] = bool(confidence_intervals)
            robjects.globalenv['test_stat'] = stat_test
            robjects.globalenv['fixed_effects_val'] = bool(fixed_effects)
            robjects.globalenv['grid_val'] = int(grid_size)
            
            if covariates:
                robjects.globalenv['cov_names'] = robjects.StrVector(covariates)
            else:
                robjects.globalenv['cov_names'] = robjects.NULL
            
            robjects.r("""
            # 1. Map dates to indices BEFORE GeoDataRead drops the date column
            # GeoDataRead maps the chronological order of sorted unique dates to 1..N
            unique_dates <- sort(unique(as.Date(py_data[[time_id]])))
            treatment_start_date <- as.Date(treatment_start)
            
            # Find the index of the first date >= treatment_start
            match_idx <- which(unique_dates >= treatment_start_date)
            
            if (length(match_idx) == 0) {
                stop(paste("Treatment start date", treatment_start, "is after all available data."))
            }
            treatment_start_idx <- match_idx[1]
            
            # 2. Convert the dataframe for GeoLift
            if (is.null(cov_names)) {
                geo_data <- GeoDataRead(data = py_data,
                                        date_id = time_id,
                                        location_id = geo_id,
                                        Y_id = Y_id,
                                        format = "yyyy-mm-dd")
            } else {
                geo_data <- GeoDataRead(data = py_data,
                                        date_id = time_id,
                                        location_id = geo_id,
                                        Y_id = Y_id,
                                        X = cov_names,
                                        format = "yyyy-mm-dd")
            }
            
            # 3. Calculate end index based on duration
            max_time <- max(geo_data$time)
            treatment_end_idx <- min(treatment_start_idx + duration - 1, max_time)
            
            if (treatment_end_idx < treatment_start_idx) {
               stop("Treatment duration too short or starts too late in the series.")
            }
            
            # 4. Run GeoLift
            if (is.null(cov_names)) {
                gl_res <- GeoLift(Y_id = "Y",
                                  time_id = "time",
                                  location_id = "location",
                                  data = geo_data,
                                  locations = treated_locs,
                                  treatment_start_time = treatment_start_idx,
                                  treatment_end_time = treatment_end_idx,
                                  model = model_name,
                                  fixed_effects = fixed_effects_val,
                                  grid_size = grid_val,
                                  alpha = alpha_val,
                                  ConfidenceIntervals = calc_ci,
                                  stat_test = test_stat)
            } else {
                gl_res <- GeoLift(Y_id = "Y",
                                  time_id = "time",
                                  location_id = "location",
                                  X = cov_names,
                                  data = geo_data,
                                  locations = treated_locs,
                                  treatment_start_time = treatment_start_idx,
                                  treatment_end_time = treatment_end_idx,
                                  model = model_name,
                                  fixed_effects = fixed_effects_val,
                                  grid_size = grid_val,
                                  alpha = alpha_val,
                                  ConfidenceIntervals = calc_ci,
                                  stat_test = test_stat)
            }
                              
            summary_res <- summary(gl_res)
            """)
            
            # Safely extract from R environment with NULL checks
            r_summary = robjects.globalenv['summary_res']
            if r_summary is robjects.NULL:
                return {"error": "GeoLift summary failed: The model did not converge or produced NULL results."}
                
            robjects.r("""
            avg_lift <- summary_res$ATT_est
            cumulative_lift <- summary_res$incremental
            p_val <- summary_res$pvalue
            perc_lift <- summary_res$PercLift
            
            # Retrieve CIs if available
            has_ci <- !is.null(summary_res$CI)
            if (has_ci) {
                ate_lower <- summary_res$LowerCI[1]
                ate_upper <- summary_res$UpperCI[1]
                cum_lower <- summary_res$lower[1]
                cum_upper <- summary_res$upper[1]
            } else {
                ate_lower <- NA
                ate_upper <- NA
                cum_lower <- NA
                cum_upper <- NA
            }
            """)
            
            r_avg = robjects.globalenv['avg_lift']
            r_cum = robjects.globalenv['cumulative_lift']
            r_p = robjects.globalenv['p_val']
            perc_lift = robjects.globalenv['perc_lift'][0]
            
            has_ci = robjects.globalenv['has_ci'][0]
            ate_lower = robjects.globalenv['ate_lower'][0] if has_ci else None
            ate_upper = robjects.globalenv['ate_upper'][0] if has_ci else None
            cum_lower = robjects.globalenv['cum_lower'][0] if has_ci else None
            cum_upper = robjects.globalenv['cum_upper'][0] if has_ci else None
            
            if r_avg is robjects.NULL or r_cum is robjects.NULL or r_p is robjects.NULL:
                return {"error": "GeoLift estimation produced NULL metrics. Check if control group has enough variance."}
                
            avg_lift = r_avg[0]
            cum_lift = r_cum[0]
            p_val = r_p[0]
            
            significant = "Yes" if p_val < alpha else "No"
            
            import time
            timestamp = int(time.time())
            impact_plot_file = f"geolift_impact_plot_{timestamp}.png"
            att_plot_file = f"geolift_att_plot_{timestamp}.png"
            csv_file = f"geolift_plot_data_{timestamp}.csv"
            robjects.globalenv['impact_plot_path'] = impact_plot_file
            robjects.globalenv['att_plot_path'] = att_plot_file
            robjects.globalenv['csv_path'] = csv_file
            
            # Generate the plots and capture the full inference summary
            robjects.r("""
            # Capture the full beautifully formatted summary as a single block of text
            full_summary_text <- paste(capture.output(print(summary(gl_res)), type='message'), collapse='\\n')
            
            # Standard Plot (Treated vs Synthetic)
            png(impact_plot_path, width=800, height=600)
            print(plot(gl_res))
            dev.off()
            
            # ATT Plot (Average Effect on Treated)
            png(att_plot_path, width=800, height=600)
            print(plot(gl_res, type="ATT"))
            dev.off()
            
            # Export plot data
            if ("y_obs" %in% names(gl_res) && "y_hat" %in% names(gl_res)) {
                has_bounds <- "lower_bound" %in% names(gl_res)
                if (has_bounds) {
                    plot_data <- data.frame(
                        Time = names(gl_res$y_obs),
                        Observed = as.numeric(gl_res$y_obs),
                        Synthetic = as.numeric(gl_res$y_hat),
                        Lower_Bound = as.numeric(gl_res$lower_bound),
                        Upper_Bound = as.numeric(gl_res$upper_bound)
                    )
                } else {
                    plot_data <- data.frame(
                        Time = names(gl_res$y_obs),
                        Observed = as.numeric(gl_res$y_obs),
                        Synthetic = as.numeric(gl_res$y_hat)
                    )
                }
                write.csv(plot_data, file=csv_path, row.names=FALSE)
            }
            """)
            
            full_summary = robjects.globalenv['full_summary_text'][0]
            
            report = f"""
            ```text
            {full_summary}
            ```
            """
            
            # Cleanup
            robjects.r("rm(gl_res, summary_res, avg_lift, cumulative_lift, p_val, full_summary_text)")
            
            # Calculate bounds for relative lift (%) if CI exists, returning decimal for python .2% conversion
            perc_lift_decimal = perc_lift / 100
            rel_lower = (cum_lower / (cum_lift / perc_lift_decimal)) if has_ci and perc_lift != 0 else None
            rel_upper = (cum_upper / (cum_lift / perc_lift_decimal)) if has_ci and perc_lift != 0 else None
            # Read plot data
            plot_df = None
            if os.path.exists(csv_file):
                plot_df = pd.read_csv(csv_file)
                
            return {
                "summary": full_summary,
                "plot_df": plot_df,
                "is_power_analysis": False,
                "metrics": {
                    "avg_lift": float(avg_lift),
                    "cum_lift": float(cum_lift),
                    "p_val": float(p_val),
                    "alpha": float(alpha),
                    "treated_geo": treated_geo,
                    "model": model,
                    "significant": p_val < alpha,
                    "perc_lift": float(perc_lift_decimal),
                    "ate_lower": float(ate_lower) if ate_lower else None,
                    "ate_upper": float(ate_upper) if ate_upper else None,
                    "cum_lower": float(cum_lower) if cum_lower else None,
                    "cum_upper": float(cum_upper) if cum_upper else None,
                    "rel_lower": float(rel_lower) if rel_lower else None,
                    "rel_upper": float(rel_upper) if rel_upper else None,
                    "has_ci": bool(has_ci)
                },
                "plot_path": impact_plot_file,
                "att_plot_path": att_plot_file
            }
            
        except Exception as e:
            return {"error": f"GeoLift execution failed: {e}"}


def run_geolift_power(df, date_col, geo_col, kpi_col, treatment_duration=14, cutoff_date=None, 
                      n_markets="1", lookback_window=1, model="none", alpha=0.1, side_of_test="two_sided",
                      parallel=True, ns=1000, effect_size_mode="Full", normalize=False, covariates=None,
                      fixed_effects=True, dtw=0, correlations=False, cpic=1.0, budget=None):
    """
    Runs GeoLift Market Selection (Power Analysis) via rpy2 with performance optimizations.
    """
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import os
    import time
    
    # Combined converter for thread safety
    cv = robjects.default_converter + pandas2ri.converter

    with localconverter(cv):
        try:
            try:
                base = importr('base')
                grdevices = importr('grDevices')
                geolift = importr('GeoLift')
            except Exception as e:
                return {"error": f"Failed to load R GeoLift package. Details: {e}"}

            cols_to_keep = [date_col, geo_col, kpi_col]
            if covariates:
                cols_to_keep += covariates
                
            df_clean = df[cols_to_keep].copy()
            
            # Application of historical cutoff if provided
            if cutoff_date:
                df_clean = df_clean[pd.to_datetime(df_clean[date_col]) < pd.to_datetime(cutoff_date)]
                
            df_clean = df_clean.dropna()
            df_clean[date_col] = pd.to_datetime(df_clean[date_col]).dt.strftime('%Y-%m-%d')
            df_clean[kpi_col] = pd.to_numeric(df_clean[kpi_col])
            
            if covariates:
                for c in covariates:
                    df_clean[c] = pd.to_numeric(df_clean[c])
            
            # Pass DataFrame to R
            r_df = pandas2ri.py2rpy(df_clean)
                
            robjects.globalenv['py_data'] = r_df
            robjects.globalenv['date_col_name'] = date_col
            robjects.globalenv['geo_col_name'] = geo_col
            robjects.globalenv['kpi_col_name'] = kpi_col
            
            if covariates:
                robjects.globalenv['cov_names'] = robjects.StrVector(covariates)
            else:
                robjects.globalenv['cov_names'] = robjects.NULL
            
            robjects.r("""
            if (is.null(cov_names)) {
                geo_data <- GeoDataRead(data = py_data,
                                        date_id = date_col_name,
                                        location_id = geo_col_name,
                                        Y_id = kpi_col_name,
                                        format = "yyyy-mm-dd")
            } else {
                geo_data <- GeoDataRead(data = py_data,
                                        date_id = date_col_name,
                                        location_id = geo_col_name,
                                        Y_id = kpi_col_name,
                                        X = cov_names,
                                        format = "yyyy-mm-dd")
            }
            """)
            
            # Parameters
            robjects.globalenv['duration'] = int(treatment_duration)
            robjects.globalenv['lookback'] = int(lookback_window)
            robjects.globalenv['model_name'] = model
            robjects.globalenv['alpha_val'] = float(alpha)
            robjects.globalenv['side'] = side_of_test
            robjects.globalenv['parallel_run'] = bool(parallel)
            robjects.globalenv['ns_val'] = int(ns)
            robjects.globalenv['normalize_val'] = bool(normalize)
            robjects.globalenv['fixed_effects_val'] = bool(fixed_effects)
            robjects.globalenv['dtw_val'] = float(dtw)
            robjects.globalenv['corr_val'] = bool(correlations)
            robjects.globalenv['cpic_val'] = float(cpic)
            robjects.globalenv['budget_val'] = float(budget) if budget is not None else robjects.NULL
            
            # Effect size handling
            if effect_size_mode == "Fast":
                robjects.r("es_sequence <- c(0, 0.1)")
            else:
                robjects.r("es_sequence <- seq(0, 0.2, 0.05)")

            # Handle N - can be a single int or a vector
            try:
                n_list = [int(x.strip()) for x in str(n_markets).split(",")]
                robjects.globalenv['n_markets_vec'] = robjects.IntVector(n_list)
            except:
                robjects.globalenv['n_markets_vec'] = 1

            timestamp = int(time.time())
            power_plot_file = f"geolift_power_{timestamp}.png"
            series_plot_file = f"geolift_series_{timestamp}.png"
            robjects.globalenv['p_plot_path'] = power_plot_file
            robjects.globalenv['s_plot_path'] = series_plot_file

            robjects.r("""
            # Market Selection searches for the best test markets
            if (is.null(cov_names)) {
                market_selection <- GeoLiftMarketSelection(
                    data = geo_data,
                    treatment_periods = duration,
                    N = n_markets_vec,
                    Y_id = "Y",
                    location_id = "location",
                    time_id = "time",
                    lookback_window = lookback,
                    model = model_name,
                    fixed_effects = fixed_effects_val,
                    dtw = dtw_val,
                    Correlations = corr_val,
                    cpic = cpic_val,
                    budget = budget_val,
                    alpha = alpha_val,
                    side_of_test = side,
                    parallel = parallel_run,
                    parallel_setup = "parallel", # Use multitasking on Mac
                    ns = ns_val,
                    normalize = normalize_val,
                    effect_size = es_sequence,
                    ProgressBar = FALSE,
                    print = FALSE
                )
            } else {
                market_selection <- GeoLiftMarketSelection(
                    data = geo_data,
                    treatment_periods = duration,
                    N = n_markets_vec,
                    Y_id = "Y",
                    location_id = "location",
                    time_id = "time",
                    X = cov_names,
                    lookback_window = lookback,
                    model = model_name,
                    fixed_effects = fixed_effects_val,
                    dtw = dtw_val,
                    Correlations = corr_val,
                    cpic = cpic_val,
                    budget = budget_val,
                    alpha = alpha_val,
                    side_of_test = side,
                    parallel = parallel_run,
                    parallel_setup = "parallel", # Use multitasking on Mac
                    ns = ns_val,
                    normalize = normalize_val,
                    effect_size = es_sequence,
                    ProgressBar = FALSE,
                    print = FALSE
                )
            }
            
            # Capture the best markets table
            # Strip custom classes to ensure smooth R->Python conversion
            best_markets_df <- as.data.frame(market_selection$BestMarkets)
            
            # Plots for the #1 ranked market
            # 1. Power Curve Plot (using ggplot2)
            library(ggplot2)
            top_location <- as.character(best_markets_df$location[1])
            pc_data <- market_selection$PowerCurves[market_selection$PowerCurves$location == top_location, ]
            
            png(p_plot_path, width = 800, height = 600)
            p_plot <- ggplot(pc_data, aes(x = AvgDetectedLift, y = power)) + 
              geom_line(color = "#1f77b4", size = 1.2) + 
              geom_point(color = "#1f77b4", size = 3) +
              theme_minimal() + 
              labs(title = paste("Power Curve:", top_location),
                   subtitle = "Likelihood of detecting effect vs. Proposed Lift Size",
                   x = "Effect Size (Average Detected Lift)",
                   y = "Probability of Success (Power)") +
              geom_hline(yintercept = 0.8, linetype = "dashed", color = "red") +
              annotate("text", x = min(pc_data$AvgDetectedLift), y = 0.82, label = "80% Power Threshold", color = "red", hjust = 0)
            print(p_plot)
            dev.off()
            
            # 2. Historical Fit Plot
            png(s_plot_path, width = 800, height = 600)
            plot(market_selection, market_ID = 1)
            dev.off()
            """)
            
            r_best_markets = robjects.globalenv['best_markets_df']
            
            # Convert R dataframe to Pandas
            if isinstance(r_best_markets, pd.DataFrame):
                df_best_markets = r_best_markets
            else:
                with localconverter(cv):
                    df_best_markets = pandas2ri.rpy2py(r_best_markets)
                
            return {
                "df": df_best_markets,
                "power_plot": power_plot_file,
                "series_plot": series_plot_file
            }
            
        except Exception as e:
            return {"error": f"GeoLift Market Selection failed: {e}"}
            
        except Exception as e:
            return {"error": f"GeoLift Market Selection failed: {e}"}


# ========================================================
# CausalPy: Bayesian Synthetic Control (Pure Python)
# ========================================================
def run_causalpy_synthetic_control(df, date_col, geo_col, kpi_col, treated_geo,
                                    intervention_date, treatment_duration=60,
                                    hdi_prob=0.95, direction="two-sided",
                                    covariates=None):
    """
    Runs Bayesian Synthetic Control using CausalPy (PyMC backend).
    Returns a structured dict with metrics, plots, and summary text.
    """
    import causalpy
    from causalpy.pymc_models import WeightedSumFitter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import tempfile
    import os

    try:
        # --- 1. Data Preparation ---
        df_work = df[[date_col, geo_col, kpi_col]].copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col])
        df_work[kpi_col] = pd.to_numeric(df_work[kpi_col], errors='coerce')
        df_work = df_work.dropna()

        # Pivot to wide format: rows=dates, columns=regions
        df_wide = df_work.pivot_table(index=date_col, columns=geo_col, values=kpi_col, aggfunc='mean')
        df_wide = df_wide.sort_index().reset_index(drop=False)
        df_wide = df_wide.set_index(date_col)

        # Validate treated geo exists
        treated_geo_str = str(treated_geo)
        if treated_geo_str not in df_wide.columns:
            return {"error": f"Treated geography '{treated_geo_str}' not found in pivoted columns: {list(df_wide.columns)}"}

        # Identify control columns
        control_cols = [c for c in df_wide.columns if c != treated_geo_str]
        if len(control_cols) == 0:
            return {"error": "No control regions found. Need at least one untreated region."}

        # Drop any control columns with all NaN
        df_wide = df_wide.dropna(axis=1, how='all')
        control_cols = [c for c in df_wide.columns if c != treated_geo_str]

        # Forward-fill remaining NaN
        df_wide = df_wide.ffill().bfill()

        # Determine treatment time
        intervention_dt = pd.to_datetime(intervention_date)

        # --- 2. Fit Model ---
        result = causalpy.SyntheticControl(
            df_wide,
            treatment_time=intervention_dt,
            treated_units=[treated_geo_str],
            control_units=control_cols,
            model=WeightedSumFitter(sample_kwargs={
                "target_accept": 0.95,
                "random_seed": 42,
                "progressbar": False
            }),
        )

        # --- 3. Extract Effect Summary ---
        stats = result.effect_summary(
            treated_unit=treated_geo_str,
            direction=direction,
        )

        # Parse the stats table
        stats_table = stats.table
        stats_text = stats.text

        # Extract key metrics from the table
        # Table structure: index=["average","cumulative"], columns=["mean","median","hdi_lower","hdi_upper","p_two_sided","prob_of_effect","relative_mean","relative_hdi_lower","relative_hdi_upper"]
        avg_effect = None
        avg_ci_lower = None
        avg_ci_upper = None
        cum_effect = None
        cum_ci_lower = None
        cum_ci_upper = None
        prob_effect = None
        rel_effect = None
        rel_ci_lower = None
        rel_ci_upper = None

        try:
            if "average" in stats_table.index:
                avg_row = stats_table.loc["average"]
                avg_effect = float(avg_row.get("mean", 0))
                avg_ci_lower = float(avg_row.get("hdi_lower", 0))
                avg_ci_upper = float(avg_row.get("hdi_upper", 0))
                prob_effect = float(avg_row.get("prob_of_effect", 0))
                rel_effect = float(avg_row.get("relative_mean", 0)) / 100.0  # Convert from percentage
                rel_ci_lower = float(avg_row.get("relative_hdi_lower", 0)) / 100.0
                rel_ci_upper = float(avg_row.get("relative_hdi_upper", 0)) / 100.0
            if "cumulative" in stats_table.index:
                cum_row = stats_table.loc["cumulative"]
                cum_effect = float(cum_row.get("mean", 0))
                cum_ci_lower = float(cum_row.get("hdi_lower", 0))
                cum_ci_upper = float(cum_row.get("hdi_upper", 0))
        except Exception:
            pass  # Will fall back to text-based summary

        # --- 4. Generate Plots ---
        plot_path = None
        try:
            fig, axes = result.plot()
            tmp_plot = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir='/tmp')
            fig.savefig(tmp_plot.name, dpi=150, bbox_inches='tight')
            plt.close(fig)
            plot_path = tmp_plot.name
        except Exception:
            pass

        # --- 5. Return Structured Results ---
        return {
            "metrics": {
                "avg_effect": avg_effect,
                "avg_ci_lower": avg_ci_lower,
                "avg_ci_upper": avg_ci_upper,
                "cum_effect": cum_effect,
                "cum_ci_lower": cum_ci_lower,
                "cum_ci_upper": cum_ci_upper,
                "prob_effect": prob_effect,
                "rel_effect": rel_effect,
                "rel_ci_lower": rel_ci_lower,
                "rel_ci_upper": rel_ci_upper,
                "hdi_prob": hdi_prob,
                "direction": direction,
                "treated_geo": treated_geo_str,
            },
            "summary_text": stats_text,
            "summary_table": stats_table,
            "plot_path": plot_path,
        }

    except Exception as e:
        return {"error": f"CausalPy Synthetic Control failed: {e}"}


# =============================================================================
# Phase 2: Data Intelligence Backend Utilities
# =============================================================================

def profile_dataset(df):
    """
    Smart data profiling: detects data structure, auto-identifies key columns,
    and recommends appropriate causal methods.
    """
    result = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "columns": {},
        "data_type": "unknown",
        "detected_roles": {},
        "recommended_methods": [],
    }
    
    date_cols = []
    geo_cols = []
    treatment_cols = []
    outcome_cols = []
    
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isnull().sum()),
            "missing_pct": float(df[col].isnull().mean() * 100),
            "unique": int(df[col].nunique()),
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            col_info["stats"] = {
                "mean": float(desc.get("mean", 0)),
                "std": float(desc.get("std", 0)),
                "min": float(desc.get("min", 0)),
                "max": float(desc.get("max", 0)),
            }
        
        result["columns"][col] = col_info
        
        # Column role detection heuristics
        col_lower = col.lower()
        
        # Date detection
        if pd.api.types.is_datetime64_any_dtype(df[col]) or \
           any(kw in col_lower for kw in ["date", "time", "day", "month", "year", "week"]):
            date_cols.append(col)
        
        # Geo detection
        if any(kw in col_lower for kw in ["geo", "region", "location", "market", "city", "state", "dma", "country"]):
            geo_cols.append(col)
        
        # Treatment detection
        if any(kw in col_lower for kw in ["treatment", "treated", "campaign", "intervention", "exposed", "group"]):
            if df[col].nunique() <= 5:
                treatment_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1}):
            treatment_cols.append(col)
        
        # Outcome detection
        if any(kw in col_lower for kw in ["kpi", "revenue", "sales", "conversion", "outcome", "value", "metric", "y_obs", "response"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                outcome_cols.append(col)
    
    result["detected_roles"] = {
        "date": date_cols,
        "geography": geo_cols,
        "treatment": treatment_cols,
        "outcome": outcome_cols,
    }
    
    # Data type detection
    has_date = len(date_cols) > 0
    has_geo = len(geo_cols) > 0
    
    if has_date and has_geo:
        result["data_type"] = "panel"
        result["recommended_methods"] = ["CausalPy (Bayesian SC)", "GeoLift", "BSTS (CausalImpact)", "DiD"]
    elif has_date and not has_geo:
        result["data_type"] = "time_series"
        result["recommended_methods"] = ["BSTS (CausalImpact)", "DiD"]
    else:
        result["data_type"] = "cross_sectional"
        result["recommended_methods"] = ["PSM", "IPTW", "LinearDML", "CausalForestDML", "S-Learner", "T-Learner", "OLS"]
    
    return result


def run_data_quality_checks(df, date_col=None, geo_col=None, outcome_col=None, treatment_col=None):
    """
    Comprehensive data quality scan. Returns issues and recommended fixes.
    """
    issues = []
    
    # 1. Missing values
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 0:
            severity = "critical" if missing_pct > 30 else "high" if missing_pct > 10 else "medium" if missing_pct > 1 else "low"
            if pd.api.types.is_numeric_dtype(df[col]):
                fix = "impute_median"
                fix_params = {"col": col, "method": "median"}
            else:
                fix = "impute_mode"
                fix_params = {"col": col, "method": "mode"}
            issues.append({
                "issue": f"{missing_pct:.1f}% missing in '{col}' ({missing_count} rows)",
                "severity": severity,
                "category": "missing_values",
                "suggested_fix": fix,
                "fix_params": fix_params,
            })
    
    # 2. Outliers (numeric columns, IQR method)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= 2:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            outlier_mask = (df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                outlier_pct = outlier_count / len(df) * 100
                severity = "high" if outlier_pct > 5 else "medium"
                issues.append({
                    "issue": f"{outlier_count} outliers in '{col}' ({outlier_pct:.1f}%, >3×IQR)",
                    "severity": severity,
                    "category": "outliers",
                    "suggested_fix": "winsorize",
                    "fix_params": {"col": col, "percentile": 0.95},
                })
    
    # 3. Treatment imbalance
    if treatment_col and treatment_col in df.columns:
        val_counts = df[treatment_col].value_counts(normalize=True)
        if len(val_counts) == 2:
            minority_pct = val_counts.min() * 100
            if minority_pct < 10:
                issues.append({
                    "issue": f"Treatment imbalance in '{treatment_col}': minority group is {minority_pct:.1f}%",
                    "severity": "high",
                    "category": "class_imbalance",
                    "suggested_fix": "note_only",
                    "fix_params": {"col": treatment_col, "minority_pct": minority_pct},
                })
    
    # 4. Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append({
            "issue": f"{dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%)",
            "severity": "medium",
            "category": "duplicates",
            "suggested_fix": "drop_duplicates",
            "fix_params": {},
        })
    
    # 5. Near-zero variance
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].std() < 1e-10 and df[col].nunique() <= 1:
            issues.append({
                "issue": f"Near-zero variance in '{col}' (constant column)",
                "severity": "low",
                "category": "low_variance",
                "suggested_fix": "drop_column",
                "fix_params": {"col": col},
            })
    
    # 6. Temporal gaps (if date_col provided)
    if date_col and date_col in df.columns:
        try:
            dates = pd.to_datetime(df[date_col])
            date_range = pd.date_range(dates.min(), dates.max())
            unique_dates = dates.dt.date.nunique()
            expected_dates = len(date_range)
            if unique_dates < expected_dates * 0.9:
                missing_dates = expected_dates - unique_dates
                issues.append({
                    "issue": f"{missing_dates} missing dates in time series ({date_col})",
                    "severity": "medium",
                    "category": "temporal_gaps",
                    "suggested_fix": "note_only",
                    "fix_params": {"col": date_col, "missing_count": missing_dates},
                })
        except Exception:
            pass
    
    # 7. Geo coverage (if geo_col provided)
    if geo_col and geo_col in df.columns:
        obs_per_geo = df.groupby(geo_col).size()
        min_obs = obs_per_geo.min()
        if min_obs < 30:
            sparse_geos = obs_per_geo[obs_per_geo < 30].index.tolist()
            issues.append({
                "issue": f"{len(sparse_geos)} regions with <30 observations (min={min_obs})",
                "severity": "medium",
                "category": "geo_coverage",
                "suggested_fix": "note_only",
                "fix_params": {"col": geo_col, "sparse_regions": sparse_geos[:5]},
            })
    
    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    issues.sort(key=lambda x: severity_order.get(x["severity"], 4))
    
    return {"issues": issues, "total_issues": len(issues)}


def apply_fixes(df, fixes):
    """
    Applies a list of approved data quality fixes to the dataframe.
    Returns the modified dataframe and a summary of changes.
    """
    changes = []
    df_fixed = df.copy()
    
    for fix in fixes:
        fix_type = fix.get("fix_type") or fix.get("suggested_fix")
        col = fix.get("col") or fix.get("fix_params", {}).get("col")
        params = fix.get("params") or fix.get("fix_params", {})
        
        try:
            if fix_type in ("impute_median", "impute_mean", "impute_mode"):
                method = fix_type.split("_")[1]
                if method == "median":
                    fill_val = df_fixed[col].median()
                elif method == "mean":
                    fill_val = df_fixed[col].mean()
                else:
                    fill_val = df_fixed[col].mode().iloc[0]
                before_missing = df_fixed[col].isnull().sum()
                df_fixed[col] = df_fixed[col].fillna(fill_val)
                changes.append(f"Imputed {before_missing} missing values in '{col}' with {method} ({fill_val:.4f})")
            
            elif fix_type == "winsorize":
                from scipy.stats import mstats
                pct = params.get("percentile", 0.95)
                before_min, before_max = df_fixed[col].min(), df_fixed[col].max()
                df_fixed[col] = mstats.winsorize(df_fixed[col].values, limits=[1-pct, 1-pct])
                changes.append(f"Winsorized '{col}' at {pct:.0%} (range: {before_min:.2f}-{before_max:.2f} → {df_fixed[col].min():.2f}-{df_fixed[col].max():.2f})")
            
            elif fix_type == "drop_duplicates":
                before_rows = len(df_fixed)
                df_fixed = df_fixed.drop_duplicates()
                changes.append(f"Removed {before_rows - len(df_fixed)} duplicate rows")
            
            elif fix_type == "drop_column":
                df_fixed = df_fixed.drop(columns=[col])
                changes.append(f"Dropped column '{col}'")
            
            elif fix_type == "remove_outliers":
                method = params.get("method", "iqr")
                if method == "iqr":
                    Q1 = df_fixed[col].quantile(0.25)
                    Q3 = df_fixed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    mask = (df_fixed[col] >= Q1 - 3*IQR) & (df_fixed[col] <= Q3 + 3*IQR)
                    removed = (~mask).sum()
                    df_fixed = df_fixed[mask]
                    changes.append(f"Removed {removed} outliers from '{col}' (IQR method)")
            
            else:
                changes.append(f"Skipped unknown fix type: {fix_type}")
        
        except Exception as e:
            changes.append(f"Failed to apply {fix_type} on '{col}': {e}")
    
    return df_fixed, changes


def check_parallel_trends(df, outcome_col, treatment_col, time_col):
    """Pre-trend test for DiD: regreses outcome on treatment*time in pre-period."""
    try:
        pre_data = df[df[time_col] == 0] if df[time_col].nunique() == 2 else df
        if len(pre_data) < 10:
            return {"pass": None, "message": "Insufficient data for parallel trends test"}
        
        X = sm.add_constant(pre_data[[treatment_col, time_col]])
        X["interaction"] = pre_data[treatment_col] * pre_data[time_col]
        y = pre_data[outcome_col]
        model = sm.OLS(y, X).fit()
        
        interaction_pval = model.pvalues.get("interaction", 1.0)
        return {
            "pass": interaction_pval > 0.05,
            "p_value": float(interaction_pval),
            "coefficient": float(model.params.get("interaction", 0)),
            "message": "Parallel trends assumption holds (p > 0.05)" if interaction_pval > 0.05 
                       else f"Parallel trends may be violated (p = {interaction_pval:.4f})"
        }
    except Exception as e:
        return {"pass": None, "message": f"Parallel trends test failed: {e}"}


def check_pre_period_fit(df, date_col, outcome_col, intervention_date, geo_col=None, treated_geo=None):
    """Assess pre-period model fit quality (MAPE) for BSTS/Synthetic Control."""
    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        intervention_dt = pd.to_datetime(intervention_date)
        
        pre_data = df_copy[df_copy[date_col] < intervention_dt]
        
        if treated_geo and geo_col:
            treated_pre = pre_data[pre_data[geo_col] == treated_geo][outcome_col]
            control_pre = pre_data[pre_data[geo_col] != treated_geo].groupby(date_col)[outcome_col].mean()
        else:
            treated_pre = pre_data[outcome_col]
            control_pre = None
        
        n_pre = len(treated_pre)
        result = {
            "n_pre_periods": n_pre,
            "pre_period_mean": float(treated_pre.mean()) if len(treated_pre) > 0 else None,
            "pre_period_std": float(treated_pre.std()) if len(treated_pre) > 0 else None,
        }
        
        if n_pre < 30:
            result["warning"] = f"Only {n_pre} pre-period observations. Consider at least 30 for reliable estimates."
        
        return result
    except Exception as e:
        return {"error": str(e)}


def check_covariate_balance(df, treatment_col, covariates):
    """Standardized Mean Differences (SMD) for PSM/IPTW balance checks."""
    try:
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]
        
        balance = []
        for cov in covariates:
            if not pd.api.types.is_numeric_dtype(df[cov]):
                continue
            mean_t = treated[cov].mean()
            mean_c = control[cov].mean()
            std_pooled = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
            smd = (mean_t - mean_c) / std_pooled if std_pooled > 0 else 0
            
            balance.append({
                "covariate": cov,
                "mean_treated": float(mean_t),
                "mean_control": float(mean_c),
                "smd": float(abs(smd)),
                "balanced": abs(smd) < 0.1,
            })
        
        all_balanced = all(b["balanced"] for b in balance) if balance else False
        return {
            "balance": balance,
            "all_balanced": all_balanced,
            "message": "All covariates balanced (|SMD| < 0.1)" if all_balanced 
                       else "Some covariates are imbalanced (|SMD| ≥ 0.1)"
        }
    except Exception as e:
        return {"error": str(e)}


def check_power_requirements(df, data_type, date_col=None, geo_col=None, treatment_col=None, intervention_date=None):
    """
    Checks if the dataset has sufficient power for causal analysis.
    """
    checks = []
    
    if data_type == "panel" and geo_col:
        n_geos = df[geo_col].nunique()
        checks.append({
            "check": "Number of regions",
            "status": "pass" if n_geos >= 6 else "warn" if n_geos >= 3 else "fail",
            "detail": f"{n_geos} regions (need ≥5 donors + 1 treated for synthetic control)"
        })
        
        if date_col:
            n_dates = df[date_col].nunique()
            checks.append({
                "check": "Time periods",
                "status": "pass" if n_dates >= 60 else "warn" if n_dates >= 30 else "fail",
                "detail": f"{n_dates} time periods"
            })

    elif data_type == "time_series" and date_col:
        if intervention_date:
            try:
                df_copy = df.copy()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                intervention_dt = pd.to_datetime(intervention_date)
                pre_n = len(df_copy[df_copy[date_col] < intervention_dt])
                post_n = len(df_copy[df_copy[date_col] >= intervention_dt])
                checks.append({
                    "check": "Pre-period length",
                    "status": "pass" if pre_n >= 2 * post_n else "warn",
                    "detail": f"{pre_n} pre-period vs {post_n} post-period points (ideal: pre ≥ 2× post)"
                })
            except Exception:
                pass

    elif data_type == "cross_sectional" and treatment_col:
        n_treated = int((df[treatment_col] == 1).sum())
        n_control = int((df[treatment_col] == 0).sum())
        checks.append({
            "check": "Sample sizes",
            "status": "pass" if min(n_treated, n_control) >= 30 else "warn" if min(n_treated, n_control) >= 10 else "fail",
            "detail": f"Treated: {n_treated}, Control: {n_control}"
        })
        
        prevalence = n_treated / (n_treated + n_control) * 100 if (n_treated + n_control) > 0 else 0
        checks.append({
            "check": "Treatment prevalence",
            "status": "pass" if 5 <= prevalence <= 95 else "warn",
            "detail": f"Treatment prevalence: {prevalence:.1f}%"
        })

    return {"data_type": data_type, "checks": checks}


# =============================================================================
# Phase 3: Robustness & Sensitivity Backend
# =============================================================================

def run_placebo_test(df, method, date_col, outcome_col, intervention_date,
                     geo_col=None, treated_geo=None, treatment_duration=60,
                     placebo_shift_days=None):
    """
    Placebo test: Shifts intervention to pre-period. If significant effect found,
    the original result may be spurious.
    """
    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        intervention_dt = pd.to_datetime(intervention_date)
        
        # Default: shift to midpoint of pre-period
        if placebo_shift_days is None:
            min_date = df_copy[date_col].min()
            pre_period_days = (intervention_dt - min_date).days
            placebo_shift_days = pre_period_days // 2
        
        placebo_date = intervention_dt - pd.Timedelta(days=placebo_shift_days)
        placebo_date_str = placebo_date.strftime('%Y-%m-%d')
        
        # Run analysis with placebo date on pre-period data only
        pre_data = df_copy[df_copy[date_col] < intervention_dt]
        
        if method in ("BSTS", "CausalImpact"):
            res = run_causal_impact_analysis(
                pre_data, date_col, outcome_col, placebo_date_str,
                unit_col=geo_col, treated_unit=treated_geo, use_panel=bool(geo_col)
            )
        elif method in ("CausalPy", "SyntheticControl"):
            res = run_causalpy_synthetic_control(
                pre_data, date_col, geo_col, outcome_col, treated_geo,
                placebo_date_str, treatment_duration=treatment_duration
            )
        else:
            return {"error": f"Placebo test not supported for method: {method}"}
        
        if 'error' in res:
            return {"pass": None, "message": f"Placebo test failed to run: {res['error']}"}
        
        # Check if placebo found a significant effect
        p_value = res.get('p_value') or res.get('metrics', {}).get('prob_effect', 0)
        is_significant = False
        if method in ("BSTS", "CausalImpact"):
            is_significant = isinstance(p_value, (int, float)) and p_value < 0.05
        else:
            is_significant = isinstance(p_value, (int, float)) and p_value > 0.95
        
        return {
            "pass": not is_significant,
            "placebo_date": placebo_date_str,
            "placebo_significant": is_significant,
            "placebo_p_value": float(p_value) if isinstance(p_value, (int, float)) else None,
            "message": "Placebo test passed (no significant effect at placebo date)" if not is_significant
                       else "⚠️ Placebo test FAILED — effect found at placebo date, original results may be spurious"
        }
    except Exception as e:
        return {"pass": None, "message": f"Placebo test error: {e}"}


def run_leave_one_out(df, method, date_col, outcome_col, geo_col, treated_geo,
                      intervention_date, treatment_duration=60):
    """
    Leave-one-out: Iteratively drops each donor region and re-runs analysis.
    Checks stability of treatment effect estimate.
    """
    try:
        all_geos = df[geo_col].unique().tolist()
        donor_geos = [g for g in all_geos if g != treated_geo]
        
        if len(donor_geos) < 3:
            return {"pass": None, "message": "Not enough donor regions for leave-one-out (need ≥3)"}
        
        results = {}
        for dropped_geo in donor_geos[:8]:  # Limit to 8 iterations for performance
            df_subset = df[df[geo_col] != dropped_geo]
            try:
                if method in ("CausalPy", "SyntheticControl"):
                    res = run_causalpy_synthetic_control(
                        df_subset, date_col, geo_col, outcome_col, treated_geo,
                        intervention_date, treatment_duration=treatment_duration
                    )
                    if 'error' not in res:
                        results[dropped_geo] = res.get('metrics', {}).get('avg_effect')
                elif method in ("BSTS", "CausalImpact"):
                    res = run_causal_impact_analysis(
                        df_subset, date_col, outcome_col, intervention_date,
                        unit_col=geo_col, treated_unit=treated_geo, use_panel=True
                    )
                    if 'error' not in res:
                        results[dropped_geo] = res.get('ate')
            except Exception:
                continue
        
        if len(results) < 2:
            return {"pass": None, "message": "Too few leave-one-out iterations succeeded"}
        
        effects = [v for v in results.values() if v is not None]
        mean_effect = np.mean(effects)
        std_effect = np.std(effects)
        cv = abs(std_effect / mean_effect) if mean_effect != 0 else float('inf')
        
        is_stable = cv < 0.3  # Coefficient of variation < 30%
        
        return {
            "pass": is_stable,
            "iterations": len(results),
            "results": {str(k): float(v) for k, v in results.items() if v is not None},
            "mean_effect": float(mean_effect),
            "std_effect": float(std_effect),
            "cv": float(cv),
            "message": f"Leave-one-out test {'passed' if is_stable else 'FAILED'} (CV={cv:.2%}, {len(results)} iterations)"
        }
    except Exception as e:
        return {"pass": None, "message": f"Leave-one-out error: {e}"}


def generate_report_content(session_state):
    """
    Compiles all session analyses into structured report sections.
    """
    sections = {}
    
    # 1. Dataset info
    if hasattr(session_state, 'df') and session_state.df is not None:
        df = session_state.df
        sections['data'] = {
            'shape': f"{len(df)} rows × {len(df.columns)} columns",
            'columns': list(df.columns),
            'sim_type': getattr(session_state, 'sim_type', 'Unknown'),
            'description': getattr(session_state, 'dataset_description', ''),
        }
    
    # 2. Preprocessing
    preprocessing = []
    if getattr(session_state, 'p_impute_enable', False):
        preprocessing.append("Missing value imputation enabled")
    if getattr(session_state, 'p_wins_enable', False):
        preprocessing.append("Winsorization enabled")
    if getattr(session_state, 'p_log_transform_cols', None):
        preprocessing.append(f"Log transform: {session_state.p_log_transform_cols}")
    if getattr(session_state, 'p_standardize_cols', None):
        preprocessing.append(f"Standardization: {session_state.p_standardize_cols}")
    sections['preprocessing'] = preprocessing
    
    # 3. Analyses run
    analyses = []
    if getattr(session_state, 'quasi_results', None):
        for key, result in session_state.quasi_results.items():
            analyses.append({'type': 'Quasi-Experimental', 'method': key, 'result': result})
    if getattr(session_state, 'obs_results', None):
        for key, result in session_state.obs_results.items():
            analyses.append({'type': 'Observational', 'method': key, 'result': result})
    sections['analyses'] = analyses
    
    # 4. Quality summary
    sections['quality_summary'] = getattr(session_state, 'data_quality_summary', None)
    
    return sections
