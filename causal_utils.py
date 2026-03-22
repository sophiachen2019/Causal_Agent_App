
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
    regions = ['North', 'South', 'East', 'West']
    total_days = 400
    start_date = pd.to_datetime('2023-01-01')
    date_range = pd.date_range(start=start_date, periods=total_days)
    data_list = []
    
    global_trend = np.linspace(100, 150, total_days)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(total_days) / 7)
    monthly_seasonality = 20 * np.sin(2 * np.pi * np.arange(total_days) / 30)
    
    for region in regions:
        regional_trend = global_trend + np.random.normal(0, 10)
        noise = np.random.normal(0, 5, total_days)
        metric = regional_trend + weekly_seasonality + monthly_seasonality + noise
        
        intervention_day = 300
        if region == 'North':
            lift = np.zeros(total_days)
            lift[intervention_day:] = 30 + np.cumsum(np.random.normal(0.5, 0.1, total_days - intervention_day))
            metric += lift
        
        region_df = pd.DataFrame({
            'Date': date_range, 'Region': region, 'Daily_Revenue': metric,
            'Marketing_Spend': np.random.normal(50, 5, total_days),
            'Is_Post_Intervention': (np.arange(total_days) >= intervention_day).astype(int),
            'Is_Treated_Region': 1 if region == 'North' else 0
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
            robjects.globalenv['plot_path'] = plot_file

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
            rel_effect <- sum_data$RelEffect[1] # Average Relative Effect
            
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
            
            # report_text in R might be a vector of strings depending on print method, we'll try to join it
            report_r = robjects.r('report_text')
            report_str = "\n".join(report_r) if hasattr(report_r, '__iter__') and not isinstance(report_r, str) else str(report_r)

        return {
            'object': None, # No python object anymore
            'plot_path': plot_file,
            'summary': "R CausalImpact model summarized below.",
            'report': report_str,
            'p_value': p_val,
            'ate': ate,
            'ate_lower': ate_lower,
            'ate_upper': ate_upper,
            'cumulative_abs': cum,
            'cumulative_lower': cum_lower,
            'cumulative_upper': cum_upper,
            'relative_effect': rel_effect
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def run_geolift_analysis(df, date_col, geo_col, treated_geo, kpi_col, intervention_date, 
                          treatment_duration=14, model="none", alpha=0.1, 
                          confidence_intervals=False, stat_test="Total", covariates=None):
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
            """)
            
            r_avg = robjects.globalenv['avg_lift']
            r_cum = robjects.globalenv['cumulative_lift']
            r_p = robjects.globalenv['p_val']
            
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
            robjects.globalenv['impact_plot_path'] = impact_plot_file
            robjects.globalenv['att_plot_path'] = att_plot_file
            
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
            """)
            
            full_summary = robjects.globalenv['full_summary_text'][0]
            
            report = f"""
            ### GeoLift Analysis Results
            **Treated Geography**: {treated_geo}
            **Model Type**: {model}
            
            **Average Estimated Treatment Effect**: {avg_lift:.2f}
            **Cumulative Lift**: {cum_lift:.2f}
            **P-Value**: {p_val:.4f}
            
            **Statistically Significant (p < {alpha})?**: {significant}
            
            #### Full Inference Summary
            ```text
            {full_summary}
            ```
            """
            
            # Cleanup
            robjects.r("rm(gl_res, summary_res, avg_lift, cumulative_lift, p_val, full_summary_text)")
            
            return {
                "summary": report,
                "plot_path": impact_plot_file,
                "att_plot_path": att_plot_file
            }
            
        except Exception as e:
            return {"error": f"GeoLift execution failed: {e}"}


def run_geolift_power(df, date_col, geo_col, kpi_col, treatment_duration=14, cutoff_date=None, 
                      n_markets="1", lookback_window=1, model="none", alpha=0.1, side_of_test="two_sided",
                      parallel=True, ns=1000, effect_size_mode="Full", normalize=False, covariates=None):
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
