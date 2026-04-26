import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from econml.dml import LinearDML, CausalForestDML
import matplotlib.pyplot as plt
import plotly.express as px
from dowhy import CausalModel
from scipy import stats
import statsmodels.api as sm

def get_index(columns, default_name, default_idx):
    if default_name in columns:
        return list(columns).index(default_name)
    return default_idx if default_idx < len(columns) else 0


# Import custom utils with explicit reload to ensure updates are picked up
import causal_utils
import chatbot_utils
import feedback_utils
import data_generation_utils
import importlib
importlib.reload(causal_utils)
importlib.reload(chatbot_utils)
importlib.reload(feedback_utils)
importlib.reload(data_generation_utils)
from causal_utils import generate_script

# --- 1. Data Simulation ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'quasi_analysis_run' not in st.session_state:
    st.session_state.quasi_analysis_run = False
if 'quasi_results' not in st.session_state:
    st.session_state.quasi_results = None
if 'quasi_method_run' not in st.session_state:
    st.session_state.quasi_method_run = None
if 'bucketing_ops' not in st.session_state:
    st.session_state.bucketing_ops = []
if 'filtering_ops' not in st.session_state:
    st.session_state.filtering_ops = []

# --- Preprocessing State Initialization ---
if 'impute_enable' not in st.session_state: st.session_state.impute_enable = False
if 'num_impute_method' not in st.session_state: st.session_state.num_impute_method = "Mean"
if 'num_custom_val' not in st.session_state: st.session_state.num_custom_val = 0.0
if 'cat_impute_method' not in st.session_state: st.session_state.cat_impute_method = "Mode"
if 'cat_custom_val' not in st.session_state: st.session_state.cat_custom_val = "Missing"
if 'winsorize_enable' not in st.session_state: st.session_state.winsorize_enable = False
if 'winsorize_cols' not in st.session_state: st.session_state.winsorize_cols = []
if 'percentile' not in st.session_state: st.session_state.percentile = 0.05
if 'log_transform_cols' not in st.session_state: st.session_state.log_transform_cols = []
if 'standardize_cols' not in st.session_state: st.session_state.standardize_cols = []
if 'override_ops' not in st.session_state: st.session_state.override_ops = []
if 'resampling_ops' not in st.session_state: st.session_state.resampling_ops = []



# Removing cache to fix stale data issues with Arrow
# @st.cache_data
def simulate_standard_data(n_samples=1000):
    np.random.seed(42)
    
    # Confounders
    # Customer Segment: 0 = SMB, 1 = Enterprise
    customer_segment = np.random.binomial(1, 0.3, n_samples)
    
    # Historical Usage: Continuous variable
    historical_usage = np.random.normal(50, 15, n_samples) + (customer_segment * 20)
    
    # Instrument: Marketing Nudge (Randomly assigned, affects adoption but not value directly)
    marketing_nudge = np.random.binomial(1, 0.5, n_samples)
    
    # Time Period: Quarter (0 = Pre, 1 = Post) - for DiD
    quarter = np.random.binomial(1, 0.5, n_samples)

    # Treatment: Feature Adoption (Binary)
    # Probability of adoption depends on segment, usage, AND marketing nudge
    prob_adoption = 1 / (1 + np.exp(-( -2 + 0.5 * customer_segment + 0.05 * historical_usage + 1.5 * marketing_nudge)))
    feature_adoption = np.random.binomial(1, prob_adoption, n_samples)
    
    # Outcome: Account Value
    # True causal effect of feature adoption is $500
    # Also depends on segment, usage, and time (trend)
    account_value = (
        200 
        + 500 * feature_adoption 
        + 1000 * customer_segment 
        + 10 * historical_usage 
        + 50 * quarter # Time trend
        + np.random.normal(0, 50, n_samples)
    )

    # Date Generation (Simulating 2 years of data)
    start_date = pd.to_datetime('2023-01-01')
    dates = start_date + pd.to_timedelta(np.random.randint(0, 730, n_samples).astype(int), unit='D')
    dates = pd.to_datetime(dates).floor('D') # Ensure no fractional seconds


    # Outcome: Conversion (Binary)
    # Base prob depends on segment. Treatment increases prob by ~10%
    prob_conversion = 1 / (1 + np.exp(-( -1 + 0.5 * customer_segment + 0.5 * feature_adoption)))
    conversion = np.random.binomial(1, prob_conversion, n_samples)
    
    df = pd.DataFrame({
        'Customer_Segment': customer_segment,
        'Historical_Usage': historical_usage,
        'Marketing_Nudge': marketing_nudge,
        'Quarter': quarter,
        'Feature_Adoption': feature_adoption,
        'Account_Value': account_value,
        'Conversion': conversion,
        'Date': dates
    })
    
    # Enforce Data Types
    df['Customer_Segment'] = df['Customer_Segment'].astype(int)
    df['Historical_Usage'] = df['Historical_Usage'].astype(float)
    df['Marketing_Nudge'] = df['Marketing_Nudge'].astype(int)
    df['Quarter'] = df['Quarter'].astype(int)
    df['Feature_Adoption'] = df['Feature_Adoption'].astype(int)
    df['Account_Value'] = df['Account_Value'].astype(float)
    df['Conversion'] = df['Conversion'].astype(int)
    df['Date'] = pd.to_datetime(dates).floor('D') # Strict midnight
    df['Date'] = df['Date'].dt.tz_localize(None) # Remove any timezone
    df['Date'] = df['Date'].astype('datetime64[ns]') # Force ns for Arrow
    
    return df


# @st.cache_data
def simulate_bsts_demo_data():
    """Generates multi-region time series data with a clear intervention for BSTS/Synthetic Control demo."""
    np.random.seed(42)
    regions = [f'Region_{i}' for i in range(1, 41)]
    total_days = 364
    start_date = pd.to_datetime('2023-01-01')
    date_range = pd.date_range(start=start_date, periods=total_days)
    
    data_list = []
    
    # Shared Global Trend and Seasonality
    global_trend = np.linspace(100, 150, total_days)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(total_days) / 7)
    monthly_seasonality = 20 * np.sin(2 * np.pi * np.arange(total_days) / 30)
    
    for region in regions:
        # Regional variation in trend - Widen the convex hull naturally so Region_1 sits comfortably inside
        regional_offset = np.random.normal(0, 20)
        regional_trend = global_trend + regional_offset
        
        # Noise - Reduced to simulate highly correlated/reliable donor markets
        noise = np.random.normal(0, 2, total_days)
        
        # Base Metric (e.g. Daily Revenue)
        metric = regional_trend + weekly_seasonality + monthly_seasonality + noise
        
        intervention_day = 304
        if region == 'Region_1':
            # Add a cumulative lift starting from intervention_day
            lift = np.zeros(total_days)
            lift[intervention_day:] = 30 + np.cumsum(np.random.normal(0.5, 0.1, total_days - intervention_day))
            metric += lift
        
        region_df = pd.DataFrame({
            'Date': pd.to_datetime(date_range).floor('D'), 'Region': region, 'Daily_Revenue': metric,
            'Marketing_Spend': np.random.normal(50, 5, total_days),
            'App_Downloads': np.random.poisson(500, total_days) + (regional_trend).astype(int),
            'Website_Traffic': np.random.normal(5000, 500, total_days) + regional_trend * 10,
            'Social_Media_Mentions': np.random.poisson(100, total_days) + (regional_trend / 2).astype(int),
            'Is_Post_Intervention': (np.arange(total_days) >= intervention_day).astype(int),
            'Is_Treated_Region': 1 if region == 'Region_1' else 0
        })
        data_list.append(region_df)
    
    df = pd.concat(data_list, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.floor('D')
    return df

def simulate_data(n_samples=1000, type="Standard"):
    if type == "BSTS Demo" or type == "GeoLift Demo (Geographic Intervention)":
        return simulate_bsts_demo_data()
    return simulate_standard_data(n_samples)

def convert_bool_to_int(df):
    # 1. Actual boolean types
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # 2. String "TRUE"/"FALSE" (case insensitive)
    # Check object columns
    obj_cols = df.select_dtypes(include=['object']).columns
    
    # Mapping dictionary
    mapping = {
        'TRUE': 1, 'FALSE': 0,
        'T': 1, 'F': 0,
        '1': 1, '0': 0,
        '1.0': 1, '0.0': 0
    }
    
    for col in obj_cols:
        # Optimization: Check first value or sample to decide if we should attempt conversion
        # to avoid expensive unique() on large columns.
        # Or just try to map and check if we introduced NaNs where there weren't any.
        
        # Fast check: are there any values that look like booleans?
        # Let's stick to the safe unique check but optimize it slightly?
        # Actually, for huge data, unique() is slow.
        # Let's try to convert and see if it works.
        
        try:
            # Attempt to map everything to upper case
            series_upper = df[col].astype(str).str.upper()
            
            # Check if all unique values are in our mapping keys (plus NaNs)
            # We can check a sample first for speed
            sample = series_upper.dropna().head(100)
            if not set(sample.unique()).issubset(mapping.keys()):
                continue # Skip if sample has non-boolean values
                
            # If sample passed, check full unique (still safer than blind conversion)
            # Or trust the sample? Let's check full unique but only if sample passed.
            unique_vals = set(series_upper.dropna().unique())
            if unique_vals.issubset(mapping.keys()):
                df[col] = series_upper.map(mapping).fillna(df[col])
                df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass
            
    return df

# --- Streamlit UI ---
# --- Streamlit UI ---
st.set_page_config(page_title="Causal Inference Application", layout="wide")

MAX_API_CALLS = 50
if 'api_call_count' not in st.session_state:
    st.session_state.api_call_count = 0

def check_api_limit():
    if "user_api_key" in st.session_state and st.session_state.user_api_key:
        return True # Bypass limit
    if st.session_state.api_call_count >= MAX_API_CALLS:
        st.error(f"⚠️ You have reached the limit of {MAX_API_CALLS} AI requests for this session. To continue, enter your API Key in the AI Assistant tab.")
        return False
    st.session_state.api_call_count += 1
    return True


# --- UI Modernization Injection ---
st.markdown("""
<style>
    /* Modernize standard Streamlit elements */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border-radius: 8px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.02) !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);
        background-color: #ffffff;
    }
    /* Tabs styling */
    button[data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    /* DataFrame subtle shadows */
    [data-testid="stDataFrame"] {
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

col_title, col_right = st.columns([7, 1])

with col_title:
    st.markdown("<h1 style='margin-top: -20px; font-size: 2.5rem;'>Causal Inference Application</h1>", unsafe_allow_html=True)
    st.markdown("**Builder:** [Sophia Chen](https://www.shunqinchen.com?utm_source=streamlit&utm_medium=application&utm_campaign=causal_inference) ")

with col_right:
    import base64
    with open("logo.png", "rb") as logo_file:
        logo_base64 = base64.b64encode(logo_file.read()).decode()
    st.markdown(f'<div style="text-align: right;"><img src="data:image/png;base64,{logo_base64}" style="width: 70px; height: auto; margin-top: -10px; margin-bottom: 5px;"></div>', unsafe_allow_html=True)

    # Float popover down to align horizontally with the Streamlit tabs row
    st.markdown("""
        <style>
        div[data-testid="column"]:last-child div[data-testid="stPopover"] {
            position: absolute;
            right: 0px;
            transform: translateY(4.0rem);
            z-index: 999;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Feedback Popover
    with st.popover("📝 Feedback"):
        st.markdown("### Send us your thoughts!")
        with st.form("feedback_form"):
            user_email = st.text_input("Your Email (Optional)")
            feedback_text = st.text_area("Comments/Bugs")
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                if feedback_text:
                    success, msg = feedback_utils.send_email(feedback_text, user_email)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                        # Fallback mailto
                        subject = "Causal Agent App Feedback"
                        body = f"Feedback: {feedback_text}"
                        mailto_link = f"mailto:sophiachen2012@gmail.com?subject={subject}&body={body}"
                        st.markdown(f"[Click here to send manually]({mailto_link})")
                else:
                    st.warning("Please enter some feedback.")

def get_app_metadata():
    """Parses metadata from requirements.txt"""
    history = []
    try:
        with open("requirements.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# Release:"):
                    # Format: # Release: <Version> | <Date> | <Note>
                    parts = line.split("|")
                    if len(parts) >= 3:
                        version = parts[0].split(":", 1)[1].strip()
                        date = parts[1].strip()
                        note = parts[2].strip()
                        history.append({
                            "Version": version,
                            "Release Date": date,
                            "Release Note": note
                        })
    except FileNotFoundError:
        pass
    return history

history = get_app_metadata()
latest_version = history[0] if history else {"Version": "Unknown", "Release Date": "Unknown", "Release Note": "Unknown"}

# Load Data
# --- Tabs Setup ---
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
tab_guide, tab_eda, tab_config, tab_chat = st.tabs(["📘 User Guide", "📊 Exploratory Analysis", "🔬 Causal Analysis", "💬 AI Assistant"])

# ==========================================
# TAB 2: Exploratory Analysis
# ==========================================
with tab_eda:
    st.header("Exploratory Data Analysis")
    
    # --- Data Source ---
    st.subheader("1. Data Source")
    data_source = st.radio("Data Source", ["Simulated Data", "Upload CSV"], horizontal=True)
    
    if data_source == "Simulated Data":
        st.markdown("---")
        sim_options = ["🧠 Dynamic AI Engine", "📊 Standard (Cross-sectional/DiD)", "📈 BSTS / GeoLift Demo (Multi-region Time Series)"]
        # Use session state to persist selection
        if 'sim_type' not in st.session_state:
            st.session_state.sim_type = "Dynamic"
            
        if st.session_state.sim_type == "BSTS Demo":
            current_idx = 2
        elif st.session_state.sim_type == "Standard":
            current_idx = 1
        else:
            current_idx = 0
            
        simulate_type = st.radio("Simulation Type", sim_options, index=current_idx)
        
        # Check if we need to reset/re-simulate
        if "Dynamic" in simulate_type:
            target_type = "Dynamic"
        elif "Standard" in simulate_type:
            target_type = "Standard"
        else:
            target_type = "BSTS Demo"

        if st.session_state.sim_type != target_type:
            if target_type == "Dynamic":
                # Placeholder with labeled columns so all tab UIs render properly
                st.session_state.df = pd.DataFrame({
                    "User_ID": pd.Series(dtype='str'),
                    "Treatment": pd.Series(dtype='int'),
                    "Outcome": pd.Series(dtype='float'),
                    "Confounder_1": pd.Series(dtype='float'),
                    "Confounder_2": pd.Series(dtype='int'),
                    "Date": pd.Series(dtype='datetime64[ns]')
                })
                st.session_state.dataset_description = None
            else:
                st.session_state.df = simulate_data(type=target_type)
                
            st.session_state.sim_type = target_type
            if target_type != "Dynamic":
                st.session_state.raw_df = st.session_state.df.copy()
            st.session_state.bucketing_ops = []
            st.session_state.filtering_ops = []
            st.session_state.dataset_description = None
            st.session_state.dataset_questions = []
            st.session_state.data_quality_summary = None
            st.session_state.chart_suggestions = None
            st.rerun()
            
        if target_type == "Dynamic":
            st.markdown("##### 🧠 Configure Synthetic Data Engine")
            
            common_themes = ["eCommerce", "FinTech", "Healthcare", "Marketplace", "SaaS", "Social Media", "Gaming", "Other..."]
            common_functions = ["Marketing/Acquisition", "Pricing/Monetization", "Product/Feature Launch", "Retention/Engagement", "Operations/Support", "Other..."]
            
            # Row 1: Theme & Business Function
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                selected_theme = st.selectbox("Company Theme", common_themes)
                if selected_theme == "Other...":
                    theme = st.text_input("Specify Company Theme", placeholder="e.g. Real Estate")
                else:
                    theme = selected_theme
                    
            with col_d2:
                selected_func = st.selectbox("Business Function", common_functions)
                if selected_func == "Other...":
                    func = st.text_input("Specify Business Function", placeholder="e.g. Recommendation System")
                else:
                    func = selected_func

            # Row 2: Study Design & Treatment Assignment & Outcome Type
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                study_design = st.selectbox(
                    "Study Design",
                    ["Cross-sectional (Observational)", "Longitudinal (Time Series / Panel)"],
                    help="Cross-sectional: one row per unit. Longitudinal: multiple time points per unit/region."
                )
            with col_s2:
                if "Cross-sectional" in study_design:
                    treatment_options = ["Self-selected / Organic", "Randomized (A/B Test)"]
                else:
                    treatment_options = ["Time-based Rollout (Before/After)", "Geographic Rollout (Geo-test)"]
                treatment_assignment = st.selectbox(
                    "Treatment Assignment",
                    treatment_options,
                    help="How was the treatment/intervention assigned?"
                )
            with col_s3:
                outcome_type = st.selectbox(
                    "Outcome Type",
                    ["Continuous (e.g. Revenue, Score)", "Binary (e.g. Converted, Churned)"],
                    help="Determines the outcome variable distribution."
                )

            # Row 3: Effect Size & Sample Size & Data Quality
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                effect_size = st.select_slider(
                    "True Treatment Effect Size",
                    options=["Small", "Medium", "Large"],
                    value="Medium",
                    help="Embedded ground truth — use to verify if your analysis recovers the known effect."
                )
            with col_e2:
                n_samples = st.select_slider(
                    "Sample Size",
                    options=[500, 1000, 2000, 5000, 10000],
                    value=1000
                )
            with col_e3:
                data_quality = st.select_slider(
                    "Data Quality",
                    options=["Low", "Medium", "High"],
                    value="High",
                    help="Low/Medium quality introduces missing values, outliers, and noise for you to clean up."
                )

            if st.button("Synthesize Dataset", type="primary"):
                if not check_api_limit():
                    st.stop()
                if theme and func:
                    with st.spinner("AI is constructing Data Generating Processes..."):
                        try:
                            df_result, desc, questions = data_generation_utils.generate_dynamic_dataset(
                                theme, func, chatbot_utils.get_api_key(),
                                n_samples=n_samples,
                                study_design=study_design,
                                treatment_assignment=treatment_assignment,
                                outcome_type=outcome_type,
                                effect_size=effect_size,
                                data_quality=data_quality
                            )
                            st.session_state.df = df_result
                            st.session_state.raw_df = df_result.copy()
                            st.session_state.dataset_description = desc
                            st.session_state.dataset_questions = questions
                            st.session_state.data_quality_summary = None  # Reset on new data
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to generate dataset: {str(e)}")
                else:
                    st.warning("Please provide both a Theme and a Business Function.")
                    
        st.markdown("---")
    
    # Initialize Session State
    if 'df' not in st.session_state:
        st.session_state.df = simulate_data(type=st.session_state.get('sim_type', 'Standard'))
    
    if 'bucketing_ops' not in st.session_state:
        st.session_state.bucketing_ops = []
    
    if 'filtering_ops' not in st.session_state:
        st.session_state.filtering_ops = []
    


    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            # Check if it's a new file
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if 'uploaded_file_id' not in st.session_state or st.session_state.uploaded_file_id != file_id:
                # New file loaded
                # New file loaded
                df = pd.read_csv(uploaded_file)
                df = convert_bool_to_int(df) # Auto-convert booleans
                st.session_state.raw_df = df.copy() # Store raw data
                st.session_state.df = df.copy() # Initialize working df
                st.session_state.uploaded_file_id = file_id
                # Reset ops
                st.session_state.bucketing_ops = []
                st.session_state.filtering_ops = []
                st.session_state.filtering_ops = []
                st.rerun()
            else:
                # Same file, keep existing session state df (which might have edits)
                df = st.session_state.df
        else:
            st.info("Awaiting CSV upload. Using simulated data for preview.")
            # Fallback to simulated if no file, but don't overwrite if we had one before?
            # Actually, if no file is uploaded, we should probably show simulated or empty.
            if 'df' not in st.session_state:
                 st.session_state.df = simulate_data(type=st.session_state.get('sim_type', 'Standard'))
            df = st.session_state.df
    else:
        # Simulated Data
        # If switching to simulated, we might want to reset if we were previously on Upload
        # For now, simplistic approach: if we don't have a df, simulate it.
        if 'df' not in st.session_state:
             st.session_state.df = simulate_data(type=st.session_state.get('sim_type', 'Standard'))
             st.session_state.raw_df = st.session_state.df.copy()
        
        # If the user explicitly wants to reset simulated data, we can add a button
        col_reset_btn, col_download_btn = st.columns([1, 4])
        with col_reset_btn:
            if st.button("Reset Simulated Data"):
                st.session_state.df = simulate_data(type=st.session_state.get('sim_type', 'Standard'))
                st.session_state.raw_df = st.session_state.df.copy()
                st.session_state.bucketing_ops = []
                st.session_state.filtering_ops = []
                st.rerun()
        
        with col_download_btn:
            if 'df' in st.session_state and st.session_state.df is not None:
                csv = st.session_state.df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Dataset (CSV)",
                    data=csv,
                    file_name=f"causal_data_{st.session_state.get('sim_type', 'Simulated').replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                )
            
        df = st.session_state.df

    # --- Data Preview ---
    st.subheader("2. Data Preview")
    if len(df) > 0:
        # Show existing AI summary or offer to generate one
        if st.session_state.get('dataset_description'):
            desc_text = f"🧠 **AI Summary:** {st.session_state.dataset_description}"
            questions = st.session_state.get('dataset_questions', [])
            if questions:
                desc_text += "\n\n**💬 Questions this dataset can help answer:**"
                for i, q in enumerate(questions, 1):
                    desc_text += f"\n{i}. {q}"
            st.info(desc_text)
        else:
            if st.button("🧠 Generate AI Dataset Summary", key="ai_preview_btn"):
                if not check_api_limit():
                    st.stop()
                with st.spinner("AI is analyzing your dataset..."):
                    preview_summary = data_generation_utils.generate_dataset_preview_summary(df, chatbot_utils.get_api_key())
                    st.session_state.dataset_description = preview_summary.get("description", "")
                    st.session_state.dataset_questions = preview_summary.get("questions", [])
                    st.rerun()
    df_preview = df.head(10).copy()
    for col in df_preview.columns:
        if pd.api.types.is_datetime64_any_dtype(df_preview[col]):
            df_preview[col] = df_preview[col].dt.strftime('%Y-%m-%d')
    st.dataframe(df_preview)

    # --- Data Summary ---
    st.subheader("3. Data Summary")
    if len(df) == 0:
        st.info("No data loaded yet. Generate or upload a dataset to see summary statistics.")
    else:
      # AI Data Quality Summary (on-demand)
      if st.button("🧠 Generate AI Data Quality Summary", key="ai_quality_btn"):
          if not check_api_limit():
              st.stop()
          with st.spinner("AI is analyzing data quality..."):
              summary = data_generation_utils.generate_data_quality_summary(df, chatbot_utils.get_api_key())
              st.session_state.data_quality_summary = summary
      if st.session_state.get('data_quality_summary'):
          st.info(st.session_state.data_quality_summary)

      with st.expander("Show Summary Statistics", expanded=False):
        df_summary = df.copy()
        for col in df_summary.columns:
            unique_vals = df_summary[col].dropna().unique()
            if len(unique_vals) <= 2:
                is_binary = all(val in [0, 1, True, False] for val in unique_vals)
                if is_binary:
                     df_summary[col] = df_summary[col].astype(str)

        st.markdown("**Numeric Statistics**")
        st.dataframe(df_summary.describe().round(2).astype(str))
        
        st.markdown("**Categorical Statistics**")
        cat_cols = df_summary.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                st.markdown(f"**{col}**")
                counts = df_summary[col].value_counts()
                percent = df_summary[col].value_counts(normalize=True) * 100
                summary_df = pd.DataFrame({'Count': counts, 'Percentage': percent})
                st.dataframe(summary_df.style.format({'Percentage': '{:.2f}%'}))
        else:
            st.info("No categorical variables found.")
        
        st.markdown("**Missing Values**")
        missing_info = pd.DataFrame({
            'Missing Count': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(missing_info.style.format({'Missing Percentage': '{:.2f}%'}))

      with st.expander("Show Correlation Matrix", expanded=False):
        st.markdown("**Correlation Matrix**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            corr_cols = st.multiselect("Select Variables", numeric_cols, default=numeric_cols)
            
            if corr_cols:
                corr_matrix = df[corr_cols].corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
            else:
                st.info("Please select at least one variable.")
        else:
            st.info("Not enough numeric variables to compute correlation.")

    # --- Data Preprocessing ---
    st.subheader("4. Data Preprocessing")
    if df.empty or len(df.columns) == 0:
        st.info("No data loaded yet. Generate or upload a dataset first.")
    else:
      with st.expander("Preprocessing Options", expanded=False):
        st.markdown("### Transformations")
        
        # 1. Missing Value Imputation
        st.markdown("#### Missing Value Imputation")
        impute_enable = st.checkbox("Enable Imputation", value=st.session_state.impute_enable, key="impute_enable")
        
        if impute_enable:
            col1, col2 = st.columns(2)
            with col1:
                num_impute_method = st.selectbox(
                    "Numeric Imputation Method",
                    ["Mean", "Median", "Zero", "Custom Value"],
                    index=["Mean", "Median", "Zero", "Custom Value"].index(st.session_state.num_impute_method),
                    key="num_impute_method"
                )
                if num_impute_method == "Custom Value":
                    num_custom_val = st.number_input("Custom Value (Numeric)", value=st.session_state.num_custom_val, key="num_custom_val")
            
            with col2:
                cat_impute_method = st.selectbox(
                    "Categorical Imputation Method",
                    ["Mode", "Missing Indicator", "Custom Value"],
                    index=["Mode", "Missing Indicator", "Custom Value"].index(st.session_state.cat_impute_method),
                    key="cat_impute_method"
                )
                if cat_impute_method == "Custom Value":
                    cat_custom_val = st.text_input("Custom Value (Categorical)", value=st.session_state.cat_custom_val, key="cat_custom_val")

            # Apply Imputation
            # Numeric
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                if num_impute_method == "Mean":
                    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                elif num_impute_method == "Median":
                    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                elif num_impute_method == "Zero":
                    df[num_cols] = df[num_cols].fillna(0)
                elif num_impute_method == "Custom Value":
                    df[num_cols] = df[num_cols].fillna(num_custom_val)
            
            # Categorical
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            if len(cat_cols) > 0:
                if cat_impute_method == "Mode":
                    for col in cat_cols:
                        if not df[col].mode().empty:
                            df[col] = df[col].fillna(df[col].mode()[0])
                elif cat_impute_method == "Missing Indicator":
                    df[cat_cols] = df[cat_cols].fillna("Missing")
                elif cat_impute_method == "Custom Value":
                    df[cat_cols] = df[cat_cols].fillna(cat_custom_val)
            
            st.info("Missing values imputed.")

        # 2. Winsorization
        st.markdown("#### Winsorization (Outlier Handling)")
        winsorize_enable = st.checkbox("Enable Winsorization", value=st.session_state.winsorize_enable, key="winsorize_enable")
        
        if winsorize_enable:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            winsorize_cols = st.multiselect(
                "Select columns to winsorize", 
                numeric_cols, 
                default=st.session_state.winsorize_cols,
                key="winsorize_cols"
            )
            
            if winsorize_cols:
                percentile = st.slider(
                    "Percentile Threshold", 
                    min_value=0.01, max_value=0.25, 
                    value=st.session_state.percentile, 
                    step=0.01, 
                    help="Clips values at the p-th and (1-p)-th percentiles.",
                    key="percentile"
                )
                
                for col in winsorize_cols:
                    lower = df[col].quantile(percentile)
                    upper = df[col].quantile(1 - percentile)
                    df[col] = df[col].clip(lower=lower, upper=upper)
                
                st.info(f"Winsorization applied to {', '.join(winsorize_cols)} at {percentile*100:.0f}% threshold.")

        # 3. Log Transformation
        st.markdown("#### Log Transformation")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        log_transform_cols = st.multiselect(
            "Apply Log Transformation (np.log1p)", 
            numeric_cols, 
            default=st.session_state.log_transform_cols,
            help="Applies log(x+1) to selected columns.",
            key="log_transform_cols"
        )
        
        if log_transform_cols:
            for col in log_transform_cols:
                # Ensure no negative values for log
                if (df[col] < 0).any():
                    st.warning(f"Column '{col}' contains negative values. Log transformation skipped for this column.")
                else:
                    df[col] = np.log1p(df[col])
            st.info(f"Log transformation applied to: {', '.join(log_transform_cols)}")

        # 4. Standardization
        st.markdown("#### Standardization")
        standardize_cols = st.multiselect(
            "Standardize Variables (StandardScaler)", 
            numeric_cols, 
            default=st.session_state.standardize_cols,
            help="Scales variables to have mean=0 and std=1.",
            key="standardize_cols"
        )
        
        if standardize_cols:
            scaler = StandardScaler()
            df[standardize_cols] = scaler.fit_transform(df[standardize_cols])
            st.info(f"Standardization applied to: {', '.join(standardize_cols)}")

        # 5. Variable Bucketing (Binning)
        st.markdown("#### Variable Bucketing (Binning)")
        with st.expander("Create Bucketed Variables", expanded=False):
            bucket_col = st.selectbox("Select Column to Bucket", numeric_cols, key="bucket_col")
            n_bins = st.number_input("Number of Bins", min_value=2, max_value=20, value=4, step=1, key="n_bins")
            bin_method = st.radio("Binning Method", ["Equal Width (cut)", "Equal Frequency (qcut)"], key="bin_method")
            new_col_name = st.text_input("New Column Name", value=f"{bucket_col}_binned" if bucket_col else "binned_col", key="new_col_name")
            
            if st.button("Create Buckets"):
                if bucket_col and new_col_name:
                    try:
                        if bin_method == "Equal Width (cut)":
                            st.session_state.df[new_col_name] = pd.cut(st.session_state.df[bucket_col], bins=n_bins, precision=1).astype(str)
                            method_code = "cut"
                        else:
                            st.session_state.df[new_col_name] = pd.qcut(st.session_state.df[bucket_col], q=n_bins, duplicates='drop', precision=1).astype(str)
                            method_code = "qcut"
                        
                        # Store operation for export script
                        op = {
                            "col": bucket_col,
                            "n_bins": n_bins,
                            "method": method_code,
                            "new_col": new_col_name
                        }
                        st.session_state.bucketing_ops.append(op)
                        
                        st.success(f"Created '{new_col_name}' with {n_bins} bins.")
                        st.write(st.session_state.df[new_col_name].value_counts().sort_index())
                        st.rerun()
                    except Exception as e:
                        st.error(f"Bucketing failed: {e}")
                else:
                    st.error("Please select a column and provide a name.")

        # 6. Conditional Override
        st.markdown("#### Conditional Override")
        with st.expander("Update Values Based on Condition", expanded=False):
            all_cols = df.columns.tolist()
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                target_col_over = st.selectbox("Target Column to Update", all_cols, key="override_target_sel")
                cond_col_over = st.selectbox("Condition Column", all_cols, key="override_cond_col_sel")
            with col_o2:
                operator_over = st.selectbox("Operator", [">", "<", "==", "!=", ">=", "<="], key="override_op_sel")
                # Type detection for value input
                if pd.api.types.is_numeric_dtype(df[cond_col_over]):
                    val_over = st.number_input("Threshold Value", value=0.0, key="override_val_input")
                else:
                    val_over = st.text_input("Match Value", key="override_val_input")
            
            if pd.api.types.is_numeric_dtype(df[target_col_over]):
                new_val_over = st.number_input("Replacement Value", value=0.0, key="override_new_val_input")
            else:
                new_val_over = st.text_input("Replacement Value", key="override_new_val_input")
                
            if st.button("Apply Override", key="override_btn_exec"):
                try:
                    import operator as op_lib
                    ops = {">": op_lib.gt, "<": op_lib.lt, "==": op_lib.eq, "!=": op_lib.ne, ">=": op_lib.ge, "<=": op_lib.le}
                    mask = ops[operator_over](st.session_state.df[cond_col_over], val_over)
                    count = mask.sum()
                    st.session_state.df.loc[mask, target_col_over] = new_val_over
                    st.success(f"Updated {count} rows in '{target_col_over}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Override failed: {e}")

        # 7. Imbalance Handling (Resampling)
        st.markdown("#### Imbalance Handling (Resampling)")
        with st.expander("Over/Down Sampling for Binary Targets", expanded=False):
            binary_cols = [c for c in df.columns if df[c].nunique() == 2]
            if binary_cols:
                resample_target = st.selectbox("Select Binary Column", binary_cols, key="resample_target_sel")
                resample_type = st.radio("Resampling Method", ["Random Over-sampling (Minority)", "Random Down-sampling (Majority)"], key="resample_type_sel")
                
                if st.button("Apply Resampling", key="resample_btn_exec"):
                    try:
                        counts = st.session_state.df[resample_target].value_counts()
                        minority_class = counts.idxmin()
                        majority_class = counts.idxmax()
                        
                        df_minority = st.session_state.df[st.session_state.df[resample_target] == minority_class]
                        df_majority = st.session_state.df[st.session_state.df[resample_target] == majority_class]
                        
                        if "Over-sampling" in resample_type:
                            df_minority_upsampled = df_minority.sample(len(df_majority), replace=True, random_state=42)
                            st.session_state.df = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
                            st.success(f"Balanced classes! New sample size: {len(st.session_state.df)}")
                        else:
                            df_majority_downsampled = df_majority.sample(len(df_minority), replace=False, random_state=42)
                            st.session_state.df = pd.concat([df_minority, df_majority_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
                            st.success(f"Balanced classes! New sample size: {len(st.session_state.df)}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Resampling failed: {e}")
            else:
                st.info("No binary columns found for resampling.")

        # 8. Data Filtering
        st.markdown("#### Data Filtering")
        with st.expander("Filter Data", expanded=False):
            filter_col = st.selectbox("Select Column to Filter", df.columns, key="filter_col")
            filter_op = st.selectbox("Condition", ["==", "!=", ">", "<", ">=", "<=", "contains"], key="filter_op")
            
            # Dynamic input based on column type
            if pd.api.types.is_numeric_dtype(df[filter_col]):
                filter_val = st.number_input("Value", value=0.0, key="filter_val_num")
            else:
                filter_val = st.text_input("Value", key="filter_val_text")
                
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if st.button("Apply Filter"):
                    try:
                        initial_len = len(st.session_state.df)
                        
                        if filter_op == "==":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] == filter_val]
                        elif filter_op == "!=":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] != filter_val]
                        elif filter_op == ">":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] > filter_val]
                        elif filter_op == "<":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] < filter_val]
                        elif filter_op == ">=":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] >= filter_val]
                        elif filter_op == "<=":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] <= filter_val]
                        elif filter_op == "contains":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col].astype(str).str.contains(str(filter_val), na=False)]
                        
                        final_len = len(st.session_state.df)
                        
                        # Store operation
                        op = {
                            "col": filter_col,
                            "op": filter_op,
                            "val": filter_val
                        }
                        st.session_state.filtering_ops.append(op)
                        
                        st.success(f"Filter applied. Rows reduced from {initial_len} to {final_len}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Filtering failed: {e}")
            
            with col_f2:
                if st.button("Reset Data"):
                    # Reload from raw_df
                    if 'raw_df' in st.session_state:
                         st.session_state.df = st.session_state.raw_df.copy()
                    else:
                         st.session_state.df = simulate_data(type=st.session_state.get('sim_type', 'Standard'))
                         st.session_state.raw_df = st.session_state.df.copy()
                    
                    st.session_state.bucketing_ops = []
                    st.session_state.filtering_ops = []
                    st.rerun()

        # 7. Duplicate Removal
        st.markdown("#### Duplicate Removal")
        if len(df) > 0:
            dup_count = df.duplicated().sum()
            st.write(f"Duplicate rows found: **{dup_count}** / {len(df)}")
            if dup_count > 0:
                dup_subset = st.multiselect(
                    "Check duplicates based on columns (empty = all columns)",
                    df.columns.tolist(),
                    key="dup_subset_cols"
                )
                if st.button("Remove Duplicates", key="remove_dups"):
                    before = len(st.session_state.df)
                    if dup_subset:
                        st.session_state.df = st.session_state.df.drop_duplicates(subset=dup_subset)
                    else:
                        st.session_state.df = st.session_state.df.drop_duplicates()
                    after = len(st.session_state.df)
                    st.success(f"Removed {before - after} duplicate rows.")
                    st.rerun()

        # 8. One-Hot Encoding
        st.markdown("#### One-Hot Encoding")
        cat_cols_for_encoding = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Exclude ID-like and date columns
        cat_cols_for_encoding = [c for c in cat_cols_for_encoding if 'id' not in c.lower() and 'date' not in c.lower()]
        if cat_cols_for_encoding:
            encode_cols = st.multiselect(
                "Select categorical columns to one-hot encode",
                cat_cols_for_encoding,
                help="Creates binary dummy columns for each category. Original column is dropped.",
                key="onehot_cols"
            )
            if encode_cols and st.button("Apply One-Hot Encoding", key="apply_onehot"):
                try:
                    st.session_state.df = pd.get_dummies(st.session_state.df, columns=encode_cols, drop_first=True, dtype=int)
                    st.success(f"One-hot encoded: {', '.join(encode_cols)}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Encoding failed: {e}")
        else:
            st.info("No categorical columns available for encoding.")

    # --- Chart Builder ---
    st.subheader("5. Visualization (Chart Builder)")
    _skip_viz = df.empty or len(df.columns) == 0
    if _skip_viz:
        st.info("No data loaded yet. Generate or upload a dataset to build charts.")
        df_plot_source = pd.DataFrame({"(no data)": []})
    else:
        df_plot_source = df
        # AI Chart Suggestions (on-demand)
        if st.button("🧠 Suggest Charts for This Dataset", key="ai_chart_btn"):
            if not check_api_limit():
                st.stop()
            with st.spinner("AI is generating chart suggestions..."):
                chart_suggestion = data_generation_utils.generate_chart_suggestions(df, chatbot_utils.get_api_key())
                st.session_state.chart_suggestions = chart_suggestion
        if st.session_state.get('chart_suggestions'):
            st.info(st.session_state.chart_suggestions)

    st.markdown("#### Chart Configuration")
    chart_type = st.selectbox("Chart Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Pie Chart"])
    col_x, col_y, col_color = st.columns(3)
    
    with col_x:
        x_var = st.selectbox("X Variable", df_plot_source.columns)
    
    # Initialize variables
    y_vars = None
    right_y_vars = []
    use_dual_axis = False

    with col_y:
        # Y variable selection (support multiple)
        if chart_type in ["Histogram", "Pie Chart"]:
            y_vars = None
            st.markdown("*N/A for this chart type*")
        else:
            # Multi-select for Y variables
            # Default to first non-X numeric column if possible
            numeric_cols = df_plot_source.select_dtypes(include=[np.number]).columns.tolist()
            default_y = [c for c in numeric_cols if c != x_var][:1]
            if not default_y and len(df_plot_source.columns) > 1:
                default_y = [df_plot_source.columns[1]]
                
            y_vars = st.multiselect("Y Variable(s)", df_plot_source.columns, default=default_y)
            
            # Application of Dual Axis Logic
            if chart_type in ["Line Chart", "Bar Chart"] and y_vars and len(y_vars) > 1:
                 use_dual_axis = st.checkbox("Enable Dual Axis (Right Y-Axis)", value=False)
                 if use_dual_axis:
                     right_y_vars = st.multiselect("Select Variables for Right Axis", y_vars)

    with col_color:
        # Color only available if Single Y variable is selected OR if Dual Axis is NOT used (simplification)
        if y_vars and len(y_vars) > 1:
            st.markdown("*Disabled when multiple Y variables are selected*")
            color_var = None
        else:
            options = [None] + list(df_plot_source.columns)
            color_var = st.selectbox("Color/Group (Optional)", options)
        
        # Add Bar Mode selection
        bar_mode = "stack" # Default
        if chart_type == "Bar Chart" and color_var is not None:
            bar_mode = st.radio("Bar Mode", ["Stacked", "Grouped"], horizontal=True, index=0).lower()
            if bar_mode == "stacked": bar_mode = "stack"
            elif bar_mode == "grouped": bar_mode = "group"

    # --- Facet Options ---
    col_fc1, col_fc2 = st.columns([3, 1])
    with col_fc1:
        facet_col = st.selectbox("Split Charts By (Facet)", [None] + list(df_plot_source.columns), key="facet_selector")
    with col_fc2:
        st.markdown("###") # Spacer
        if st.button("Reset Facet"):
            st.session_state['facet_selector'] = None
            st.rerun()

    # Aggregation Options (Only if NOT using Pivot, as Pivot already aggregates)
    enable_aggregation = st.checkbox("Aggregate Data", key="agg_std")
    if enable_aggregation:
        agg_method = st.selectbox("Aggregation Method", ["Mean", "Sum", "Count", "Median", "Min", "Max"], key="agg_meth_std")
        
        if y_vars:
            try:
                # Group by X and Color (if exists) AND Facet (if exists)
                groups = [x_var]
                if color_var:
                    groups.append(color_var)
                if facet_col:
                    groups.append(facet_col)
                    
                # Aggegate all selected Ys
                # Note: We need to include facet_col in groupby to keep it for splitting later
                df_plot = df_plot_source.groupby(groups)[y_vars].agg(agg_method.lower()).reset_index()
                st.info(f"Plotting {agg_method} of {', '.join(y_vars)}")
            except Exception as e:
                st.error(f"Aggregation failed: {e}")
                df_plot = df_plot_source
        else:
             st.warning("Aggregation requires Y variables.")
             df_plot = df_plot_source
    else:
        df_plot = df_plot_source

    # --- Plotting Loop (Facet Support) ---
    
    # Define plotting function to reuse
    def plot_chart(data, x, y, color, bar_mode="stack", title_suffix=""):
        # Fix for continuous color scale on binary/categorical variables
        # If the color variable has few unique values, treat it as categorical (string)
        # This prevents Streamlit/Altair from using a continuous gradient for 0/1 variables.
        if color and color in data.columns:
            unique_count = data[color].nunique()
            # Threshold of 10 should cover most categorical use cases like Region, Segment, etc.
            # Binary variables (2 values) will definitely be caught here.
            if unique_count <= 10 or data[color].dtype == 'object' or data[color].dtype.name == 'category':
                 data = data.copy()
                 data[color] = data[color].astype(str)

        # Dual Axis Plotting (Matplotlib)
        if use_dual_axis and y and len(y) > 1:
            # Check for duplicates in X and aggregate if needed for line plot
            if data[x].duplicated().any():
                st.warning(f"Data contains duplicate values for X-axis ({x}). Aggregating by mean for Dual Axis chart.")
                # Select only numeric columns + X for aggregation to avoid errors
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                cols_to_agg = list(set(numeric_cols + y)) 
                # Ensure X is preserved if it's not the index
                if x in data.columns:
                     data_sorted = data.groupby(x)[cols_to_agg].mean().reset_index()
                else: 
                     # If X is index
                     data_sorted = data.groupby(level=0)[cols_to_agg].mean().reset_index()
            else:
                data_sorted = data.sort_values(by=x)

            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Identify Left vs Right vars
            left_vars = [v for v in y if v not in right_y_vars]
            right_vars = [v for v in y if v in right_y_vars]
            
            # Plot Left Axis
            if left_vars:
                # Use a color cycle or distinct colors
                colors = plt.cm.tab10(np.linspace(0, 1, len(left_vars) + len(right_vars)))
                
                for i, col in enumerate(left_vars):
                    ax1.plot(data_sorted[x], data_sorted[col], label=f"{col} (Left)", marker='o', color=colors[i])
            
            ax1.set_xlabel(x)
            ax1.set_ylabel(", ".join(left_vars))
            ax1.tick_params(axis='y')
            # Combine legends later or put separately
            lines1, labels1 = ax1.get_legend_handles_labels()

            # Plot Right Axis
            lines2, labels2 = [], []
            if right_vars:
                ax2 = ax1.twinx()
                for i, col in enumerate(right_vars):
                     # Offset color index
                     color_idx = len(left_vars) + i
                     if color_idx >= len(colors): color_idx = i % len(colors)
                     
                     ax2.plot(data_sorted[x], data_sorted[col], label=f"{col} (Right)", linestyle='--', alpha=0.7, color=colors[color_idx])
                
                ax2.set_ylabel(", ".join(right_vars))
                ax2.tick_params(axis='y')
                lines2, labels2 = ax2.get_legend_handles_labels()
            
            # Unified Legend
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title(f"Dual Axis Chart {title_suffix}")
            st.pyplot(fig)
            return

        if chart_type == "Scatter Plot":
            fig = px.scatter(data, x=x, y=y, color=color, title=f"Scatter Plot {title_suffix}")
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Line Chart":
            fig = px.line(data, x=x, y=y, color=color, title=f"Line Chart {title_suffix}")
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Bar Chart":
            fig = px.bar(data, x=x, y=y, color=color, barmode=bar_mode, title=f"Bar Chart {title_suffix}")
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Histogram":
            fig, ax = plt.subplots()
            if color:
                for label, group in data.groupby(color):
                    ax.hist(group[x], alpha=0.5, label=str(label), bins=20)
                ax.legend()
            else:
                ax.hist(data[x], bins=20)
            ax.set_title(f"Histogram of {x} {title_suffix}")
            st.pyplot(fig)
        elif chart_type == "Box Plot":
            fig, ax = plt.subplots()
            if color:
                # Boxplot with grouping
                box_data = []
                labels = []
                for label, group in data.groupby(color):
                    if y:
                         # Use first Y for boxplot if multiple?
                         box_data.append(group[y[0]])
                    else:
                         box_data.append(group[x]) 
                    labels.append(label)
                ax.boxplot(box_data, labels=labels)
            else:
                if y:
                     # Multiple boxplots for multiple Ys
                     box_data = [data[col] for col in y]
                     ax.boxplot(box_data, labels=y)
                else:
                    ax.boxplot(data[x])
            ax.set_title(f"Box Plot {title_suffix}")
            st.pyplot(fig)
        elif chart_type == "Pie Chart":
            fig, ax = plt.subplots()
            if color:
                 counts = data[color].value_counts()
                 ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            else:
                 counts = data[x].value_counts()
                 ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            ax.set_title(f"Pie Chart {title_suffix}")
            st.pyplot(fig)

    if facet_col:
        # Get unique values and sort
        unique_facets = sorted(df_plot[facet_col].unique())
        st.write(f"Displaying {len(unique_facets)} charts for: **{facet_col}**")
        
        for facet_val in unique_facets:
            st.markdown(f"#### {facet_col}: {facet_val}")
            # Filter data
            df_facet = df_plot[df_plot[facet_col] == facet_val]
            if len(df_facet) > 0:
                plot_chart(df_facet, x_var, y_vars, color_var, bar_mode=bar_mode, title_suffix=f"({facet_val})")
            else:
                st.warning(f"No data for {facet_val}")
    else:
        # Single Chart
        plot_chart(df_plot, x_var, y_vars, color_var, bar_mode=bar_mode)


# ==========================================
# TAB 3: Analysis Configuration (Unified)
# ==========================================
with tab_config:
    st.header("Analysis Configuration")
    
    # Analysis Type Selector
    analysis_type = st.radio(
        "Analysis Type",
        ["🔍 Observational (Cross-Sectional)", "📈 Quasi-Experimental (Time Series / Panel)"],
        horizontal=True,
        help="Observational: one-time snapshot data. Quasi-Experimental: data with time/region dimensions."
    )
    
    _is_observational = "Observational" in analysis_type
    
    if len(df) == 0:
        st.info("⚠️ No data loaded yet. Please go to the **Exploratory Analysis** tab to load or generate a dataset first.")
    else:
        col_ai1, col_ai2 = st.columns(2)
        with col_ai1:
            _tab_type = "observational" if _is_observational else "quasi_experimental"
            if st.button("🧠 Recommend Method", key="config_recommend_btn"):
                if not check_api_limit():
                    st.stop()
                with st.spinner("AI is analyzing your data..."):
                    rec = data_generation_utils.generate_method_recommendation(df, chatbot_utils.get_api_key(), tab_type=_tab_type)
                    st.session_state.config_method_rec = rec
        if st.session_state.get('config_method_rec'):
            st.info(st.session_state.config_method_rec)
    
    st.divider()
    
    # ==========================================
    # Observational Analysis Configuration
    # ==========================================
    if _is_observational:
        st.subheader("Observational Analysis")
        
        # Initialize variables
        time_period = None
        
        estimation_method = st.selectbox(
            "Estimation Method",
            [
                "Linear/Logistic Regression (OLS/Logit)",
                "Propensity Score Matching (PSM)",
                "Inverse Propensity Weighting (IPTW)",
                "Linear Double Machine Learning (LinearDML)", 
                "Generalized Random Forests (CausalForestDML)",
                "Meta-Learner: S-Learner",
                "Meta-Learner: T-Learner"
            ]
        )

        # Configuration Guide button (now aware of selected method)
        if st.button("🧠 Configuration Guide", key="config_guide_obs_btn"):
            if not check_api_limit():
                st.stop()
            with st.spinner("AI is generating guidance..."):
                _existing_rec = st.session_state.get('config_method_rec', '')
                guidance = data_generation_utils.generate_config_guidance(df, chatbot_utils.get_api_key(), method=estimation_method, method_recommendation=_existing_rec)
                st.session_state.config_guide = guidance
        if st.session_state.get('config_guide'):
            st.success(st.session_state.config_guide)

        treatment = st.selectbox("Treatment (Action)", df.columns, index=get_index(df.columns, 'Feature_Adoption', 2))
    
        # -----------------------------------------------------------
        # Handle Categorical Treatment
        # -----------------------------------------------------------
        if df[treatment].dtype == 'object' or df[treatment].dtype.name == 'category':
            st.info(f"Detected categorical treatment: {treatment}. Encoding as binary.")
            unique_vals = df[treatment].unique()
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                control_val = st.selectbox("Control Value (0)", unique_vals, index=0)
            with col_t2:
                treat_val = st.selectbox("Treatment Value (1)", unique_vals, index=1 if len(unique_vals) > 1 else 0)
                
            if control_val == treat_val:
                st.error("Control and Treatment values must be different.")
                st.stop()
                
            # Create a temporary encoded column for analysis
            df['Treatment_Encoded'] = df[treatment].apply(lambda x: 1 if x == treat_val else 0)
            # Update treatment variable to point to new column
            treatment = 'Treatment_Encoded'
        # ----------------------------------------------------------- 
        outcome = st.selectbox("Outcome (Result)", df.columns, index=get_index(df.columns, 'Account_Value', 3))
        
        # Check for Binary Outcome
        is_binary_outcome = False
        if df[outcome].nunique() == 2:
            is_binary_outcome = True
            st.info(f"Detected binary outcome: `{outcome}`. Using Classification models for estimation.")

        default_confounders = [c for c in ['Customer_Segment', 'Historical_Usage'] if c in df.columns]
        confounders = st.multiselect("Confounders (Common Causes)", df.columns, default=default_confounders)
        

        enable_bootstrap = st.checkbox("Enable Bootstrapping (Calculate Standard Errors)", value=True)
        if enable_bootstrap:
            n_iterations = st.number_input("Bootstrap Iterations", min_value=10, max_value=500, value=50, step=10, help="Number of resampling iterations for SE estimation.")
        else:
            n_iterations = 0

        def on_run_click():
            st.session_state.analysis_run = True

        st.button("Run Causal Analysis", type="primary", on_click=on_run_click)


    # ==========================================
    # Quasi-Experimental Analysis Configuration
    # ==========================================
    else:
        st.subheader("Quasi-Experimental Analysis")
        st.markdown("Methods for **Panel Data** (User + Time) or **Aggregate Time Series**.")
        
        quasi_method = st.selectbox(
            "Analysis Method",
            ["Difference-in-Differences (DiD)", "CausalImpact (Bayesian Time Series)", "GeoLift (Synthetic Control)", "CausalPy (Bayesian Synthetic Control)"],
            help="DiD requires Control Group + Pre/Post. CausalImpact requires Pre-Period time series. GeoLift/CausalPy use Synthetic Control for geo experiments."
        )
        st.session_state._quasi_method = quasi_method

        # Configuration Guide button (now aware of selected method)
        if st.button("🧠 Configuration Guide", key="config_guide_btn"):
            if not check_api_limit():
                st.stop()
            with st.spinner("AI is generating guidance..."):
                _existing_rec = st.session_state.get('config_method_rec', '')
                guidance = data_generation_utils.generate_config_guidance(df, chatbot_utils.get_api_key(), method=quasi_method, method_recommendation=_existing_rec)
                st.session_state.config_guide = guidance
        if st.session_state.get('config_guide'):
            st.success(st.session_state.config_guide)
    
        # --- Difference-in-Differences ---
        if quasi_method == "Difference-in-Differences (DiD)":
            st.subheader("Configuration: Difference-in-Differences")
            
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                did_treatment = st.selectbox("Treatment Column (Intervention)", df.columns, key="did_t")
                did_outcome = st.selectbox("Outcome Column", df.columns, key="did_y")
            with col_q2:
                did_time = st.selectbox("Time Period Column (0=Pre, 1=Post)", df.columns, key="did_time", help="Must be a binary indicator or boolean where 1 indicates Post-Intervention period.")
                did_confounders = st.multiselect("Control Variables", [c for c in df.columns if c not in [did_treatment, did_outcome, did_time]], key="did_conf")

            # Check Binary Outcome
            did_is_binary = df[did_outcome].nunique() == 2
            did_use_logit = False
            if did_is_binary:
                did_use_logit = st.checkbox("Use Logit Model (for Binary Outcome)", value=True, key="did_logit")

            if st.button("Run DiD Analysis", type="primary"):
                st.write("---")
                with st.spinner("Running DiD Estimation..."):
                    results = causal_utils.run_did_analysis(
                        df, did_treatment, did_outcome, did_time, did_confounders, 
                        is_binary_outcome=did_is_binary, use_logit=did_use_logit
                    )
                    
                    if 'error' in results:
                        st.error(f"Analysis Failed: {results['error']}")
                    else:
                        st.session_state.quasi_results = results
                        st.session_state.quasi_analysis_run = True
                        st.session_state.quasi_method_run = "Difference-in-Differences (DiD)"
                        # Store params for export
                        st.session_state.did_params = {
                            'treatment': did_treatment,
                            'outcome': did_outcome,
                            'time': did_time,
                            'confounders': did_confounders,
                            'is_binary': did_is_binary,
                            'use_logit': did_use_logit
                        }
                        st.success("Analysis Complete! Scroll down to see results.")


        # --- CausalImpact ---
        elif quasi_method == "CausalImpact (Bayesian Time Series)":
            st.subheader("Configuration: CausalImpact")
            st.info("This method aggregates your User-Level data into a Daily/Weekly Time Series.")
            
            # Row 1: Date and Intervention
            col_r1_1, col_r1_2 = st.columns(2)
            with col_r1_1:
                date_cols = df.select_dtypes(include=['datetime', 'object']).columns
                ci_date_col = st.selectbox("Date Column", date_cols, index=get_index(date_cols, "Date", 0), key="ci_date")
            
            with col_r1_2:
                # We need to know the range to offer a date picker
                try:
                    min_date = pd.to_datetime(df[ci_date_col]).min()
                    max_date = pd.to_datetime(df[ci_date_col]).max()
                    
                    # Default intervention date
                    if 'Is_Treated_Region' in df.columns:
                        default_int = pd.to_datetime('2023-11-01')
                    else:
                        default_int = min_date + (max_date - min_date) / 2
                    
                    ci_intervention = st.date_input(
                        "Intervention Date (Start of Treatment)", 
                        value=default_int,
                        min_value=min_date,
                        max_value=max_date,
                        help="The model trains on data BEFORE this date, and predicts what SHOULD happen AFTER."
                    )
                except:
                    st.warning("Could not parse Date column. Please process it in Exploratory tab first.")
                    ci_intervention = None

            # Row 2: Outcome and Control Variables
            col_r2_1, col_r2_2 = st.columns(2)
            with col_r2_1:
                num_cols = df.select_dtypes(include=[np.number]).columns
                ci_outcome = st.selectbox("Outcome Column (to aggregate)", num_cols, index=get_index(num_cols, "Daily_Revenue", 0), key="ci_y")
            
            with col_r2_2:
                 if st.session_state.get('sim_type') == 'BSTS Demo':
                      candidates = [c for c in df.columns if c not in [ci_date_col, ci_outcome]]
                 else:
                      # Note: ci_unit_col is not defined yet, so we only exclude date/outcome
                      candidates = [c for c in df.columns if c not in [ci_date_col, ci_outcome] if pd.api.types.is_numeric_dtype(df[c])]

                 ci_covariates = st.multiselect("Control Variables (Covariates)", candidates, help="Select additional variables (e.g. Marketing Spend, Weather) to help the model predict the counterfactual.")

            # --- Unit / Filter Configuration ---
            st.markdown("##### Data Scope / Filtering")
            col_p1, col_p2 = st.columns(2)
            
            # Candidate columns for Unit ID
            candidates_unit = [c for c in df.columns if c not in [ci_date_col, ci_outcome]]
            ci_unit_col = st.selectbox("Unit Identifier Column (Optional)", ["None"] + candidates_unit, index=get_index(["None"] + candidates_unit, "Region", 0), help="Select if you want to filter for a specific unit OR run Panel Analysis.")
            
            ci_treated_unit = None
            if ci_unit_col != "None":
                unique_units = list(df[ci_unit_col].unique())
                ci_treated_unit = st.selectbox("Select Target/Treated Unit", unique_units, index=get_index(unique_units, "Region_1", 0), help="The specific unit/group to analyze.")
            else:
                ci_unit_col = None
                
            use_panel_data = st.checkbox("Run as Panel Data / Synthetic Control", value=True if ci_unit_col else False, help="If checked, uses other units as controls. If unchecked, performs simple time-series analysis on the Target Unit (or entire data).")

            if st.button("Run CausalImpact", type="primary"):
                if ci_intervention:
                    st.write("---")
                    with st.spinner("Aggregating Data & Running Bayesian Structural Time Series..."):
                        results = causal_utils.run_causal_impact_analysis(
                            df, ci_date_col, ci_outcome, ci_intervention, 
                            unit_col=ci_unit_col, treated_unit=ci_treated_unit,
                            use_panel=use_panel_data,
                            covariates=ci_covariates
                        )
                        
                        if 'error' in results:
                            st.error(f"Analysis Failed: {results['error']}")
                        else:
                            st.session_state.quasi_results = results
                            st.session_state.quasi_analysis_run = True
                            st.session_state.quasi_method_run = "CausalImpact (Bayesian Time Series)"
                            # Store params for export
                            st.session_state.ci_params = {
                                'date_col': ci_date_col,
                                'outcome': ci_outcome,
                                'intervention': ci_intervention,
                                'unit_col': ci_unit_col,
                                'treated_unit': ci_treated_unit,
                                'use_panel': use_panel_data
                            }
                            st.success("Analysis Complete! Scroll down to see results.")


        # --- GeoLift Analysis ---
        elif quasi_method == "GeoLift (Synthetic Control)":
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
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    power_date = st.selectbox("Date Column", df.columns, index=get_index(df.columns, 'Date', 0), key="pow_date")
                    power_geo = st.selectbox("Geography/Location Column", df.columns, index=get_index(df.columns, 'Region', 1), key="pow_geo")
                    
                    # Predictor/Covariate extraction excluding identifiers (added default calculation for covariates)
                    available_covariates_geo = [c for c in df.columns if c not in [power_date, power_geo]]
                    pow_covariates = st.multiselect("Covariates (Optional)", available_covariates_geo, help="Select additional features to improve Synthetic Control fit.", key="pow_cov")
                    
                with col_p2:
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    
                    # Dynamic filter to prevent KPI from being a covariate and vice versa
                    kpi_options = [c for c in num_cols if c not in pow_covariates]
                    power_kpi = st.selectbox("Outcome Column", kpi_options, index=get_index(kpi_options, 'KPI', 0) if len(kpi_options) > 0 else 0, key="pow_kpi")
                
                power_duration = st.number_input("Expected Treatment Duration (Days)", min_value=1, value=60, key="pow_dur", help="How long do you expect the marketing test to run?")
                
                # Cutoff date logic for testing on datasets with existing treatment
                pow_min_date = df[power_date].min()
                pow_max_date = df[power_date].max()
                pow_default_cutoff = pow_max_date
                
                # For the demo, default to Nov 1st 2023 if available
                if 'Region' in df.columns and 'Date' in df.columns:
                    demo_cutoff = pd.to_datetime('2023-11-01')
                    if demo_cutoff >= pow_min_date and demo_cutoff <= pow_max_date:
                        pow_default_cutoff = demo_cutoff
                
                power_cutoff_date = st.date_input(
                    "Historical Data Cutoff", 
                    value=pow_default_cutoff, 
                    min_value=pow_min_date.date() if hasattr(pow_min_date, 'date') else None,
                    max_value=pow_max_date.date() if hasattr(pow_max_date, 'date') else None,
                    help="Simulation will only use data up to this date. Useful for testing on datasets that already contain intervention data.",
                    key="pow_cutoff"
                )

                with st.expander("Advanced Configuration"):
                    st.markdown("""
                    **⚡ Performance Tips**:
                    - **Number of Test Markets (N)**: Testing combinations (e.g. '1, 2') massively increases computation time compared to just '1' because it builds synthetic controls for every possible pair.
                    - **Lookback Window**: Increasing this acts as a direct multiplier on runtime. A lookback of 5 takes 5x longer than a lookback of 1, because it simulates the test backwards 5 independent times.
                    - **Model**: 'none' (standard Synthetic Control) is the fastest. Adding 'Ridge' or 'GSYN' augmentation significantly improves precision on noisy data but can be 3-5x slower to optimize.
                    """)
                    
                    pow_n_markets = st.text_input("Number of Test Markets (comma-separated)", value="1", help="e.g. '1' to test single regions, or '1, 2' to test groups of 1 and 2.")
                    pow_lookback = st.number_input("Lookback Window", min_value=1, value=1, help="Number of historical samples. Set to 1 for fast iteration, 5+ for final decisions.")
                    pow_model = st.selectbox("Augmentation Model", ["none", "Ridge", "GSYN", "best"], index=0, help="'Ridge' is recommended for small panels, 'GSYN' for large panels, and 'best' will test all and return the optimal model.")
                    pow_alpha = st.slider("Significance Level (Alpha)", 0.01, 0.2, 0.1, help="Standard is 0.1 for GeoLift.")
                    pow_side = st.selectbox("Side of Test", ["two_sided", "one_sided"], index=0)
                    pow_parallel = st.checkbox("Enable Parallel Processing", value=True, help="Uses multiple CPU cores to speed up simulations.")

                    st.write("---")
                    st.markdown("**🛡️ Performance & Precision Overrides**")
                    pow_fixed_effects = st.checkbox("Fixed Effects", value=True, help="Controls for geographic and time unobservables.")
                    pow_dtw = st.number_input("Dynamic Time Warping (DTW)", value=0, help="Addresses misaligned time-series data.")
                    pow_correlations = st.checkbox("Analyze Correlations", value=False, help="Uses correlated features to improve fit.")
                    pow_cpic = st.number_input("Cost Per Incremental Conversion (CPIC)", value=1.0)
                    pow_budget_input = st.number_input("Budget (Optional)", value=0.0)
                    pow_budget = pow_budget_input if pow_budget_input > 0 else None
                    pow_es_mode = st.radio("Simulation Density", ["Fast (0%, 10%)", "Full (0%, 5%, 10%, 15%, 20%)"], index=0, 
                                            help="Fast mode tests fewer points, significantly reducing runtime for exploration.")
                    pow_ns = st.selectbox("Resamples (NS)", [100, 1000], index=0, 
                                            help="1000 is for final publication; 100 is sufficient for market selection.")
                    pow_normalize = st.checkbox("Normalize Data", value=False, help="Scaling outcome data can speed up computation on large raw values.")
                    
                st.info("Power Analysis will search through all unique geographies in the dataset to find the best candidate for treatment.")
                
                if st.button("Run Power Analysis", type="primary"):
                    st.write("---")
                    with st.spinner("Running R GeoLiftMarketSelection (Augmentation models and multiple lookbacks increase runtime)..."):
                        try:
                            results_power = causal_utils.run_geolift_power(
                                df, power_date, power_geo, power_kpi, 
                                treatment_duration=power_duration, 
                                cutoff_date=str(power_cutoff_date),
                                n_markets=pow_n_markets,
                                lookback_window=pow_lookback,
                                model=pow_model,
                                alpha=pow_alpha,
                                side_of_test=pow_side,
                                parallel=pow_parallel,
                                ns=pow_ns,
                                effect_size_mode=pow_es_mode.split()[0], # "Fast" or "Full",
                                normalize=pow_normalize,
                                covariates=pow_covariates,
                                fixed_effects=pow_fixed_effects,
                                dtw=pow_dtw,
                                correlations=pow_correlations,
                                cpic=pow_cpic,
                                budget=pow_budget
                            )
                            
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
                    
                    # Predictor/Covariate extraction excluding identifiers
                    available_covariates = [c for c in df.columns if c not in [geo_lift_date, geo_lift_geo]]
                    est_covariates = st.multiselect("Covariates (Optional)", available_covariates, help="Select additional features to improve Synthetic Control fit.", key="est_cov")
                    
                with col_g2:
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    
                    kpi_est_options = [c for c in num_cols if c not in est_covariates]
                    geo_lift_kpi = st.selectbox("Outcome Column", kpi_est_options, index=get_index(kpi_est_options, 'KPI', 0) if len(kpi_est_options) > 0 else 0, key="gl_kpi")
                    
                    treated_geo_options = []
                    if geo_lift_geo in df.columns:
                        treated_geo_options = df[geo_lift_geo].dropna().astype(str).unique().tolist()
                    geo_lift_treated = st.selectbox("Treated Geography", treated_geo_options, key="gl_treat")
                
                min_date = df[geo_lift_date].min()
                max_date = df[geo_lift_date].max()
                default_int = min_date + (max_date - min_date) / 2
                
                try:
                    # Let's try to infer if they are using the demo dataset
                    if 'Region' in df.columns and 'Date' in df.columns:
                         default_int_val = pd.to_datetime('2023-11-01')
                         if default_int_val >= min_date and default_int_val <= max_date:
                             default_int = default_int_val
                except:
                     pass
                     
                geo_lift_intervention_date = st.date_input("Intervention Start Date", value=default_int, min_value=min_date.date() if hasattr(min_date, 'date') else None, max_value=max_date.date() if hasattr(max_date, 'date') else None, key="gl_int")
                geo_lift_duration = st.number_input("Treatment Duration (Days)", min_value=1, value=60, key="gl_dur2")

                with st.expander("Estimation Configuration"):
                    est_model = st.selectbox("Augmentation Model", ["none", "Ridge", "GSYN", "best"], index=0, key="gl_model_est")
                    est_alpha = st.slider("Significance Level (Alpha)", 0.01, 0.2, 0.1, help="Threshold for statistical significance.", key="gl_alpha_est")
                    est_ci = st.checkbox("Calculate Confidence Intervals", value=True, help="Adds uncertainty estimation but increases runtime.", key="gl_ci_est")
                    est_test = st.selectbox("Statistical Test", ["Total", "Negative", "Positive"], index=0, help="'Total' is standard for overall lift.", key="gl_test_est")
                    est_fixed_effects = st.checkbox("Fixed Effects", value=True, key="gl_fe_est")
                    est_grid_size = st.number_input("Grid Size (CI Precision)", value=250, key="gl_grid_est")
        
                if st.button("Run GeoLift Analysis", type="primary"):
                    st.write("---")
                    with st.spinner("Initializing R Environment and Running GeoLift..."):
                        try:
                            results = causal_utils.run_geolift_analysis(
                                df, geo_lift_date, geo_lift_geo, geo_lift_treated, geo_lift_kpi, 
                                str(geo_lift_intervention_date), 
                                treatment_duration=geo_lift_duration,
                                model=est_model,
                                alpha=est_alpha,
                                confidence_intervals=est_ci,
                                stat_test=est_test,
                                covariates=est_covariates,
                                fixed_effects=est_fixed_effects,
                                grid_size=est_grid_size
                            )
                            st.session_state.quasi_results = {
                                'method': 'GeoLift (Synthetic Control)',
                                'result': results,
                                'date_col': geo_lift_date,
                                'geo_col': geo_lift_geo,
                                'kpi_col': geo_lift_kpi,
                                'treated_geo': geo_lift_treated,
                                'intervention_date': str(geo_lift_intervention_date),
                                'treatment_duration': geo_lift_duration
                            }
                            st.session_state.quasi_analysis_run = True
                            st.session_state.quasi_method_run = "GeoLift (Synthetic Control)"
                            st.success("GeoLift Analysis Complete! Scroll down to see results.")
                        except Exception as e:
                            st.error(f"GeoLift Analysis failed: {e}")

        # --- CausalPy (Bayesian Synthetic Control) ---
        elif quasi_method == "CausalPy (Bayesian Synthetic Control)":
            st.subheader("Configuration: CausalPy (Bayesian Synthetic Control)")
            st.caption("Pure Python • PyMC Backend • Proper Two-sided HDI Intervals")
            
            col_cp1, col_cp2 = st.columns(2)
            with col_cp1:
                cp_date = st.selectbox("Date Column", df.columns, index=get_index(df.columns, 'Date', 0), key="cp_date")
                cp_geo = st.selectbox("Geography/Location Column", df.columns, index=get_index(df.columns, 'Region', 1), key="cp_geo")
                
            with col_cp2:
                num_cols = df.select_dtypes(include=[np.number]).columns
                cp_kpi = st.selectbox("Outcome (KPI) Column", num_cols, index=get_index(num_cols, 'KPI', 0) if len(num_cols) > 0 else 0, key="cp_kpi")
                
                treated_geo_options_cp = []
                if cp_geo in df.columns:
                    treated_geo_options_cp = df[cp_geo].dropna().astype(str).unique().tolist()
                cp_treated = st.selectbox("Treated Geography", treated_geo_options_cp, key="cp_treat")
                
                # Covariates for CausalPy
                cp_covariates = st.multiselect(
                    "Covariates (Controls/Predictors)", 
                    [c for c in num_cols if c not in [cp_kpi]], 
                    help="Additional time-series features to help construct the synthetic control.",
                    key="cp_covs"
                )
            
            min_date_cp = df[cp_date].min()
            max_date_cp = df[cp_date].max()
            default_int_cp = min_date_cp + (max_date_cp - min_date_cp) / 2
            
            try:
                if 'Region' in df.columns and 'Date' in df.columns:
                    demo_val = pd.to_datetime('2023-11-01')
                    if demo_val >= min_date_cp and demo_val <= max_date_cp:
                        default_int_cp = demo_val
            except:
                pass
            
            cp_intervention_date = st.date_input("Intervention Start Date", value=default_int_cp, 
                                                  min_value=min_date_cp.date() if hasattr(min_date_cp, 'date') else None,
                                                  max_value=max_date_cp.date() if hasattr(max_date_cp, 'date') else None,
                                                  key="cp_int")
            cp_duration = st.number_input("Treatment Duration (Days)", min_value=1, value=60, key="cp_dur")

            with st.expander("Bayesian Inference Settings"):
                cp_direction = st.selectbox("Effect Direction", ["two-sided", "increase", "decrease"], index=0, 
                                            help="Two-sided tests for any effect; 'increase'/'decrease' for directional hypotheses.", key="cp_dir")
                cp_hdi = st.slider("HDI Credible Interval Width", 0.80, 0.99, 0.95, step=0.01, 
                                    help="Width of the Highest Density Interval (Bayesian CI).", key="cp_hdi")
                
                st.divider()
                st.markdown("**Advanced PyMC Sampling Parameters**")
                col_mc1, col_mc2 = st.columns(2)
                with col_mc1:
                    cp_tune = st.number_input("Tuning Steps", min_value=100, max_value=4000, value=1000, step=100, help="Number of MCMC tuning steps. Reduce to 500 for faster execution.", key="cp_tune")
                    cp_draws = st.number_input("MCMC Draws", min_value=100, max_value=4000, value=1000, step=100, help="Number of posterior samples. Reduce to 500 for faster execution.", key="cp_draws")
                with col_mc2:
                    cp_chains = st.number_input("Chains", min_value=1, max_value=8, value=4, step=1, help="Parallel MCMC chains. Reduce to 2 for smaller instances.", key="cp_chains")
                    cp_target_accept = st.slider("Target Acceptance", 0.70, 0.99, 0.95, step=0.01, help="PyMC target_accept. Lowering to 0.85 greatly speeds up convergence but may diverge on complex data.", key="cp_target_accept")
            
            if st.button("Run CausalPy Analysis", type="primary"):
                st.write("---")
                with st.spinner("Fitting Bayesian Synthetic Control model via PyMC (this may take a few minutes)..."):
                    try:
                        cp_results = causal_utils.run_causalpy_synthetic_control(
                            df, cp_date, cp_geo, cp_kpi, cp_treated,
                            str(cp_intervention_date),
                            treatment_duration=cp_duration,
                            hdi_prob=cp_hdi,
                            direction=cp_direction,
                            covariates=cp_covariates,
                            tune=cp_tune,
                            draws=cp_draws,
                            chains=cp_chains,
                            target_accept=cp_target_accept
                        )
                        st.session_state.quasi_results = {
                            'method': 'CausalPy (Bayesian Synthetic Control)',
                            'result': cp_results,
                            'date_col': cp_date,
                            'geo_col': cp_geo,
                            'kpi_col': cp_kpi,
                            'treated_geo': cp_treated,
                            'intervention_date': str(cp_intervention_date),
                            'treatment_duration': cp_duration
                        }
                        st.session_state.quasi_analysis_run = True
                        st.session_state.quasi_method_run = "CausalPy (Bayesian Synthetic Control)"
                        st.success("CausalPy Analysis Complete! Scroll down to see results.")
                    except Exception as e:
                        st.error(f"CausalPy Analysis failed: {e}")

        # --- SHARED RESULTS & EXPORT MODULE ---
        if st.session_state.get('quasi_analysis_run', False) and st.session_state.get('quasi_results') is not None:
            st.divider()
            results = st.session_state.quasi_results
            quasi_method_run = st.session_state.quasi_method_run
            
            if quasi_method_run == "Difference-in-Differences (DiD)":
                # --- Methodology and Formula ---
                st.markdown("### 📖 Methodology: Difference-in-Differences — *[Angrist & Pischke (2009)](https://www.mostlyharmlesseconometrics.com/)*")
                st.markdown(r"""
                **Difference-in-Differences (DiD)** is a quasi-experimental design that uses longitudinal data from treatment and control groups to obtain an appropriate counterfactual to estimate a causal effect.
                
                It compares the changes in outcomes over time between a population that is enrolled in a program (the intervention group) and a population that is not (the control group).
                
                **The Regression Model:**
                """)
                st.latex(r"Y = \beta_0 + \beta_1 T + \beta_2 Post + \beta_{DiD} (T \times Post) + \epsilon")
                st.markdown(r"""
                Where:
                - $T$: Treatment indicator (1 if in treatment group, 0 otherwise).
                - $Post$: Time indicator (1 if post-intervention, 0 if pre-intervention).
                - $T \times Post$: The interaction term.
                - $\beta_{DiD}$: The **Difference-in-Differences coefficient**, representing the causal effect.
                """)
                st.divider()

                st.subheader(f"Results: {quasi_method_run}")
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Coefficient (Interaction)", f"{results['coefficient']:.4f}")
                m2.metric("P-value", f"{results['p_value']:.4f}")
                m3.metric("95% CI", f"[{results['ci_lower']:.2f}, {results['ci_upper']:.2f}]")
                
                if 'odds_ratio' in results:
                    st.info(f"Odds Ratio: **{results['odds_ratio']:.4f}**")
                
                # Summary
                with st.expander("Show Regression Summary", expanded=True):
                    st.text(results['summary'])
                    
            elif quasi_method_run == "CausalImpact (Bayesian Time Series)":
                # --- Methodology and Formula ---
                st.markdown(r"""
                **CausalImpact** is built on Bayesian Structural Time Series (BSTS) models, as detailed in **[Scott and Varian (2014)](https://doi.org/10.1504/IJMMNO.2014.059942)**. It decomposes the time series into trend, seasonal, and regression components, using a state-space formulation to estimate the counterfactual.
                
                **The Observation Equation:**
                """)
                st.latex(r"y_t = \mu_t + \tau_t + \beta^T \mathbf{x}_t + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma_\epsilon^2)")
                
                st.markdown(r"**The State Equations (Dynamics):**")
                st.latex(r"\begin{aligned} \mu_t &= \mu_{t-1} + \delta_{t-1} + u_t, \quad &u_t \sim N(0, \sigma_u^2) \quad (\text{Trend}) \\ \delta_t &= \delta_{t-1} + v_t, \quad &v_t \sim N(0, \sigma_v^2) \quad (\text{Slope}) \\ \tau_t &= -\sum_{s=1}^{S-1} \tau_{t-s} + w_t, \quad &w_t \sim N(0, \sigma_w^2) \quad (\text{Seasonality}) \end{aligned}")
                
                st.markdown(r"""
                Where:
                - $y_t$: Observed outcome at time $t$.
                - $\mu_t$: Level/Trend component.
                - $\delta_t$: Slope of the trend (Local Linear Trend).
                - $\tau_t$: Seasonal component with $S$ seasons.
                - $\beta^T \mathbf{x}_t$: Regression component (external predictors).
                - **Spike-and-Slab Prior**: Used for feature selection in $\beta$, allowing for sparse regression on large sets of control variables.

                The **Causal Effect** $\tau_t$ (not to be confused with seasonal $\tau_t$) is the difference between observed and predicted counterfactual:
                """)
                st.latex(r"\text{Effect}_t = y_t - \hat{y}_t(0)")
                st.divider()

                st.subheader(f"Results: {quasi_method_run}")
                metrics = results.get('metrics', {})
                if not metrics:
                    st.error("Model did not return valid metrics.")
                else:
                    c1, c2, c3 = st.columns(3)
                    p = metrics['p_value']
                    with c1:
                        st.metric("Average Effect (ATE)", f"{metrics['ate']:,.2f}", f"95% CI: [{metrics['ate_lower']:,.2f}, {metrics['ate_upper']:,.2f}] | p={p:.3f}", delta_color="off")
                    with c2:
                        st.metric("Cumulative Effect", f"{metrics['cum_abs']:,.2f}", f"95% CI: [{metrics['cum_lower']:,.2f}, {metrics['cum_upper']:,.2f}] | p={p:.3f}", delta_color="off")
                    with c3:
                        st.metric("Relative Lift", f"{metrics['rel_effect']:+.2%}", f"95% CI: [{metrics['rel_lower']:+.2%}, {metrics['rel_upper']:+.2%}] | p={p:.3f}", delta_color="off")
                
                    st.subheader("Report")
                    with st.expander("Read Detailed Report", expanded=False):
                        st.markdown(results['report'])
                
                # Plotting
                st.subheader("Visualization")
                try:
                    st.image(results['plot_path'], caption="CausalImpact Results: Original, Pointwise, and Cumulative", use_container_width=True)
                    if 'plot_df' in results and results['plot_df'] is not None:
                        with st.expander("View Underlying Plot Data", expanded=False):
                            st.dataframe(results['plot_df'], use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render plot from R: {e}")

            elif quasi_method_run == "GeoLift Power Analysis (Market Selection)":
                st.divider()

                if "error" in results['result']:
                    st.error(results['result']['error'])
                else:
                    st.subheader("Results: GeoLift Market Selection")
                    
                    st.markdown(r"""
                    **Methodology (Market Selection & Power Analysis):**
                    GeoLift's Market Selection uses historical data to identify the best candidate markets for an experiment. It evaluates possible treated units by checking their pre-treatment fit (using Synthetic Control) and their statistical power (calculating the Minimum Detectable Effect).
                    
                    **Methodology References:**
                    - Augmented Synthetic Control Method (*[Ben-Michael, Feller, and Roth, 2021](https://doi.org/10.1080/01621459.2021.1929245)*)
                    - Generalized Synthetic Control Method (GSYN) (*[Yiqing Xu, 2017](https://doi.org/10.1017/pan.2016.2)*)
                    - Meta Open Source GeoLift implementation (*[Arturo Deza, Nicolas Cruces, and Jose Benitez, 2023](https://github.com/facebookincubator/GeoLift)*)
                    
                    **Augmentation Models**:
                    - **Ridge**: Adds a penalty to the synthetic control weights to handle smaller datasets. Highly recommended for datasets with few locations or noisy signals.
                    - **GSYN (Generalized Synthetic Control)**: Uses a latent factor model to estimate counterfactuals. Best for larger datasets with many locations, as it can capture complex unobserved time-varying trends.
                    
                    **Formula Definition:**
                    The power analysis evaluates the probability of rejecting the null hypothesis $H_0: \text{Lift} = 0$:
                    $$ \text{Power} = P(\text{p-value} < \alpha \mid H_1) $$
                    where $H_1$ is a simulated effect of size $\delta$.
                    """)
                    st.divider()
                    
                    st.markdown("Below are the ranked candidate markets based on pre-treatment fit and required MDE:")
                    
                    if 'df' in results['result']:
                        st.dataframe(
                            results['result']['df'], 
                            use_container_width=True,
                            column_config={
                                "AvgScaledL2Imbalance": st.column_config.NumberColumn(
                                    "AvgScaledL2Imbalance",
                                    help="A high Scaled Imbalance (near 1.0) often means your control markets are already so similar to your test market that the synthetic weighting adds little extra precision. If your visual fit is tight, the test is likely still valid."
                                )
                            }
                        )
                        
                        st.subheader("Diagnostic Plots (#1 Ranked Market)")
                        col_plot1, col_plot2 = st.columns(2)
                        with col_plot1:
                            if 'power_plot' in results['result']:
                                st.image(results['result']['power_plot'], caption="Power Curve: Effect Size vs Power", use_container_width=True)
                        with col_plot2:
                            if 'series_plot' in results['result']:
                                st.image(results['result']['series_plot'], caption="Historical Fit: Treated vs Synthetic", use_container_width=True)
                                
                        st.markdown(r"""
                        #### 💡 How to Interpret & Select Markets
                        To select the best candidate for your experiment, look for rows that balance these criteria:
                        
                        1.  **Lower RMSE (Pre-treatment Fit)**: Look at the **`AvgScaledL2Imbalance`** column. This measures how well the synthetic control matches your test market historically. 
                            *   *Benchmark*: Values **below 0.2** indicate a exceptionally strong, unique fit. 
                            *   *The Scaling "Trap"*: Because this metric is scaled against a simple average of all your donor regions, highly correlated datasets will naturally produce scores near 1.0. If the plot above shows the lines nearly perfectly overlapping and the `abs_lift_in_zero` column (pre-test hallucination) is near 0, your test is highly reliable and you can safely disregard a high L2 score!
                            *   *How to improve a bad visual fit*: Try adding `Ridge` integration or filter your donor pool to remove identical "carbon-copy" cities, giving the algorithm more mathematical room to optimize.
                        2.  **Lower MDE (Sensitivity)**: Look at the **`Average_MDE`** column. This is the smallest effect size the test can reliably detect. 
                        3.  **Higher Power**: Look at the **`Power`** column. Standard practice is to aim for power $\ge$ 0.8 (80%).
                        
                        **Understanding the Plots**:
                        *   **Power Curve**: Shows how statistical power scales up with the size of your effect. You want the curve to sharply rise and securely cross the 0.8 (80%) horizontal line as far left (at a low minimum detectable effect) as possible.
                        *   **Historical Fit**: Shows your actual target market's data (solid line) vs the algorithm's constructed Synthetic Control (dashed line) during the pre-period. These lines should consistently overlap without crossing or "fanning out" too wildly.
                        
                        **Note on Locations**: If you see multiple regions listed for a single location (e.g., "Region_1, Region_2"), it means GeoLift recommends testing these regions **together as a cluster** to achieve the required statistical power.
                        """)

            elif quasi_method_run == "GeoLift (Synthetic Control)":
                st.divider()
                
                if "error" in results['result']:
                    st.error(results['result']['error'])
                else:
                    st.subheader("Results: GeoLift (Synthetic Control)")
                    
                    st.markdown(r"""
                    **Methodology (Augmented Synthetic Control):**
                    GeoLift uses the Augmented Synthetic Control Method (ASCM) to estimate the causal effect of an intervention at the geographic level. It constructs a "synthetic" version of the treated location by finding a weighted combination of untreated locations that best matches the pre-treatment time series.
                    
                    *Methodology References:* 
                    - Augmented Synthetic Control Method (*[Ben-Michael, Feller, and Roth, 2021](https://doi.org/10.1080/01621459.2021.1929245)*)
                    - Meta Open Source GeoLift implementation (*[Arturo Deza, Nicolas Cruces, and Jose Benitez, 2023](https://github.com/facebookincubator/GeoLift)*)
                    
                    **Formula Definition:**
                    Let region $1$ be treated and $2, \dots, N$ be the donor pool. ASCM seeks weights $W$ to minimize:
                    $$ \min_W \sum_{t=1}^{T_0} (Y_{1t} - \sum_{j=2}^N w_j Y_{jt})^2 $$
                    The augmented estimator adds a ridge adjustment:
                    $$ \hat{\tau}_{1t} = Y_{1t} - (\sum_{j=2}^N \hat{w}_j Y_{jt} + (\mathbf{X}_{1t} - \sum_{j=2}^N \hat{w}_j \mathbf{X}_{jt})^T \hat{\beta}) $$
                    """)
                    st.divider()
                    
                    if 'metrics' in results['result']:
                        metrics = results['result']['metrics']
                        
                        st.subheader("Statistical Summary")
                        c1, c2, c3 = st.columns(3)
                        p = metrics['p_val']
                        with c1:
                            ci_str = f"95% CI: [{metrics['ate_lower']:,.2f}, {metrics['ate_upper']:,.2f}] | p={p:.3f}" if metrics.get('has_ci') else f"p={p:.3f}"
                            st.metric("Average Effect (ATT)", f"{metrics['avg_lift']:,.2f}", ci_str, delta_color="off")
                        with c2:
                            ci_str_cum = f"95% CI: [{metrics['cum_lower']:,.2f}, {metrics['cum_upper']:,.2f}] | p={p:.3f}" if metrics.get('has_ci') else f"p={p:.3f}"
                            st.metric("Cumulative Impact", f"{metrics['cum_lift']:,.2f}", ci_str_cum, delta_color="off")
                        with c3:
                            ci_str_rel = f"95% CI: [{metrics['rel_lower']:+.2%}, {metrics['rel_upper']:+.2%}] | p={p:.3f}" if metrics.get('has_ci') and metrics.get('rel_lower') is not None else f"p={p:.3f}"
                            st.metric("Relative Lift", f"{metrics['perc_lift']:+.2%}", ci_str_rel, delta_color="off")
                            
                        direction = "increase" if metrics['cum_lift'] > 0 else "decrease"
                        if metrics['significant']:
                            sig_text = f"The results are **statistically significant** ($p = {metrics['p_val']:.4f} < {metrics['alpha']}$). "
                            conclusion = f"This indicates that the intervention had a measurable and statistically significant **{direction}** on the target metric for **{metrics['treated_geo']}**."
                        else:
                            sig_text = f"The results are **NOT statistically significant** ($p = {metrics['p_val']:.4f} \ge {metrics['alpha']}$). "
                            conclusion = f"This indicates that there is insufficient evidence to conclude the intervention had a meaningful impact on **{metrics['treated_geo']}**, as the observed differences fall within the expected noise of the synthetic control."
                            
                        st.info(f"**Interpretation:** During the post-intervention period, the **{metrics['treated_geo']}** market experienced an average estimated treatment effect of **{metrics['avg_lift']:,.2f}** per period, compounding to a total cumulative impact of **{metrics['cum_lift']:,.2f}**. {sig_text}{conclusion}")
                    
                        with st.expander("Show Raw Model Output"):
                            st.markdown(results['result']['summary'])
                    
                    if 'plot_path' in results['result'] or 'att_plot_path' in results['result']:
                        if 'plot_path' in results['result']:
                            try:
                                st.image(results['result']['plot_path'], caption="GeoLift Results: Treated vs Synthetic Control", use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not load standard plot: {e}")
                        if 'att_plot_path' in results['result']:
                            try:
                                st.image(results['result']['att_plot_path'], caption="Average Treatment Effect on the Treated (ATT)", use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not load ATT plot: {e}")
                        
                        if 'plot_df' in results['result'] and results['result']['plot_df'] is not None:
                            with st.expander("View Underlying Plot Data", expanded=False):
                                st.dataframe(results['result']['plot_df'], use_container_width=True)

            elif quasi_method_run == "CausalPy (Bayesian Synthetic Control)":
                st.divider()
                
                if "error" in results['result']:
                    st.error(results['result']['error'])
                else:
                    st.subheader("Results: CausalPy (Bayesian Synthetic Control)")
                    
                    st.markdown(r"""
                    **Methodology (Bayesian Synthetic Control):**
                    CausalPy uses a Bayesian approach to the Synthetic Control Method. It constructs a counterfactual by fitting a weighted combination of control units to the treated unit's pre-treatment data using PyMC's probabilistic programming framework. Unlike frequentist approaches, this yields full posterior distributions over the causal effect, providing proper two-sided Highest Density Interval (HDI) credible intervals and posterior tail probabilities.
                    
                    *Methodology References:* 
                    - Synthetic Control Methods for Comparative Case Studies (*[Abadie, Diamond, Hainmueller, 2010](https://doi.org/10.1198/jasa.2009.ap09707)*)
                    - Inferring Causal Impact Using Bayesian Structural Time-Series Models (*[Brodersen et al., 2015](https://doi.org/10.1214/14-AOAS788)*)
                    - CausalPy open-source library (*[pymc-labs/CausalPy](https://github.com/pymc-labs/CausalPy)*)
                    
                    **Formula Definition:**
                    Given treated unit $Y_1$ and $J$ control units $Y_2, \dots, Y_{J+1}$, the synthetic control seeks weights $\mathbf{w}$ to minimize:
                    $$ \min_{\mathbf{w}} \sum_{t=1}^{T_0} (Y_{1t} - \sum_{j=2}^{J+1} w_j Y_{jt})^2, \quad w_j \ge 0, \quad \sum_j w_j = 1 $$
                    The Bayesian approach places priors on $\mathbf{w}$ and uses MCMC sampling to obtain posterior distributions of the treatment effect $\hat{\tau}_{1t} = Y_{1t} - \hat{Y}_{1t}^{SC}$.
                    """)
                    st.divider()
                    
                    if 'metrics' in results['result']:
                        metrics = results['result']['metrics']
                        
                        st.subheader("Statistical Summary")
                        c1, c2, c3 = st.columns(3)
                        
                        with c1:
                            avg_val = metrics.get('avg_effect')
                            avg_lo = metrics.get('avg_ci_lower')
                            avg_hi = metrics.get('avg_ci_upper')
                            if avg_val is not None:
                                ci_str = f"{int(metrics.get('hdi_prob', 0.95)*100)}% HDI: [{avg_lo:,.2f}, {avg_hi:,.2f}]" if avg_lo is not None else ""
                                st.metric("Average Effect (ATT)", f"{avg_val:,.2f}", ci_str, delta_color="off")
                            else:
                                st.metric("Average Effect (ATT)", "N/A")
                        
                        with c2:
                            cum_val = metrics.get('cum_effect')
                            cum_lo = metrics.get('cum_ci_lower')
                            cum_hi = metrics.get('cum_ci_upper')
                            if cum_val is not None:
                                ci_str_cum = f"{int(metrics.get('hdi_prob', 0.95)*100)}% HDI: [{cum_lo:,.2f}, {cum_hi:,.2f}]" if cum_lo is not None else ""
                                st.metric("Cumulative Impact", f"{cum_val:,.2f}", ci_str_cum, delta_color="off")
                            else:
                                st.metric("Cumulative Impact", "N/A")
                        
                        with c3:
                            rel_val = metrics.get('rel_effect')
                            rel_lo = metrics.get('rel_ci_lower')
                            rel_hi = metrics.get('rel_ci_upper')
                            if rel_val is not None:
                                ci_str_rel = f"{int(metrics.get('hdi_prob', 0.95)*100)}% HDI: [{rel_lo:+.2%}, {rel_hi:+.2%}]" if rel_lo is not None else ""
                                st.metric("Relative Lift", f"{rel_val:+.2%}", ci_str_rel, delta_color="off")
                            else:
                                st.metric("Relative Lift", "N/A")
                        
                        # Posterior probability
                        prob = metrics.get('prob_effect')
                        if prob is not None:
                            direction = metrics.get('direction', 'two-sided')
                            st.info(f"**Posterior Probability:** The probability of a causal effect ({direction}) is **{prob:.3f}** for **{metrics.get('treated_geo', 'N/A')}**.")
                    
                    # Summary text
                    if 'summary_text' in results['result'] and results['result']['summary_text']:
                        with st.expander("Show Full Bayesian Summary"):
                            st.markdown(results['result']['summary_text'])
                    
                    # Summary table
                    if 'summary_table' in results['result'] and results['result']['summary_table'] is not None:
                        with st.expander("Show Effect Summary Table"):
                            st.dataframe(results['result']['summary_table'], use_container_width=True)
                    
                    # Plot
                    if 'plot_path' in results['result'] and results['result']['plot_path']:
                        try:
                            st.image(results['result']['plot_path'], caption="CausalPy: Treated vs Synthetic Control with Credible Intervals", use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not load plot: {e}")

            # --- EXPORT SECTION ---
            st.divider()
            st.subheader("Export Data and Script")
            
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                csv_quasi = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Data with Results (CSV)",
                    data=csv_quasi,
                    file_name='quasi_analysis_results.csv',
                    mime='text/csv',
                )
            
            # Script Generation
            # Extract variables from state
            p_impute_enable = st.session_state.get('impute_enable', False)
            p_num_method = st.session_state.get('num_impute_method', "Mean")
            p_num_val = st.session_state.get('num_custom_val', 0.0)
            p_cat_method = st.session_state.get('cat_impute_method', "Mode")
            p_cat_val = st.session_state.get('cat_custom_val', "Missing")
            p_wins_enable = st.session_state.get('winsorize_enable', False)
            p_wins_cols = st.session_state.get('winsorize_cols', [])
            p_percentile = st.session_state.get('percentile', 0.05)
            p_log_cols = st.session_state.get('log_transform_cols', [])
            p_std_cols = st.session_state.get('standardize_cols', [])

            if quasi_method_run == "Difference-in-Differences (DiD)":
                p = st.session_state.did_params
                script_quasi = generate_script(
                    data_source=data_source,
                    treatment=p['treatment'],
                    outcome=p['outcome'],
                    confounders=p['confounders'],
                    time_period=p['time'],
                    estimation_method=quasi_method_run,
                    impute_enable=p_impute_enable,
                    num_impute_method=p_num_method if p_impute_enable else None,
                    num_custom_val=p_num_val if p_impute_enable else None,
                    cat_impute_method=p_cat_method if p_impute_enable else None,
                    cat_custom_val=p_cat_val if p_impute_enable else None,
                    winsorize_enable=p_wins_enable,
                    winsorize_cols=p_wins_cols,
                    percentile=p_percentile,
                    log_transform_cols=p_log_cols,
                    standardize_cols=p_std_cols,
                    n_iterations=50,
                    use_logit=p['use_logit']
                )
            elif quasi_method_run == "CausalImpact (Bayesian Time Series)":
                p = st.session_state.ci_params
                ts_params_script = {
                    'date_col': p['date_col'],
                    'intervention_date': str(p['intervention']),
                    'enabled': True,
                    'use_panel': p.get('use_panel', False),
                    'is_bsts_demo': st.session_state.get('sim_type') == "BSTS Demo"
                }
                script_quasi = generate_script(
                    data_source=data_source,
                    treatment=None,
                    outcome=p['outcome'],
                    confounders=[],
                    time_period=None,
                    estimation_method=quasi_method_run,
                    impute_enable=p_impute_enable,
                    num_impute_method=p_num_method if p_impute_enable else None,
                    num_custom_val=p_num_val if p_impute_enable else None,
                    cat_impute_method=p_cat_method if p_impute_enable else None,
                    cat_custom_val=p_cat_val if p_impute_enable else None,
                    winsorize_enable=p_wins_enable,
                    winsorize_cols=p_wins_cols,
                    percentile=p_percentile,
                    log_transform_cols=p_log_cols,
                    standardize_cols=p_std_cols,
                    n_iterations=50,
                    ts_params=ts_params_script,
                    unit_col=p['unit_col'],
                    treated_unit=p['treated_unit']
                )
            elif "GeoLift" in quasi_method_run:
                res_dict = st.session_state.get('quasi_results', {})
                date_col = res_dict.get('date_col', 'date')
                geo_col = res_dict.get('geo_col', 'location')
                kpi_col = res_dict.get('kpi_col', 'Y')
                treated_geo = res_dict.get('treated_geo', 'Treated_Location')
                int_date = res_dict.get('intervention_date', '2023-01-01')
                dur = res_dict.get('treatment_duration', 60)

                script_quasi = f'''# Python Script for {quasi_method_run} using rpy2
# Make sure to install: pip install rpy2 pandas
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# 1. Load your data
# df = pd.read_csv("your_data.csv")
# (For example purposes, replace this with your actual data loading)
# Ensure the date column is a string format 'YYYY-MM-DD' and KPI is numeric.
# df['{date_col}'] = pd.to_datetime(df['{date_col}']).dt.strftime('%Y-%m-%d')
# df['{kpi_col}'] = pd.to_numeric(df['{kpi_col}'])

# 2. Transfer Data to R Environment
# (Uncomment the lines below when you have your 'df')
"""
cv = robjects.default_converter + pandas2ri.converter
with localconverter(cv):
        r_df = pandas2ri.py2rpy(df)
robjects.globalenv['py_data'] = r_df
"""

# 3. GeoLift R Execution Block
robjects.r("""
library(GeoLift)
library(dplyr)

# Read and Format Data
# geo_data <- GeoDataRead(data = py_data,
#                         date_id = "{date_col}",
#                         location_id = "{geo_col}",
#                         Y_id = "{kpi_col}",
#                         format = "yyyy-mm-dd")
'''
                if "Market Selection" in quasi_method_run:
                    script_quasi += f'''
# Run Power Analysis / Market Selection
# results <- GeoLiftMarketSelection(data = geo_data,
#                                   treatment_periods = {dur},
#                                   N = c(1),
#                                   Y_id = "Y",
#                                   location_id = "location",
#                                   time_id = "time",
#                                   lookback_window = 1,
#                                   effect_size = seq(0, 0.2, 0.05),
#                                   model = "none",
#                                   parallel = TRUE)

# View best markets table
# print(head(results$BestMarkets))
# plot(results, market_ID = 1)
""")
'''
                else:
                    script_quasi += f'''
# Map intervention date to time index
# time_index <- unique(geo_data$time)
# all_dates <- unique(geo_data$date)
# treatment_start_date <- as.Date("{int_date}")
# 
# valid_start_times <- time_index[as.Date(all_dates) >= treatment_start_date]
# treatment_start_idx <- min(valid_start_times)
# treatment_end_idx <- treatment_start_idx + {dur} - 1

# Run Impact Estimation
# results <- GeoLift(Y_id = "Y", 
#                    time_id = "time", 
#                    location_id = "location",
#                    data = geo_data, 
#                    locations = c("{treated_geo}"),
#                    treatment_start_time = treatment_start_idx, 
#                    treatment_end_time = treatment_end_idx)

# summary(results)
# plot(results)
""")
'''

            elif "CausalPy" in quasi_method_run:
                res_dict = st.session_state.get('quasi_results', {})
                date_col = res_dict.get('date_col', 'date')
                geo_col = res_dict.get('geo_col', 'location')
                kpi_col = res_dict.get('kpi_col', 'Y')
                treated_geo = res_dict.get('treated_geo', 'Treated_Location')
                int_date = res_dict.get('intervention_date', '2023-01-01')
                dur = res_dict.get('treatment_duration', 60)

                script_quasi = f'''# Python Script for CausalPy Bayesian Synthetic Control
# pip install causalpy pandas

import pandas as pd
import causalpy
from causalpy.pymc_models import WeightedSumFitter

# 1. Load your data
df = pd.read_csv("your_data.csv")
df["{date_col}"] = pd.to_datetime(df["{date_col}"])

# 2. Pivot to wide format (rows=dates, columns=regions)
df_wide = df.pivot_table(index="{date_col}", columns="{geo_col}", values="{kpi_col}", aggfunc="mean")
df_wide = df_wide.sort_index()

# 3. Define treated and control units
treated_units = ["{treated_geo}"]
control_units = [c for c in df_wide.columns if c != "{treated_geo}"]

# 4. Fit the Bayesian Synthetic Control model
result = causalpy.SyntheticControl(
    df_wide,
    treatment_time=pd.Timestamp("{int_date}"),
    treated_units=treated_units,
    control_units=control_units,
    model=WeightedSumFitter(sample_kwargs={{"random_seed": 42}}),
)

# 5. Get effect summary
stats = result.effect_summary(treated_unit="{treated_geo}", direction="two-sided")
print(stats.text)
print(stats.table)

# 6. Plot results
result.plot()
'''
                
            with col_ex2:
                st.download_button(
                    label="Download Python Script",
                    data=script_quasi,
                    file_name='causal_analysis_script.py',
                    mime='text/x-python',
                )
                
            with st.expander("View Generated Script"):
                st.code(script_quasi, language='python')



# ==========================================
# ==========================================
# TAB 1: User Guide
# ==========================================
with tab_guide:
    with open("user_guide.html", "r", encoding="utf-8") as f:
        user_guide_html = f.read()
    st.html(user_guide_html)
    
    if history:
        # Display Latest Version
        st.markdown(f"**Latest Version:** {latest_version['Version']} ({latest_version['Release Date']})")
        st.info(f"**Release Note:** {latest_version['Release Note']}")
        
        # Display History Table
        with st.expander("See Full History"):
            history_df = pd.DataFrame(history)
            st.table(history_df)
    else:
        st.warning("No version history found.")

# Observational Analysis Results Block
if st.session_state.get('analysis_run', False):
    with tab_config:

        with st.container(): # Main results container 
            # Ideally we dedent the whole block, but to minimize diff noise let's just remove the check.
            # Actually, let's just remove the if/else and dedent.
            pass

            st.divider()
            st.header("Causal Analysis Pipeline")
        
            # --- Step 1: Model ---
            st.subheader("1. Causal Model")
            st.markdown("**Methodology:** Structural Causal Model (SCM) — *[Pearl (2009)](https://doi.org/10.1017/CBO9780511803161)*")
            st.markdown("We define a Directed Acyclic Graph (DAG) $G = (V, E)$ where:")
            st.markdown(f"- $V$: Variables including Treatment (`{treatment}`), Outcome (`{outcome}`), and Confounders.")
            st.markdown("- $E$: Causal edges representing direct effects.")
            st.markdown("Assumption: **Causal Markov Assumption** (each variable is independent of its non-descendants given its parents).")
        
            # Only use confounders as effect modifiers for HTE-capable ML methods
            # For OLS/Logit, we want a simple adjustment without interaction terms in the main ATE model.
            if estimation_method in ["Linear Double Machine Learning (LinearDML)", "Generalized Random Forests (CausalForestDML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner"]:
                modifiers = confounders
            else:
                modifiers = []

                modifiers = []



                st.divider()
                st.subheader("Overall Analysis (Aggregated)")

            with st.spinner("Building Causal Graph..."):
                model = CausalModel(
                    data=df,
                    treatment=treatment,
                    outcome=outcome,
                    common_causes=confounders,
                    instruments=None,
                    effect_modifiers=modifiers
                )
        
            st.success("Model built successfully!")
            st.markdown("**Assumptions:**")
            st.write(f"Treatment: `{treatment}` causes Outcome: `{outcome}`")
            st.write(f"Confounders: `{', '.join(confounders)}` affect both.")
                    # Visualize DAG explicitly with Graphviz DOT syntax
            try:
                dot_graph = "digraph {\n"
                dot_graph += '  rankdir=LR;\n'
                dot_graph += '  node [fontname="Helvetica, Arial, sans-serif"];\n'
                
                # Target & Outcome with nice colors
                dot_graph += f'  "{treatment}" [style=filled, fillcolor="#bae6fd", shape=box];\n'
                dot_graph += f'  "{outcome}" [style=filled, fillcolor="#bbf7d0", shape=box];\n'
                dot_graph += f'  "{treatment}" -> "{outcome}" [label=" Causal Effect?", color="#0ea5e9", penwidth=2.0];\n'
                
                # Confounders
                for c in confounders:
                    dot_graph += f'  "{c}" [style=filled, fillcolor="#f3f4f6", shape=ellipse];\n'
                    dot_graph += f'  "{c}" -> "{treatment}" [color="#6b7280"];\n'
                    dot_graph += f'  "{c}" -> "{outcome}" [color="#6b7280"];\n'
                dot_graph += "}"
                
                with st.expander("📍 View Causal Graph (DAG)", expanded=True):
                    st.graphviz_chart(dot_graph)
            except Exception as e:
                st.caption(f"Could not render DAG: {e}")

            # --- Step 2: Identify ---
            st.subheader("2. Identification")
            st.markdown("**Methodology:** Backdoor Criterion — *[Pearl (2009)](https://doi.org/10.1017/CBO9780511803161)*")
            st.markdown("We aim to identify the causal effect $P(Y|do(T))$ from observational data $P(Y, T, X)$.")
            st.markdown("If a set of variables $X$ satisfies the Backdoor Criterion, we can use the **Adjustment Formula**:")
            st.latex(r"P(Y|do(T)) = \sum_X P(Y|T, X)P(X)")
        
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # check if successfully identified
            is_identified = False
            for stim_type in ['backdoor', 'frontdoor', 'iv']:
                if identified_estimand.estimands.get(stim_type) is not None:
                    is_identified = True
                    break
            
            if is_identified:
                st.success(f"✅ Causal Relationship Identified! (Type: {identified_estimand.estimand_type})")
            else:
                st.error("❌ Causal Relationship NOT Identified. The current DAG and confounders do not allow for identifying the effect.")

            with st.expander("Show Identification Results", expanded=False):
                st.code(str(identified_estimand), language="text")
        
            # --- Step 3: Estimate (using EconML / DML) ---
            st.subheader("3. Estimation")
            with st.spinner(f"Estimating Causal Effect using {estimation_method}..."):
            
                use_logit = False # Initialize to avoid NameError in Refutation
                estimate = None # Initialize to avoid UnboundLocalError
                if estimation_method == "Linear Double Machine Learning (LinearDML)":
                    st.markdown("##### Methodology: Double Machine Learning (DML) — *[Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097)*")
                    st.markdown("DML removes the effect of confounders ($X$) from both treatment ($T$) and outcome ($Y$) using ML models.")
                
                    st.markdown("**Step 1: Residualize Outcome**")
                    st.latex(r"Y_{res} = Y - E[Y|X]")
                
                    st.markdown("**Step 2: Residualize Treatment**")
                    st.latex(r"T_{res} = T - E[T|X]")
                
                    st.markdown("**Step 3: Estimate Causal Effect**")
                    st.latex(r"Y_{res} = \theta \cdot T_{res} + \epsilon")
                    st.caption("Where $\\theta$ is the Average Treatment Effect (ATE).")

                    # We use LinearDML from EconML
                    # It uses ML models to residualize treatment and outcome, then runs linear regression on residuals
                    
                    # For DML, even if the outcome is binary, we use a Regressor for model_y (LPM approach)
                    # because LinearDML expects a continuous residual or probability estimate, 
                    # and passing a Classifier can cause errors if EconML expects a Regressor interface.
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
                    st.markdown("#### Methodology: Propensity Score Matching (PSM) — *[Rosenbaum & Rubin (1983)](https://doi.org/10.1093/biomet/70.1.41)*")
                    if is_binary_outcome:
                        st.caption("ℹ️ **Binary Outcome**: Estimate represents Risk Difference (Difference in Proportions).")
                    st.markdown("PSM matches treated units with control units that have similar probability of receiving treatment.")
                
                    st.markdown("**Step 1: Estimate Propensity Score**")
                    st.latex(r"e(x) = P(T=1|X=x)")
                
                    st.markdown("**Step 2: Match Units**")
                    st.markdown("Find control unit $j$ for treated unit $i$ such that $e(x_i) \approx e(x_j)$.")
                
                    st.markdown("**Step 3: Estimate ATE**")
                    st.latex(r"ATE = \frac{1}{N} \sum_{i=1}^{N} (Y_i(1) - Y_{match(i)}(0))")
                
                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.propensity_score_matching"
                    )

                elif estimation_method == "Inverse Propensity Weighting (IPTW)":
                    st.markdown("##### Methodology: Inverse Propensity Weighting (IPTW) — *[Austin (2011)](https://doi.org/10.1080/00273171.2011.568786)*")
                    if is_binary_outcome:
                        st.caption("ℹ️ **Binary Outcome**: Estimate represents Risk Difference (Weighted Difference in Proportions).")
                    st.markdown("IPTW re-weights the data to create a pseudo-population where treatment is independent of confounders.")
                
                    st.markdown("**Step 1: Estimate Propensity Score**")
                    st.latex(r"e(x) = P(T=1|X=x)")
                
                    st.markdown("**Step 2: Calculate Weights**")
                    st.latex(r"w_i = \frac{T_i}{e(x_i)} + \frac{1-T_i}{1-e(x_i)}")
                
                    st.markdown("**Step 3: Estimate ATE**")
                    st.latex(r"ATE = \frac{1}{N} \sum_{i=1}^{N} w_i Y_i")
                
                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.propensity_score_weighting"
                    )

                elif "Meta-Learner" in estimation_method:
                    learner_type = estimation_method.split(": ")[1]
                    st.markdown(f"##### Methodology: {learner_type} — *[Künzel et al. (2019)](https://doi.org/10.1073/pnas.1804597116)*")
                
                    if learner_type == "S-Learner":
                        st.markdown("S-Learner (Single Learner) treats treatment as a feature in a single ML model.")
                        st.latex(r"f(X, T) \approx Y")
                        st.latex(r"ATE = E[f(X, 1) - f(X, 0)]")
                        method_name = "backdoor.econml.metalearners.SLearner"
                        
                        if is_binary_outcome:
                            overall_model = RandomForestClassifier(n_jobs=-1, random_state=42)
                        else:
                            overall_model = RandomForestRegressor(n_jobs=-1, random_state=42)
                            
                        init_params = {"overall_model": overall_model}
                    else: # T-Learner
                        st.markdown("T-Learner (Two Learners) fits separate models for treated and control groups.")
                        st.latex(r"\mu_1(X) \approx E[Y|T=1, X], \quad \mu_0(X) \approx E[Y|T=0, X]")
                        st.latex(r"ATE = E[\mu_1(X) - \mu_0(X)]")
                        method_name = "backdoor.econml.metalearners.TLearner"
                        
                        if is_binary_outcome:
                            models = RandomForestClassifier(n_jobs=-1, random_state=42)
                        else:
                            models = RandomForestRegressor(n_jobs=-1, random_state=42)
                            
                        init_params = {"models": models}

                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name=method_name,
                        method_params={
                            "init_params": init_params,
                            "fit_params": {}
                        }
                    )

                elif estimation_method == "Generalized Random Forests (CausalForestDML)":
                    st.markdown("##### Methodology: Generalized Random Forests (CausalForestDML) — *[Wager & Athey (2018)](https://doi.org/10.1080/01621459.2017.1319839)*")
                    st.markdown("Causal Forests extend Random Forests to estimate heterogeneous treatment effects (CATE) using an honest splitting criterion.")
                    st.latex(r"\hat{\tau}(x) = \frac{\sum \alpha_i(x) (Y_i - \hat{m}(X_i)) (T_i - \hat{e}(X_i))}{\sum \alpha_i(x) (T_i - \hat{e}(X_i))^2}")
                
                    # Use Regressor for model_y (LPM) to avoid errors with binary outcomes
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


                elif estimation_method == "Difference-in-Differences (DiD)":
                    st.markdown("#### Methodology: Difference-in-Differences (DiD)")
                    
                    use_logit = False
                    if is_binary_outcome:
                        use_logit = True
                        st.caption("ℹ️ **Binary Outcome**: Using **Logit Model** (Logistic Regression) to estimate **Odds Ratio**.")
                    
                    if not time_period:
                        st.error("Please select a Time Period in the sidebar.")
                        st.stop()
                    
                    st.markdown("DiD compares the changes in outcomes over time between a treatment group and a control group.")
                    if use_logit:
                        st.latex(r"\text{Logit Model: } \log\left(\frac{P(Y=1)}{1-P(Y=1)}\right) = \beta_0 + \beta_1 T + \beta_2 \text{Time} + \beta_3 (T \times \text{Time}) + \dots")
                        st.latex(r"\text{Odds Ratio (Interaction)} = e^{\beta_3}")
                    else:
                        st.latex(r"ATE = (E[Y|T=1, Post] - E[Y|T=1, Pre]) - (E[Y|T=0, Post] - E[Y|T=0, Pre])")
                    
                    # Manual DiD Implementation (OLS or Logit)
                    # Create interaction term
                    df_did = df.copy()
                    df_did['DiD_Interaction'] = df_did[treatment] * df_did[time_period]
                    
                    # Define X (Treatment, Time, Interaction, Confounders)
                    X_cols = [treatment, time_period, 'DiD_Interaction']
                    if confounders:
                        X_cols.extend(confounders)
                    X_did = df_did[X_cols]
                    X_did = sm.add_constant(X_did)
                    y_did = df_did[outcome]
                    
                    if use_logit:
                        try:
                            did_model = sm.Logit(y_did, X_did).fit(disp=0)
                            did_coeff = did_model.params['DiD_Interaction']
                            did_pvalue = did_model.pvalues['DiD_Interaction']
                            did_conf_int = did_model.conf_int().loc['DiD_Interaction']
                            odds_ratio = np.exp(did_coeff)
                            
                            st.markdown(f"**Estimated Odds Ratio (Interaction): {odds_ratio:.4f}**")
                            st.markdown(f"**Log-Odds Coefficient:** {did_coeff:.4f}")
                            st.markdown(f"**P-value:** {did_pvalue:.4f}")
                            st.markdown(f"**95% CI (Log-Odds):** [{did_conf_int[0]:.4f}, {did_conf_int[1]:.4f}]")
                            
                            # Convert OR to Risk Difference (Approximate)
                            # Baseline Risk (Control Group in Post Period - or overall Control)
                            # Let's use the overall Control group mean as a reference baseline
                            baseline_risk = df_did[df_did[treatment] == 0][outcome].mean()
                            if 0 < baseline_risk < 1:
                                def or_to_rd(odds_ratio, p0):
                                    return (odds_ratio * p0 / (1 - p0 + (odds_ratio * p0))) - p0
                                
                                implied_risk_diff = or_to_rd(odds_ratio, baseline_risk)
                                
                                # Convert CI
                                or_lower = np.exp(did_conf_int[0])
                                or_upper = np.exp(did_conf_int[1])
                                rd_lower = or_to_rd(or_lower, baseline_risk)
                                rd_upper = or_to_rd(or_upper, baseline_risk)
                                
                                st.markdown(f"**Implied Risk Difference (at Baseline Risk {baseline_risk:.2%}):** {implied_risk_diff:+.2%}")
                                st.markdown(f"**Implied RD 95% CI:** [{rd_lower:+.2%}, {rd_upper:+.2%}]")
                                st.caption("Note: Implied Risk Difference is calculated based on the observed control group mean.")
                            
                            # Create a dummy estimate object for compatibility (optional, or just skip standard display)
                            estimate = type('obj', (object,), {'value': did_coeff, 'estimator': type('obj', (object,), {'__str__': lambda self: "Logistic Regression (DiD)"})()})
                        except Exception as e:
                            st.error(f"Logit Model Failed: {e}")
                            st.stop()
                        
                    else:
                        try:
                            did_model = sm.OLS(y_did, X_did).fit()
                            if 'DiD_Interaction' in did_model.params:
                                did_estimate = did_model.params['DiD_Interaction']
                                st.markdown(f"**Estimated ATE (Interaction Coefficient): {did_estimate:.4f}**")
                                st.write(did_model.summary())
                                estimate = type('obj', (object,), {'value': did_estimate})
                            else:
                                st.error("DiD Interaction term dropped (likely due to collinearity). Check your data.")
                                st.stop()
                        except Exception as e:
                            st.error(f"LPM (OLS) Failed: {e}")
                            st.stop()
                    
                    # We skip standard DoWhy estimation for DiD because we did it manually
                    # But we need 'estimate' object for Refutation if we were to run it (Refutation disabled for DiD)
                    
                    # Let's use statsmodels for the DiD specific regression to get the p-value and summary.
                    import statsmodels.api as sm
                    


                elif estimation_method == "Linear/Logistic Regression (OLS/Logit)":
                    st.markdown("##### Methodology: OLS/Logit")
                    
                    use_logit = False
                    if is_binary_outcome:
                        use_logit = True
                        st.caption("ℹ️ **Binary Outcome**: Using **Logit Model** (Logistic Regression) to estimate **Odds Ratio**.")
                    
                    st.markdown("Simple comparison of average outcomes between treatment and control groups.")
                    
                    if use_logit:
                        st.latex(r"\text{Odds Ratio} = \frac{P(Y=1|T=1)/P(Y=0|T=1)}{P(Y=1|T=0)/P(Y=0|T=0)}")
                        
                        try:
                            # Manual Logit Implementation for A/B
                            X_ab_cols = [treatment]
                            if confounders:
                                X_ab_cols.extend(confounders)
                            X_ab = df[X_ab_cols]
                            X_ab = sm.add_constant(X_ab)
                            y_ab = df[outcome]
                            
                            ab_model = sm.Logit(y_ab, X_ab).fit(disp=0)
                            ab_coeff = ab_model.params[treatment]
                            ab_pvalue = ab_model.pvalues[treatment]
                            ab_conf_int = ab_model.conf_int().loc[treatment]
                            odds_ratio = np.exp(ab_coeff)
                            
                            st.markdown(f"**Estimated Odds Ratio: {odds_ratio:.4f}**")
                            st.markdown(f"**Log-Odds Coefficient:** {ab_coeff:.4f}")
                            st.markdown(f"**P-value:** {ab_pvalue:.4f}")
                            st.markdown(f"**95% CI (Log-Odds):** [{ab_conf_int[0]:.4f}, {ab_conf_int[1]:.4f}]")
                            
                            # Convert OR to Risk Difference
                            # Baseline Risk (Control Group)
                            baseline_risk = df[df[treatment] == 0][outcome].mean()
                            if 0 < baseline_risk < 1:
                                def or_to_rd(odds_ratio, p0):
                                    return (odds_ratio * p0 / (1 - p0 + (odds_ratio * p0))) - p0
                                    
                                implied_risk_diff = or_to_rd(odds_ratio, baseline_risk)
                                
                                # Convert CI
                                or_lower = np.exp(ab_conf_int[0])
                                or_upper = np.exp(ab_conf_int[1])
                                rd_lower = or_to_rd(or_lower, baseline_risk)
                                rd_upper = or_to_rd(or_upper, baseline_risk)
                                
                                st.markdown(f"**Implied Risk Difference (at Baseline Risk {baseline_risk:.2%}):** {implied_risk_diff:+.2%}")
                                st.markdown(f"**Implied RD 95% CI:** [{rd_lower:+.2%}, {rd_upper:+.2%}]")
                                st.caption("Note: Implied Risk Difference is calculated based on the observed control group mean.")
                            
                            estimate = type('obj', (object,), {'value': ab_coeff})
                        except Exception as e:
                            st.error(f"Logit Model Failed: {e}")
                            st.stop()
                        
                    else:
                        st.latex(r"ATE = E[Y|T=1] - E[Y|T=0]")
                        try:
                            estimate = model.estimate_effect(
                                identified_estimand,
                                method_name="backdoor.linear_regression",
                                test_significance=True
                            )
                            if estimate is None:
                                st.warning("Debug: First estimation attempt returned None. Retrying without test_significance...")
                                try:
                                    estimate = model.estimate_effect(
                                        identified_estimand,
                                        method_name="backdoor.linear_regression",
                                        test_significance=False
                                    )
                                except Exception as e:
                                    st.error(f"Debug: Retry failed with error: {e}")

                            if estimate is None:
                                st.error("Debug: model.estimate_effect returned None for backdoor.linear_regression")
                                st.write("Debug Info:")
                                st.write(f"Treatment: {treatment}")
                                st.write(f"Outcome: {outcome}")
                                st.write(f"Confounders: {confounders}")
                                st.write("Data Types:")
                                st.write(df[[treatment, outcome] + confounders].dtypes)
                                st.write("Data Head:")
                                st.write(df[[treatment, outcome] + confounders].head())
                                st.write("Identified Estimand:")
                                st.write("Identified Estimand:")
                                st.write(identified_estimand)
                                st.stop() # Stop here to ensure user sees the debug info
                            
                            # Extract and Display Detailed Stats
                            try:
                                # DoWhy's LinearRegressionEstimator uses statsmodels internally
                                # Accessing the internal model to get summary stats
                                if hasattr(estimate.estimator, 'model'):
                                    sm_model = estimate.estimator.model
                                    # The treatment coefficient name might vary, usually it's the treatment name
                                    # But DoWhy might rename it. Let's check params.
                                    # Actually, estimate.value is the coefficient.
                                    # We can try to find the corresponding index or name.
                                    
                                    # Simpler approach: Use the test_significance results if available
                                    # But DoWhy's test_significance printout is just a print.
                                    
                                    # Let's try to show the full summary which is very informative
                                    st.markdown("**Regression Results:**")
                                    st.write(sm_model.summary())
                                    
                                    # Explicitly show SE and CI for Treatment
                                    # We need to identify the treatment variable name in the model
                                    # DoWhy usually keeps it as is or adds prefix.
                                    # Let's look at pvalues index
                                    treat_var = treatment
                                    if treat_var in sm_model.pvalues:
                                        se = sm_model.bse[treat_var]
                                        pval = sm_model.pvalues[treat_var]
                                        ci = sm_model.conf_int().loc[treat_var]
                                        
                                        st.markdown(f"**Standard Error:** {se:.4f}")
                                        st.markdown(f"**P-value:** {pval:.4f}")
                                        st.markdown(f"**95% CI:** [{ci[0]:.4f}, {ci[1]:.4f}]")
                            except Exception as e:
                                st.warning(f"Could not extract detailed stats: {e}")
                        except Exception as e:
                            st.error(f"LPM Estimation Failed: {e}")
                            st.stop()
                    
                    if confounders:
                        st.info("Note: Confounders included in regression (CUPED / Variance Reduction).")
                    else:
                        st.info("Note: Simple Difference in Means (Unadjusted).")
                    

            
            
            
            if estimate is None:
                st.error("Estimation failed (returned None). Please check your data and settings.")
                st.stop()

            ate = estimate.value
        
            # --- Extract Standard Error & CI ---
            se = None
            ci = None
        
            try:

            
                # 2. Other Methods (Check generic attributes)
                if se is None:
                    if hasattr(estimate, 'stderr'):
                        se = estimate.stderr
                
                    if hasattr(estimate, 'get_confidence_intervals'):
                        try:
                            ci_res = estimate.get_confidence_intervals(confidence_level=0.95)
                            if ci_res is not None:
                                # Handle different return formats (tuple, array, etc.)
                                if isinstance(ci_res, (list, tuple, np.ndarray)) and len(ci_res) == 2:
                                    ci = (float(ci_res[0]), float(ci_res[1]))
                        except Exception:
                            pass # CI extraction failed
                        
            except Exception as e:
                st.warning(f"Could not extract Standard Error: {e}")

            # --- Bootstrapping for SE ---
            # Bootstrapping is now default and configured in sidebar
        # --- 4. Bootstrapping (if enabled) ---
            se = None
            ci_lower = None
            ci_upper = None
            
            if n_iterations > 0:
                bootstrap_estimates = []
                progress_bar = st.progress(0)
            
                with st.spinner(f"Running {n_iterations} bootstrap iterations..."):
                    for i in range(n_iterations):
                        # Resample with replacement
                        df_resampled = df.sample(frac=1, replace=True, random_state=i) # Use i as seed for reproducibility of the set
                    
                        # Re-define model on resampled data
                        # Note: We must re-instantiate CausalModel to avoid state leakage
                        # Use same modifier logic as main model
                        if estimation_method in ["Linear Double Machine Learning (LinearDML)", "Generalized Random Forests (CausalForestDML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner"]:
                            modifiers_boot = confounders
                        else:
                            modifiers_boot = []
    
                        model_boot = CausalModel(
                            data=df_resampled,
                            treatment=treatment,
                            outcome=outcome,
                            common_causes=confounders,
                            instruments=None,
                            effect_modifiers=modifiers_boot
                        )
                    
                        identified_estimand_boot = model_boot.identify_effect(proceed_when_unidentifiable=True)
                    
                        # Re-estimate
                        # We need to use the exact same method and params
                        # This duplication is a bit verbose but necessary to ensure same config
                        try:
                            if estimation_method == "Linear Double Machine Learning (LinearDML)":
                                est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.econml.dml.LinearDML",
                                    method_params={
                                        "init_params": {
                                            "model_y": RandomForestRegressor(random_state=42),
                                            "model_t": RandomForestClassifier(random_state=42),
                                            "discrete_treatment": True,
                                            "random_state": 42
                                        },
                                        "fit_params": {}
                                    }
                                )
                            elif estimation_method == "Propensity Score Matching (PSM)":
                                    est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.propensity_score_matching"
                                )
                            elif estimation_method == "Inverse Propensity Weighting (IPTW)":
                                    est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.propensity_score_weighting"
                                )
                            elif "Meta-Learner" in estimation_method:
                                learner_type = estimation_method.split(": ")[1]
                                if learner_type == "S-Learner":
                                    method_name = "backdoor.econml.metalearners.SLearner"
                                    init_params = {"overall_model": RandomForestRegressor(random_state=42)}
                                else:
                                    method_name = "backdoor.econml.metalearners.TLearner"
                                    init_params = {"models": RandomForestRegressor(random_state=42)}
                            
                                est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name=method_name,
                                    method_params={
                                        "init_params": init_params,
                                        "fit_params": {}
                                    }
                                )
                            elif estimation_method == "Generalized Random Forests (CausalForestDML)":
                                    est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.econml.dml.CausalForestDML",
                                    method_params={
                                        "init_params": {
                                            "model_y": RandomForestRegressor(random_state=42),
                                            "model_t": RandomForestClassifier(random_state=42),
                                            "discrete_treatment": True,
                                            "random_state": 42
                                        },
                                        "fit_params": {}
                                    }
                                )

                            elif estimation_method == "Linear/Logistic Regression (OLS/Logit)":
                                    est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.linear_regression",
                                    test_significance=False # Speed up
                                )
                                    # For Logit, we might need to extract the coefficient differently if we want OR
                                    # But for SE calculation, we usually want SE of the coefficient (log-odds)
                                    # est_boot.value should be the coefficient.
    
                        
                            bootstrap_estimates.append(est_boot.value)
                    
                        except Exception:
                            pass # Skip failed iterations
                    
                        progress_bar.progress((i + 1) / n_iterations)
            
                if len(bootstrap_estimates) > 0:
                    se = np.std(bootstrap_estimates)
                    ci_lower = np.percentile(bootstrap_estimates, 2.5)
                    ci_upper = np.percentile(bootstrap_estimates, 97.5)
                    st.success(f"Bootstrapping complete. Used {len(bootstrap_estimates)} successful iterations.")
                else:
                    st.error("Bootstrapping failed for all iterations.")
            else:
                st.info("Bootstrapping disabled. Standard Errors and Confidence Intervals will not be calculated.")

            # Display Metrics
            col_ate, col_se = st.columns(2)
            with col_ate:
                st.metric(label="Average Treatment Effect (ATE)", value=f"{ate:.2f}")
            with col_se:
                if se is not None:
                    st.metric(label="Standard Error (SE)", value=f"{se:.2f}")
                else:
                    st.metric(label="Standard Error (SE)", value="N/A", help="SE not available for this method/configuration.")
        
            if ci is not None:
                st.caption(f"**95% Confidence Interval:** [{ci[0]:.2f}, {ci[1]:.2f}]")
                st.caption("(Computed via Bootstrapping)")
        
            st.info(
                f"**Interpretation:** On average, `{treatment}` increases `{outcome}` by **{ate:.2f}** "
                "after accounting for confounding variables."
            )
            
            # AI Results Interpretation
            if st.button("🧠 AI: Interpret These Results", key="ai_interpret_obs"):
                if not check_api_limit():
                    st.stop()
                with st.spinner("AI is interpreting your results..."):
                    _metrics = {
                        "Average Treatment Effect (ATE)": f"{ate:.4f}",
                        "Method": estimation_method,
                    }
                    if 'ci' in dir() and ci:
                        _metrics["95% Confidence Interval"] = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
                    interpretation = data_generation_utils.generate_results_interpretation(
                        chatbot_utils.get_api_key(), estimation_method, _metrics,
                        treatment=treatment, outcome=outcome
                    )
                    st.session_state.obs_interpretation = interpretation
            if st.session_state.get('obs_interpretation'):
                st.success(st.session_state.obs_interpretation)

            # --- Heterogeneity Analysis (Moved from Step 5) ---
            st.divider()
            st.subheader("Heterogeneity Analysis")
            
            hte_feature = None
        
            # Check if method supports Heterogeneous Treatment Effects (CATE)
            cate_methods = [
                "Linear Double Machine Learning (LinearDML)",
                "Meta-Learner: S-Learner",
                "Meta-Learner: T-Learner",
                "Generalized Random Forests (CausalForestDML)"
            ]
        
            if estimation_method in cate_methods:
                st.markdown("#### Individual Treatment Effects (ITE)")
                st.markdown("Distribution of causal effects across the population.")
            
                try:
                    # EconML estimators in DoWhy are wrapped. 
                    # We need to pass the effect modifiers (X) to predict ITE.
                    # For this simple app, we'll use the confounders as X.
                    if not confounders:
                        st.warning("No confounders selected. Cannot estimate Heterogeneous Treatment Effects (ITE) based on confounders.")
                        ite = None
                    else:
                        X_test = df[confounders]
                        
                        # Accessing the underlying EconML estimator can be tricky via DoWhy's unified API
                        # But estimate.estimator object usually exposes it.
                        # However, DoWhy's CausalEstimator might not directly expose 'effect' for all.
                        # We will try to use the `estimate.estimator.effect(X)` if available.
                    
                        if hasattr(estimate.estimator, 'effect'):
                             ite = estimate.estimator.effect(X_test)
                        elif hasattr(estimate, 'estimator_instance') and hasattr(estimate.estimator_instance, 'effect'):
                             ite = estimate.estimator_instance.effect(X_test)
                        else:
                            # Fallback for some DoWhy/EconML versions
                            ite = None
                            st.warning("Could not extract ITEs from this estimator version.")

                    if ite is not None:
                        # Flatten if necessary
                        ite = ite.flatten()
                    
                        fig, ax = plt.subplots()
                        ax.hist(ite, bins=30, alpha=0.7, color='green')
                        ax.axvline(estimate.value, color='red', linestyle='--', label='ATE')
                        ax.set_title("Distribution of Individual Treatment Effects")
                        ax.set_xlabel("Treatment Effect")
                        ax.set_ylabel("Frequency")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                        # Add ITE to dataframe for download
                        df_results = df.copy()
                        df_results['Estimated_ITE'] = ite
                    
                        # Feature Importance (Causal Forest only)
                        if estimation_method == "Generalized Random Forests (CausalForestDML)":
                            st.markdown("#### Feature Importance")
                            # EconML CausalForest has feature_importances_
                            if hasattr(estimate.estimator, 'feature_importances_'):
                                importances = estimate.estimator.feature_importances_
                                feat_names = confounders
                            
                                fig, ax = plt.subplots()
                                y_pos = np.arange(len(feat_names))
                                ax.barh(y_pos, importances, align='center')
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(feat_names)
                                ax.invert_yaxis()  # labels read top-to-bottom
                                ax.set_title("Feature Importance for Heterogeneity")
                                ax.set_title("Feature Importance for Heterogeneity")
                                st.pyplot(fig)
                                plt.close(fig)

                        # --- Universal HTE Summary ---
                        st.markdown("**Effect Modification**")
                        st.markdown("Analysis of how covariates influence the estimated Individual Treatment Effects (ITE).")
                        
                        # Allow heterogeneity analysis on any column (except treatment/outcome/time)
                        valid_covariates_cate = [c for c in df.columns if c not in [treatment, outcome, time_period, 'Treatment_Encoded']]
                        
                        if not valid_covariates_cate:
                            st.warning("No covariates available for heterogeneity analysis.")
                        else:
                            st.markdown("**Analyzing heterogeneity for all available features...**")
                            hte_results_cate = []
                            progress_bar_cate = st.progress(0)
                            
                            # Create a temporary dataframe for regression
                            df_ite = df.copy()
                            df_ite['ITE'] = ite
                            
                            for i, feature in enumerate(valid_covariates_cate):
                                try:
                                    # Simple Linear Regression: ITE ~ Feature
                                    X_feat = sm.add_constant(df_ite[feature])
                                    y_feat = df_ite['ITE']
                                    
                                    model_feat = sm.OLS(y_feat, X_feat).fit()
                                    
                                    coef = model_feat.params[feature]
                                    pval = model_feat.pvalues[feature]
                                    
                                    hte_results_cate.append({
                                        "Feature": feature,
                                        "Effect Modification (Slope)": coef,
                                        "P-value": pval,
                                        "Significant (p<0.05)": "Yes" if pval < 0.05 else "No"
                                    })
                                except Exception as e:
                                    continue
                                
                                progress_bar_cate.progress((i + 1) / len(valid_covariates_cate))
                            
                            progress_bar_cate.empty()
                            
                            if hte_results_cate:
                                hte_df_cate = pd.DataFrame(hte_results_cate).sort_values("P-value")
                                st.dataframe(hte_df_cate.style.format({
                                    "Effect Modification (Slope)": "{:.4f}",
                                    "P-value": "{:.4f}"
                                }))
                                
                                # Highlight significant findings
                                sig_features_cate = hte_df_cate[hte_df_cate["P-value"] < 0.05]
                                if not sig_features_cate.empty:
                                    best_feat_cate = sig_features_cate.iloc[0]["Feature"]
                                    st.success(f"Found significant heterogeneity! The treatment effect is most strongly modified by **{best_feat_cate}**.")
                                else:
                                    st.info("No significant heterogeneity found across the analyzed features.")
                            else:
                                st.warning("Could not compute heterogeneity for any feature.")
                        

                except Exception as e:
                    st.error(f"Error calculating ITEs: {e}")
                    df_results = df.copy()

            elif estimation_method in ["Linear/Logistic Regression (OLS/Logit)", "Difference-in-Differences (DiD)"]:
                st.markdown("Analyze how the treatment effect varies across different subgroups.")
                
                analyze_hte = True # Always enabled by default
                
                if analyze_hte:
                    # Allow heterogeneity analysis on any column (except treatment/outcome/time)
                    valid_covariates = [c for c in df.columns if c not in [treatment, outcome, time_period, 'Treatment_Encoded', 'DiD_Interaction', 'DiD_Interaction', 'HTE_Interaction']]
                    
                    if not valid_covariates:
                        st.warning("No covariates available for heterogeneity analysis.")
                        hte_feature = None
                        hte_results = []
                    else:
                        st.markdown("**Analyzing heterogeneity for all available features...**")
                        hte_results = []
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        
                        for i, feature in enumerate(valid_covariates):
                            try:
                                if estimation_method == "Linear/Logistic Regression (OLS/Logit)":
                                    # Model: Y ~ T + X + T*X + Confounders
                                    df['HTE_Interaction'] = df[treatment] * df[feature]
                                    X_hte = df[[treatment, feature, 'HTE_Interaction']]
                                    other_confounders = [c for c in confounders if c != feature]
                                    if other_confounders:
                                        X_hte = pd.concat([X_hte, df[other_confounders]], axis=1)
                                    X_hte = sm.add_constant(X_hte)
                                    y_hte = df[outcome]
                                    hte_model = sm.OLS(y_hte, X_hte).fit()
                                    
                                    coef = hte_model.params['HTE_Interaction']
                                    pval = hte_model.pvalues['HTE_Interaction']
                                    
                                elif estimation_method == "Difference-in-Differences (DiD)":
                                    # Model: Y ~ T + Post + T*Post + X + T*X + Post*X + T*Post*X + Confounders
                                    df['T_X'] = df[treatment] * df[feature]
                                    df['Post_X'] = df[time_period] * df[feature]
                                    df['Triple_Interaction'] = df['DiD_Interaction'] * df[feature]
                                    
                                    X_hte = df[[treatment, time_period, 'DiD_Interaction', feature, 'T_X', 'Post_X', 'Triple_Interaction']]
                                    other_confounders = [c for c in confounders if c != feature]
                                    if other_confounders:
                                        X_hte = pd.concat([X_hte, df[other_confounders]], axis=1)
                                    X_hte = sm.add_constant(X_hte)
                                    y_hte = df[outcome]
                                    hte_model = sm.OLS(y_hte, X_hte).fit()
                                    
                                    coef = hte_model.params['Triple_Interaction']
                                    pval = hte_model.pvalues['Triple_Interaction']
                                
                                hte_results.append({
                                    "Feature": feature,
                                    "Interaction Coefficient": coef,
                                    "P-value": pval,
                                    "Significant (p<0.05)": "Yes" if pval < 0.05 else "No"
                                })
                            except Exception as e:
                                # Skip if error (e.g. collinearity)
                                continue
                            
                            progress_bar.progress((i + 1) / len(valid_covariates))
                        
                        progress_bar.empty()
                        
                        if hte_results:
                            hte_df = pd.DataFrame(hte_results).sort_values("P-value")
                            st.dataframe(hte_df.style.format({
                                "Interaction Coefficient": "{:.4f}",
                                "P-value": "{:.4f}"
                            }))
                            
                            # Highlight significant findings
                            sig_features = hte_df[hte_df["P-value"] < 0.05]
                            if not sig_features.empty:
                                best_feat = sig_features.iloc[0]["Feature"]
                                st.success(f"Found significant heterogeneity! The treatment effect varies most significantly by **{best_feat}**.")
                                hte_feature = best_feat # Set hte_feature for script export
                            else:
                                st.info("No significant heterogeneity found across the analyzed features.")
                                hte_feature = None # No single best feature
                        else:
                            st.warning("Could not compute heterogeneity for any feature.")

                df_results = df.copy()

            else:
                st.info(f"Individual Treatment Effects are not directly available for {estimation_method} in this view.")
                df_results = df.copy()

            csv = df_results.to_csv(index=False).encode('utf-8')


            # --- Step 4: Refute ---
            # Refutation (Step 4)
            st.subheader("4. Refutation")
            
            if estimation_method == "Difference-in-Differences (DiD)" or (is_binary_outcome and use_logit):
                st.warning("Refutation tests are not currently supported for this method/configuration (Manual Implementation).")
            else:
                st.markdown("##### 📖 Methodology: Robustness & Refutation — *[Sharma & Kiciman (2020)](https://arxiv.org/abs/2011.04216)*")
                st.markdown("""
                Robustness checks are essential to validate that the causal estimate is not a result of chance or model misspecification. 
                We implement the refutation framework proposed by **[Sharma & Kiciman (2020)](https://arxiv.org/abs/2011.04216)** in the `DoWhy` library.
                """)
                
                st.markdown("**1. Random Common Cause Test**")
                st.markdown("We add a random variable $W_{random}$ as a common cause to the dataset. Since $W_{random}$ is independent of the true process, the new estimate should not change significantly.")
                st.latex(r"ATE_{new} \approx ATE_{original}")
                
                st.markdown("**2. Placebo Treatment Refuter**")
                st.markdown("Based on the 'Placebo Test' framework popularized in econometrics by **[Angrist & Pischke (2009)](https://www.researchgate.net/publication/51992844_Mostly_Harmless_Econometrics_An_Empiricist's_Companion)**, we replace the true treatment variable $T$ with an independent random variable $T_{placebo}$. Since the placebo treatment is random, it should have no effect on the outcome.")
                st.latex(r"ATE_{placebo} \approx 0")

                try:
                    with st.spinner("Running Refutation Tests..."):
                        # 1. Random Common Cause
                        refute_results_rcc = model.refute_estimate(
                            identified_estimand,
                            estimate,
                            method_name="random_common_cause"
                        )
                        
                        # 2. Placebo Treatment
                        refute_results_placebo = model.refute_estimate(
                            identified_estimand,
                            estimate,
                            method_name="placebo_treatment_refuter",
                            placebo_type="permute"
                        )
                    
                    # Display Results: Random Common Cause
                    st.write("**Test 1: Add Random Common Cause**")
                    st.write(f"Original Effect: {refute_results_rcc.estimated_effect:.2f}")
                    st.write(f"New Effect: {refute_results_rcc.new_effect:.2f}")
                    st.write(f"P-value: {refute_results_rcc.refutation_result['p_value']:.2f}")
                    
                    if refute_results_rcc.refutation_result['p_value'] > 0.05:
                         st.success("✅ Random Common Cause: Passed (Estimate is stable).")
                    else:
                         st.warning("⚠️ Random Common Cause: Warning (Estimate might be sensitive).")

                    st.divider()

                    # Display Results: Placebo Treatment
                    st.write("**Test 2: Placebo Treatment**")
                    st.markdown("Replaces the treatment with a random variable. The new effect should be close to 0.")
                    st.write(f"Original Effect: {refute_results_placebo.estimated_effect:.2f}")
                    st.write(f"New Effect (Placebo): {refute_results_placebo.new_effect:.2f}")
                    st.write(f"P-value: {refute_results_placebo.refutation_result['p_value']:.2f}")
                    
                    # For Placebo, we want the new effect to be close to 0, so p-value should ideally be > 0.05 
                    # (null hypothesis: effect is 0). 
                    # DoWhy's p-value here tests if the new effect is significantly different from 0.
                    # Wait, DoWhy's p-value interpretation depends on the test.
                    # For placebo, we want the effect to be insignificant (p > 0.05).
                    
                    if refute_results_placebo.refutation_result['p_value'] > 0.05:
                        st.success("✅ Placebo Treatment: Passed (Effect is indistinguishable from 0).")
                    else:
                        st.warning("⚠️ Placebo Treatment: Warning (Placebo effect is significant).")

                except Exception as e:
                    st.error(f"Refutation failed: {e}")



            # --- Step 5: Export Data and Script ---
            st.subheader("5. Export Data and Script")
            
            st.download_button(
                label="Download Data with Results as CSV",
                data=csv,
                file_name='causal_analysis_results.csv',
                mime='text/csv',
            )
        


            # Generate the script
            # We need to pass all current state variables
            # Note: Some variables like 'percentile' are only defined if winsorize_enable is True
            # We'll use defaults or current values.
        

            # Debug: Check if variables are available
            # st.write(f"Debug: data_source={data_source}")
            # st.write(f"Debug: control_val={control_val if 'control_val' in locals() else 'Not Found'}")
            
            try:
                # Prepare Preprocessing Params from State
                p_impute_enable = st.session_state.get('impute_enable', False)
                p_num_method = st.session_state.get('num_impute_method', "Mean")
                p_num_val = st.session_state.get('num_custom_val', 0.0)
                p_cat_method = st.session_state.get('cat_impute_method', "Mode")
                p_cat_val = st.session_state.get('cat_custom_val', "Missing")
                p_wins_enable = st.session_state.get('winsorize_enable', False)
                p_wins_cols = st.session_state.get('winsorize_cols', [])
                p_percentile = st.session_state.get('percentile', 0.05)
                p_log_cols = st.session_state.get('log_transform_cols', [])
                p_std_cols = st.session_state.get('standardize_cols', [])

                # Prepare TS Params
                ts_params = {
                    'enabled': enable_ts_analysis if 'enable_ts_analysis' in locals() else False,
                    'date_col': ts_date_col if 'ts_date_col' in locals() else None,
                    'freq': ts_freq if 'ts_freq' in locals() else None,
                    'is_bsts_demo': st.session_state.get('sim_type') == "BSTS Demo"
                }
                
                script = generate_script(
                    data_source=data_source,
                    treatment=treatment,
                    outcome=outcome,
                    confounders=confounders,
                    time_period=time_period,
                    estimation_method=estimation_method,
                    impute_enable=p_impute_enable,
                    num_impute_method=p_num_method if p_impute_enable else None,
                    num_custom_val=p_num_val if p_impute_enable else 0.0,
                    cat_impute_method=p_cat_method if p_impute_enable else None,
                    cat_custom_val=p_cat_val if p_impute_enable else "Missing",
                    winsorize_enable=p_wins_enable,
                    winsorize_cols=p_wins_cols if p_wins_enable else [],
                    percentile=p_percentile if p_wins_enable else 0.05,
                    log_transform_cols=p_log_cols,
                    standardize_cols=p_std_cols,
                    n_iterations=n_iterations,
                    control_val=control_val if 'control_val' in locals() else None,
                    treat_val=treat_val if 'treat_val' in locals() else None,
                    hte_features=valid_covariates if 'valid_covariates' in locals() else (valid_covariates_cate if 'valid_covariates_cate' in locals() else None),
                    use_logit=use_logit,
                    bucketing_ops=st.session_state.bucketing_ops,
                    filtering_ops=st.session_state.filtering_ops,
                    ts_params=ts_params
                )

            
                st.download_button(
                    label="Download Python Script",
                    data=script,
                    file_name="causal_analysis.py",
                    mime="text/x-python"
                )

                with st.expander("View Generated Python Script"):
                    st.code(script, language='python')
            except Exception as e:
                st.error(f"Error generating script: {e}")


# ==========================================
# TAB 5: AI Assistant
# ==========================================
with tab_chat:
    st.header("💬 Causal Inference Assistant")
    
    # --- API Assistant Logic ---
    SYSTEM_KEY = None
    try:
        SYSTEM_KEY = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        import os
        SYSTEM_KEY = os.getenv("GOOGLE_API_KEY")

    requests_left = MAX_API_CALLS - st.session_state.api_call_count
    is_free_tier_active = requests_left > 0
    try:
        SYSTEM_KEY = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        import os
        SYSTEM_KEY = os.getenv("GOOGLE_API_KEY")

    # Display Status
    if is_free_tier_active:
        st.caption(f"🎁 Free Trial: {requests_left}/{MAX_API_CALLS} requests remaining.")
    else:
        if "user_api_key" in st.session_state:
             st.caption("🔑 Using your provided API Key.")
        else:
             st.caption("🛑 Free limit reached.")

    # 1. Initialize History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Causal Inference Assistant. I can help you understand your data, suggest causal models, or explain how to use this app. What would you like to do?"}
        ]

    # 2. Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # 3. Quick Start Questions
    if st.session_state.messages[-1]["role"] != "user":
        st.caption("Common Questions:")
        bq1, bq2, bq3, bq4 = st.columns(4)
        if bq1.button("📊 Summarize Data"):
            st.session_state.messages.append({"role": "user", "content": "Can you summarize the current dataset and its key features?"})
            st.rerun()
        if bq2.button("📈 Visualize Data"):
            st.session_state.messages.append({"role": "user", "content": "Can you help me visualize the relationship between the treatment and outcome?"})
            st.rerun()
        if bq3.button("💡 Suggest Method"):
            st.session_state.messages.append({"role": "user", "content": "Based on this data, what causal estimation method would you recommend?"})
            st.rerun()
        if bq4.button("❓ Explain LinearDML"):
            st.session_state.messages.append({"role": "user", "content": "Explain how Linear Double Machine Learning (LinearDML) works in simple terms."})
            st.rerun()

    # 4. Handle User Input
    if prompt := st.chat_input("Ask about your data or causal inference..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # 5. Generate Response
    if st.session_state.messages[-1]["role"] == "user":
        
        # --- API Key Selection Logic ---
        active_key = None
        increment_usage = False
        
        if is_free_tier_active and SYSTEM_KEY:
            active_key = SYSTEM_KEY
            increment_usage = True
        elif "user_api_key" in st.session_state:
            active_key = st.session_state.user_api_key
        else:
            # Need Key
            with st.chat_message("assistant"):
                st.warning(f"🛑 You have used your {MAX_API_CALLS} free requests.")
                st.write("To continue exploring, please enter your own Google API Key.")
                st.markdown("[Get a key here](https://aistudio.google.com/app/apikey)")
                
                user_key = st.text_input("Enter Google API Key:", type="password", key="api_key_input")
                if user_key:
                    st.session_state.user_api_key = user_key
                    st.rerun()
                st.stop() # Wait for input

        # Configure API with the selected key
        # (Implicitly handled by passing active_key to utils)
            
        with st.chat_message("user"):
            st.markdown(st.session_state.messages[-1]["content"])
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Gather Context
            data_context = chatbot_utils.get_data_context(st.session_state.df)
            app_context = chatbot_utils.get_app_context()
            
            # Stream Response
            try:
                full_response = ""
                # Updating to gemini-3-flash-preview (Flash 2.0/3.0 Preview) as requested
                model_name = "gemini-3-flash-preview"
                
                response_stream = chatbot_utils.chat_stream(
                    model_name=model_name,
                    messages=st.session_state.messages,
                    data_context=data_context, 
                    app_context=app_context,
                    api_key=active_key
                )
                
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                        
                message_placeholder.markdown(full_response)
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                if increment_usage:
                    st.session_state.api_call_count += 1 # Global counter increment

                st.rerun() # Rerun to show buttons again
                
            except Exception as e:
                st.error(f"Error generating response: {e}") 
                st.info("Note: If using a custom key, ensure it has access to the selected model.")
