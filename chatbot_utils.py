from google import genai
from google.genai import types
import streamlit as st
import pandas as pd
import os

def get_api_key():
    """Retrieves the Gemini API key from Streamlit secrets or environment variables."""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        return os.getenv("GOOGLE_API_KEY")

def get_data_context(df):
    """
    Generates a concise summary of the dataframe for the LLM.
    Includes column names, types, missing values, and basic stats.
    """
    if df is None or df.empty:
        return "No data loaded."
    
    buffer = []
    buffer.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    buffer.append("\nColumns & Data Types:")
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        buffer.append(f"- {col} ({dtype}): {missing} missing values")
        
        # Add a snippet of stats for numeric, or unique count for categorical
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            buffer.append(f"  Stats: Mean={desc['mean']:.2f}, Min={desc['min']:.2f}, Max={desc['max']:.2f}")
        else:
            unique_count = df[col].nunique()
            buffer.append(f"  Unique Values: {unique_count}")
            # If few unique values, list them
            if unique_count < 10:
                vals = df[col].unique().tolist()
                buffer.append(f"  Values: {vals}")

    return "\n".join(buffer)

def get_app_context():
    """
    Returns text context about the application's capabilities.
    This helps the AI understand what the user can do in the UI.
    """
    return """
    Application Structure:
    - Tab 1: 📘 User Guide & Methodology (Includes "Results Interpretation" tables for all methods).
    - Tab 2: 📊 Exploratory Analysis (Profiling, Imputation, Winsorization, Log/Std Transforms).
    - Tab 3: 🔍 Observational Analysis (Cross-sectional data, User-level analysis).
    - Tab 4: 📈 Quasi-Experimental Analysis (Longitudinal/Panel data, Time-series analysis).
    - Tab 5: 💬 AI Assistant (You are here).

    Detailed Capabilities:
    1. Observational Analysis (Tab 3):
       - Methods: OLS, Logit, PSM, IPTW, LinearDML, CausalForestDML, Meta-Learners (S/T).
       - Validations: Refutation tests (Random Common Cause, Placebo Treatment).
       - Features: ATE estimation, HTE (Heterogeneity) analysis.
    2. Quasi-Experimental Analysis (Tab 4):
       - Methods: Difference-in-Differences (DiD), Interrupted/Bayesian Time Series (ITS/BSTS), GeoLift (Synthetic Control), and CausalPy (Bayesian Synthetic Control).
       - When to use: When the intervention is not randomly assigned at the user level, but rather implemented over time or separated across specific geographic regions (GeoLift/CausalPy).
       - Impact Estimation: ITS/BSTS, GeoLift, and CausalPy return metric scorecards including ATE, Cumulative Lift, and Relative Lift with Confidence/Credible Intervals.
       - CausalPy: Pure-Python Bayesian Synthetic Control using PyMC. Provides proper two-sided HDI credible intervals, posterior probability of effect, and relative lift. Recommended over GeoLift for robust uncertainty quantification without R dependencies.
       - Support: Panel Data (Synthetic Control style) or Aggregate Time Series.
    3. Reproducibility:
       - Every analysis in Tab 3 and Tab 4 has a "Data & Script Export" section to download CSV results and a Python reproduction script.
    """

import causal_utils
import numpy as np

def run_observational_analysis(treatment: str, outcome: str, confounders: list[str], method_name: str, is_binary_outcome: bool = False) -> str:
    """
    Runs an observational causal analysis (like OLS, PSM, LinearDML) on the currently loaded data.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded in the application."
    df = st.session_state.df
    
    # Map friendly names to actual method names expected by causal_utils
    # Valid methods: "Linear/Logistic Regression (OLS/Logit)", "Propensity Score Matching (PSM)", "Inverse Propensity Weighting (IPTW)", "Linear Double Machine Learning (LinearDML)", "Generalized Random Forests (CausalForestDML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner"
    valid_methods = [
        "Linear/Logistic Regression (OLS/Logit)", "Propensity Score Matching (PSM)",
        "Inverse Propensity Weighting (IPTW)", "Linear Double Machine Learning (LinearDML)",
        "Generalized Random Forests (CausalForestDML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner"
    ]
    # Simple fuzzy match or let LLM pass exact strings
    if method_name not in valid_methods:
        # try to fallback
        if "OLS" in method_name or "Linear" in method_name and "DML" not in method_name: method_name = "Linear/Logistic Regression (OLS/Logit)"
        elif "PSM" in method_name: method_name = "Propensity Score Matching (PSM)"
        elif "IPTW" in method_name: method_name = "Inverse Propensity Weighting (IPTW)"
        elif "LinearDML" in method_name: method_name = "Linear Double Machine Learning (LinearDML)"
        elif "CausalForest" in method_name: method_name = "Generalized Random Forests (CausalForestDML)"
        elif "S-Learner" in method_name: method_name = "Meta-Learner: S-Learner"
        elif "T-Learner" in method_name: method_name = "Meta-Learner: T-Learner"
        else: return f"Error: '{method_name}' is not a valid method. Choose from {valid_methods}"

    try:
        ate, ci_lower, ci_upper = causal_utils.calculate_period_effect(df, treatment, outcome, confounders, method_name, is_binary_outcome)
        if np.isnan(ate):
            return "Estimation failed. Check if confounders are valid and variance exists."
        
        return f"Observational Analysis ({method_name}) Results:\\nATE (Average Treatment Effect): {ate:.4f}\\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
    except Exception as e:
        return f"Error running analysis: {e}"

def run_did_analysis(treatment_col: str, outcome_col: str, time_col: str, confounders: list[str], is_binary_outcome: bool = False, use_logit: bool = False) -> str:
    """
    Runs a Difference-in-Differences (DiD) Analysis manually using OLS/Logit interaction.
    Assumptions: The 'time_col' should be a binary period indicator (0=Pre, 1=Post).
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    df = st.session_state.df
    
    try:
        res = causal_utils.run_did_analysis(df, treatment_col, outcome_col, time_col, confounders, is_binary_outcome, use_logit)
        if 'error' in res:
            return f"Error: {res['error']}"
        
        output = f"DiD Analysis ({res['method']}) Results:\\n"
        output += f"Interaction Coefficient (DiD Estimate): {res['coefficient']:.4f}\\n"
        output += f"P-Value: {res['p_value']:.4f}\\n"
        output += f"95% CI: [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]\\n"
        if 'odds_ratio' in res:
             output += f"Odds Ratio: {res['odds_ratio']:.4f}\\n"
             
        return output
    except Exception as e:
        return f"Error running DiD: {e}"

def run_time_series_bsts(date_col: str, outcome_col: str, intervention_date: str, unit_col: str = None, treated_unit: str = None, use_panel: bool = False, covariates: list[str] = None) -> str:
    """
    Runs CausalImpact (BSTS) analysis on time series data. Best for single unit testing or panel data.
    intervention_date must be YYYY-MM-DD.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    df = st.session_state.df
    
    try:
        res = causal_utils.run_causal_impact_analysis(df, date_col, outcome_col, intervention_date, unit_col, treated_unit, use_panel, covariates)
        if 'error' in res:
            return f"Error processing CausalImpact: {res['error']}"
        
        output = "BSTS (CausalImpact) Analysis Results:\\n"
        output += f"P-Value: {res['p_value']:.4f}\\n"
        output += f"Average Absolute Effect (Lift): {res['ate']:.2f} (95% CI: [{res['ate_lower']:.2f}, {res['ate_upper']:.2f}])\\n"
        output += f"Relative Effect: {res['relative_effect']:.2%}\\n"
        output += f"Cumulative Absolute Effect: {res['cumulative_abs']:.2f} (95% CI: [{res['cumulative_lower']:.2f}, {res['cumulative_upper']:.2f}])"
        return output
    except Exception as e:
         return f"Error running BSTS: {e}"

def run_synthetic_control_geolift(date_col: str, geo_col: str, treated_geo: str, kpi_col: str, intervention_date: str, treatment_duration: int, covariates: list[str] = None) -> str:
    """
    Runs GeoLift (Synthetic Control) on Panel Data. Best for geographic A/B testing where a market is treated.
    intervention_date must be YYYY-MM-DD.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    df = st.session_state.df
    
    try:
        res = causal_utils.run_geolift_analysis(df, date_col, geo_col, treated_geo, kpi_col, intervention_date, treatment_duration=treatment_duration, covariates=covariates)
        if 'error' in res:
            return f"Error in GeoLift: {res['error']}"
        
        return res['summary'] # Comes properly formatted in Markdown
    except Exception as e:
        return f"Error running GeoLift: {e}"

def run_causalpy_analysis(date_col: str, geo_col: str, treated_geo: str, kpi_col: str, intervention_date: str, treatment_duration: int = 60, direction: str = "two-sided", covariates: list = None) -> str:
    """
    Runs CausalPy Bayesian Synthetic Control on Panel Data. Pure Python alternative to GeoLift with proper two-sided HDI intervals and posterior probabilities.
    intervention_date must be YYYY-MM-DD. direction can be "two-sided", "increase", or "decrease".
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    df = st.session_state.df
    
    try:
        res = causal_utils.run_causalpy_synthetic_control(df, date_col, geo_col, kpi_col, treated_geo, intervention_date, treatment_duration=treatment_duration, direction=direction, covariates=covariates)
        if 'error' in res:
            return f"Error in CausalPy: {res['error']}"
        
        metrics = res.get('metrics', {})
        output = "CausalPy (Bayesian Synthetic Control) Results:\\n"
        if metrics.get('avg_effect') is not None:
            output += f"Average Treatment Effect (ATT): {metrics['avg_effect']:.4f}\\n"
            output += f"95% HDI: [{metrics.get('avg_ci_lower', 0):.4f}, {metrics.get('avg_ci_upper', 0):.4f}]\\n"
        if metrics.get('cum_effect') is not None:
            output += f"Cumulative Impact: {metrics['cum_effect']:.4f}\\n"
        if metrics.get('rel_effect') is not None:
            output += f"Relative Lift: {metrics['rel_effect']:.2%}\\n"
        if metrics.get('prob_effect') is not None:
            output += f"Posterior Probability of Effect: {metrics['prob_effect']:.4f}\\n"
        if res.get('summary_text'):
            output += f"\\nFull Summary:\\n{res['summary_text'][:500]}"
        return output
    except Exception as e:
        return f"Error running CausalPy: {e}"


def load_synthetic_dataset(theme: str, business_function: str) -> str:
    """
    Generates a new synthetic dataset based on a given business theme (e.g. eCommerce) 
    and business function (e.g. Marketing) and loads it into the application's global state. 
    Use this tool when the user asks to generate a new dataset.
    """
    import data_generation_utils
    try:
        api_key = get_api_key()
        df, description, questions = data_generation_utils.generate_dynamic_dataset(theme, business_function, api_key)
        st.session_state.df = df
        st.session_state.raw_df = df.copy()
        st.session_state.bucketing_ops = []
        st.session_state.filtering_ops = []
        st.session_state.sim_type = "Dynamic"
        st.session_state.dataset_description = description
        st.session_state.dataset_questions = questions
        st.session_state.data_quality_summary = None
        questions_text = " ".join([f"({i+1}) {q}" for i, q in enumerate(questions)]) if questions else ""
        return f"Successfully generated a dataset with {len(df)} rows and {len(df.columns)} columns based on the '{theme}' theme and '{business_function}' function. {description} Suggested questions: {questions_text}"
    except Exception as e:
        return f"Failed to generate dataset: {str(e)}"

def chat_stream(model_name, messages, data_context, app_context, api_key):
    """
    Streams response from Gemini model using the new google.genai SDK.
    """
    # System Prompt
    system_prompt = f"""
    You are an expert Causal Inference Statistician and Data Scientist assistant embedded in the "Causal Inference Agent" application.
    
    Your goals:
    1. Help users understand the currently loaded dataset and identify the right columns for analysis.
    2. Suggest appropriate causal models based on data structure:
       - If they have cross-sectional data (one row per user), suggest **Observational Analysis (Tab 3)**.
       - If they have longitudinal/time-series data (multiple timestamps), suggest **Quasi-Experimental Analysis (Tab 4)** and refer to methods like DiD or CausalImpact.
    3. You have TOOLS available to directly run causal inference methods on the CURRENTLY LOADED dataset. When a user asks you to "Run an analysis", use your tools and provide the results. 
       - If the user's request is ambiguous about parameters (e.g. they don't specify confounders), make a reasonable Data Science assumption based on the dataset columns, or clarify with them.
    4. Explain causal concepts (ATE, HTE, Confounding, Instrumental Variables, Regression Discontinuity, etc.) and interpret the statistical results you generate using your tools.
    5. Guide them to the correct UI Tab when applicable.
    6. Mention the "Results Interpretation" guides in the **User Guide (Tab 1)** for interpreting statistical metrics like P-values, Confidence Intervals, and Relative Lift.

    Context:
    --- APP CONTEXT ---
    {app_context}
    
    --- CURRENT LOADED DATASET ---
    {data_context}
    
    Instructions:
    - Be concise and professional.
    - Use markdown (bold, tables, lists) to improve readability.
    - When suggesting variables or running tools, STRICTLY refer to the columns present in the 'CURRENT LOADED DATASET'.
    - If the user asks for code, provide Python code compatible with the libraries used (pandas, econml, dowhy, causalimpact, statsmodels, etc.).
    - You are encouraged to answer general causal inference questions, not just those related to the app's current functionality.
    """

    
    # Initialize Client
    client = genai.Client(api_key=api_key)
    
    # Prepare history for Gemini
    # Gemini 2.0/new SDK uses 'user' and 'model' roles.
    # Note: New SDK handles Chat formatting easier, but we need to convert Streamlit format.
    
    formatted_history = []
    
    # Prepend System Prompt to history or use config
    # The new SDK supports 'config' with 'system_instruction'.
    
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.7,
        tools=[run_observational_analysis, run_did_analysis, run_time_series_bsts, run_synthetic_control_geolift, run_causalpy_analysis, load_synthetic_dataset]
    )

    # Convert messages
    # Exclude the very last message which is the new prompt (we send it separately or let chat handle it?)
    # Streamlit messages include the latest user prompt usually if we appended it before calling this.
    # Let's check call site. Yes, keys are appended.
    # But for 'chats.create', we pass history (excluding the new message) and then send the new message.
    
    history_messages = messages[:-1]
    current_message = messages[-1]["content"]
    
    for msg in history_messages:
        role = "user" if msg["role"] == "user" else "model"
        
        # Streamlit message formatting handling for tool calls
        # The new SDK handles function calling via normal parts.
        # Simple extraction of text
        formatted_history.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["content"])]
        ))
    
    chat = client.chats.create(
        model=model_name,
        history=formatted_history,
        config=config
    )
    
    response_stream = chat.send_message_stream(current_message)
    
    for chunk in response_stream:
        # We can just yield the chunk text directly for Streamlit, the SDK handles function execution internally
        # if the python functions are passed.
        yield chunk
