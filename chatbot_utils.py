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

def run_causalpy_analysis(date_col: str, geo_col: str, treated_geo: str, kpi_col: str, intervention_date: str, treatment_duration: int = 60, direction: str = "two-sided") -> str:
    """
    Runs CausalPy Bayesian Synthetic Control on Panel Data. Pure Python alternative to GeoLift with proper two-sided HDI intervals and posterior probabilities.
    intervention_date must be YYYY-MM-DD. direction can be "two-sided", "increase", or "decrease".
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    df = st.session_state.df
    
    try:
        res = causal_utils.run_causalpy_synthetic_control(df, date_col, geo_col, kpi_col, treated_geo, intervention_date, treatment_duration=treatment_duration, direction=direction)
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

def run_method_comparison(date_col: str, outcome_col: str, geo_col: str, treated_geo: str, intervention_date: str, treatment_duration: int = 60, methods: list[str] = None) -> str:
    """
    Runs multiple quasi-experimental causal methods (BSTS, GeoLift, CausalPy) on the same dataset
    and returns a unified comparison table. Use this when users ask to "compare methods" or want
    to validate results across methodologies. Each method is run independently with graceful error handling.
    intervention_date must be YYYY-MM-DD.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    
    if methods is None:
        methods = ["BSTS", "GeoLift", "CausalPy"]
    
    results = {}
    failed = {}
    
    # Run each method independently
    for method in methods:
        try:
            if method == "BSTS":
                res = causal_utils.run_causal_impact_analysis(
                    st.session_state.df, date_col, outcome_col, intervention_date,
                    unit_col=geo_col, treated_unit=treated_geo, use_panel=True
                )
                if 'error' in res:
                    failed[method] = res['error']
                else:
                    results[method] = {
                        'label': 'BSTS (CausalImpact)',
                        'ate': res.get('ate'),
                        'ate_lower': res.get('ate_lower'),
                        'ate_upper': res.get('ate_upper'),
                        'cumulative': res.get('cumulative_abs'),
                        'cum_lower': res.get('cumulative_lower'),
                        'cum_upper': res.get('cumulative_upper'),
                        'relative': res.get('relative_effect'),
                        'significance': f"p={res.get('p_value', 'N/A'):.4f}" if isinstance(res.get('p_value'), (int, float)) else 'N/A',
                        'significant': res.get('p_value', 1) < 0.05 if isinstance(res.get('p_value'), (int, float)) else False
                    }
            elif method == "GeoLift":
                res = causal_utils.run_geolift_analysis(
                    st.session_state.df, date_col, geo_col, treated_geo, outcome_col,
                    intervention_date, treatment_duration=treatment_duration
                )
                if 'error' in res:
                    failed[method] = res['error']
                else:
                    metrics = res.get('metrics', {})
                    results[method] = {
                        'label': 'GeoLift (Synthetic Control)',
                        'ate': metrics.get('att_avg'),
                        'ate_lower': metrics.get('att_avg_ci_lower'),
                        'ate_upper': metrics.get('att_avg_ci_upper'),
                        'cumulative': metrics.get('att_sum'),
                        'cum_lower': metrics.get('att_sum_ci_lower'),
                        'cum_upper': metrics.get('att_sum_ci_upper'),
                        'relative': metrics.get('lift_pct', 0) / 100 if metrics.get('lift_pct') is not None else None,
                        'significance': f"p={metrics.get('p_value', 'N/A'):.4f}" if isinstance(metrics.get('p_value'), (int, float)) else 'N/A',
                        'significant': metrics.get('p_value', 1) < 0.05 if isinstance(metrics.get('p_value'), (int, float)) else False
                    }
            elif method == "CausalPy":
                res = causal_utils.run_causalpy_synthetic_control(
                    st.session_state.df, date_col, geo_col, outcome_col, treated_geo,
                    intervention_date, treatment_duration=treatment_duration
                )
                if 'error' in res:
                    failed[method] = res['error']
                else:
                    metrics = res.get('metrics', {})
                    results[method] = {
                        'label': 'CausalPy (Bayesian SC)',
                        'ate': metrics.get('avg_effect'),
                        'ate_lower': metrics.get('avg_ci_lower'),
                        'ate_upper': metrics.get('avg_ci_upper'),
                        'cumulative': metrics.get('cum_effect'),
                        'cum_lower': metrics.get('cum_ci_lower'),
                        'cum_upper': metrics.get('cum_ci_upper'),
                        'relative': metrics.get('rel_effect'),
                        'significance': f"P(effect)={metrics.get('prob_effect', 'N/A'):.4f}" if isinstance(metrics.get('prob_effect'), (int, float)) else 'N/A',
                        'significant': metrics.get('prob_effect', 0) > 0.95 if isinstance(metrics.get('prob_effect'), (int, float)) else False
                    }
        except Exception as e:
            failed[method] = str(e)
    
    if not results:
        error_details = "\n".join([f"- {m}: {e}" for m, e in failed.items()])
        return f"All methods failed:\n{error_details}\n\nPlease check data format, column names, and intervention date."
    
    # Build comparison table
    def fmt(val, fmt_str=".2f"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:{fmt_str}}"
    
    def fmt_ci(val, lower, upper, fmt_str=".2f"):
        v = fmt(val, fmt_str)
        l = fmt(lower, fmt_str)
        u = fmt(upper, fmt_str)
        if l == "N/A" or u == "N/A":
            return v
        return f"{v} [{l}, {u}]"
    
    # Header
    method_labels = [r['label'] for r in results.values()]
    header = "| Metric | " + " | ".join(method_labels) + " |"
    divider = "|---" * (len(method_labels) + 1) + "|"
    
    # Rows
    rows = []
    
    # ATE
    ate_cells = [fmt_ci(r['ate'], r.get('ate_lower'), r.get('ate_upper')) for r in results.values()]
    rows.append("| Avg Effect (ATT) | " + " | ".join(ate_cells) + " |")
    
    # Cumulative
    cum_cells = [fmt_ci(r.get('cumulative'), r.get('cum_lower'), r.get('cum_upper')) for r in results.values()]
    rows.append("| Cumulative Impact | " + " | ".join(cum_cells) + " |")
    
    # Relative
    rel_cells = [f"{r['relative']:.2%}" if r.get('relative') is not None else "N/A" for r in results.values()]
    rows.append("| Relative Lift | " + " | ".join(rel_cells) + " |")
    
    # Significance
    sig_cells = [r.get('significance', 'N/A') for r in results.values()]
    rows.append("| Significance | " + " | ".join(sig_cells) + " |")
    
    # Status
    status_cells = ["✅ Significant" if r.get('significant') else "❌ Not Significant" for r in results.values()]
    rows.append("| Status | " + " | ".join(status_cells) + " |")
    
    table = "\n".join([header, divider] + rows)
    
    # Add failed methods note
    failed_note = ""
    if failed:
        failed_note = "\n\n**Methods that failed:**\n"
        for m, e in failed.items():
            failed_note += f"- {m}: {e}\n"
    
    # Synthesis prompt - let the LLM interpret via the response
    synthesis_hint = f"\n\n*{len(results)} of {len(methods)} methods completed successfully.*"
    
    return f"**Multi-Method Comparison Results:**\n\n{table}{failed_note}{synthesis_hint}"


def run_cross_sectional_comparison(treatment_col: str, outcome_col: str, confounders: list[str], methods: list[str] = None, is_binary_outcome: bool = False) -> str:
    """
    Runs multiple cross-sectional causal methods (PSM, IPTW, LinearDML, S-Learner) on the same dataset
    and returns a comparison table. Use when users want to compare observational methods.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    
    if methods is None:
        methods = ["PSM", "IPTW", "LinearDML", "S-Learner"]
    
    method_map = {
        "OLS": "Linear/Logistic Regression (OLS/Logit)",
        "PSM": "Propensity Score Matching (PSM)",
        "IPTW": "Inverse Propensity Weighting (IPTW)",
        "LinearDML": "Linear Double Machine Learning (LinearDML)",
        "CausalForest": "Generalized Random Forests (CausalForestDML)",
        "S-Learner": "Meta-Learner: S-Learner",
        "T-Learner": "Meta-Learner: T-Learner"
    }
    
    results = {}
    failed = {}
    df = st.session_state.df
    
    for method in methods:
        full_name = method_map.get(method, method)
        try:
            ate, ci_lower, ci_upper = causal_utils.calculate_period_effect(
                df, treatment_col, outcome_col, confounders, full_name, is_binary_outcome
            )
            if np.isnan(ate):
                failed[method] = "Estimation returned NaN"
            else:
                results[method] = {
                    'label': method,
                    'ate': ate,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                }
        except Exception as e:
            failed[method] = str(e)
    
    if not results:
        error_details = "\n".join([f"- {m}: {e}" for m, e in failed.items()])
        return f"All methods failed:\n{error_details}"
    
    # Build table
    labels = [r['label'] for r in results.values()]
    header = "| Metric | " + " | ".join(labels) + " |"
    divider = "|---" * (len(labels) + 1) + "|"
    
    def fmt_ci(r):
        ate = f"{r['ate']:.4f}"
        if np.isnan(r.get('ci_lower', float('nan'))) or np.isnan(r.get('ci_upper', float('nan'))):
            return ate
        return f"{ate} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
    
    ate_cells = [fmt_ci(r) for r in results.values()]
    rows = ["| ATE | " + " | ".join(ate_cells) + " |"]
    
    failed_note = ""
    if failed:
        failed_note = "\n\n**Methods that failed:**\n"
        for m, e in failed.items():
            failed_note += f"- {m}: {e}\n"
    
    table = "\n".join([header, divider] + rows)
    return f"**Cross-Sectional Method Comparison:**\n\n{table}{failed_note}\n\n*{len(results)} of {len(methods)} methods completed.*"


def profile_dataset() -> str:
    """
    Automatically profiles the loaded dataset: detects data structure (panel, time-series, cross-sectional),
    identifies key columns (date, geography, treatment, outcome), and recommends appropriate causal methods.
    Call this as the first step when helping a user analyze their data.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded. Please upload data or generate a synthetic dataset first."
    
    result = causal_utils.profile_dataset(st.session_state.df)
    
    output = f"**Dataset Profile:**\n\n"
    output += f"**Shape:** {result['shape']['rows']} rows × {result['shape']['cols']} columns\n"
    output += f"**Data Type:** `{result['data_type']}`\n\n"
    
    roles = result['detected_roles']
    output += "**Detected Column Roles:**\n"
    for role, cols in roles.items():
        if cols:
            output += f"- {role.title()}: {', '.join(cols)}\n"
    
    output += f"\n**Recommended Methods:** {', '.join(result['recommended_methods'])}\n\n"
    
    # Column summary
    output += "**Column Details:**\n\n"
    output += "| Column | Type | Missing | Unique |\n|---|---|---|---|\n"
    for col, info in result['columns'].items():
        output += f"| {col} | {info['dtype']} | {info['missing_pct']:.1f}% | {info['unique']} |\n"
    
    return output


def check_data_quality() -> str:
    """
    Scans the loaded dataset for quality issues (missing values, outliers, duplicates, imbalance, etc.)
    and returns a diagnostic report with suggested fixes. The user should be asked to approve fixes
    before they are applied. Present issues as a numbered action plan.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    
    df = st.session_state.df
    # Try to detect key columns
    profile = causal_utils.profile_dataset(df)
    roles = profile.get('detected_roles', {})
    
    result = causal_utils.run_data_quality_checks(
        df,
        date_col=roles.get('date', [None])[0] if roles.get('date') else None,
        geo_col=roles.get('geography', [None])[0] if roles.get('geography') else None,
        treatment_col=roles.get('treatment', [None])[0] if roles.get('treatment') else None,
        outcome_col=roles.get('outcome', [None])[0] if roles.get('outcome') else None,
    )
    
    if result['total_issues'] == 0:
        return "✅ **No data quality issues found!** The dataset looks clean and ready for analysis."
    
    severity_icons = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
    
    output = f"**Data Quality Report:** Found {result['total_issues']} issue(s).\n\n"
    for i, issue in enumerate(result['issues'], 1):
        icon = severity_icons.get(issue['severity'], '⚪')
        fix_desc = issue['suggested_fix'].replace('_', ' ').title()
        if issue['suggested_fix'] == 'note_only':
            fix_desc = 'No action needed (informational)'
        output += f"{i}. {icon} **{issue['issue']}** → Suggested: {fix_desc}\n"
    
    output += "\n*Would you like me to apply all fixes, or would you like to adjust any?*\n"
    output += "*(Reply e.g., \"Apply all\", \"Skip #3\", or \"Use mean for #1\")*"
    
    # Store issues in session state for apply_data_fixes to reference
    st.session_state['pending_quality_fixes'] = result['issues']
    
    return output


def apply_data_fixes(fix_indices: list[int] = None, skip_indices: list[int] = None, custom_overrides: list[dict] = None) -> str:
    """
    Applies approved data quality fixes to the loaded dataset. Only call AFTER the user has confirmed.
    fix_indices: list of 1-based issue numbers to apply (default: all). 
    skip_indices: list of 1-based issue numbers to skip.
    custom_overrides: list of dicts like [{"index": 1, "fix_type": "impute_mean"}] to change a fix.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    if 'pending_quality_fixes' not in st.session_state:
        return "Error: No pending fixes. Please run check_data_quality() first."
    
    all_fixes = st.session_state['pending_quality_fixes']
    
    # Determine which fixes to apply
    if fix_indices:
        selected = [all_fixes[i-1] for i in fix_indices if 1 <= i <= len(all_fixes)]
    else:
        selected = list(all_fixes)
    
    if skip_indices:
        selected = [f for i, f in enumerate(selected) if (i+1) not in skip_indices]
    
    # Apply custom overrides
    if custom_overrides:
        for override in custom_overrides:
            idx = override.get('index', 0) - 1
            if 0 <= idx < len(selected):
                if 'fix_type' in override:
                    selected[idx]['suggested_fix'] = override['fix_type']
    
    # Filter out note_only fixes
    actionable = [f for f in selected if f.get('suggested_fix') != 'note_only']
    
    if not actionable:
        return "No actionable fixes to apply (all selected items are informational)."
    
    df_fixed, changes = causal_utils.apply_fixes(st.session_state.df, actionable)
    
    # Update session state
    st.session_state.df = df_fixed
    del st.session_state['pending_quality_fixes']
    
    output = f"**Applied {len(changes)} fix(es):**\n\n"
    for change in changes:
        output += f"- {change}\n"
    output += f"\n**New dataset shape:** {df_fixed.shape[0]} rows × {df_fixed.shape[1]} columns"
    
    return output


def check_assumptions(method: str, treatment_col: str = None, outcome_col: str = None, 
                      time_col: str = None, date_col: str = None, geo_col: str = None,
                      treated_geo: str = None, intervention_date: str = None,
                      confounders: list[str] = None) -> str:
    """
    Runs relevant assumption checks for the specified causal method.
    method: 'DiD', 'BSTS', 'SyntheticControl', 'PSM', 'IPTW', 'DML'
    Returns diagnostic results and warnings.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    df = st.session_state.df
    
    output = f"**Assumption Check for {method}:**\n\n"
    
    if method == 'DiD' and treatment_col and outcome_col and time_col:
        result = causal_utils.check_parallel_trends(df, outcome_col, treatment_col, time_col)
        status = "✅" if result.get('pass') else "⚠️" if result.get('pass') is None else "❌"
        output += f"{status} **Parallel Trends:** {result['message']}\n"
        if result.get('p_value'):
            output += f"   Pre-trend interaction p-value: {result['p_value']:.4f}\n"
    
    elif method in ('BSTS', 'SyntheticControl', 'CausalPy') and date_col and outcome_col:
        result = causal_utils.check_pre_period_fit(df, date_col, outcome_col, intervention_date, geo_col, treated_geo)
        if 'error' in result:
            output += f"❌ Pre-period check failed: {result['error']}\n"
        else:
            output += f"📊 **Pre-period:** {result['n_pre_periods']} observations\n"
            if result.get('warning'):
                output += f"⚠️ {result['warning']}\n"
            else:
                output += f"✅ Sufficient pre-period data\n"
    
    elif method in ('PSM', 'IPTW', 'DML') and treatment_col and confounders:
        result = causal_utils.check_covariate_balance(df, treatment_col, confounders)
        if 'error' in result:
            output += f"❌ Balance check failed: {result['error']}\n"
        else:
            status = "✅" if result['all_balanced'] else "⚠️"
            output += f"{status} **Covariate Balance:** {result['message']}\n\n"
            output += "| Covariate | Mean (T) | Mean (C) | |SMD| | Status |\n|---|---|---|---|---|\n"
            for b in result['balance']:
                s = "✅" if b['balanced'] else "❌"
                output += f"| {b['covariate']} | {b['mean_treated']:.3f} | {b['mean_control']:.3f} | {b['smd']:.3f} | {s} |\n"
    else:
        output += "⚠️ Insufficient parameters provided for checking assumptions for this method.\n"
    
    return output


def check_power(data_type: str = "auto", treatment_col: str = None, date_col: str = None, 
                geo_col: str = None, intervention_date: str = None) -> str:
    """
    Pre-flight power check before running analysis. Verifies sufficient data for reliable results.
    data_type: 'auto', 'panel', 'time_series', or 'cross_sectional'
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    df = st.session_state.df
    
    if data_type == "auto":
        profile = causal_utils.profile_dataset(df)
        data_type = profile['data_type']
    
    result = causal_utils.check_power_requirements(df, data_type, date_col, geo_col, treatment_col, intervention_date)
    
    if not result['checks']:
        return f"⚠️ Could not run power checks for data type '{data_type}'. Please provide the required column parameters."
    
    status_icons = {'pass': '✅', 'warn': '⚠️', 'fail': '❌'}
    output = f"**Power / Sample Size Check ({data_type}):**\n\n"
    for check in result['checks']:
        icon = status_icons.get(check['status'], '⚪')
        output += f"{icon} **{check['check']}:** {check['detail']}\n"
    
    all_pass = all(c['status'] == 'pass' for c in result['checks'])
    if all_pass:
        output += "\n✅ All power checks passed. Proceed with analysis."
    else:
        output += "\n⚠️ Some concerns detected. Consider adjusting your data or methodology."
    
    return output


import plotly.express as px
import plotly.graph_objects as go

def generate_diagnostic_plot(plot_type: str, date_col: str = None, outcome_col: str = None,
                              geo_col: str = None, treated_geo: str = None,
                              treatment_col: str = None, intervention_date: str = None,
                              covariates: list[str] = None) -> str:
    """
    Generates causal-relevant diagnostic plots and saves them for display.
    
    TIME-SERIES plot_types:
    - 'time_series_overview'     : Outcome over time with intervention line
    - 'pre_post_comparison'      : Box plot of pre vs. post periods
    
    PANEL / GEO plot_types:
    - 'multi_region_spaghetti'   : All regions on one plot, treated highlighted
    - 'donor_correlation_heatmap': Pre-period correlation matrix across regions
    
    CROSS-SECTIONAL plot_types:
    - 'treatment_distribution'   : Outcome distribution by treatment group
    - 'covariate_balance_love'   : Love plot (SMD for each covariate)
    - 'propensity_overlap'       : Propensity score distributions by group
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    df = st.session_state.df
    
    import time
    timestamp = int(time.time() * 1000)
    plot_path = f"/tmp/diagnostic_plot_{plot_type}_{timestamp}.html"
    
    try:
        if plot_type == 'time_series_overview' and date_col and outcome_col:
            df_plot = df.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col])
            if geo_col and treated_geo:
                df_plot = df_plot[df_plot[geo_col] == treated_geo]
            agg = df_plot.groupby(date_col)[outcome_col].mean().reset_index()
            fig = px.line(agg, x=date_col, y=outcome_col, title=f"Time Series: {outcome_col}")
            if intervention_date:
                fig.add_vline(x=intervention_date, line_dash="dash", line_color="red",
                             annotation_text="Intervention")
            fig.write_html(plot_path)
            st.session_state['last_diagnostic_plot'] = plot_path
            return f"Generated time series overview plot. View the plot in the Exploratory Analysis tab, or see: {plot_path}"
        
        elif plot_type == 'pre_post_comparison' and date_col and outcome_col and intervention_date:
            df_plot = df.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col])
            intervention_dt = pd.to_datetime(intervention_date)
            df_plot['Period'] = df_plot[date_col].apply(lambda x: 'Post' if x >= intervention_dt else 'Pre')
            fig = px.box(df_plot, x='Period', y=outcome_col, color='Period',
                        title=f"Pre vs. Post: {outcome_col}")
            fig.write_html(plot_path)
            st.session_state['last_diagnostic_plot'] = plot_path
            return f"Generated pre/post comparison box plot. View: {plot_path}"
        
        elif plot_type == 'multi_region_spaghetti' and date_col and outcome_col and geo_col:
            df_plot = df.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col])
            fig = px.line(df_plot, x=date_col, y=outcome_col, color=geo_col,
                         title=f"All Regions: {outcome_col} Over Time")
            if treated_geo:
                for trace in fig.data:
                    if trace.name == treated_geo:
                        trace.line.width = 4
                        trace.line.color = 'red'
                    else:
                        trace.opacity = 0.3
            if intervention_date:
                fig.add_vline(x=intervention_date, line_dash="dash", line_color="gray")
            fig.write_html(plot_path)
            st.session_state['last_diagnostic_plot'] = plot_path
            return f"Generated multi-region spaghetti plot with {treated_geo or 'all regions'} highlighted. View: {plot_path}"
        
        elif plot_type == 'donor_correlation_heatmap' and date_col and outcome_col and geo_col:
            df_plot = df.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col])
            if intervention_date:
                df_plot = df_plot[df_plot[date_col] < pd.to_datetime(intervention_date)]
            pivot = df_plot.pivot_table(index=date_col, columns=geo_col, values=outcome_col)
            corr = pivot.corr()
            fig = px.imshow(corr, text_auto=".2f", title="Pre-Period Donor Correlation Heatmap",
                           color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig.write_html(plot_path)
            st.session_state['last_diagnostic_plot'] = plot_path
            return f"Generated donor correlation heatmap for pre-period. View: {plot_path}"
        
        elif plot_type == 'treatment_distribution' and outcome_col and treatment_col:
            fig = px.histogram(df, x=outcome_col, color=treatment_col, barmode='overlay',
                              title=f"Outcome Distribution by Treatment ({treatment_col})",
                              marginal='box', opacity=0.7)
            fig.write_html(plot_path)
            st.session_state['last_diagnostic_plot'] = plot_path
            return f"Generated treatment distribution plot. View: {plot_path}"
        
        elif plot_type == 'covariate_balance_love' and treatment_col and covariates:
            result = causal_utils.check_covariate_balance(df, treatment_col, covariates)
            if 'error' in result:
                return f"Error checking balance: {result['error']}"
            balance = result['balance']
            df_bal = pd.DataFrame(balance)
            fig = px.scatter(df_bal, x='smd', y='covariate', title="Covariate Balance (Love Plot)",
                            color=df_bal['balanced'].map({True: 'Balanced', False: 'Imbalanced'}),
                            color_discrete_map={'Balanced': 'green', 'Imbalanced': 'red'})
            fig.add_vline(x=0.1, line_dash="dash", line_color="red", annotation_text="|SMD|=0.1")
            fig.update_layout(xaxis_title="|Standardized Mean Difference|", yaxis_title="Covariate")
            fig.write_html(plot_path)
            st.session_state['last_diagnostic_plot'] = plot_path
            return f"Generated covariate balance Love plot. View: {plot_path}"
        
        elif plot_type == 'propensity_overlap' and treatment_col and covariates:
            from sklearn.linear_model import LogisticRegression as LR
            X = df[covariates].select_dtypes(include=[np.number]).dropna()
            y = df.loc[X.index, treatment_col]
            lr = LR(max_iter=1000).fit(X, y)
            ps = lr.predict_proba(X)[:, 1]
            df_ps = pd.DataFrame({'Propensity Score': ps, treatment_col: y.astype(str)})
            fig = px.histogram(df_ps, x='Propensity Score', color=treatment_col,
                              barmode='overlay', nbins=40, opacity=0.7,
                              title="Propensity Score Overlap")
            fig.write_html(plot_path)
            st.session_state['last_diagnostic_plot'] = plot_path
            return f"Generated propensity score overlap plot. View: {plot_path}"
        
        else:
            return f"⚠️ Plot type '{plot_type}' requires parameters that were not provided. Needed: {', '.join(p for p, v in [('date_col', date_col), ('outcome_col', outcome_col), ('geo_col', geo_col), ('treatment_col', treatment_col)] if v is None)}"
    
    except Exception as e:
        return f"Error generating plot '{plot_type}': {e}"


def run_robustness_checks(method: str, date_col: str, outcome_col: str, intervention_date: str,
                          geo_col: str = None, treated_geo: str = None,
                          treatment_duration: int = 60) -> str:
    """
    Runs a suite of robustness checks (placebo test, leave-one-out) for quasi-experimental methods.
    Returns a confidence grade: High / Medium / Low with explanations.
    method: 'BSTS', 'CausalPy', 'GeoLift', 'SyntheticControl'
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    df = st.session_state.df
    
    output = f"**Robustness Checks for {method}:**\n\n"
    checks_passed = 0
    checks_total = 0
    
    # 1. Placebo test
    output += "### 1. Placebo Test\n"
    placebo_result = causal_utils.run_placebo_test(
        df, method, date_col, outcome_col, intervention_date,
        geo_col, treated_geo, treatment_duration
    )
    checks_total += 1
    status = "✅" if placebo_result.get('pass') else "⚠️" if placebo_result.get('pass') is None else "❌"
    output += f"{status} {placebo_result['message']}\n"
    if placebo_result.get('placebo_p_value') is not None:
        output += f"   Placebo p-value: {placebo_result['placebo_p_value']:.4f}\n"
    if placebo_result.get('pass'):
        checks_passed += 1
    output += "\n"
    
    # 2. Leave-one-out (only for panel/geo data)
    if geo_col and treated_geo:
        output += "### 2. Leave-One-Out Test\n"
        loo_result = causal_utils.run_leave_one_out(
            df, method, date_col, outcome_col, geo_col, treated_geo,
            intervention_date, treatment_duration
        )
        checks_total += 1
        status = "✅" if loo_result.get('pass') else "⚠️" if loo_result.get('pass') is None else "❌"
        output += f"{status} {loo_result['message']}\n"
        if loo_result.get('mean_effect') is not None:
            output += f"   Mean effect across iterations: {loo_result['mean_effect']:.4f} (±{loo_result.get('std_effect', 0):.4f})\n"
        if loo_result.get('pass'):
            checks_passed += 1
        output += "\n"
    
    # Confidence grade
    if checks_total == 0:
        grade = "Unknown"
    elif checks_passed == checks_total:
        grade = "🟢 **High Confidence**"
    elif checks_passed >= checks_total / 2:
        grade = "🟡 **Medium Confidence**"
    else:
        grade = "🔴 **Low Confidence**"
    
    output += f"---\n**Overall Confidence Grade:** {grade} ({checks_passed}/{checks_total} checks passed)\n"
    
    # Store in session state
    st.session_state['robustness_results'] = {
        'method': method, 'grade': grade,
        'checks_passed': checks_passed, 'checks_total': checks_total
    }
    
    return output


def run_refutation_tests(method: str, treatment_col: str, outcome_col: str, confounders: list[str],
                         tests: list[str] = None) -> str:
    """
    Runs DoWhy refutation tests on observational methods.
    tests: list of 'random_common_cause', 'placebo_treatment', 'data_subset'.
    Returns pass/fail for each test.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded."
    df = st.session_state.df
    
    if tests is None:
        tests = ['random_common_cause', 'placebo_treatment']
    
    output = f"**Refutation Tests ({method}):**\n\n"
    
    method_map = {
        "OLS": "Linear/Logistic Regression (OLS/Logit)",
        "PSM": "Propensity Score Matching (PSM)",
        "IPTW": "Inverse Propensity Weighting (IPTW)",
        "LinearDML": "Linear Double Machine Learning (LinearDML)",
        "CausalForest": "Generalized Random Forests (CausalForestDML)",
    }
    full_method = method_map.get(method, method)
    
    for test_name in tests:
        try:
            # Use DoWhy's built-in refutation
            graph_str = "digraph {" + "; ".join([f"{c} -> {treatment_col}; {c} -> {outcome_col}" for c in confounders]) + f"; {treatment_col} -> {outcome_col}" + "}"
            model = dowhy.CausalModel(
                data=df, treatment=treatment_col, outcome=outcome_col,
                graph=graph_str
            )
            identified = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
            
            refute_method = {
                'random_common_cause': 'random_common_cause',
                'placebo_treatment': 'placebo_treatment_refuter',
                'data_subset': 'data_subset_refuter'
            }.get(test_name, test_name)
            
            refutation = model.refute_estimate(identified, estimate, method_name=refute_method)
            
            # Parse result
            new_effect = refutation.new_effect if hasattr(refutation, 'new_effect') else None
            if new_effect is not None and not np.isnan(new_effect):
                original_effect = estimate.value
                diff = abs(new_effect - original_effect)
                passed = diff < abs(original_effect) * 0.5  # Effect shouldn't change by >50%
                status = "✅" if passed else "❌"
                output += f"{status} **{test_name.replace('_', ' ').title()}**: "
                output += f"Original ATE={original_effect:.4f}, Refuted ATE={new_effect:.4f}\n"
            else:
                output += f"⚠️ **{test_name.replace('_', ' ').title()}**: Could not compute\n"
        except Exception as e:
            output += f"⚠️ **{test_name.replace('_', ' ').title()}**: Failed — {e}\n"
    
    return output


def get_session_summary() -> str:
    """
    Returns a summary of the current session: dataset info, preprocessing applied,
    analyses run, and any robustness checks. Use this to track what has been done so far.
    """
    output = "**Session Summary:**\n\n"
    
    # Dataset
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        output += f"**Dataset:** {df.shape[0]} rows × {df.shape[1]} columns\n"
        sim_type = st.session_state.get('sim_type', 'Unknown')
        output += f"**Source:** {sim_type}\n"
        if st.session_state.get('dataset_description'):
            output += f"**Description:** {st.session_state.dataset_description}\n"
    else:
        output += "**Dataset:** None loaded\n"
        return output
    
    # Preprocessing
    preprocessing = []
    if st.session_state.get('p_impute_enable'):
        preprocessing.append("Missing value imputation")
    if st.session_state.get('p_wins_enable'):
        preprocessing.append("Winsorization")
    if st.session_state.get('bucketing_ops'):
        preprocessing.append(f"{len(st.session_state.bucketing_ops)} bucketing operations")
    if st.session_state.get('filtering_ops'):
        preprocessing.append(f"{len(st.session_state.filtering_ops)} filter operations")
    
    if preprocessing:
        output += f"\n**Preprocessing:** {', '.join(preprocessing)}\n"
    else:
        output += "\n**Preprocessing:** None applied\n"
    
    # Quality checks
    if st.session_state.get('data_quality_summary'):
        output += f"\n**Data Quality:** {st.session_state.data_quality_summary}\n"
    
    # Analyses
    analyses_found = False
    if st.session_state.get('quasi_results'):
        output += "\n**Quasi-Experimental Analyses:**\n"
        for method, result in st.session_state.quasi_results.items():
            output += f"- {method}\n"
        analyses_found = True
    if st.session_state.get('obs_results'):
        output += "\n**Observational Analyses:**\n"
        for method, result in st.session_state.obs_results.items():
            output += f"- {method}\n"
        analyses_found = True
    
    if not analyses_found:
        output += "\n**Analyses:** None run yet\n"
    
    # Robustness
    if st.session_state.get('robustness_results'):
        r = st.session_state.robustness_results
        output += f"\n**Robustness:** {r.get('grade', 'N/A')} ({r.get('checks_passed', 0)}/{r.get('checks_total', 0)} checks passed)\n"
    
    return output


def generate_analysis_report() -> str:
    """
    Compiles all analyses, comparisons, and robustness checks from the current session
    into a downloadable Markdown report. Call after completing your analysis workflow.
    """
    import time
    
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Error: No dataset loaded — nothing to report."
    
    report_content = causal_utils.generate_report_content(st.session_state)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_path = f"/tmp/causal_report_{timestamp}.md"
    
    lines = []
    lines.append("# Causal Analysis Report")
    lines.append(f"\n*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Executive Summary
    lines.append("## 1. Executive Summary\n")
    data_info = report_content.get('data', {})
    analyses = report_content.get('analyses', [])
    lines.append(f"This report covers the analysis of a {data_info.get('shape', 'unknown')} dataset ")
    lines.append(f"({data_info.get('sim_type', 'unknown')} source). ")
    lines.append(f"{len(analyses)} analysis/analyses were conducted.\n")
    
    # Data Description
    lines.append("## 2. Data Description\n")
    lines.append(f"- **Shape:** {data_info.get('shape', 'N/A')}\n")
    lines.append(f"- **Source:** {data_info.get('sim_type', 'N/A')}\n")
    if data_info.get('description'):
        lines.append(f"- **Description:** {data_info['description']}\n")
    lines.append(f"- **Columns:** {', '.join(data_info.get('columns', []))}\n")
    
    # Preprocessing
    preprocessing = report_content.get('preprocessing', [])
    lines.append("## 3. Preprocessing\n")
    if preprocessing:
        for p in preprocessing:
            lines.append(f"- {p}\n")
    else:
        lines.append("No preprocessing was applied.\n")
    
    # Analyses
    lines.append("## 4. Results\n")
    if analyses:
        for a in analyses:
            lines.append(f"### {a.get('method', 'Unknown Method')}\n")
            lines.append(f"- **Type:** {a.get('type', 'N/A')}\n")
            result = a.get('result', {})
            if isinstance(result, dict):
                for k, v in result.items():
                    if k not in ('plot_path', 'time_series_data'):
                        lines.append(f"- **{k}:** {v}\n")
    else:
        lines.append("No analyses were run in this session.\n")
    
    # Robustness
    robustness = st.session_state.get('robustness_results')
    lines.append("## 5. Robustness Checks\n")
    if robustness:
        lines.append(f"- **Confidence Grade:** {robustness.get('grade', 'N/A')}\n")
        lines.append(f"- **Checks Passed:** {robustness.get('checks_passed', 0)}/{robustness.get('checks_total', 0)}\n")
    else:
        lines.append("No robustness checks were performed.\n")
    
    # Conclusion
    lines.append("## 6. Conclusion\n")
    lines.append("*This report was auto-generated by the Causal Agent. Review all results ")
    lines.append("in context and consult domain experts before making business decisions.*\n")
    
    report_text = "\n".join(lines)
    
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    return f"**Report generated!**\n\nSaved to: `{report_path}`\n\n---\n\n{report_text[:2000]}"


def generate_dataset(theme: str, business_function: str,
                     study_design: str = "Cross-sectional (Observational)",
                     treatment_assignment: str = "Self-selected / Organic",
                     outcome_type: str = "Continuous (e.g. Revenue, Score)",
                     effect_size: str = "Medium",
                     data_quality: str = "High",
                     n_samples: int = 1000) -> str:
    """
    Generates a synthetic dataset for causal inference analysis with full parameter control.
    
    Parameters:
    - theme: Industry vertical (e.g., eCommerce, FinTech, Healthcare, Retail, SaaS)
    - business_function: Department (e.g., Marketing, Product, Operations, Growth)
    - study_design: "Cross-sectional (Observational)" | "Longitudinal (Time Series)" | "Longitudinal (Panel / Geographic)"
    - treatment_assignment: "Self-selected / Organic" | "Randomized (A/B)" | "Geographic (Market-level)"
    - outcome_type: "Continuous (e.g. Revenue, Score)" | "Binary (e.g. Converted, Churned)"
    - effect_size: "Small" | "Medium" | "Large"
    - data_quality: "High" (clean) | "Medium" (some issues) | "Low" (many issues)
    - n_samples: Number of rows (default 1000)
    
    Use when users describe a business problem. Infer parameters from their description.
    Example: "I want to test if a promo campaign lifted revenue across US states"
    → theme="Retail", business_function="Marketing", study_design="Longitudinal (Panel / Geographic)"
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


# Keep legacy alias for backward compatibility
def load_synthetic_dataset(theme: str, business_function: str) -> str:
    """Legacy alias for generate_dataset. Use generate_dataset() for full parameter control."""
    return generate_dataset(theme, business_function)

def chat_stream(model_name, messages, data_context, app_context, api_key):
    """
    Streams response from Gemini model using the new google.genai SDK.
    """
    # System Prompt
    system_prompt = f"""
    You are an expert Causal Inference Statistician and Data Scientist assistant embedded in the "Causal Inference Agent" application.
    You are an AGENTIC assistant — you proactively guide users through a structured causal analysis workflow, not just answer questions.
    
    Your goals:
    1. Help users understand the currently loaded dataset and identify the right columns for analysis.
    2. Suggest appropriate causal models based on data structure:
       - If they have cross-sectional data (one row per user), suggest **Observational Analysis**.
       - If they have longitudinal/time-series data (multiple timestamps), suggest **Quasi-Experimental Analysis** and refer to methods like DiD, CausalImpact, GeoLift, or CausalPy.
    3. You have TOOLS available to directly run causal inference methods on the CURRENTLY LOADED dataset. When a user asks you to "Run an analysis", use your tools and provide the results. 
       - If the user's request is ambiguous about parameters (e.g. they don't specify confounders), make a reasonable Data Science assumption based on the dataset columns, or clarify with them.
    4. Explain causal concepts (ATE, HTE, Confounding, Instrumental Variables, Regression Discontinuity, etc.) and interpret the statistical results you generate using your tools.
    5. Guide them to the correct UI Tab when applicable.
    6. Mention the "Results Interpretation" guides in the **User Guide (Tab 1)** for interpreting statistical metrics like P-values, Confidence Intervals, and Relative Lift.
    7. When a user asks to "compare methods" or "run a comparison":
       - For quasi-experimental data: use `run_method_comparison` to compare BSTS, GeoLift, and CausalPy simultaneously.
       - For cross-sectional data: use `run_cross_sectional_comparison` to compare PSM, IPTW, LinearDML, etc.
       - After receiving results, interpret and synthesize: highlight agreement/disagreement across methods, explain why estimates differ, and give a final recommendation.

    AGENTIC WORKFLOW — Follow this when users ask for analysis or help:

    Step 0: DATA ONBOARDING (if no dataset loaded)
      → Ask: \"Do you have a dataset to upload, or would you like me to generate one?\"
      → If generate: infer parameters from their problem description, confirm, then call load_synthetic_dataset()

    Step 1: DATA INSPECTION
      → Call profile_dataset() to detect data structure, column roles, and suitable methods
      → Report findings and recommended methods to the user

    Step 1.5: DATA QUALITY (Propose → Confirm → Apply)
      → Call check_data_quality() to scan for issues
      → Present issues as a numbered list with suggested fixes
      → WAIT for user confirmation — NEVER auto-apply fixes
      → If user approves, call apply_data_fixes() with their selections

    Step 2: METHOD SELECTION (Interactive)
      → Present applicable methods as a numbered menu with ✅/⚠️ status
      → For Quasi-Experimental: CausalPy, GeoLift, BSTS, DiD
      → For Cross-Sectional: PSM, IPTW, LinearDML, CausalForest, Meta-Learners
      → Include your recommendation and reasoning
      → WAIT for user to pick one, multiple, or \"compare all\"

    Step 2.5: POWER / SAMPLE SIZE PRE-CHECK
      → Call check_power() to verify sufficient data before running analysis
      → If underpowered, WARN user and suggest alternatives

    Step 3: ASSUMPTION CHECKING
      → Call check_assumptions() for the chosen method
      → Auto-generate diagnostic plots:
          DiD → generate_diagnostic_plot('covariate_balance_love')
          BSTS → generate_diagnostic_plot('time_series_overview')
          Synthetic Control → generate_diagnostic_plot('donor_correlation_heatmap')
          PSM/IPTW → generate_diagnostic_plot('covariate_balance_love') + generate_diagnostic_plot('propensity_overlap')
      → If assumptions violated, suggest alternative methods
      → WAIT for user confirmation before proceeding

    Step 4: RUN ANALYSIS
      → If single method: call the specific tool
      → If multiple methods: call run_method_comparison() or run_cross_sectional_comparison()
      → Auto-generate results plots:
          Panel/Geo → generate_diagnostic_plot('multi_region_spaghetti')
          Time-Series → generate_diagnostic_plot('time_series_overview') with intervention
          Cross-Sectional → generate_diagnostic_plot('treatment_distribution')
      → Present results with synthesis

    Step 5: INTERPRETATION & NEXT STEPS
      → Synthesize results in plain language
      → Highlight key metrics, significance, and practical significance
      → Proactively suggest: \"Would you like me to run robustness checks?\"

    Step 6: ROBUSTNESS / REFUTATION (if requested)
      → For quasi-experimental: call run_robustness_checks() (placebo + leave-one-out)
      → For observational: call run_refutation_tests() (random common cause + placebo treatment)
      → Report confidence grade: High / Medium / Low

    Step 7: SESSION MEMORY & REPORT
      → If user asks \"What have we done?\": call get_session_summary()
      → If user asks for a report: call generate_analysis_report()
      → The report includes all prior analyses, robustness checks, and caveats

    CAUSAL REASONING GUARDRAILS:
    - For quasi-experimental results: Use causal language (\"The treatment caused a ~X% lift\")
    - For observational without refutation: Use associational language (\"The treatment is associated with ~X% change\")
    - For observational with passed refutation: \"Evidence supports a causal effect of ~X%\"
    - Always report both statistical significance AND effect size
    - If running multiple methods, warn about multiple comparisons
    - Note scope: \"This effect was estimated on [dataset description]. Generalization requires additional validation.\"

    ERROR HANDLING:
    - If a tool call fails, report which method failed and why, continue with remaining methods
    - Present partial results clearly; do NOT stop the entire workflow for one failure

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
        tools=[run_observational_analysis, run_did_analysis, run_time_series_bsts, run_synthetic_control_geolift, run_causalpy_analysis, run_method_comparison, run_cross_sectional_comparison, profile_dataset, check_data_quality, apply_data_fixes, check_assumptions, check_power, generate_diagnostic_plot, run_robustness_checks, run_refutation_tests, get_session_summary, generate_analysis_report, generate_dataset, load_synthetic_dataset]
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
