import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import json

def generate_dynamic_dataset(
    theme: str, 
    business_function: str, 
    api_key: str, 
    n_samples: int = 1000,
    study_design: str = "Cross-sectional (Observational)",
    treatment_assignment: str = "Self-selected / Organic",
    outcome_type: str = "Continuous (e.g. Revenue, Score)",
    effect_size: str = "Medium",
    data_quality: str = "High"
) -> tuple[pd.DataFrame, str]:
    """
    Uses Gemini to dynamically generate a synthetic dataset configured for causal inference.
    Supports both cross-sectional and longitudinal study designs.
    Returns a tuple of (DataFrame, description_string).
    """
    
    is_longitudinal = "Longitudinal" in study_design
    is_binary_outcome = "Binary" in outcome_type
    is_geo = "Geographic" in treatment_assignment
    
    # Map effect size to numeric guidance
    effect_map = {"Small": "a subtle, hard-to-detect", "Medium": "a moderate, detectable", "Large": "a strong, obvious"}
    effect_desc = effect_map.get(effect_size, "a moderate")
    
    if is_longitudinal:
        # --- Longitudinal / Time Series / Panel prompt ---
        if is_geo:
            n_regions = max(10, n_samples // 365)  # sensible number of regions
            n_days = min(365, n_samples // n_regions)
            data_shape_desc = f"{n_regions} geographic regions over {n_days} days"
        else:
            n_days = min(365, n_samples)
            data_shape_desc = f"a single or small number of units over {n_days} days"
        
        outcome_guidance = "a binary 0/1 indicator" if is_binary_outcome else "a continuous metric"
        
        quality_map = {
            "High": "Provide clean, high-quality data with no missing values and minimal noise.",
            "Medium": "Inject moderate noise and approximately 5-10% missing values in some columns.",
            "Low": "Inject significant issues: 15-20% missing values, extreme outliers (3x-10x mean), and high measurement noise."
        }
        quality_desc = quality_map.get(data_quality, quality_map["High"])
        
        prompt = f"""
        You are an expert Data Scientist designing a synthetic LONGITUDINAL / TIME SERIES causal dataset.
        Theme: '{theme}', Business Function: '{business_function}'.
        Treatment Assignment: '{treatment_assignment}'.
        Data Quality: '{data_quality}'. {quality_desc}
        The dataset represents {data_shape_desc}.
        The outcome should be {outcome_guidance}.
        The treatment effect should be {effect_desc}.

        Output a strictly formatted JSON object:
        {{
            "regions": ["Region_1", "Region_2"],  // list of region/unit names (use 1 if single-unit time series)
            "n_days": {n_days},
            "start_date": "2023-01-01",
            "intervention_day": 250, // day index when treatment starts (0-indexed)
            "treated_regions": ["Region_1"], // which regions receive the treatment
            "kpi": {{
                "name": "outcome_kpi_name",
                "base_value": 100,
                "trend_slope": 0.1,
                "seasonality_amplitude": 10,
                "noise_std": 3
            }},
            "treatment_effect": {{
                "additive_lift": 20,
                "cumulative_growth": 0.5
            }},
            "covariates": [
                {{"name": "covariate_name", "base_value": 50, "noise_std": 5}}
            ],
            "binary_outcome": {str(is_binary_outcome).lower()},
            "description": "1-2 sentence summary of this synthetic dataset including the theme, KPI, treatment, and study design.",
            "questions": ["Causal question 1?", "Causal question 2?", "Causal question 3?"]
        }}
        """
    else:
        # --- Cross-sectional / Observational prompt ---
        outcome_guidance = "binary (0/1)" if is_binary_outcome else "continuous"
        
        prompt = f"""
        You are an expert Data Scientist and Economist designing a synthetic causal dataset.
        The dataset is for a company with Theme: '{theme}' and Business Function: '{business_function}'.
        Treatment Assignment: '{treatment_assignment}'.
        The outcome variable should be {outcome_guidance}.
        The treatment effect should be {effect_desc}.
        
        The dataset should have {n_samples} rows and approximately 6 to 8 columns, including:
        1. A binary 'Treatment' variable (0 or 1).
        2. An outcome variable ({outcome_guidance}).
        3. Several confounding variables (both continuous and categorical/binary) that affect BOTH the Treatment probability and the Outcome.
        
        DATA QUALITY REQUIREMENTS ({data_quality}):
        {quality_desc}
        If quality is Low or Medium, explicitly ensure some columns in the JSON 'rows' have null values or extreme outliers.
        
        Output a strictly formatted JSON object:
        {{
            "columns": [
                {{"name": "col_name", "type": "categorical_binary", "base_prob": 0.5}},
                {{"name": "col_name_2", "type": "continuous", "mean": 100, "std": 15}}
            ],
            "treatment": {{
                "name": "treatment_col_name",
                "base_prob": 0.3,
                "confounder_effects": [
                    {{"confounder": "col_name", "effect_on_logit": 0.5}},
                    {{"confounder": "col_name_2", "effect_on_logit": 0.05}}
                ]
            }},
            "outcome": {{
                "name": "outcome_col_name",
                "base_value": 50,
                "treatment_effect": 20, 
                "confounder_effects": [
                    {{"confounder": "col_name", "effect_on_outcome": 10}},
                    {{"confounder": "col_name_2", "effect_on_outcome": 2}}
                ],
                "noise_std": 5,
                "binary": {str(is_binary_outcome).lower()}
            }},
            "date_column": {{
                "include": true,
                "name": "Date",
                "start_date": "2023-01-01",
                "range_days": 365
            }},
            "description": "1-2 sentence summary of this synthetic dataset including the theme, treatment, outcome, and key confounders.",
            "questions": ["Causal question 1?", "Causal question 2?", "Causal question 3?"]
        }}
        """
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7
            )
        )
        
        spec = json.loads(response.text)
        np.random.seed(42)
        
        if is_longitudinal:
            df = _build_longitudinal_df(spec, is_binary_outcome)
        else:
            df = _build_cross_sectional_df(spec, n_samples, is_binary_outcome)
        
        description = spec.get("description", f"Synthetic causal dataset for {theme} / {business_function}.")
        questions = spec.get("questions", [])
        return df, description, questions
        
    except Exception as e:
        print(f"Error generating data: {str(e)}")
        raise RuntimeError(f"Failed to generate dynamic dataset via Gemini: {str(e)}")


def _build_cross_sectional_df(spec: dict, n_samples: int, is_binary: bool) -> pd.DataFrame:
    """Builds a cross-sectional DataFrame from the Gemini JSON spec."""
    df = pd.DataFrame()
    
    # 0. Unique identifier
    df["User_ID"] = [f"U{str(i+1).zfill(4)}" for i in range(n_samples)]
    # 1. Confounders
    for col_spec in spec.get("columns", []):
        if col_spec["type"] == "categorical_binary":
            df[col_spec["name"]] = np.random.binomial(1, col_spec.get("base_prob", 0.5), n_samples)
        elif col_spec["type"] == "continuous":
            df[col_spec["name"]] = np.random.normal(col_spec.get("mean", 0), col_spec.get("std", 1), n_samples)
    
    # 2. Treatment
    trt_spec = spec["treatment"]
    trt_logit = np.full(n_samples, -1.0)
    for effect in trt_spec.get("confounder_effects", []):
        conf = effect["confounder"]
        if conf in df.columns:
            if df[conf].dtype == float:
                norm = (df[conf] - df[conf].mean()) / df[conf].std()
                trt_logit += effect["effect_on_logit"] * norm
            else:
                trt_logit += effect["effect_on_logit"] * df[conf]
    prob_treatment = 1 / (1 + np.exp(-trt_logit))
    df[trt_spec["name"]] = np.random.binomial(1, prob_treatment, n_samples)
    
    # 3. Outcome
    out_spec = spec["outcome"]
    outcome = np.full(n_samples, float(out_spec.get("base_value", 0.0)))
    outcome += float(out_spec.get("treatment_effect", 0.0)) * df[trt_spec["name"]]
    for effect in out_spec.get("confounder_effects", []):
        conf = effect["confounder"]
        if conf in df.columns:
            outcome += effect["effect_on_outcome"] * df[conf]
    outcome += np.random.normal(0, out_spec.get("noise_std", 1.0), n_samples)
    
    if is_binary or out_spec.get("binary", False):
        prob = 1 / (1 + np.exp(-outcome))
        df[out_spec["name"]] = np.random.binomial(1, prob, n_samples)
    else:
        df[out_spec["name"]] = outcome
    
    # 4. Dates
    date_spec = spec.get("date_column", {})
    if date_spec.get("include", False):
        start = pd.to_datetime(date_spec.get("start_date", "2023-01-01"))
        days = date_spec.get("range_days", 365)
        date_offsets = np.random.randint(0, days, n_samples)
        df[date_spec.get("name", "Date")] = start + pd.to_timedelta(date_offsets, unit='D')
    
    # Clean types
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    
    return df


def _build_longitudinal_df(spec: dict, is_binary: bool) -> pd.DataFrame:
    """Builds a longitudinal / panel DataFrame from the Gemini JSON spec."""
    regions = spec.get("regions", ["Region_1"])
    n_days = spec.get("n_days", 365)
    start_date = pd.to_datetime(spec.get("start_date", "2023-01-01"))
    intervention_day = spec.get("intervention_day", int(n_days * 0.7))
    treated_regions = set(spec.get("treated_regions", [regions[0]]))
    
    kpi_spec = spec.get("kpi", {})
    trt_effect = spec.get("treatment_effect", {})
    covariates_spec = spec.get("covariates", [])
    
    date_range = pd.date_range(start=start_date, periods=n_days)
    
    # Shared global patterns
    global_trend = np.linspace(
        kpi_spec.get("base_value", 100), 
        kpi_spec.get("base_value", 100) + kpi_spec.get("trend_slope", 0.1) * n_days, 
        n_days
    )
    seasonality = kpi_spec.get("seasonality_amplitude", 10) * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    data_list = []
    for region in regions:
        regional_offset = np.random.normal(0, 10)
        noise = np.random.normal(0, kpi_spec.get("noise_std", 3), n_days)
        metric = global_trend + regional_offset + seasonality + noise
        
        # Apply treatment effect to treated regions
        is_treated = region in treated_regions
        if is_treated:
            lift = np.zeros(n_days)
            additive = trt_effect.get("additive_lift", 20)
            growth = trt_effect.get("cumulative_growth", 0.5)
            post_days = n_days - intervention_day
            if post_days > 0:
                lift[intervention_day:] = additive + np.cumsum(
                    np.random.normal(growth, abs(growth) * 0.2, post_days)
                )
            metric += lift
        
        row = {
            "Date": pd.to_datetime(date_range).floor('D'),
            "Region": region,
            kpi_spec.get("name", "KPI"): metric,
            "Is_Post_Intervention": (np.arange(n_days) >= intervention_day).astype(int),
            "Is_Treated_Region": 1 if is_treated else 0
        }
        
        # Add covariates
        for cov in covariates_spec:
            row[cov["name"]] = np.random.normal(
                cov.get("base_value", 50), cov.get("noise_std", 5), n_days
            ) + global_trend * 0.1
        
        region_df = pd.DataFrame(row)
        data_list.append(region_df)
    
    df = pd.concat(data_list, ignore_index=True)
    
    # Add unique row identifier
    df.insert(0, "Row_ID", [f"R{str(i+1).zfill(5)}" for i in range(len(df))])
    
    # Handle binary outcome
    if is_binary or spec.get("binary_outcome", False):
        kpi_name = kpi_spec.get("name", "KPI")
        prob = 1 / (1 + np.exp(-(df[kpi_name] - df[kpi_name].median()) / df[kpi_name].std()))
        df[kpi_name] = np.random.binomial(1, prob, len(df))
    
    df['Date'] = pd.to_datetime(df['Date']).dt.floor('D')
    return df


def generate_data_quality_summary(df: pd.DataFrame, api_key: str) -> str:
    """
    Uses AI to generate a data quality summary with preprocessing suggestions.
    Includes correlation analysis for confounding detection.
    """
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.value_counts().to_dict()
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(1)
    missing_report = missing[missing > 0]
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    stats_summary = ""
    corr_summary = ""
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2).to_string()
        stats_summary = f"Numeric summary:\n{desc}\n"
        skew = df[numeric_cols].skew().round(2).to_dict()
        stats_summary += f"Skewness: {skew}\n"
        
        # Correlation matrix for confounding detection
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().round(2)
            high_corr = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    r = corr.iloc[i, j]
                    if abs(r) > 0.5:
                        high_corr.append(f"{numeric_cols[i]} <-> {numeric_cols[j]}: r={r}")
            corr_summary = f"High correlations (|r|>0.5): {high_corr if high_corr else 'None'}\n"
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_summary = ""
    if cat_cols:
        for col in cat_cols[:5]:
            cat_summary += f"{col}: {df[col].nunique()} unique, top={df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}\n"
    
    dup_count = df.duplicated().sum()
    
    profile = f"""
    Dataset: {n_rows} rows, {n_cols} columns
    Dtypes: {dtypes}
    Duplicates: {dup_count}
    Missing values: {missing_report.to_dict() if len(missing_report) > 0 else 'None'}
    Missing %: {missing_pct[missing_pct > 0].to_dict() if len(missing_pct[missing_pct > 0]) > 0 else 'None'}
    {stats_summary}
    {corr_summary}
    {cat_summary}
    """
    
    prompt = f"""
    You are a senior data scientist reviewing a dataset for causal inference analysis.
    Here is the data profile:
    {profile}
    
    Provide a concise markdown summary (max 5-6 bullet points) covering:
    1. **Data Overview**: Key characteristics (size, types, balance)
    2. **Data Quality Issues**: Missing values, duplicates, outliers (based on skewness/range), class imbalance
    3. **Correlation Insights**: Highlight any strongly correlated variable pairs that may indicate confounding or multicollinearity
    4. **Preprocessing Recommendations**: Specific suggestions like imputation, winsorization, log transforms, standardization, one-hot encoding, or duplicate removal — reference the "Data Preprocessing" section below
    5. **Readiness for Causal Analysis**: Is the dataset suitable as-is, or does it need cleaning first?
    
    Keep it practical and actionable. Use bullet points. Do NOT use code blocks.
    """
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return response.text
    except Exception as e:
        return f"Could not generate summary: {str(e)}"


def generate_method_recommendation(df: pd.DataFrame, target: str, treatment: str, api_key: str) -> str:
    """
    Uses AI to analyze the dataset relative to the explicit target and treatment variables
    and recommends the optimal Causal Inference method.
    """
    import numpy as np
    
    n_rows, n_cols = df.shape
    cols = df.columns.tolist()
    
    target_info = f"Numeric, {df[target].nunique()} unique" if np.issubdtype(df[target].dtype, np.number) else f"Categorical, {df[target].nunique()} unique"
    treatment_info = f"Numeric, {df[treatment].nunique()} unique" if np.issubdtype(df[treatment].dtype, np.number) else f"Categorical, {df[treatment].nunique()} unique"
    
    date_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c]) or 'date' in c.lower()]
    
    profile = f"""
    Dataset Map:
    - Total Rows: {n_rows}
    - Total Columns: {n_cols}
    - Potential Time Series Columns: {date_cols}
    
    Explicit Analytical Goal:
    - TARGET METRIC (Y): '{target}' ({target_info})
    - TREATMENT INTERVENTION (X): '{treatment}' ({treatment_info})
    """
    
    prompt = f"""
    You are a senior econometrician and causal inference expert.
    The user wants to measure the causal impact of their Treatment variable on their Target variable.
    
    Here is the structural map of their uploaded dataset:
    {profile}
    
    Your job is to recommend the OPTIMAL causal inference method from the following list:
    - Difference-in-Differences (DiD)
    - Bayesian Structural Time Series (CausalImpact)
    - Synthetic Control (GeoLift / CausalPy)
    - Double Machine Learning (DML) / Causal Forests
    - Meta-Learners (S-Learner, T-Learner)
    - Propensity Score Matching (PSM) / Inverse Propensity Weighting (IPTW)
    - Traditional OLS / Logistic Regression
    
    Provide a concise (2-3 paragraphs) recommendation. 
    1. First, state clearly which method(s) you recommend.
    2. Then explain *why* it is the best fit mathematically based on the characteristics of the Target, Treatment, and the presence (or absence) of Time Series variables.
    3. Quickly note any structural constraints (e.g., "Since Treatment has >2 values, PSM won't work, but DML will.").
    """
    
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return response.text
    except Exception as e:
        return f"Could not generate recommendation: {str(e)}"

def generate_chart_suggestions(df: pd.DataFrame, api_key: str) -> str:
    """
    Uses AI to suggest relevant chart configurations for the loaded dataset.
    References the app's actual Chart Builder UI controls.
    """
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        columns_info.append(f"{col} ({dtype}, {nunique} unique)")
    
    cols_str = ", ".join(columns_info)
    
    prompt = f"""
    You are a data visualization expert. A user has a dataset with these columns:
    {cols_str}
    
    The user has a Chart Builder with these options:
    - Chart Types: Scatter Plot, Line Chart, Bar Chart, Histogram, Box Plot, Pie Chart
    - X Variable, Y Variable(s), Color/Group variable (optional)
    - Facet (split charts by a column)
    - Aggregation (Mean, Sum, Count, Median)
    - Dual Axis support for Line/Bar charts
    
    Suggest 3-4 specific, practical charts the user should create to explore this data for causal inference. 
    For each suggestion, provide:
    1. A short title
    2. The exact settings: Chart Type, X, Y, Color, and any aggregation to use
    3. Why it's useful for causal analysis (1 sentence)
    
    Format as a numbered list. Be very specific with column names. Do NOT use code blocks.
    """
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4)
        )
        return response.text
    except Exception as e:
        return f"Could not generate suggestions: {str(e)}"


def generate_dataset_preview_summary(df: pd.DataFrame, api_key: str) -> dict:
    """
    Generates an AI description and causal questions for any loaded dataset.
    Returns a dict with 'description' and 'questions' keys.
    """
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        sample = str(df[col].dropna().head(3).tolist())
        columns_info.append(f"{col} ({dtype}, {nunique} unique, sample: {sample})")
    
    cols_str = "\n".join(columns_info)
    
    prompt = f"""
    You are a causal inference expert. A user has loaded a dataset with {len(df)} rows and {len(df.columns)} columns.
    
    Columns:
    {cols_str}
    
    Respond in valid JSON with exactly two keys:
    1. "description": A 1-2 sentence summary of what this dataset represents, the likely treatment/intervention, outcome, and confounders.
    2. "questions": A list of exactly 3 causal questions this dataset could help answer.
    
    Example:
    {{"description": "This dataset tracks user behavior...", "questions": ["Does X cause Y?", "...", "..."]}}
    
    Return ONLY the JSON object, no markdown fences.
    """
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        import json
        text = response.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return {
            "description": result.get("description", ""),
            "questions": result.get("questions", [])
        }
    except Exception as e:
        return {
            "description": f"Could not generate summary: {str(e)}",
            "questions": []
        }


def generate_method_recommendation(df: pd.DataFrame, api_key: str, tab_type: str = "observational") -> str:
    """
    Recommends the best causal inference method based on data structure.
    tab_type: 'observational' or 'quasi_experimental'
    """
    columns_info = []
    for col in df.columns[:15]:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        columns_info.append(f"{col} ({dtype}, {nunique} unique)")
    
    cols_str = ", ".join(columns_info)
    n_rows = len(df)
    has_date = any('date' in c.lower() or 'time' in c.lower() for c in df.columns)
    has_region = any(k in c.lower() for c in df.columns for k in ['region', 'geo', 'city', 'state', 'market'])
    
    if tab_type == "observational":
        methods = """
        Available methods:
        1. Linear/Logistic Regression (OLS/Logit) — simple, interpretable, assumes linearity
        2. Propensity Score Matching (PSM) — matches treated/control units, works well with binary treatment
        3. Inverse Propensity Weighting (IPTW) — reweights samples, handles selection bias
        4. Linear Double Machine Learning (LinearDML) — debiased ML, handles high-dimensional confounders
        5. Generalized Random Forests (CausalForestDML) — non-parametric, discovers heterogeneous treatment effects
        6. Meta-Learner: S-Learner — single model, simple but may miss treatment heterogeneity
        7. Meta-Learner: T-Learner — separate models for treated/control, captures heterogeneity
        """
    else:
        methods = """
        Available methods:
        1. Difference-in-Differences (DiD) — requires treatment/control groups + pre/post periods, parallel trends assumption
        2. CausalImpact (Bayesian Time Series) — single treated time series with covariates, builds counterfactual forecast
        3. GeoLift (Synthetic Control) — multi-region geographic data, constructs synthetic control from donor regions (uses R via rpy2)
        4. CausalPy (Bayesian Synthetic Control) — pure Python alternative to GeoLift using PyMC. Proper two-sided HDI credible intervals, posterior probability of effect. Recommended for robust uncertainty quantification.
        """
    
    prompt = f"""
    You are a causal inference expert. A user has a dataset with {n_rows} rows.
    Columns: {cols_str}
    Has date/time columns: {has_date}
    Has geographic/region columns: {has_region}
    
    {methods}
    
    Based on the data structure, recommend the BEST method and briefly explain why. Also mention:
    - Which column is likely the treatment/intervention
    - Which column is likely the outcome
    - Key assumptions to verify
    
    Keep it to 4-5 bullet points maximum. Be practical and specific to this dataset. Do NOT use code blocks.
    """
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return response.text
    except Exception as e:
        return f"Could not generate recommendation: {str(e)}"


def generate_config_guidance(df: pd.DataFrame, api_key: str, method: str, treatment: str = None, outcome: str = None, method_recommendation: str = None) -> str:
    """
    Provides specific configuration guidance for the selected causal method.
    """
    columns_info = []
    for col in df.columns[:15]:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        columns_info.append(f"{col} ({dtype}, {nunique} unique)")
    
    cols_str = ", ".join(columns_info)
    
    context = f"Treatment: {treatment}" if treatment else ""
    context += f", Outcome: {outcome}" if outcome else ""
    
    rec_context = ""
    if method_recommendation:
        rec_context = f"""
    IMPORTANT: A prior method recommendation was generated for this dataset:
    ---
    {method_recommendation[:500]}
    ---
    Your configuration guide MUST be consistent with this recommendation. Focus your guidance on the method: {method}.
    Do NOT suggest a different method or talk about other methods unless the user has explicitly selected one.
    """
    
    prompt = f"""
    You are a causal inference expert helping a user configure their analysis.
    
    Method selected: {method}
    Dataset columns: {cols_str}
    {context}
    Dataset size: {len(df)} rows
    {rec_context}
    
    Provide a brief configuration guide (4-5 bullet points) covering:
    1. **Variable Selection**: Which columns should be treatment, outcome, and confounders (if applicable)? Be specific.
    2. **Key Assumptions**: What assumptions must hold for this method? How to check them.
    3. **Potential Pitfalls**: Common mistakes with this method and how to avoid them.
    4. **Expected Output**: What to look for in the results.
    """
    
    if "CausalPy" in method:
        prompt += """
    5. **Advanced Parameters (Speed Up)**: Explicitly mention that users can open the 'Advanced PyMC Sampling Parameters' expander and lower 'MCMC Draws' and 'Tuning Steps' to 500, or reduce 'Target Acceptance' to 0.85 to dramatically speed up the Bayesian inference.
    """
    
    prompt += """
    Keep it concise and actionable. Do NOT use code blocks.
    """
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return response.text
    except Exception as e:
        return f"Could not generate guidance: {str(e)}"


def generate_results_interpretation(api_key: str, method: str, metrics: dict, treatment: str = "", outcome: str = "") -> str:
    """
    Generates a plain-English interpretation of causal analysis results.
    """
    metrics_str = "\n".join([f"  - {k}: {v}" for k, v in metrics.items()])
    
    prompt = f"""
    You are a causal inference expert explaining results to a business stakeholder.
    
    Method: {method}
    Treatment: {treatment}
    Outcome: {outcome}
    Key Metrics:
    {metrics_str}
    
    Provide a clear, non-technical interpretation (4-5 bullet points) covering:
    1. **Main Finding**: What is the estimated causal effect? Is it statistically significant?
    2. **Practical Significance**: Is the effect size meaningful for business decisions?
    3. **Confidence**: How certain can we be about these results? (confidence intervals, p-values)
    4. **Limitations**: What caveats should decision-makers keep in mind?
    5. **Recommendation**: Based on these results, what action should be taken?
    
    Use simple language. Avoid jargon. Do NOT use code blocks.
    """
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return response.text
    except Exception as e:
        return f"Could not generate interpretation: {str(e)}"
