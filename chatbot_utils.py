import google.generativeai as genai
import streamlit as st
import pandas as pd
import os

def configure_genai():
    """Configures the Gemini API key from Streamlit secrets or environment variables."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        # Fallback for local development if not in secrets/env (optional, or raise error)
        # For now, let's assume it must be present
        return False
    
    genai.configure(api_key=api_key)
    return True

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
    Application: Causal Inference Agent
    Capabilities:
    1. Data Simulation & Upload: Users can upload CSVs or use simulated SaaS data.
    2. preprocessing: Imputation, Winsorization, Log Transform, Standardization, Bucketing, Filtering.
    3. Visualization: Scatter, Line, Bar, Histogram, Box Plot, Pie Chart.
    4. Causal Analysis:
       - Estimators: OLS, Logit, DiD, PSM, IPTW, LinearDML, CausalForestDML, S-Learner, T-Learner.
       - Features: Binary/Continuous Treatment, Binary/Continuous Outcome, Confounder selection.
       - Validation: Refutation tests (Random Common Cause, Placebo Treatment).
       - HTE: Heterogeneous Treatment Effects analysis.
    """

def chat_stream(model_name, messages, data_context, app_context):
    """
    Streams response from Gemini model.
    """
    # System Prompt
    system_prompt = f"""
    You are an expert Causal Inference Statistician and Data Scientist assistant embedded in the "Causal Inference Agent" application.
    
    Your goals:
    1. Help users understand the dataset.
    2. Suggest appropriate causal models (Treatment, Outcome, Confounders).
    3. Explain causal concepts (ATE, HTE, Backdoor Criterion) and specific methods (LinearDML, DiD, etc.).
    4. Guide them on how to use the specific UI features of this app.

    Context:
    --- APP CONTEXT ---
    {app_context}
    
    --- CURRENT DATASET ---
    {data_context}
    
    Instructions:
    - Be concise and helpful.
    - Use markdown for formatting.
    - When suggesting variables, strictly refer to the columns present in the 'CURRENT DATASET'.
    - If the user asks for code, provide Python code compatible with the libraries used (pandas, econml, dowhy).
    """
    
    # Prepare history for Gemini
    # Gemini expects 'user' and 'model' roles. Streamlit uses 'user' and 'assistant'.
    gemini_history = []
    
    # Add system prompt as the first part of the conversation (or as system instruction if supported, 
    # but strictly prepending to the first message is often safer/easier for simple chat).
    # Actually, let's look at `google.generativeai` chat history format.
    # Ideally, we start a chat session.
    
    model = genai.GenerativeModel(model_name)
    
    # transform streamlit messages to gemini format
    # Streamlit: {"role": "user"/"assistant", "content": "..."}
    # Gemini: {"role": "user"/"model", "parts": ["..."]}
    
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
        
    # We want to inject the system prompt. A robust way with `chat` is to put it 
    # in the history as a user message, or start the chat with it.
    # However, since we are doing a stateless-like call where we pass full history each time 
    # (or rather, we send the history to `start_chat`), let's construct the chat object.
    
    # IMPORTANT: The system instructions can be impactful. 
    # Let's prepend it to the very first message if history exists, 
    # otherwise send it as the first message.
    
    if not gemini_history:
        # Should not happen if this is called after user input
        gemini_history.append({"role": "user", "parts": [system_prompt]})
    else:
        # Prepend system context to the latest detailed data context?
        # Actually, best practice: system instruction argument in `GenerativeModel` creation 
        # is available in newer APIs, but let's stick to prepending to history or context.
        # Let's add it as a "developer" or "system" type instruction if possible?
        # Standard approach: Prepend to the first message.
        first_part = gemini_history[0]["parts"][0]
        gemini_history[0]["parts"][0] = system_prompt + "\n\nUser: " + first_part
        
    chat = model.start_chat(history=gemini_history[:-1]) # All but last
    response = chat.send_message(gemini_history[-1]["parts"][0], stream=True)
    
    return response
