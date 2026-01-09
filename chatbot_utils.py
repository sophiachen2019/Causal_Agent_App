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

def chat_stream(model_name, messages, data_context, app_context, api_key):
    """
    Streams response from Gemini model using the new google.genai SDK.
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
        temperature=0.7
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
    
    return response_stream
