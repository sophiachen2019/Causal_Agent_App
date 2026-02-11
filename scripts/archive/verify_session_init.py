import streamlit as st
import pandas as pd
import numpy as np

# Mock Streamlit functions
class MockSessionState(dict):
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value

if 'session_state' not in globals():
    st.session_state = MockSessionState()

# Simulate the initialization logic
print("--- Testing Session Initialization ---")

# 1. Initial State (Empty)
if 'df' not in st.session_state:
    print("Initializing 'df'...")
    st.session_state.df = pd.DataFrame({'A': [1, 2, 3]})

if 'bucketing_ops' not in st.session_state:
    print("Initializing 'bucketing_ops'...")
    st.session_state.bucketing_ops = []

# 2. Verify Existence
if 'df' in st.session_state and isinstance(st.session_state.df, pd.DataFrame):
    print("✅ 'df' initialized correctly.")
else:
    print("❌ 'df' NOT initialized.")

if 'bucketing_ops' in st.session_state and isinstance(st.session_state.bucketing_ops, list):
    print("✅ 'bucketing_ops' initialized correctly.")
else:
    print("❌ 'bucketing_ops' NOT initialized.")

# 3. Simulate Re-run (State should persist)
print("\n--- Testing Persistence (Mock Re-run) ---")
# Logic: If keys exist, they shouldn't be re-initialized (or overwritten if we had logic for that)
# Our app logic:
if 'df' not in st.session_state:
    print("Re-initializing 'df' (Unexpected for persistence)...")
    st.session_state.df = pd.DataFrame({'B': [4, 5, 6]})
else:
    print("'df' persists.")

if 'bucketing_ops' not in st.session_state:
    print("Re-initializing 'bucketing_ops' (Unexpected for persistence)...")
    st.session_state.bucketing_ops = []
else:
    print("'bucketing_ops' persists.")
