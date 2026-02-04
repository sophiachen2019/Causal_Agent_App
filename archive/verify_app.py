import sys
from unittest.mock import MagicMock

# Mock streamlit
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st

# Mock sidebar
mock_st.sidebar = MagicMock()
mock_st.sidebar.__enter__ = MagicMock(return_value=None)
mock_st.sidebar.__exit__ = MagicMock(return_value=None)

# Mock expander
mock_st.expander = MagicMock()
mock_st.expander.__enter__ = MagicMock(return_value=None)
mock_st.expander.__exit__ = MagicMock(return_value=None)

# Mock spinner
mock_st.spinner = MagicMock()
mock_st.spinner.__enter__ = MagicMock(return_value=None)
mock_st.spinner.__exit__ = MagicMock(return_value=None)

# Mock columns
def mock_columns(n):
    return [MagicMock() for _ in range(n)]
mock_st.columns = MagicMock(side_effect=mock_columns)

# Mock inputs
mock_st.number_input.return_value = 50
mock_st.slider.return_value = 0.05
mock_st.selectbox.return_value = "Double Machine Learning (LinearDML)"
mock_st.checkbox.return_value = False # Disable optional steps
mock_st.radio.return_value = "Simulated Data"
mock_st.button.return_value = True

# Mock multiselect with side_effect to handle different calls
# Order of calls in app (assuming checkbox=False):
# 1. log_transform_cols
# 2. standardize_cols
# 3. confounders
# We want the first two to be empty to skip loops, and third to be valid.
mock_st.multiselect.side_effect = [[], [], ['Customer_Segment']]

# Mock dowhy
mock_dowhy = MagicMock()
sys.modules['dowhy'] = mock_dowhy
mock_model = MagicMock()
mock_dowhy.CausalModel.return_value = mock_model

# Mock estimate
mock_estimate = MagicMock()
mock_estimate.value = 100.0
mock_estimate.stderr = 5.0
mock_estimate.get_confidence_intervals.return_value = (90.0, 110.0)
mock_model.estimate_effect.return_value = mock_estimate

# Mock refute
mock_refute = MagicMock()
mock_refute.estimated_effect = 100.0
mock_refute.new_effect = 99.0
mock_refute.refutation_result = {'p_value': 0.6}
mock_model.refute_estimate.return_value = mock_refute

# Mock pandas
import pandas as pd
# Ensure real pandas is used if possible, but our mocks should handle flow.

try:
    import causal_agent_app
    print("Import and Execution successful")
except NameError as e:
    print(f"NameError: {e}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
