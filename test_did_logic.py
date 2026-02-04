import pandas as pd
import numpy as np
from dowhy import CausalModel
import warnings
warnings.filterwarnings("ignore")

# Simulate DiD Data
np.random.seed(42)
n = 1000
# 2 Time Periods (0, 1)
time = np.random.binomial(1, 0.5, n)
# 2 Groups (Control=0, Treat=1)
group = np.random.binomial(1, 0.5, n)
# Interaction (Treatment happens in period 1 for group 1)
# DiD = (E[Y|G=1,T=1] - E[Y|G=1,T=0]) - (E[Y|G=0,T=1] - E[Y|G=0,T=0])
# Here 'treatment' in the causal sense is the Interaction term for the ATT.
# But usually we define Treatment as Group assignment, and Time as a confounder/modifier?
# Standard DiD Regression: Y = a + b1*Group + b2*Time + b3*(Group*Time) + e
# b3 is the effect.

# Let's see how the app does it.
# App uses: treatment='Feature_Adoption' (which is the Group assignment? or the actual treatment status?)
# In the simulation:
# quarter = 0 or 1
# feature_adoption = binary (depends on quarter?)

# Let's mimic the app's simulation for DiD
# Quarter (0=Pre, 1=Post)
quarter = np.random.binomial(1, 0.5, n)
# Customer Segment (Group)
segment = np.random.binomial(1, 0.5, n)
# Treatment (Feature Adoption) - In a DiD, Treatment is usually Group * Time (if we care about the effect of the intervention that started at Time 1)
# OR Treatment is Group, and we look at the interaction.
# The app selects "Treatment" as "Feature_Adoption".
# If "Feature_Adoption" is 1 only for (Group=Treat, Time=Post), then it IS the interaction term.
# Let's check the simulation logic in the app.
# prob_adoption depends on segment, usage, nudge.
# It does NOT explicitly depend on Quarter in the reverted simulation.
# Wait, the reverted simulation has:
# prob_adoption = 1 / (1 + np.exp(-( -2 + 0.5 * customer_segment ...)))
# It does NOT depend on Quarter.
# So Feature_Adoption is assigned based on Segment (Group) but invariant to Time?
# If Feature_Adoption is just "Group Assignment", then we need an interaction with Time.
# If Feature_Adoption is "Actual Treatment" (0 in Pre, 1 in Post for Treated Group), then it IS the interaction.

# Let's assume Feature_Adoption is the "Treatment Status" (0 or 1).
# If it's a DiD, we expect Feature_Adoption to be 0 for everyone in Pre-period.
# The current simulation allows Feature_Adoption to be 1 in Pre-period (Quarter=0).
# This violates the standard DiD setup where treatment starts at T=1.
# UNLESS Feature_Adoption is just "Assigned to Treatment Group".

print("Analyzing Simulation Logic for DiD...")
# In the app:
# treatment = st.selectbox(..., 'Feature_Adoption')
# time_period = st.selectbox(..., 'Quarter')

# If Feature_Adoption is "Group", we need Group * Time.
# If Feature_Adoption is "Active Treatment", we just regress Y on Feature_Adoption + Unit/Time fixed effects?

# Let's run the model as the app does:
df = pd.DataFrame({
    'Feature_Adoption': np.random.binomial(1, 0.5, n),
    'Quarter': np.random.binomial(1, 0.5, n),
    'Revenue': np.random.normal(100, 10, n)
})

model = CausalModel(
    data=df,
    treatment='Feature_Adoption',
    outcome='Revenue',
    common_causes=['Quarter'] # Time period is treated as a confounder
)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(f"Identified Estimand: {identified_estimand.estimand_type}")

# Manual DiD Estimation (mimicking app logic)
import statsmodels.api as sm

print("\n--- Testing Manual DiD Logic ---")
df['DiD_Interaction'] = df['Feature_Adoption'] * df['Quarter']
X_did = df[['Feature_Adoption', 'Quarter', 'DiD_Interaction']]
X_did = sm.add_constant(X_did)
y_did = df['Revenue']

did_model = sm.OLS(y_did, X_did).fit()
print(did_model.summary())

did_estimate = did_model.params['DiD_Interaction']
print(f"\nEstimated DiD Effect: {did_estimate:.4f}")

# True effect is 0 in this random data, but we check if the code runs and produces an estimate.
if not np.isnan(did_estimate):
    print("SUCCESS: DiD Estimate calculated.")
else:
    print("FAILURE: DiD Estimate is NaN.")
