import re

with open('causal_agent_app.py', 'r') as f:
    content = f.read()

methodology_block = """        st.markdown(\"\"\"
        **Methodology (Augmented Synthetic Control):**
        GeoLift uses the Augmented Synthetic Control Method (ASCM) developed by Meta to estimate the causal effect of an intervention at the geographic level. It constructs a "synthetic" version of the treated location by finding a weighted combination of untreated locations that best matches the pre-treatment time series of the treated location.
        
        **Formula Definition:**
        Let $Y_{it}$ be the outcome for geography $i$ at time $t$. Let region $1$ be the treated region, and regions $2, \dots, N$ be the donor pool.
        
        The Synthetic Control estimator seeks weights $W = (w_2, \dots, w_N)$ such that the pre-intervention distance is minimized:
        $$ \min_W \sum_{t=1}^{T_0} (Y_{1t} - \sum_{j=2}^N w_j Y_{jt})^2 $$
        subject to $w_j \ge 0$ and $\sum w_j = 1$.
        
        In **Augmented Synthetic Control** (used by GeoLift), ridge regression is added to adjust for poor pre-treatment fit:
        $$ \hat{\\tau}_{1t} = Y_{1t} - (\sum_{j=2}^N \hat{w}_j Y_{jt} + (\mathbf{X}_{1t} - \sum_{j=2}^N \hat{w}_j \mathbf{X}_{jt})^T \hat{\\beta}) $$
        
        **Reference:**
        [GeoLift: Meta's open source solution for Geo-based Experimentation](https://github.com/facebookincubator/GeoLift)
        \"\"\")"""

# 1. Remove from pre-run input section
if methodology_block in content:
    content = content.replace(methodology_block, "")
    
# 2. Add to post-run results section
post_run_target = """        elif quasi_method_run == "GeoLift (Synthetic Control via R)":
            st.divider()"""
            
if post_run_target in content:
    content = content.replace(post_run_target, post_run_target + "\n\n" + methodology_block + "\n            st.divider()")

with open('causal_agent_app.py', 'w') as f:
    f.write(content)

print("Moved GeoLift methodology to results section")
