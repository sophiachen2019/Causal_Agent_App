# Causal Agent App

A powerful, interactive web application for Causal Inference and Quasi-Experimental Analysis, designed to help data scientists and researchers identify causal relationships and estimate treatment effects with confidence.

Try it out here: https://causalinferenceagent.streamlit.app/

## 🚀 Key Features

- **Observational Analysis**: Implements robust causal estimation methods including OLS/Logit, Propensity Score Matching (PSM), Inverse Propensity Weighting (IPTW), Double Machine Learning (DML), and Meta-Learners (S/T-Learner).
- **Quasi-Experimental Analysis**: Dedicated modules for **Difference-in-Differences (DiD)**, **Interrupted/Bayesian Time Series (ITS/BSTS)**, **GeoLift (Synthetic Control)**, and **CausalPy (Bayesian Synthetic Control)** for panel and geographic split-testing data.
- **Unified Impact Estimation**: GeoLift and ITS/BSTS share a fully standardized metric scorecard UI natively compiling robust 95% Confidence Intervals for Absolute, Cumulative, and Relative Lifts. 
- **AI Assistant**: An integrated chatbot powered by Gemini (Google GenAI) to provide guidance on methodology, interpret results, and suggest best practices.
- **Reproducible Research**: One-click generation of Python scripts to reproduce any analysis performed in the UI.
- **Data Preprocessing**: Built-in tools for imputation, winsorization, log transformations, and standardization.
- **Interactive EDA**: Detailed data profiling and visualizations to understand distributions and correlations before modeling.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Causal_Agent_App
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys (Optional for Chatbot)**:
   Create a `.streamlit/secrets.toml` file or set environment variables:
   ```toml
   GOOGLE_API_KEY = "your-google-api-key"
   ```

## 💻 Usage

Run the application using Streamlit:
```bash
streamlit run causal_agent_app.py
```

Navigate through the tabs:
1. **📘 User Guide**: Learn about the methodologies and how to interpret results.
2. **📊 Exploratory Analysis**: Profile your data, perform transformations and visualizations.
3. **🔍 Observational Analysis**: Run cross-sectional causal models with refutation tests.
4. **📈 Quasi-Experimental Analysis**: Analyze impact over time with DiD, BSTS, or GeoLift.
5. **💬 AI Assistant**: Chat with the agent for expert advice.

## 📁 Project Structure

- `causal_agent_app.py`: The main Streamlit application entry point and UI logic.
- `causal_utils.py`: Core backend functions for causal estimation and script generation.
- `chatbot_utils.py`: Integration with Google Gemini for the AI Assistant.
- `feedback_utils.py`: Module for collecting user feedback.
- `requirements.txt`: Python dependencies.
- `archive/`: Archived development, test, and debug scripts.

## 📦 Dependencies

- `streamlit`: UI Framework
- `dowhy`: Causal modeling and refutation
- `econml`: Advanced ML-based causal estimation
- `causalimpact`: Bayesian Structural Time Series
- `rpy2`: R to Python interface (requires R and the GeoLift package installed)
- `statsmodels`: Statistical modeling
- `google-genai`: AI Assistant integration
- `pandas`, `numpy`, `matplotlib`, `seaborn`: Data processing and visualization

## 📝 Version History

- **v5.4.0 (Latest)**: Implemented interactive Global Target & Intervention objective capture; decoupled Spatiotemporal analysis routers; enhanced Data Preview & Summary UX framework.
- **v5.3.0**: Implemented 7-method targeted Data Quality Readiness framework algorithm; migrated Chart Builder (Histogram, Box Plot, Pie, Dual Axis) entirely to Plotly interactive graph objects.
- **v5.2.0**: Extended CausalPy (Bayesian Synthetic Control) with covariate support, Advanced PyMC Sampling Parameter tuning, and proper manual `posterior_predictive` metric sampling; added Bar Mode toggle (Stacked/Grouped) to Chart Builder; refined AI Configuration Guide consistency; resolved `zoo` and `arviz==0.21.0` dependency breakage on Streamlit Community Cloud.
- **v5.1.0**: Added CausalPy (Bayesian Synthetic Control) as pure-Python alternative to GeoLift; updated User Guide and AI knowledge base.
- **v5.0.0**: Added data quality simulation engine (Low/Medium/High) and interactive AI-assistant experience for data exploration and causal analysis; advanced data preprocessing (Conditional Override, Resampling); backend API rate limiting; dataset download.
- **v4.5.0**: Restructured synthetic control and temporal causality modules to dynamically ingest multiple covariates, documented rigorous 95% Confidence Intervals for all metrics, and migrated core plots to Plotly Express.
- **v4.4.0**: Modernized UI with custom theme configuration and new HTML-based user guide.
- **v4.3.0**: Added missing advanced parameters for GeoLift analysis and embedded paper reference links.
- **v4.2.0**: Standardized inference scorecards for GeoLift and CausalImpact, added underlying time series UI data tables, and enhanced Python script exporter template.
- **v4.1.0**: Added covariate support for GeoLift models and migrated CausalImpact to the official R implementation.
- **v4.0.0**: Added GeoLift (Synthetic Control) for Market Selection and Impact Estimation.

## 📄 License

[Insert License Info Here]
