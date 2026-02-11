# Causal Agent App

A powerful, interactive web application for Causal Inference and Quasi-Experimental Analysis, designed to help data scientists and researchers identify causal relationships and estimate treatment effects with confidence.

Try it out here: https://causalinferenceagent.streamlit.app/

## 🚀 Key Features

- **Observational Analysis**: Implements robust causal estimation methods including OLS/Logit, Propensity Score Matching (PSM), Inverse Propensity Weighting (IPTW), Double Machine Learning (DML), and Meta-Learners (S/T-Learner).
- **Quasi-Experimental Analysis**: Dedicated modules for **Difference-in-Differences (DiD)** and **CausalImpact (Bayesian Structural Time Series)** for panel and time-series data.
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
4. **📈 Quasi-Experimental Analysis**: Analyze impact over time with DiD or BSTS.
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
- `statsmodels`: Statistical modeling
- `google-genai`: AI Assistant integration
- `pandas`, `numpy`, `matplotlib`, `seaborn`: Data processing and visualization

## 📄 License

[Insert License Info Here]
