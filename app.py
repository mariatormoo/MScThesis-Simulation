# Import Libraries
import streamlit as st
from forecasting import run_forecasting_module
from scenario_planning import run_scenario_module
from ai_reporting import run_ai_reporting_module

# Set Page Configuration
st.set_page_config(page_title="CFO Simulation Tool", layout="wide")

# Set Design
st.title("🧠 CFO Simulation Tool")
st.markdown("A prototype to explore AI-enhanced CFO tools.")

# Set Modules
module = st.sidebar.radio("Choose a module:", [
    "1. Financial Forecasting",
    "2. Scenario Planning",
    "3. Generative AI Decision Reports"
])

if module == "1. Financial Forecasting":
    run_forecasting_module()

elif module == "2. Scenario Planning":
    run_scenario_module()

elif module == "3. Generative AI Decision Reports":
    run_ai_reporting_module()
