# scenario_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

def run_scenario_module():
    st.header("📊 Scenario Planning (What-If Analysis)")

    # Input assumptions
    inflation = st.slider("Inflation Rate (%)", 0, 15, 5)
    sales_change = st.slider("Sales Change (%)", -50, 50, 0)
    interest_rate = st.slider("Interest Rate (%)", 0, 10, 2)

    base_sales = 100000
    base_costs = 60000
    base_interest = 2000

    adjusted_sales = base_sales * (1 + sales_change / 100)
    adjusted_costs = base_costs * (1 + inflation / 100)
    adjusted_interest = base_interest * (1 + interest_rate / 100)

    profit = adjusted_sales - adjusted_costs - adjusted_interest

    # Output
    st.metric("Adjusted Sales", f"${adjusted_sales:,.2f}")
    st.metric("Adjusted Costs", f"${adjusted_costs:,.2f}")
    st.metric("Adjusted Interest", f"${adjusted_interest:,.2f}")
    st.metric("Projected Profit", f"${profit:,.2f}")

    # Visual chart
    fig = go.Figure(go.Waterfall(
        name="20XX",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Sales", "Costs", "Interest", "Profit"],
        textposition="outside",
        text=[f"${adjusted_sales:,.0f}", f"-${adjusted_costs:,.0f}", f"-${adjusted_interest:,.0f}", f"${profit:,.0f}"],
        y=[adjusted_sales, -adjusted_costs, -adjusted_interest, profit],
    ))

    fig.update_layout(title="Scenario Waterfall Analysis")
    st.plotly_chart(fig)
