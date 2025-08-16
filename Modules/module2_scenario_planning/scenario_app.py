from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st


# =========================
# Helper Functions
# =========================

# Waterfall Chart
def _waterfall_chart(adjusted_sales: float, adjusted_costs: float, adjusted_interest: float, profit: float) -> go.Figure:
    fig = go.Figure(
        go.Waterfall(
            name="Scenario",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Sales", "Costs", "Interest", "Profit"],
            textposition="outside",
            text=[
                f"${adjusted_sales:,.0f}",
                f"-${adjusted_costs:,.0f}",
                f"-${adjusted_interest:,.0f}",
                f"${profit:,.0f}",
            ],
            y=[adjusted_sales, -adjusted_costs, -adjusted_interest, profit],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )
    # Plot Big Title
    st.subheader("Scenario Waterfall Analysis")
    #fig.update_layout(title="Scenario Waterfall Analysis", margin=dict(l=20, r=20, t=40, b=20))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


# Tornado Sensitivity
def _tornado_sensitivity(base_sales: float, base_costs: float, base_interest: float, inflation: float, sales_change: float, interest_rate: float) -> go.Figure:
    """Compute simple +/-10% sensitivity on each driver and plot as tornado bars."""

    def profit_fn(i_rate, s_change, r_rate):
        # Calculate Scenario Profit based on adjusted values, costs, and interest rate.
        adj_sales = base_sales * (1 + s_change / 100)
        adj_costs = base_costs * (1 + i_rate / 100)
        adj_interest = base_interest * (1 + r_rate / 100)
        profit = adj_sales - adj_costs - adj_interest
        return profit

    drivers = [
        ("Inflation", inflation, lambda v: profit_fn(v, sales_change, interest_rate)),
        ("Sales Change", sales_change, lambda v: profit_fn(inflation, v, interest_rate)),
        ("Interest Rate", interest_rate, lambda v: profit_fn(inflation, sales_change, v)),
    ]

    lows, highs, labels = [], [], []
    for name, val, fn in drivers:
        p_low = fn(val - 10)
        p_high = fn(val + 10)
        lows.append(p_low)
        highs.append(p_high)
        labels.append(name)

    base_profit = profit_fn(inflation, sales_change, interest_rate)
    delta_low = [p - base_profit for p in lows]
    delta_high = [p - base_profit for p in highs]

    fig = go.Figure()
    fig.add_bar(y=labels, x=delta_low, orientation="h", name="-10%", base=0)
    fig.add_bar(y=labels, x=delta_high, orientation="h", name="+10%", base=0)
    # Plot Big Title
    st.subheader("Tornado Sensitivity Analysis (±10%)")
    fig.update_layout(
        #title="Tornado Sensitivity (±10%)",
        barmode="overlay",
        xaxis_title="Δ Profit",
        margin=dict(l=80, r=20, t=40, b=30),
    )
    return fig


# Monte Carlo Simulation
def _monte_carlo(base_sales: float, base_costs: float, base_interest: float, inflation: float, sales_change: float, interest_rate: float, runs: int = 1000, seed: int | None = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    infl_samp = rng.normal(loc=inflation, scale=2.0, size=runs)
    sales_samp = rng.normal(loc=sales_change, scale=5.0, size=runs)
    rate_samp = rng.normal(loc=interest_rate, scale=1.0, size=runs)
    
    sales = base_sales * (1 + sales_samp / 100)
    costs = base_costs * (1 + infl_samp / 100)
    interest = base_interest * (1 + rate_samp / 100)
    profit = sales - costs - interest
    
    return pd.DataFrame({
        "inflation": infl_samp,
        "sales_change": sales_samp,
        "interest_rate": rate_samp,
        "sales": sales,
        "costs": costs,
        "interest": interest,
        "profit": profit,
    })


# =========================
# Main Page with Tabs
# =========================

def run_scenario_module():
    """Run the Scenario Planning app with tabs for What-If, Monte Carlo, and Sensitivity Analysis."""
    
    # Set up Streamlit page config
    st.set_page_config(
        page_title="Scenario Planning (What‑If & Risk)",
        page_icon="📊",
        layout="wide",
    )
    st.caption("Tweak drivers, compare scenarios, run a quick Monte Carlo, and see what moves the needle.")

    # Read URL Parameters for Sliders
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        qp_infl = float(qp.get("infl", 5.0))
        qp_sales = float(qp.get("sales", 0.0))
        qp_rate = float(qp.get("rate", 2.0))
    except Exception:
        params = st.experimental_get_query_params()
        qp_infl = float((params.get("infl", [5.0]) or [5.0])[0])
        qp_sales = float((params.get("sales", [0.0]) or [0.0])[0])
        qp_rate = float((params.get("rate", [2.0]) or [2.0])[0])
    
    # Base Values
    BASE_SALES = 100000.0
    BASE_COSTS = 60000.0
    BASE_INTEREST = 2000.0


    # Page Tabs
    tabs = st.tabs(["🔧 What-If Scenario", "🧮 Scenario Comparison", "🎲 Monte Carlo", "🌪️ Sensitivity Analysis", "📖 Executive Summary"])

    # --- Tab 1: Scenario Inputs & Waterfall ---
    with tabs[0]:
        # Inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            inflation = st.slider("Inflation Rate (%)", 0.0, 15.0, float(qp_infl), 0.1, help="Impact on costs and prices.")
        with col2:
            sales_change = st.slider("Sales Change (%)", -50.0, 50.0, float(qp_sales), 0.5, help="Impact on sales volume or price changes.")
        with col3:
            interest_rate = st.slider("Interest Rate (%)", 0.0, 10.0, float(qp_rate), 0.1, help="Impact on financing costs.")

        # Adjusted Calculations
        adjusted_sales = BASE_SALES * (1 + sales_change / 100)
        adjusted_costs = BASE_COSTS * (1 + inflation / 100)
        adjusted_interest = BASE_INTEREST * (1 + interest_rate / 100)
        profit = adjusted_sales - adjusted_costs - adjusted_interest

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Adjusted Sales", f"${adjusted_sales:,.2f}")
        m2.metric("Adjusted Costs", f"${adjusted_costs:,.2f}")
        m3.metric("Adjusted Interest", f"${adjusted_interest:,.2f}")
        m4.metric("Projected Profit", f"${profit:,.2f}")

        st.subheader("🌊 Scenario Waterfall")
        st.plotly_chart(_waterfall_chart(adjusted_sales, adjusted_costs, adjusted_interest, profit), use_container_width=True)

    # --- Tab 2: Scenario Comparison ---
    with tabs[1]:
        st.subheader("Compare Preset Scenarios")

        presets = {
            "Baseline": dict(inflation=inflation, sales_change=sales_change, interest_rate=interest_rate),
            "Optimistic": dict(inflation=max(inflation - 2, 0), sales_change=min(sales_change + 10, 50), interest_rate=max(interest_rate - 0.5, 0)),
            "Pessimistic": dict(inflation=min(inflation + 2, 15), sales_change=max(sales_change - 10, -50), interest_rate=min(interest_rate + 0.5, 10)),
        }

        chosen = st.multiselect("Pick scenarios", list(presets.keys()), default=["Baseline", "Optimistic", "Pessimistic"])
        rows = []
        for name in chosen:
            p = presets[name]
            s = BASE_SALES * (1 + p["sales_change"] / 100)
            c = BASE_COSTS * (1 + p["inflation"] / 100)
            r = BASE_INTEREST * (1 + p["interest_rate"] / 100)
            rows.append(dict(Scenario=name, Sales=s, Costs=c, Interest=r, Profit=s - c - r))

        if rows:
            df = pd.DataFrame(rows).set_index("Scenario")
            st.dataframe(df.style.format("${:,.0f}"))
            fig = go.Figure()
            for col in ["Sales", "Costs", "Interest", "Profit"]:
                fig.add_bar(name=col, x=df.index, y=df[col])
            # Plot Big Title
            st.subheader("Scenario Comparison Bar Plot")
            #fig.update_layout(barmode="group", title="Scenario Comparison", xaxis_title="Scenario", yaxis_title="Amount")
            fig.update_layout(barmode="group", xaxis_title="Scenario", yaxis_title="Amount")
            st.plotly_chart(fig, use_container_width=True)
        
        # Export CSV
        st.subheader("Export Scenario Data")
        csv = df.to_csv(index=True)
        st.download_button("⬇️ Download scenario table (CSV)", data=csv, file_name="scenario_compare.csv", mime="text/csv")



    # --- Tab 3: Monte Carlo Simulation ---
    with tabs[2]:
        st.subheader("🎲 Quixk Monte Carlo Simulation (1,000 runs default)")
        runs = st.slider("Runs", 200, 5000, 1000, 100)
    
        with st.status("Running simulation…", expanded=False):
            mc_df = _monte_carlo(BASE_SALES, BASE_COSTS, BASE_INTEREST, inflation, sales_change, interest_rate, runs=runs)
    
        p5, p50, p95 = np.percentile(mc_df["profit"], [5, 50, 95])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P5", f"${p5:,.0f}")
        c2.metric("Median", f"${p50:,.0f}")
        c3.metric("P95", f"${p95:,.0f}")
        c4.metric("Pr(loss)", f"{(mc_df['profit'] < 0).mean():.1%}")
    
        hist = go.Figure()
        hist.add_histogram(x=mc_df["profit"], nbinsx=40, name="Profit Distribution")
        #hist.add_histogram(x=mc_df["profit"], nbinsx=50, name="Profit Distribution", marker_color="skyblue")
        #hist.add_vline(x=p5, line=dict(color="red", dash="dash"), annotation_text="P5")
        #hist.add_vline(x=p50, line=dict(color="green", dash="dash"), annotation_text="Median")
        #hist.add_vline(x=p95, line=dict(color="blue", dash="dash"), annotation_text="P95")
        # Plot Big Title
        st.subheader("Monte Carlo Profit Distribution")
        #hist.update_layout(title="Profit Distribution", xaxis_title="Profit", yaxis_title="Frequency")
        hist.update_layout(xaxis_title="Profit", yaxis_title="Frequency")
        st.plotly_chart(hist, use_container_width=True)


    # --- Tab 4: Tornado Sensitivity ---
    with tabs[3]:
        # Tornado sensitivity
        with st.expander("🌪️ What moves profit most? (±10%)", expanded=False):
            st.plotly_chart(_tornado_sensitivity(BASE_SALES, BASE_COSTS, BASE_INTEREST, inflation, sales_change, interest_rate), use_container_width=True)


    # --- Tab 5: Executive Summary / Download ---
    with tabs[4]:
        # Executive summary download of the Monte Carlo 
        summary_md = f"""
        ## Executive Summary
        - Inflation: {inflation:.1f}% | Sales Δ: {sales_change:.1f}% | Rate: {interest_rate:.1f}%
        - Profit: ${profit:,.0f} (P5 ${p5:,.0f} · Median ${p50:,.0f} · P95 ${p95:,.0f})
        - Loss probability: {(mc_df['profit'] < 0).mean():.1%}
        """
        st.download_button("📥 Download Monte Carlo Summary (MD)", summary_md, "scenario_summary.md")

    

if __name__ == "__main__":
    # Run the scenario module directly if this script is executed
    run_scenario_module()

    

# WORKING ALL TOGETHER

def run_scenario_module():
    st.header("📊 Scenario Planning (What‑If & Risk)")
    st.caption("Tweak drivers, compare scenarios, run a quick Monte Carlo, and see what moves the needle.")

    # Read from URL if present
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        qp_infl = float(qp.get("infl", 5.0))
        qp_sales = float(qp.get("sales", 0.0))
        qp_rate = float(qp.get("rate", 2.0))
    except Exception:
        params = st.experimental_get_query_params()
        qp_infl = float((params.get("infl", [5.0]) or [5.0])[0])
        qp_sales = float((params.get("sales", [0.0]) or [0.0])[0])
        qp_rate = float((params.get("rate", [2.0]) or [2.0])[0])

    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        inflation = st.slider("Inflation Rate (%)", 0.0, 15.0, float(qp_infl), 0.1)
    with col2:
        sales_change = st.slider("Sales Change (%)", -50.0, 50.0, float(qp_sales), 0.5)
    with col3:
        interest_rate = st.slider("Interest Rate (%)", 0.0, 10.0, float(qp_rate), 0.1)

    # Update URL helper
    if st.button("🔗 Update URL with sliders"):
        try:
            st.query_params.update({"infl": inflation, "sales": sales_change, "rate": interest_rate})  # type: ignore[attr-defined]
        except Exception:
            st.experimental_set_query_params(infl=inflation, sales=sales_change, rate=interest_rate)
        st.toast("URL updated – copy from your browser.")

    base_sales = 100_000.0
    base_costs = 60_000.0
    base_interest = 2_000.0

    adjusted_sales = base_sales * (1 + sales_change / 100)
    adjusted_costs = base_costs * (1 + inflation / 100)
    adjusted_interest = base_interest * (1 + interest_rate / 100)
    profit = adjusted_sales - adjusted_costs - adjusted_interest

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Adjusted Sales", f"${adjusted_sales:,.2f}")
    m2.metric("Adjusted Costs", f"${adjusted_costs:,.2f}")
    m3.metric("Adjusted Interest", f"${adjusted_interest:,.2f}")
    m4.metric("Projected Profit", f"${profit:,.2f}")

    st.plotly_chart(_waterfall_chart(adjusted_sales, adjusted_costs, adjusted_interest, profit), use_container_width=True)

    st.divider()

    # Scenario presets & comparison
    st.subheader("Compare Scenarios")
    presets = {
        "Baseline": dict(inflation=inflation, sales_change=sales_change, interest_rate=interest_rate),
        "Optimistic": dict(inflation=max(inflation - 2, 0), sales_change=min(sales_change + 10, 50), interest_rate=max(interest_rate - 0.5, 0)),
        "Pessimistic": dict(inflation=min(inflation + 2, 15), sales_change=max(sales_change - 10, -50), interest_rate=min(interest_rate + 0.5, 10)),
    }
    chosen = st.multiselect("Pick scenarios", list(presets.keys()), default=["Baseline", "Optimistic", "Pessimistic"])
    rows = []
    for name in chosen:
        p = presets[name]
        s = base_sales * (1 + p["sales_change"] / 100)
        c = base_costs * (1 + p["inflation"] / 100)
        r = base_interest * (1 + p["interest_rate"] / 100)
        rows.append(dict(Scenario=name, Sales=s, Costs=c, Interest=r, Profit=s - c - r))
    if rows:
        df = pd.DataFrame(rows).set_index("Scenario")
        st.dataframe(df.style.format("${:,.0f}"))
        fig = go.Figure()
        for col in ["Sales", "Costs", "Interest", "Profit"]:
            fig.add_bar(name=col, x=df.index, y=df[col])
        # Plot Big Title
        st.subheader("Scenario Comparison Bar Plot")
        #fig.update_layout(barmode="group", title="Scenario Comparison", xaxis_title="Scenario", yaxis_title="Amount")
        fig.update_layout(barmode="group", xaxis_title="Scenario", yaxis_title="Amount")
        st.plotly_chart(fig, use_container_width=True)

        # Export CSV
        csv = df.to_csv(index=True)
        st.download_button("⬇️ Download scenario table (CSV)", data=csv, file_name="scenario_compare.csv", mime="text/csv")


    st.divider()

    # Monte Carlo
    with st.expander("🎲 Quick Monte Carlo (1,000 runs)", expanded=False):
        runs = st.slider("Runs", 200, 5000, 1000, 100)
        with st.status("Running simulation…", expanded=False):
            mc_df = _monte_carlo(base_sales, base_costs, base_interest, inflation, sales_change, interest_rate, runs=runs)
        p5, p50, p95 = np.percentile(mc_df["profit"], [5, 50, 95])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P5", f"${p5:,.0f}")
        c2.metric("Median", f"${p50:,.0f}")
        c3.metric("P95", f"${p95:,.0f}")
        c4.metric("Pr(loss)", f"{(mc_df['profit'] < 0).mean():.1%}")
        hist = go.Figure()
        hist.add_histogram(x=mc_df["profit"], nbinsx=40, name="Profit dist.")
        # Plot Big Title
        st.subheader("Monte Carlo Profit Distribution")
        #hist.update_layout(title="Profit Distribution", xaxis_title="Profit", yaxis_title="Frequency")
        hist.update_layout(xaxis_title="Profit", yaxis_title="Frequency")
        st.plotly_chart(hist, use_container_width=True)

        # Executive summary download on this page
        summary_md = f"""
        ## Executive Summary
        - Inflation: {inflation:.1f}% | Sales Δ: {sales_change:.1f}% | Rate: {interest_rate:.1f}%
        - Profit: ${profit:,.0f} (P5 ${p5:,.0f} · Median ${p50:,.0f} · P95 ${p95:,.0f})
        - Loss probability: {(mc_df['profit'] < 0).mean():.1%}
        """
        st.download_button("📥 Download Summary (MD)", summary_md, "scenario_summary.md")

    st.divider()

    # Tornado sensitivity
    with st.expander("🌪️ What moves profit most? (±10%)", expanded=False):
        st.plotly_chart(_tornado_sensitivity(base_sales, base_costs, base_interest, inflation, sales_change, interest_rate), use_container_width=True)

    # Export
    if rows:
        csv = df.to_csv(index=True)
        st.download_button("⬇️ Download scenario table (CSV)", data=csv, file_name="scenario_compare.csv", mime="text/csv")
