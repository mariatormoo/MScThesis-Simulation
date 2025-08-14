# Import Libraries
from __future__ import annotations

import textwrap
from typing import Optional
import importlib
import streamlit as st
#from Modules.module1_financial_forecasting.forecasting_app import run_forecasting_module
#from Modules.module2_scenario_planning.scenario_app import run_scenario_module
#from Modules.module3_generative_AI.ai_reporting_app import run_ai_reporting_module

#Streamlit App
#- Beautiful hero header
#- Sidebar navigation with deep-links (URL query parameters)
#- Homepage with module cards
#- Safe + Lazy Module Loading
#- Global settings panel + demo mode + reset
#- Friendly Status/Toasts and helpful error hanldling
#- Footer with contact information

# ------------------------------
# Set Page Configuration + Theme-friendly Tweaks
# ------------------------------
st.set_page_config(
    page_title="CFO Simulation Tool: AI-Powered Toolkit",
    page_icon="🧠",  # Emoji icon for the page
    layout="wide",  # Full-width layout
    initial_sidebar_state="expanded",  # Sidebar is expanded by default
    menu_items={
        'Get Help': "https://docs.streamlit.io/",
        'Report a Bug': "https://github.com/streamlit/streamlit/issues",
        'About': "CFO Simulation Tool with basic features. "
        "This application includes different modules addressing key areas the CFOs are involved in. "
        "It aims to show how even simple AI-based tools could impact aspects like accuracy, efficiency, and decision-making. ",
    }
)

# Add a logo or image if needed
#st.sidebar.image("/Users/mariatormo02/Downloads/logan-armstrong-hVhfqhDYciU-unsplash.jpg", use_column_width=True)

# Small CSS polish (kept lightweight, theme-compatible)
st.markdown(
    """
    <style>
      .app-hero {
        background: linear-gradient(120deg, rgba(14,165,233,.14), rgba(124,58,237,.14));
        border: 1px solid rgba(125,125,125,.15);
        padding: 1.25rem 1.25rem 1rem 1.25rem;
        border-radius: 1.25rem;
        margin-bottom: 1.25rem;
      }
      .module-card {
        border: 1px solid rgba(125,125,125,.18);
        border-radius: 1rem;
        padding: 1rem;
        transition: transform .08s ease, box-shadow .12s ease;
      }
      .module-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,.08); }
      .muted { opacity: .85 }
      .pill {font-size:.8rem;padding:.25rem .6rem;border-radius:999px;border:1px solid rgba(125,125,125,.25)}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# URL state helpers (support both new + old APIs)
# ----------------------------

def get_qp(key: str, default: str) -> str:
    """Read a query param in a version-tolerant way."""
    try:
        # Streamlit >= 1.30
        return st.query_params.get(key, default)  # type: ignore[attr-defined]
    except Exception:
        # Older API
        qp = st.experimental_get_query_params()  # {k: [v]}
        return (qp.get(key, [default]) or [default])[0]


def set_qp(**kwargs) -> None:
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        for k, v in kwargs.items():
            qp[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)


# ----------------------------
# Global App/Session State
# ----------------------------
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True
if "active_page" not in st.session_state:
    st.session_state.active_page = get_qp("page", "home")



# ----------------------------
# Sidebar: Branding + Navigation + Settings
# ----------------------------
with st.sidebar:
    st.markdown("### 🧠 CFO Simulation Tool")
    st.caption("Decision Intelligence for Finance Leaders")

    # Navigation
    # Define navigation mapping (pages - keys)
    pages_dict = {
        "🏠 Home": "home",
        "📈 Financial Forecasting": "forecasting",
        "🧮 Scenario Planning": "scenario",
       "📝 Generative AI Decision Reports": "ai_reports",
    }
    
    # Ensure active_page exists in session state
    if "active_page" not in st.session_state:
        st.session_state.active_page = list(pages_dict.values())[0] # Default Home Page

    labels = list(pages_dict.keys())
    values = list(pages_dict.values())


    # Preselect index based on session state
    try:
        idx = values.index(st.session_state.active_page)
    except ValueError:
        idx = 0

    # Navigation Radio (one widget)
    choice = st.radio("Navigation", labels, index=idx, help="Select which module to explore", label_visibility="visible")
    st.caption("Tip: Use the '?' icons to see help text on individual controls.")

    # Update Session State and set query params
    st.session_state.active_page = pages_dict[choice]
    set_qp(page=st.session_state.active_page)

    st.divider()

    # Global settings
    st.subheader("⚙️ Settings")
    st.session_state.demo_mode = st.toggle("Demo mode (use sample data)", value=st.session_state.demo_mode, help="Helpful for Demo scenarios without uploading files.")

    if st.button("🔄 Reset app state"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.toast("State cleared. Reloading…")
        st.rerun()

    # Add a footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by [María Tormo Nieto](https://www.linkedin.com/in/maría-tormo-nieto) | Contact: maria.tormo@alumni.esade.edu") 

# ----------------------------
# Header
# ----------------------------
st.title("CFO Simulation Tool")
st.markdown("A prototype to explore AI-enhanced CFO tools.")
with st.container():
    st.markdown(
        """
        <div class="app-hero">
          <h3 style="margin:0">Make Better Decisions, Faster.</h3>
          <p class="muted" style="margin:.25rem 0 0 0">Forecast cash, stress-test scenarios, and auto-generate executive-ready decision reports.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Lazy imports so the app opens instantly
# ----------------------------

def _safe_run(module_name: str, runner: callable) -> None:
    """Run a module with friendly error UI."""
    try:
        with st.status(f"Loading {module_name}…", expanded=False) as s:
            runner()
            s.update(label=f"{module_name} ready", state="complete")
    except Exception as e:  # noqa: BLE001 (we want to surface everything nicely)
        st.error(f"{module_name} encountered an error.")
        st.exception(e)


# ----------------------------
# PAGES
# ----------------------------

def page_home() -> None:
    st.subheader("Select a Module")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Financial Forecasting")
        #st.markdown("Cashflow, runway, and key ratios.")
        st.markdown("<span class='pill'>Time series</span> <span class='pill'>ARIMA/Prophet/XGBoost</span> <span class='pill'>What‑ifs</span>", unsafe_allow_html=True)
        st.markdown(
            """
            **Purpose:** Quickly forecast revenues, costs, and financial metrics.

            **Features:**
            - Train AI models (ARIMA, Prophet, XGBoost) on demo or uploaded data.
            - Adjust inputs like seasonality, market trends, or shocks.
            - Compare **AI forecasts vs. manual CFO predictions**.
            - Outputs: accuracy metrics, graphs, scenario comparisons.

            **Objective:** Quantify differences in speed, accuracy, and bias detection.
            """,
            unsafe_allow_html=True,
        )
        # Open Forecasting Module selection
        if st.button("Open Forecasting", type="primary", key="go_forecasting"):
            st.session_state.active_page = "forecasting"
            set_qp(page="forecasting")
            st.rerun()
        st.markdown("---")
        st.caption("Upload Historicals or use Demo data to generate forward-looking projections with uncertainties.")

    with c2:
        st.markdown("#### 2. Scenario Planning")
        #st.markdown("Stress-test and compare strategic paths.")
        st.markdown("<span class='pill'>Best/Base/Worst Cases</span> <span class='pill'>Monte Carlo</span> <span class='pill'>Sensitivity</span> <span class='pill'>KPIs</span>", unsafe_allow_html=True)
        st.markdown(
            """
            **Purpose:** Test “what-if” scenarios for strategic planning.

            **Features:**
            - Modify drivers (inflation, interest rates, sales decline) via sliders.
            - Dynamically recalculate forecasts and visualize outcomes.
            - Visual outputs: waterfall charts, sensitivity tornado graphs, stress tests.

            **Objective:** Explore impact of assumptions on KPIs over 3–5 year horizons.
            """,
            unsafe_allow_html=True,
        )
        # Open Scenario Planning Module selection
        if st.button("Open Scenario Planning", type="primary", key="go_scenario"):
            st.session_state.active_page = "scenario"
            set_qp(page="scenario")
            st.rerun()
        st.markdown("---")
        st.caption("Define drivers and assumptions; see outcomes, distributions and risk heatmaps.")

    with c3:
        st.markdown("#### 3. Generative AI Decision Reports")
        #st.markdown("Auto-draft executive memos from your numbers.")
        st.markdown("<span class='pill'>Narratives</span> <span class='pill'>Charts</span> <span class='pill'>Action items</span>", unsafe_allow_html=True)
        st.markdown(
            """
            **Purpose:** Auto-generate executive-ready reports from your data.

            **Features:**
            - Integrates GPT-4 API for narrative generation.
            - Generate forecast summaries, risk reports, or board presentation content.
            - Include insights, risk/opportunity narratives, and recommended actions.

            **Objective:** Turn numbers and KPIs into concise, actionable reports instantly.
            """,
            unsafe_allow_html=True,
        )
        # Open AI Reports Module selection
        if st.button("Open AI Reports", type="primary", key="go_ai"):
            st.session_state.active_page = "ai_reports"
            set_qp(page="ai_reports")
            st.rerun()
        st.markdown("---")
        st.caption("Turn insights into clear, structured recommendations for the board.")

    st.divider()
    with st.expander("👀 What judges want to see (quick checklist)"):
        st.markdown(
            """
            - Immediate clarity (what it does, in 1 sentence)
            - Minimal clicks to ‘wow’ (demo data preloaded)
            - Beautiful defaults (theme, spacing, typography)
            - Speed (lazy loading + caching)
            - A clear story: **input → analysis → insight → action**
            - Export or share output (e.g., PDF/Markdown)
            """
        )

# Financial Forecasting Page
def page_forecasting() -> None:
    st.subheader("📈 Financial Forecasting")
    st.caption("Upload Data or use Demo mode, then generate forecasts with intervals.")
    def runner():
        # Lazy import keeps startup fast
        from Modules.module1_financial_forecasting.forecasting_app import (
            run_forecasting_module,
        )
        run_forecasting_module()
    _safe_run("Financial Forecasting", runner)

# Scenario Planning Page
def page_scenario() -> None:
    st.subheader("🧪 Scenario Planning")
    st.caption("Design Drivers & Assumptions, run Stress Tests, compare outcomes.")
    def runner():
        from Modules.module2_scenario_planning.scenario_app import run_scenario_module
        run_scenario_module()
    _safe_run("Scenario Planning", runner)


# Generative AI Decision Reports Page
def page_ai_reports() -> None:
    st.subheader("🤖 Generative AI Decision Reports")
    st.caption("Draft concise, board-ready narratives from your analysis.")
    def runner():
        from Modules.module3_generative_AI.ai_reporting_app import (
            run_ai_reporting_module,
        )
        run_ai_reporting_module()
    _safe_run("AI Decision Reports", runner)


# ----------------------------
# Router to different pages
# ----------------------------
page = st.session_state.active_page
if page == "home":
    page_home()
elif page == "forecasting":
    page_forecasting()
elif page == "scenario":
    page_scenario()
elif page == "ai_reports":
    page_ai_reports()
else:
    st.warning("Unknown page. Returning home…")
    set_qp(page="home")
    st.session_state.active_page = "home"
    st.rerun()

st.divider()

# Feedback Section (need to refine this, it appears on all pages)
st.markdown("🌟 Leave your feedback on this prototype:")
sentiment_mapping = ['1', '2', '3', '4', '5']
selected_sentiment = st.feedback("stars")
if selected_sentiment is not None:
    st.toast(
        f"Thank you for your feedback! You rated us {sentiment_mapping[selected_sentiment]} star(s).",
        icon="✅",
    )