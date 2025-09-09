# Streamlit UI for Module 3: Generative AI Decision Reports
from __future__ import annotations

# Import Libraries
import os
import streamlit as st


#Helper
def _get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY") 
    if api_key:
        return api_key
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None
    return None


def _get_openai_client():
    """
    Try to import the new (>=1.0) or legacy OpenAI SDK.
    Always return a tuple (mode, client) where mode is "new", "legacy" or None.
    """
    
    try:
        # new SDK Style
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None))
        return ("new", client)
    except Exception:
        try:
            # legacy SDK Style
            import openai
            return ("legacy", openai)
        except Exception:
            #st.error("OpenAI SDK not found. Please install the OpenAI package.")
            return ("stub", None)
        

# Small set of templates for different document types
TEMPLATES = {
    "Forecast Summary": """
    You are an expert data analyst. 
    Analyze the following sales data and provide a concise summary of the forecasted trends, key insights, 
    and any recommendations. Make it easy to understand for a non-technical audience.

    Write a concise CFO-ready forecast summary based on the data provided.

    Context:
     - Key metrics: {metrics}
     - Scenario Horizon: {horizon} months
     - Risk Appetite: {risk_appetite}

    Deliver 5-8 crisp bullet points with numbers where possible, in professional tone.
    """,

    "Risk Report": """
    You are a risk management expert. 
    Analyze the following sales forecast data and identify potential risks, uncertainties, and mitigation strategies.
    Provide a clear and actionable risk report that highlights key risk factors and recommendations for managing them.      
   
    Produce a risk report for the board based on the data provided.

   Context:
     - Key financial metrics: {metrics}
     - Risk Appetite: {risk_appetite}

    Include: top risks (financial, market, operational), early indicators, mitigation strategies, and rough impact ranges,
    including an impact assessment with likelihoods. Use bullet points and professional tone.
    """,

    "Board Presentation": """
    You are an expert business consultant. 

    Create a professional board presentation outline based on the following data.
    The outline should include key insights, strategic recommendations, and visual aids to effectively 
    communicate the forecast to the board of directors.     
    
    Draft a one-pager for the board.
    Include three section with headings: Outlook, Risks & Opportunities, Recommended Actions.
    Structure with clear sections and suggested visuals
    Ground it in: {metrics}. Risk appetite: {risk_appetite}.
    Use bullet points and keep it under 200 words. Use a formal and engaging tone.
    """,

}



def _call_openai(prompt: str) -> str:
    """
    Attempt to call OpenAI using either the new or legacy SDK.
    If no API key or client is available, return a helpful local stub.
    """

    mode, client = _get_openai_client()
    api_key = _get_api_key()

    # Control missing API key
    if not api_key:
        return(
            "- AI report stub (no API key found)\n"
            "- Add OPEN_API_KEY env var to eanble real generation\n"
            " - Key Takeaways: stable outlook, monitor costs, hedge rates, diversify suppliers, maintain liquidity."
        )

    try:
        # Call the appropriate OpenAI API based on the SDK version
        if mode == "new" and client:
            try:
                # New SDK Style
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert CFO assistant. Be concise, professional and actionable."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature = 0.3, # Lower temperature for more focused responses
                )
                return response.choices[0].message.content.strip()
            
            except Exception:
                return f"Error calling OpenAI API"
            
        elif mode == "legacy" and client:
            try:
                # Legacy SDK Style
                client.api_key = api_key
                response = client.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert CFO assistant. Be concise, professional and actionable."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3, # Lower temperature for more focused responses
                )
                return response.choices[0].text.strip()
            
            except Exception:
                return f"Error calling OpenAI API"
            
    except Exception:
        return f"Error initializing OpenAI client"



# Run Module
def run_genai_module():
    st.header("🧾 Generative AI: Decision Report Generator")
    st.caption("Generate CFO-style narrative reports with GPT based on your data.")

    doc_type = st.selectbox("Document Type", list(TEMPLATES.keys()), index=0)
    horizon = st.slider("Scenario Horizon (months)", min_value=1, max_value=24, value=6, step=1)
    metrics = st.text_area("Key Metrics (comma-separated)", "Revenue growth 5%, Gross Margin 42%, Cash 20m, Net Debt 5m")
    risk_appetite = st.selectbox("Risk Appetite", ["Low", "Moderate", "High"], index=1)

    if st.button("Generate Document", type="primary"):
        prompt = TEMPLATES[doc_type].format(metrics=metrics, horizon=horizon, risk_appetite=risk_appetite)
        
        with st.spinner("Generating report..."):
            report = _call_openai(prompt)

        st.subheader("Result")
        st.write(report)
        st.download_button("⬇️ Download Report (TXT)", report, file_name=f"{doc_type.replace(' ', '_').lower()}_report.txt")
        
