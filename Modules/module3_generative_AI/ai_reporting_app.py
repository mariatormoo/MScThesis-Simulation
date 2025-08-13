from __future__ import annotations

import streamlit as st
import openai
import os
from dataclasses import dataclass
from typing import Optional

# Optional (only used if the OpenAI SDK is installed). We import lazily inside the function.

@dataclass
class ReportConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.4
    audience: str = "Board of Directors"
    tone: str = "Concise & executive"
    length: str = "Medium"
    sections: tuple[str, ...] = (
        "Executive Summary",
        "KPIs",
        "Risks",
        "Opportunities",
        "Next Actions",
    )


def _read_upload(file) -> str:
    if file is None:
        return ""
    try:
        content = file.read()
        try:
            return content.decode("utf-8")
        except Exception:
            return content.decode("latin-1", errors="ignore")
    except Exception:
        return ""
    
def _build_prompt(cfg: ReportConfig, key_metrics: str, context_text: str) -> tuple[str, list[dict]]:
    """Returns (system_message, user_messages)."""
    length_map = {"Short": "~300 words", "Medium": "~600 words", "Long": "~900 words"}
    sections_md = "".join(f"- {s}" for s in cfg.sections)
    system = (
        "You are a CFO copilot that writes clear, structured, board-ready decision reports. "
        "Use bullet points where helpful, include numbers and percentages if present, and keep the tone professional. "
        "Use Markdown headings for sections. If data is missing, do not fabricate; state assumptions explicitly."
    )
    user = f"""
    Create a **{cfg.length}** ({length_map.get(cfg.length, '')}) **{cfg.tone}** report for the **{cfg.audience}**.
    **Sections to include (in this exact order):**
{sections_md}

    **Financial input (verbatim / unstructured):**
{key_metrics}

    **Additional context (optional):**
{context_text}

    Rules:
    - Start with a crisp one-paragraph Executive Summary.
    - Use Markdown headings (#, ##) and lists; avoid tables unless necessary.
    - End with a short 'Decision & Owners' list under 'Next Actions'.
    """
    return system, [{"role": "user", "content": user}]


def run_ai_reporting_module():
    st.header("🧾 Generative AI: Decision Report Generator")
    st.caption("Draft polished, executive-ready narratives from your numbers.")


    # --- Sidebar / settings
    with st.expander("Report settings", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0, help="Pick a fast, cost-effective model for demos.")
        with col2:
            temperature = st.slider("Creativity", 0.0, 1.0, 0.4, 0.05)
        with col3:
            length = st.select_slider("Length", options=["Short", "Medium", "Long"], value="Medium")
        col4, col5 = st.columns(2)
        with col4:
            audience = st.text_input("Audience", value="Board of Directors")
        with col5:
            tone = st.selectbox("Tone", ["Concise & executive", "Neutral", "Persuasive", "Analytical"], index=0)
        sections = st.multiselect(
            "Sections",
            ["Executive Summary", "KPIs", "Risks", "Opportunities", "Scenarios", "Next Actions"],
            default=["Executive Summary", "KPIs", "Risks", "Opportunities", "Next Actions"],
        )

    # --- Inputs
    api_prefill = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    api_key = st.text_input("OpenAI API key (or set in Secrets)", value=api_prefill, type="password", help="Store it in Secrets for convenience.")

    c1, c2 = st.columns([2, 1])
    with c1:
        key_metrics = st.text_area("Paste key financial metrics / notes", height=200, placeholder="Revenue YoY +22%... Cash runway 13 months... CAC payback 9.2 months...")
    with c2:
        context_file = st.file_uploader("Optional: attach context (txt/csv/md)", type=["txt", "csv", "md"])
        demo_mode = st.checkbox("Demo mode (no API call)", value=st.session_state.get("demo_mode", True))

    context_text = _read_upload(context_file)

    cfg = ReportConfig(
        model=model, temperature=temperature, audience=audience, tone=tone, length=length, sections=tuple(sections)
    )

    generate = st.button("🚀 Generate report", type="primary")

    if not generate:
        return

    system_msg, user_msgs = _build_prompt(cfg, key_metrics.strip(), context_text.strip())

    with st.status("Preparing your report…", expanded=False) as s:
        try:
            if demo_mode or not api_key:
                s.update(label="Demo mode: generating sample output…")
                # Deterministic demo output
                demo_report = f"""
# Executive Summary
- Momentum remains strong with improving gross margins and disciplined opex. Cash runway > 12 months.

## KPIs
- Revenue growth: +22% YoY; GM: 61% (+3pp); NRR: 114%; CAC payback: 9.2 months.

## Risks
- Topline sensitivity to {audience.lower()} approvals; FX headwinds; hiring pace.

## Opportunities
- Price uplift in Enterprise; self-serve funnel; working capital optimization.

## Next Actions
- Approve FY plan scenario B; prioritize pipeline hygiene; finalize debt facility renewal.
                """
                result_md = demo_report
            else:
                from openai import OpenAI  # lazy import

                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model=cfg.model,
                    temperature=cfg.temperature,
                    messages=[{"role": "system", "content": system_msg}, *user_msgs],
                    max_tokens=1200,
                )
                result_md = resp.choices[0].message.content or "(No content returned)"

            s.update(label="Formatting…", state="running")
            st.subheader("Generated Report")
            st.markdown(result_md)

            st.download_button(
                "⬇️ Download Markdown",
                data=result_md,
                file_name="decision_report.md",
                mime="text/markdown",
            )

            st.toast("Report ready")
            s.update(label="Done", state="complete")
        except Exception as e:
            st.error("Couldn't generate the report.")
            st.exception(e)



# Function to run the Generative AI Reporting module
# This function sets up the Streamlit app for generating AI-based financial reports
#def run_ai_reporting_module():
    #st.header("🧾 Generative AI: Decision Report Generator")

    #api_key = st.text_input("Enter your OpenAI API key", type="password")
    #report_type = st.selectbox("Select report type", ["Forecast Summary", "Risk Report", "Board Presentation"])
    #key_metrics = st.text_area("Enter key financial metrics or notes")

    #if st.button("Generate Report"):
        #if not api_key:
            #st.error("API key is required.")
            #return

        #openai.api_key = api_key
        #prompt = f"Create a {report_type.lower()} based on the following financial input:\n{key_metrics}"

       #with st.spinner("Generating report..."):
            #try:
                #response = openai.ChatCompletion.create(
                    #model="gpt-4",
                    #messages=[{"role": "user", "content": prompt}],
                    #temperature=0.5,
                    #max_tokens=500
                #)
                #result = response["choices"][0]["message"]["content"]
                #st.subheader("Generated Report")
                #st.markdown(result)
            #except Exception as e:
                #st.error(f"Error: {e}")