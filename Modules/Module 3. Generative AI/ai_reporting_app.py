# ai_reporting_app.py
import streamlit as st
import openai
import os

def run_ai_reporting_module():
    st.header("🧾 Generative AI: Decision Report Generator")

    api_key = st.text_input("Enter your OpenAI API key", type="password")
    report_type = st.selectbox("Select report type", ["Forecast Summary", "Risk Report", "Board Presentation"])
    key_metrics = st.text_area("Enter key financial metrics or notes")

    if st.button("Generate Report"):
        if not api_key:
            st.error("API key is required.")
            return

        openai.api_key = api_key
        prompt = f"Create a {report_type.lower()} based on the following financial input:\n{key_metrics}"

        with st.spinner("Generating report..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=500
                )
                result = response["choices"][0]["message"]["content"]
                st.subheader("Generated Report")
                st.markdown(result)
            except Exception as e:
                st.error(f"Error: {e}")
