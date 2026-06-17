"""Page 6: AI Analyst — LLM Q&A chat interface."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st

from dashboard.data_loader import load_final_risk_scores
from dashboard.styles.theme import inject_css, page_header

try:
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        GENAI_AVAILABLE = True
    else:
        GENAI_AVAILABLE = False
except ImportError:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

inject_css()
page_header(
    "🤖",
    "AI Financial Analyst",
    "Ask any question about SEC filings · Powered by Gemini 1.5 Flash",
)

if "ai_call_count" not in st.session_state:
    st.session_state.ai_call_count = 0
MAX_CALLS_PER_SESSION = 10

df = load_final_risk_scores()

# ── Sidebar: context controls ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Query Settings")

    ticker_options = ["All companies"] + (
        sorted(df["ticker"].dropna().unique().tolist())
        if not df.empty and "ticker" in df.columns
        else []
    )
    filter_ticker = st.selectbox("Focus on company (optional)", ticker_options)

    st.markdown("### Example Questions")
    examples = [
        "What are the main risk factors for Apple?",
        "Which companies mention going concern language?",
        "What revenue recognition policies are mentioned?",
        "Are there any regulatory investigation disclosures?",
        "Compare liquidity risk across tech companies",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["pending_question"] = ex

# ── Chat Interface ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Sidebar example button injects question
question = None
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

user_input = st.chat_input("Ask about any SEC filing...")
if user_input:
    question = user_input

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("🤖 Analyzing SEC filings with Gemini AI..."):
            if GENAI_AVAILABLE:
                if st.session_state.ai_call_count >= MAX_CALLS_PER_SESSION:
                    st.warning("You have reached the demo limit of 10 AI queries per session. Refresh the page to reset.")
                    st.stop()
                prompt = (
                    "You are a senior financial risk analyst specializing in SEC filings, "
                    "forensic accounting, and financial anomaly detection. "
                    "Answer this question concisely and professionally: " + question
                )
                try:
                    response = model.generate_content(prompt)
                    answer = response.text
                except Exception as e:
                    answer = f"AI temporarily unavailable: {str(e)[:100]}. Please try again."
                st.session_state.ai_call_count += 1
                st.caption(f"AI queries used: {st.session_state.ai_call_count}/{MAX_CALLS_PER_SESSION}")
            else:
                answer = "Add GEMINI_API_KEY to Streamlit secrets to enable AI."

            st.markdown(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

    if len(st.session_state.chat_history) > 20:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
