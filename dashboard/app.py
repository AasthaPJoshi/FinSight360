"""
FinSight360 — Autonomous Financial Risk Intelligence Platform
Main Streamlit application entry point.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datetime

import streamlit as st
import streamlit.components.v1 as components

from dashboard.data_loader import get_db_stats
from dashboard.styles.theme import inject_css

st.set_page_config(
    page_title="FinSight360",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem;">
        <div class="brand-logo">📊 FinSight360</div>
        <div class="brand-subtitle">Financial Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="font-size:10px;text-transform:uppercase;
    letter-spacing:0.5px;color:rgba(176,190,197,0.4);
    margin-bottom:8px;">Pipeline Status</div>
    """, unsafe_allow_html=True)

    stats = get_db_stats()
    if stats and any(v > 0 for v in stats.values()):
        cols = st.columns(2)
        with cols[0]:
            st.metric("Companies", stats.get("companies", 0))
            st.metric("Facts", f"{stats.get('financial_facts', 0):,}")
        with cols[1]:
            st.metric("Filings", stats.get("filings", 0))
            st.metric("Scored", stats.get("final_risk_scores", 0))

        st.markdown("""
        <div class="pipeline-active">
            <div class="pipeline-dot"></div>
            Pipeline Active
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No data. Run pipeline.")

    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style="position:fixed;bottom:1rem;
    font-size:10px;color:rgba(176,190,197,0.3);
    line-height:1.6;">
    © 2025 Aastha Joshi<br>
    MS Information Systems, SDSU<br>
    finsight360.streamlit.app
    </div>
    """, unsafe_allow_html=True)

# ── Main Landing Page — Particle Canvas Hero ──────────────────────────────────
HERO_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: transparent; font-family: 'Inter', -apple-system, sans-serif; overflow: hidden; }
  canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  .hero {
    position: relative; width: 100%; height: 520px;
    background: radial-gradient(ellipse at 30% 50%, #0D1B2A 0%, #060B14 100%);
    border-radius: 16px;
    border: 1px solid rgba(41,121,255,0.25);
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .content {
    position: relative; z-index: 10;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100%; text-align: center; padding: 2rem;
  }
  .badge {
    background: rgba(41,121,255,0.12);
    border: 1px solid rgba(41,121,255,0.3);
    border-radius: 20px; padding: 5px 16px;
    font-size: 11px; color: #90CAF9;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 1.5rem;
  }
  h1 {
    font-size: 58px; font-weight: 800; line-height: 1.05;
    background: linear-gradient(135deg, #FFFFFF 0%, #00E5FF 55%, #2979FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.75rem;
    filter: drop-shadow(0 0 30px rgba(0,229,255,0.3));
  }
  .subtitle {
    font-size: 15px; color: rgba(176,190,197,0.75);
    margin-bottom: 2.5rem; max-width: 580px; line-height: 1.65;
  }
  .stats { display: flex; gap: 3rem; margin-bottom: 2.25rem; }
  .stat { text-align: center; }
  .stat-num {
    font-size: 40px; font-weight: 800; line-height: 1;
    color: #00E5FF;
    text-shadow: 0 0 25px rgba(0,229,255,0.5);
  }
  .stat-label {
    font-size: 10px; color: rgba(176,190,197,0.55);
    text-transform: uppercase; letter-spacing: 1px; margin-top: 5px;
  }
  .pills { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  .pill {
    background: rgba(41,121,255,0.1);
    border: 1px solid rgba(41,121,255,0.22);
    border-radius: 6px; padding: 5px 13px;
    font-size: 11px; color: #90CAF9;
    transition: all 0.2s;
  }
  .pill:hover {
    background: rgba(41,121,255,0.2);
    border-color: rgba(0,229,255,0.4);
    color: #00E5FF;
  }
</style>
</head>
<body>
<div class="hero">
  <canvas id="c"></canvas>
  <div class="content">
    <div class="badge">⚡ Production Ready &nbsp;·&nbsp; SEC EDGAR &nbsp;·&nbsp; S&amp;P 500</div>
    <h1>FinSight360</h1>
    <div class="subtitle">
      Autonomous Financial Anomaly Detection &amp; Risk Intelligence Platform<br>
      powered by Isolation Forest, Benford's Law &amp; Gemini AI
    </div>
    <div class="stats">
      <div class="stat"><div class="stat-num">10</div><div class="stat-label">Companies</div></div>
      <div class="stat"><div class="stat-num">3</div><div class="stat-label">AI Signals</div></div>
      <div class="stat"><div class="stat-num">78</div><div class="stat-label">Tests</div></div>
      <div class="stat"><div class="stat-num">10</div><div class="stat-label">Phases</div></div>
    </div>
    <div class="pills">
      <span class="pill">🧠 Isolation Forest</span>
      <span class="pill">📐 Benford's Law</span>
      <span class="pill">🕸️ Network Graph</span>
      <span class="pill">🤖 Gemini AI</span>
      <span class="pill">⚖️ SHAP Explainability</span>
      <span class="pill">📊 DuckDB</span>
      <span class="pill">🔬 SR 11-7 Compliant</span>
    </div>
  </div>
</div>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let W, H, particles = [];

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  W = canvas.width = rect.width || 800;
  H = canvas.height = rect.height || 520;
}
resize();

class Particle {
  constructor() { this.reset(); }
  reset() {
    this.x = Math.random() * W;
    this.y = Math.random() * H;
    this.vx = (Math.random() - 0.5) * 0.35;
    this.vy = (Math.random() - 0.5) * 0.35;
    this.r = Math.random() * 1.8 + 0.4;
    this.alpha = Math.random() * 0.45 + 0.1;
    this.color = Math.random() > 0.7 ? '0,229,255' : '41,121,255';
  }
  update() {
    this.x += this.vx; this.y += this.vy;
    if (this.x < 0 || this.x > W || this.y < 0 || this.y > H) this.reset();
  }
  draw() {
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${this.color},${this.alpha})`;
    ctx.fill();
  }
}

for (let i = 0; i < 90; i++) particles.push(new Particle());

function drawLines() {
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      const dx = particles[i].x - particles[j].x;
      const dy = particles[i].y - particles[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 110) {
        ctx.beginPath();
        ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(particles[j].x, particles[j].y);
        ctx.strokeStyle = `rgba(41,121,255,${0.12 * (1 - dist / 110)})`;
        ctx.lineWidth = 0.6;
        ctx.stroke();
      }
    }
  }
}

function animate() {
  ctx.clearRect(0, 0, W, H);
  particles.forEach(p => { p.update(); p.draw(); });
  drawLines();
  requestAnimationFrame(animate);
}
animate();
</script>
</body></html>"""

components.html(HERO_HTML, height=535)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🗺️ Navigate")
    st.markdown("""
| Page | Purpose |
|------|---------|
| 📋 Executive Overview | Top-level risk summary & KPIs |
| 🌡️ Risk Heatmap | Sector × risk tier visualization |
| 🔍 Company Deep Dive | Per-company full analysis |
| 🕸️ Network Graph | Corporate relationship graph |
| 📐 Benford Analysis | Forensic digit analysis |
| 🤖 AI Analyst | Gemini-powered Q&A chat |
| ⚖️ Governance | Audit trail, SHAP, bias monitoring |
""")

with col2:
    st.markdown("### ⚙️ Signal Architecture")
    st.code(
        "SEC EDGAR 10-K/10-Q\n"
        "      ↓\n"
        "dbt Data Models\n"
        "      ↓\n"
        "┌─────────────────────────┐\n"
        "│ ML: Isolation Forest 50%│\n"
        "│ Forensic: Benford's  30%│\n"
        "│ Graph: Network Risk  20%│\n"
        "└─────────────────────────┘\n"
        "      ↓\n"
        "Final Risk Score (0-100)\n"
        "      ↓\n"
        "Gemini AI Risk Brief",
        language=None,
    )

st.divider()
st.markdown("### 🚀 Getting Started")

step_col1, step_col2, step_col3 = st.columns(3)
with step_col1:
    st.markdown("**Step 1: Ingest Data**")
    st.code("python -m finsight360.cli ingest --ticker AAPL", language="bash")

with step_col2:
    st.markdown("**Step 2: Run ML + Graph**")
    st.code(
        "python -m finsight360.cli ml-run\n"
        "python -m finsight360.cli graph-build",
        language="bash",
    )

with step_col3:
    st.markdown("**Step 3: Explore Dashboard**")
    st.code(
        "streamlit run dashboard/app.py\n"
        "# Navigate using the sidebar",
        language="bash",
    )
