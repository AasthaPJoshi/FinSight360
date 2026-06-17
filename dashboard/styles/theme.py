"""FinSight360 — Global CSS theme and UI helper functions."""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --fs-navy: #0A0F1E;
    --fs-navy-mid: #0D1B2A;
    --fs-navy-light: #1A2744;
    --fs-blue: #1565C0;
    --fs-blue-bright: #2979FF;
    --fs-cyan: #00E5FF;
    --fs-purple: #7B1FA2;
    --fs-success: #00E676;
    --fs-warning: #FFD600;
    --fs-danger: #FF1744;
    --fs-orange: #FF6D00;
    --glass-bg: rgba(13,27,42,0.7);
    --glass-border: rgba(41,121,255,0.2);
    --glass-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

.stApp {
    background: radial-gradient(ellipse at top left, #0D1B2A 0%, #0A0F1E 50%, #060B14 100%) !important;
}

.main .block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1400px !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060B14 0%, #0A0F1E 100%) !important;
    border-right: 1px solid rgba(41,121,255,0.2) !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.4) !important;
}
[data-testid="stSidebar"] * {
    color: rgba(224,230,240,0.85) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p {
    color: rgba(176,190,197,0.6) !important;
    font-size: 12px !important;
}
[data-testid="stSidebarNav"] a {
    border-radius: 8px !important;
    margin: 2px 8px !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(41,121,255,0.15) !important;
    color: #00E5FF !important;
    box-shadow: 0 0 10px rgba(41,121,255,0.2) !important;
}

[data-testid="metric-container"] {
    background: rgba(13,27,42,0.7) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(41,121,255,0.2) !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(41,121,255,0.2), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    border-color: rgba(41,121,255,0.4) !important;
}
[data-testid="metric-container"] label {
    color: rgba(176,190,197,0.7) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #00E5FF !important;
    text-shadow: 0 0 20px rgba(0,229,255,0.4) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(41,121,255,0.2) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
}

.js-plotly-plot {
    border-radius: 12px !important;
    border: 1px solid rgba(41,121,255,0.1) !important;
}

[data-testid="stSelectbox"] > div {
    border-radius: 8px !important;
    border-color: rgba(41,121,255,0.3) !important;
    background: rgba(13,27,42,0.8) !important;
}

.page-header {
    background: linear-gradient(135deg, rgba(21,101,192,0.3) 0%, rgba(41,121,255,0.15) 50%, rgba(0,229,255,0.1) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(41,121,255,0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
}
.page-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(41,121,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.page-header::after {
    content: '';
    position: absolute;
    bottom: -60%;
    right: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,229,255,0.05) 0%, transparent 70%);
    border-radius: 50%;
}
.page-header-icon {
    font-size: 40px;
    margin-bottom: 8px;
    display: block;
    filter: drop-shadow(0 0 20px rgba(41,121,255,0.5));
}
.page-header h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: white !important;
    margin: 0 !important;
    padding: 0 !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #00E5FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.page-header p {
    color: rgba(176,190,197,0.8) !important;
    font-size: 14px !important;
    margin: 6px 0 0 !important;
}

.badge-critical {
    background: rgba(198,40,40,0.15);
    color: #FF5252;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid rgba(255,82,82,0.4);
    display: inline-block;
    box-shadow: 0 0 8px rgba(255,82,82,0.2);
}
.badge-high {
    background: rgba(230,81,0,0.15);
    color: #FF6D00;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid rgba(255,109,0,0.4);
    display: inline-block;
    box-shadow: 0 0 8px rgba(255,109,0,0.2);
}
.badge-medium {
    background: rgba(249,168,37,0.15);
    color: #FFD600;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid rgba(255,214,0,0.4);
    display: inline-block;
    box-shadow: 0 0 8px rgba(255,214,0,0.2);
}
.badge-low {
    background: rgba(0,230,118,0.15);
    color: #00E676;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid rgba(0,230,118,0.4);
    display: inline-block;
    box-shadow: 0 0 8px rgba(0,230,118,0.2);
}

.stat-card {
    background: rgba(13,27,42,0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid rgba(41,121,255,0.2);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: all 0.2s ease;
}
.stat-card:hover {
    box-shadow: 0 8px 30px rgba(41,121,255,0.2);
    transform: translateY(-2px);
    border-color: rgba(41,121,255,0.4);
}
.stat-card .value {
    font-size: 36px;
    font-weight: 700;
    color: #00E5FF;
    text-shadow: 0 0 20px rgba(0,229,255,0.4);
    line-height: 1;
    margin-bottom: 4px;
}
.stat-card .label {
    font-size: 12px;
    color: rgba(176,190,197,0.7);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}

.info-card {
    background: rgba(13,27,42,0.7);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 1px solid rgba(41,121,255,0.2);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
}

.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #B0BEC5;
    margin: 1.5rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(41,121,255,0.2);
    margin-left: 8px;
}

.alert-info {
    background: rgba(21,101,192,0.15);
    border-left: 4px solid #2979FF;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 13px;
    color: #90CAF9;
    margin: 0.5rem 0;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(13,27,42,0.7);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(41,121,255,0.2) !important;
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    border: none !important;
    color: rgba(176,190,197,0.6) !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(41,121,255,0.2) !important;
    color: #00E5FF !important;
    box-shadow: 0 0 15px rgba(41,121,255,0.2) !important;
}

[data-testid="stButton"] button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
    background: rgba(41,121,255,0.15) !important;
    border: 1px solid rgba(41,121,255,0.3) !important;
    color: #90CAF9 !important;
}
[data-testid="stButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(41,121,255,0.3) !important;
    background: rgba(41,121,255,0.25) !important;
    border-color: rgba(41,121,255,0.5) !important;
    color: #00E5FF !important;
}

[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    border: 1px solid rgba(41,121,255,0.2) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    background: rgba(13,27,42,0.7) !important;
    margin-bottom: 12px !important;
}

[data-testid="stSpinner"] {
    color: #00E5FF !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(10,15,30,0.5); }
::-webkit-scrollbar-thumb { background: rgba(41,121,255,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(41,121,255,0.6); }

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
.shimmer-text {
    background: linear-gradient(90deg, #FFFFFF 0%, #00E5FF 30%, #2979FF 50%, #00E5FF 70%, #FFFFFF 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
}

.pipeline-active {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,230,118,0.1);
    border: 1px solid rgba(0,230,118,0.3);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    color: #00E676;
    margin-top: 8px;
    box-shadow: 0 0 10px rgba(0,230,118,0.1);
}
.pipeline-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00E676;
    box-shadow: 0 0 6px #00E676;
    flex-shrink: 0;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px #00E676; }
    50% { opacity: 0.5; box-shadow: 0 0 12px #00E676; }
}

.brand-logo {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #FFFFFF 0%, #00E5FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.brand-subtitle {
    font-size: 10px;
    opacity: 0.5;
    text-transform: uppercase;
    letter-spacing: 1px;
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,23,68,0.3); }
    50% { box-shadow: 0 0 0 8px rgba(255,23,68,0); }
}
.pulse-danger {
    animation: pulse-red 2s infinite;
}

.gradient-text {
    background: linear-gradient(135deg, #2979FF, #00E5FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

@keyframes countUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: countUp 0.4s ease forwards;
}

</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str):
    import streamlit as st
    st.markdown(f"""
    <div class="page-header">
        <span class="page-header-icon">{icon}</span>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def risk_badge(tier: str) -> str:
    tier_upper = str(tier).upper()
    class_map = {
        "CRITICAL": "badge-critical",
        "HIGH": "badge-high",
        "MEDIUM": "badge-medium",
        "LOW": "badge-low",
        "MINIMAL": "badge-low",
        "CLEAN": "badge-low",
    }
    css_class = class_map.get(tier_upper, "badge-low")
    return f'<span class="{css_class}">{tier_upper}</span>'


def stat_card(value: str, label: str, color: str = "#00E5FF") -> str:
    return f"""
    <div class="stat-card">
        <div class="value" style="color:{color}">{value}</div>
        <div class="label">{label}</div>
    </div>
    """


def section_title(title: str) -> str:
    return f'<div class="section-title">{title}</div>'


CHART_THEME = {
    "layout": {
        "font": {"family": "Inter, sans-serif", "size": 12, "color": "#B0BEC5"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15,23,42,0.6)",
        "margin": {"t": 40, "b": 30, "l": 20, "r": 20},
        "showlegend": True,
        "legend": {
            "bgcolor": "rgba(13,27,42,0.8)",
            "bordercolor": "rgba(41,121,255,0.2)",
            "font": {"color": "#B0BEC5"},
        },
        "xaxis": {
            "gridcolor": "rgba(41,121,255,0.1)",
            "linecolor": "rgba(41,121,255,0.2)",
            "tickfont": {"size": 11, "color": "#B0BEC5"},
        },
        "yaxis": {
            "gridcolor": "rgba(41,121,255,0.1)",
            "linecolor": "rgba(41,121,255,0.2)",
            "tickfont": {"size": 11, "color": "#B0BEC5"},
        },
    }
}

TIER_COLORS = {
    "CRITICAL": "#C62828",
    "HIGH": "#E65100",
    "MEDIUM": "#F57C00",
    "LOW": "#2E7D32",
    "MINIMAL": "#1565C0",
    "CLEAN": "#1565C0",
}
