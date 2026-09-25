import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from catboost import CatBoostClassifier

# ======================================================================================
# PAGE CONFIG
# ======================================================================================
st.set_page_config(
    page_title="VaultIQ | AI Loan Intelligence",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["🔮 Predict", "📊 Model Insights", "ℹ️ About"]
if "sidebar_nav" not in st.session_state:
    st.session_state.sidebar_nav = PAGES[0]


def set_page(target_page):
    st.session_state.sidebar_nav = target_page


# ======================================================================================
# CUSTOM CSS — PURE BLACK / GOLD MODERN THEME
# ======================================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --c-black: #020202;
        --c-panel: rgba(255,255,255,0.04);
        --c-gold: #d4af37;
        --c-gold-light: #f4d67a;
        --c-yellow: #ffd447;
        --c-text: #f4f1e8;
        --c-muted: #9d988a;
    }

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, .hero-title {
        font-family: 'Poppins', sans-serif !important;
    }

    html, body {
        background: #000000;
    }

    /* App background */
    .stApp {
        background: transparent;
        color: var(--c-text);
    }

    /* Animated network background lives in an injected iframe (see
       render_animated_background()). Force it into a fixed fullscreen
       layer behind every other element, and lift the real app content
       above it so it stays fully clickable and readable. */
    iframe {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 0 !important;
        pointer-events: none !important;
        border: none !important;
    }
    section[data-testid="stAppViewContainer"],
    section[data-testid="stSidebar"],
    header[data-testid="stHeader"] {
        position: relative;
        z-index: 1;
        background: transparent;
    }
    div[data-testid="stAppViewBlockContainer"] {
        position: relative;
        z-index: 1;
    }
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0.6) !important;
        backdrop-filter: blur(4px);
    }

    /* Hide default hamburger footer branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ============================ BRAND MARK ============================ */
    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .brand-mark {
        width: 40px;
        height: 40px;
        min-width: 40px;
        border-radius: 11px;
        background: linear-gradient(150deg, var(--c-gold), var(--c-yellow));
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 16px rgba(212,175,55,0.4);
    }
    .brand-word {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 1.2rem;
        color: #ffffff;
        line-height: 1.1;
        white-space: nowrap;
    }
    .brand-word span { color: var(--c-gold); }
    .brand-tagline {
        color: var(--c-muted);
        font-size: 0.72rem;
        letter-spacing: 0.02em;
    }
    .sidebar-brand {
        text-align: center;
        padding: 0.3rem 0 1rem 0;
    }
    .sidebar-brand .brand-row { justify-content: center; margin-bottom: 0.5rem; }
    .sidebar-brand .brand-mark { animation: mark-glow 3s ease-in-out infinite; }
    @keyframes mark-glow {
        0%, 100% { box-shadow: 0 0 12px rgba(212,175,55,0.35); }
        50%      { box-shadow: 0 0 24px rgba(255,212,71,0.6); }
    }

    /* ============================ TOP NAVBAR ============================ */
    .top-navbar-wrap {
        padding: 0.6rem 0 0.2rem 0;
        border-bottom: 1px solid rgba(212,175,55,0.14);
        margin-bottom: 1.6rem;
    }
    .st-key-topnav_predict button,
    .st-key-topnav_insights button,
    .st-key-topnav_about button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        font-weight: 600 !important;
        border-radius: 0 !important;
        padding: 0.3rem 0.1rem !important;
    }
    .st-key-topnav_predict button[kind="secondary"],
    .st-key-topnav_insights button[kind="secondary"],
    .st-key-topnav_about button[kind="secondary"] {
        color: #c9c3b3 !important;
        border-bottom: 2px solid transparent !important;
    }
    .st-key-topnav_predict button[kind="primary"],
    .st-key-topnav_insights button[kind="primary"],
    .st-key-topnav_about button[kind="primary"] {
        color: var(--c-yellow) !important;
        border-bottom: 2px solid var(--c-gold) !important;
    }
    .st-key-cta_get_started button {
        background: linear-gradient(90deg, var(--c-gold), var(--c-yellow)) !important;
        color: #0a0a0a !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 20px rgba(212,175,55,0.35) !important;
    }

    /* ============================ HERO ============================ */
    .hero-banner {
        position: relative;
        border-radius: 22px;
        overflow: hidden;
        padding: 3.4rem 2.6rem;
        margin-bottom: 1.6rem;
        background:
            radial-gradient(ellipse at 25% 15%, rgba(212,175,55,0.14), transparent 55%),
            radial-gradient(ellipse at 85% 80%, rgba(255,212,71,0.08), transparent 55%),
            #020202;
        border: 1px solid rgba(212,175,55,0.16);
    }
    .hero-eyebrow {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 30px;
        background: transparent;
        border: 1px solid rgba(212,175,55,0.55);
        color: var(--c-gold);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.7rem 0;
        line-height: 1.08;
        letter-spacing: -0.01em;
    }
    .hero-title span {
        color: var(--c-gold);
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c9c3b3;
        max-width: 640px;
        line-height: 1.65;
        font-weight: 300;
        margin-bottom: 1.8rem;
    }
    .st-key-cta_primary button {
        background: linear-gradient(90deg, var(--c-gold), var(--c-yellow)) !important;
        color: #0a0a0a !important;
        font-weight: 700 !important;
        font-size: 1.0rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.6rem !important;
        box-shadow: 0 10px 25px rgba(212,175,55,0.32) !important;
        transition: transform 0.15s ease !important;
    }
    .st-key-cta_primary button:hover { transform: translateY(-2px) !important; }
    .st-key-cta_secondary button {
        background: transparent !important;
        color: #f4f1e8 !important;
        font-weight: 700 !important;
        font-size: 1.0rem !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.6rem !important;
        transition: border-color 0.15s ease, transform 0.15s ease !important;
    }
    .st-key-cta_secondary button:hover {
        border-color: var(--c-gold) !important;
        transform: translateY(-2px) !important;
    }

    /* TRUST STRIP */
    .trust-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1.6rem;
    }
    .trust-chip {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(212,175,55,0.22);
        border-radius: 30px;
        padding: 0.4rem 0.9rem;
        font-size: 0.78rem;
        color: #e9e4d4;
        font-weight: 500;
    }

    /* DISCLAIMER — subtle small-print pill */
    .disclaimer-banner {
        display: inline-block;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(212,175,55,0.18);
        border-radius: 30px;
        padding: 0.32rem 0.85rem;
        font-size: 0.68rem;
        color: var(--c-muted);
        margin-top: 0.8rem;
    }

    /* METRIC / GLASS CARDS */
    .glass-card {
        background: var(--c-panel);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 1.4rem 1.4rem;
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(212,175,55,0.45);
        box-shadow: 0 14px 34px rgba(212,175,55,0.18);
    }
    .glass-card h3 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff;
    }
    .glass-card p {
        margin: 0.2rem 0 0 0;
        color: var(--c-muted);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* SECTION HEADERS */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.4rem 0 0.1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-caption {
        color: #b0aa98;
        font-size: 0.9rem;
        margin-bottom: 1.1rem;
        font-weight: 300;
    }

    /* FORM CONTAINER */
    .form-block {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(212,175,55,0.12);
        border-radius: 18px;
        padding: 1.6rem 1.7rem 1rem 1.7rem;
        margin-bottom: 1.3rem;
    }
    .calc-note {
        font-size: 0.78rem;
        color: var(--c-muted);
        margin: -0.4rem 0 0.8rem 0;
    }

    /* MAIN PREDICT BUTTON */
    .stButton>button {
        background: linear-gradient(90deg, var(--c-gold), var(--c-yellow));
        color: #0a0a0a;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.6rem;
        font-weight: 700;
        font-size: 1.02rem;
        letter-spacing: 0.02em;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 10px 25px rgba(212,175,55,0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 32px rgba(255,212,71,0.45);
        color: #0a0a0a;
    }

    /* RESULT CARDS */
    .result-approved {
        background: linear-gradient(135deg, rgba(212,175,55,0.20), rgba(212,175,55,0.03));
        border: 1px solid rgba(212,175,55,0.55);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-rejected {
        background: linear-gradient(135deg, rgba(224,82,82,0.20), rgba(224,82,82,0.03));
        border: 1px solid rgba(224,82,82,0.55);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.4rem 0;
        font-family: 'Poppins', sans-serif;
    }
    .result-sub {
        color: #d9d3c2;
        font-size: 0.95rem;
        font-weight: 300;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000000 0%, #0a0906 100%);
        border-right: 1px solid rgba(212,175,55,0.10);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.98rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.1rem;
        color: #cfc9b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(212,175,55,0.18) !important;
        color: var(--c-yellow) !important;
        font-weight: 700;
    }

    /* PHOTO STRIP TILES */
    .photo-tile {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        height: 150px;
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(212,175,55,0.18);
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .photo-tile:hover {
        transform: scale(1.03);
        box-shadow: 0 16px 36px rgba(212,175,55,0.3);
    }
    .photo-tile::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(0deg, rgba(0,0,0,0.95) 5%, rgba(0,0,0,0.25) 60%, rgba(0,0,0,0.05) 100%);
    }
    .photo-tile-label {
        position: absolute;
        bottom: 10px;
        left: 14px;
        z-index: 2;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
        font-family: 'Poppins', sans-serif;
    }
    .photo-tile-icon {
        position: absolute;
        top: 10px;
        left: 12px;
        z-index: 2;
        font-size: 1.4rem;
    }

    /* PHOTO MOSAIC (About page) */
    .mosaic-img {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(212,175,55,0.18);
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        transition: transform 0.25s ease;
    }
    .mosaic-img:hover { transform: translateY(-4px); }

    /* Footer */
    .footer-note {
        text-align:center;
        color: var(--c-muted);
        font-size:0.8rem;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(212,175,55,0.12);
    }

    /* MOBILE RESPONSIVENESS */
    @media (max-width: 768px) {
        .hero-banner { padding: 2rem 1.2rem; border-radius: 16px; }
        .hero-title { font-size: 1.8rem; line-height: 1.2; }
        .hero-subtitle { font-size: 0.92rem; }
        .hero-eyebrow { font-size: 0.66rem; padding: 0.26rem 0.65rem; }
        .trust-chip { font-size: 0.7rem; padding: 0.32rem 0.7rem; }
        .glass-card { padding: 1rem; }
        .glass-card h3 { font-size: 1.5rem; }
        .photo-tile { height: 100px; }
        .photo-tile-label { font-size: 0.78rem; }
        .photo-tile-icon { font-size: 1.1rem; }
        .form-block { padding: 1.1rem 1rem 0.5rem 1rem; }
        .brand-mark { width: 34px; height: 34px; min-width: 34px; }
        .brand-word { font-size: 1.0rem; }
        .section-title { font-size: 1.15rem; }
        .result-title { font-size: 1.4rem; }
        .top-navbar-wrap { padding-bottom: 0.4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================================================
# ANIMATED NETWORK BACKGROUND
# Self-contained vanilla-JS canvas animation (no external CDN needed) rendered
# in a Streamlit component. CSS above pins its iframe to a fixed, fullscreen,
# click-through layer behind the app content.
# ======================================================================================
def render_animated_background():
    components.html(
        """
        <canvas id="bg-canvas" style="display:block; width:100vw; height:100vh; background:
            radial-gradient(ellipse at 15% 15%, rgba(212,175,55,0.08), transparent 50%),
            radial-gradient(ellipse at 85% 80%, rgba(255,212,71,0.06), transparent 50%),
            #000000;">
        </canvas>
        <script>
        const canvas = document.getElementById('bg-canvas');
        const ctx = canvas.getContext('2d');
        let width, height, dpr;

        function resize() {
            dpr = window.devicePixelRatio || 1;
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width * dpr;
            canvas.height = height * dpr;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        window.addEventListener('resize', resize);
        resize();

        const NUM_NODES = 60;
        const MAX_DIST = 150;
        const nodes = [];
        for (let i = 0; i < NUM_NODES; i++) {
            nodes.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                r: Math.random() * 1.6 + 0.9,
            });
        }

        function step() {
            ctx.clearRect(0, 0, width, height);

            for (const n of nodes) {
                n.x += n.vx;
                n.y += n.vy;
                if (n.x < 0 || n.x > width) n.vx *= -1;
                if (n.y < 0 || n.y > height) n.vy *= -1;
            }

            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const a = nodes[i], b = nodes[j];
                    const dx = a.x - b.x, dy = a.y - b.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < MAX_DIST) {
                        const alpha = (1 - dist / MAX_DIST) * 0.25;
                        ctx.strokeStyle = `rgba(212,175,55, ${alpha})`;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(a.x, a.y);
                        ctx.lineTo(b.x, b.y);
                        ctx.stroke();
                    }
                }
            }

            for (const n of nodes) {
                const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 4);
                grad.addColorStop(0, 'rgba(255, 212, 71, 0.85)');
                grad.addColorStop(1, 'rgba(255, 212, 71, 0)');
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r * 4, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = '#f6e3ad';
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
                ctx.fill();
            }

            requestAnimationFrame(step);
        }
        step();
        </script>
        """,
        height=0,
    )


render_animated_background()


# ======================================================================================
# MODEL LOADING
# ======================================================================================
MODEL_CANDIDATES = [
    "catboost_loan_approval.cbm",
    "final_catboost_model.cbm",
]


@st.cache_resource(show_spinner="Loading CatBoost model...")
def load_model():
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            loaded_model = CatBoostClassifier()
            loaded_model.load_model(candidate)
            return loaded_model
    return None


model = load_model()


def resolve_prediction(model, input_df):
    """
    Return (prediction_label, approve_probability) using the model's OWN
    predict() output as the source of truth, rather than assuming
    predict_proba()'s column order matches label values. This avoids the
    approved/rejected label getting flipped when a model's internal
    class order isn't [0, 1].
    """
    proba = np.array(model.predict_proba(input_df)).flatten()

    raw_pred = np.array(model.predict(input_df)).flatten()[0]
    try:
        prediction = int(raw_pred)
    except (ValueError, TypeError):
        prediction = 1 if str(raw_pred).strip().lower() in ("1", "approved", "yes", "true") else 0

    classes = list(getattr(model, "classes_", [0, 1]))
    approve_idx = None
    for i, c in enumerate(classes):
        if c == 1 or str(c).strip().lower() in ("1", "approved", "yes", "true"):
            approve_idx = i
            break
    if approve_idx is None or approve_idx >= len(proba):
        approve_idx = int(np.argmax(proba))

    approve_prob = float(proba[approve_idx])
    return prediction, approve_prob


BRAND_MARK_SVG = """
<div class="brand-mark">
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2 L20 5.5 V11 C20 16 16.5 19.8 12 21 C7.5 19.8 4 16 4 11 V5.5 Z"
              fill="#0a0a0a"/>
        <path d="M8.2 12 L10.8 14.6 L15.8 9.4" stroke="#0a0a0a" stroke-width="1.8"
              stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    </svg>
</div>
"""

# ======================================================================================
# TOP NAVBAR
# ======================================================================================
current_page = st.session_state.sidebar_nav

st.markdown('<div class="top-navbar-wrap">', unsafe_allow_html=True)
nav_l, nav_m1, nav_m2, nav_m3, nav_r = st.columns([2.4, 0.9, 1.3, 0.9, 1.3])
with nav_l:
    st.markdown(
        f"""
        <div class="brand-row">
            {BRAND_MARK_SVG}
            <div>
                <div class="brand-word">Vault<span>IQ</span></div>
                <div class="brand-tagline">AI Loan Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with nav_m1:
    with st.container(key="topnav_predict"):
        st.button(
            "Predict", key="nav_predict_btn", use_container_width=True,
            type=("primary" if current_page == PAGES[0] else "secondary"),
            on_click=set_page, args=(PAGES[0],),
        )
with nav_m2:
    with st.container(key="topnav_insights"):
        st.button(
            "Model Insights", key="nav_insights_btn", use_container_width=True,
            type=("primary" if current_page == PAGES[1] else "secondary"),
            on_click=set_page, args=(PAGES[1],),
        )
with nav_m3:
    with st.container(key="topnav_about"):
        st.button(
            "About", key="nav_about_btn", use_container_width=True,
            type=("primary" if current_page == PAGES[2] else "secondary"),
            on_click=set_page, args=(PAGES[2],),
        )
with nav_r:
    with st.container(key="cta_get_started"):
        st.button(
            "🔍 Get Started", key="nav_get_started_btn", use_container_width=True,
            on_click=set_page, args=(PAGES[0],),
        )
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================================================
# SIDEBAR
# ======================================================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="brand-row">{BRAND_MARK_SVG}
                <div style="text-align:left;">
                    <div class="brand-word">Vault<span>IQ</span></div>
                    <div class="brand-tagline">AI Loan Intelligence</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        PAGES,
        key="sidebar_nav",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <p style="margin-bottom:0.3rem;">Model Accuracy</p>
            <h3 style="color:#ffd447;">94%</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Powered by CatBoost Gradient Boosting")

    st.markdown(
        '<div class="disclaimer-banner">🎓 Concept / portfolio project — predictions are illustrative, not real credit decisions.</div>',
        unsafe_allow_html=True,
    )

    st.image(
        "https://images.unsplash.com/photo-1560472354-b33ff0c44a43?q=80&w=800&auto=format&fit=crop",
        use_container_width=True,
    )

    if model is None:
        st.error(
            "Model file not found. Place `catboost_loan_approval.cbm` "
            "in the same folder as `app.py`."
        )

page = st.session_state.sidebar_nav

# ======================================================================================
# HERO
# ======================================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-eyebrow">🟡 AI-Powered Credit Risk Engine</div>
        <div class="hero-title">Know Your <span>Approval Odds</span><br>Instantly.</div>
        <div class="hero-subtitle">
            VaultIQ uses a CatBoost gradient-boosting model trained on applicant demographics,
            income, employment, and credit-history data to predict loan approval outcomes with
            94% accuracy — track the key factors, explore the model, and get an instant decision.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_cta_l, hero_cta_r, hero_cta_spacer = st.columns([1, 1, 2])
with hero_cta_l:
    with st.container(key="cta_primary"):
        st.button(
            "🚀 Start Application →", key="hero_cta_primary_btn", use_container_width=True,
            on_click=set_page, args=(PAGES[0],),
        )
with hero_cta_r:
    with st.container(key="cta_secondary"):
        st.button(
            "📊 View Model Insights", key="hero_cta_secondary_btn", use_container_width=True,
            on_click=set_page, args=(PAGES[1],),
        )

st.markdown(
    """
    <div class="trust-strip">
        <div class="trust-chip">⚡ Instant Decisioning</div>
        <div class="trust-chip">🧠 94% Model Accuracy</div>
        <div class="trust-chip">🔐 No Data Stored — Demo Only</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ======================================================================================
# PAGE: PREDICT
# ======================================================================================
if page == "🔮 Predict":

    top1, top2, top3, top4 = st.columns(4)
    for col, (val, label) in zip(
        [top1, top2, top3, top4],
        [
            ("94%", "Model Accuracy"),
            ("CatBoost", "Algorithm"),
            ("13", "Input Features"),
            ("Binary", "Prediction Type"),
        ],
    ):
        with col:
            st.markdown(
                f"""<div class="glass-card"><h3>{val}</h3><p>{label}</p></div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    p1, p2, p3, p4 = st.columns(4)
    photo_tiles = [
        (p1, "👤", "Personal Profile", "https://images.unsplash.com/photo-1521791136064-7986c2920216?q=80&w=500&auto=format&fit=crop"),
        (p2, "💼", "Income & Employment", "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?q=80&w=500&auto=format&fit=crop"),
        (p3, "🏦", "Loan Details", "https://images.unsplash.com/photo-1563013544-824ae1b704d3?q=80&w=500&auto=format&fit=crop"),
        (p4, "💳", "Credit History", "https://images.unsplash.com/photo-1591696205602-2f950c417cb9?q=80&w=500&auto=format&fit=crop"),
    ]
    for col, icon, label, url in photo_tiles:
        with col:
            st.markdown(
                f"""
                <div class="photo-tile" style="background-image:url('{url}');">
                    <div class="photo-tile-icon">{icon}</div>
                    <div class="photo-tile-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown('<div class="section-title">📝 Applicant Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Fill in the applicant\'s details below to generate an instant prediction.</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):

        st.markdown('<div class="form-block">', unsafe_allow_html=True)
        st.markdown("#### 👤 Personal Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        with c2:
            person_gender = st.selectbox("Gender", ["male", "female"])
        with c3:
            person_education = st.selectbox(
                "Education Level",
                ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="form-block">', unsafe_allow_html=True)
        st.markdown("#### 💰 Financial & Employment")
        c1, c2, c3 = st.columns(3)
        with c1:
            person_income = st.number_input("Annual Income ($)", min_value=1, value=55000, step=1000)
        with c2:
            person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=60, value=5, step=1)
        with c3:
            person_home_ownership = st.selectbox(
                "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="form-block">', unsafe_allow_html=True)
        st.markdown("#### 🏦 Loan Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000, step=500)
        with c2:
            loan_intent = st.selectbox(
                "Loan Intent",
                ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
            )
        with c3:
            loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=11.5, step=0.1)

        st.markdown(
            '<div class="calc-note">💡 Loan-to-Income ratio is calculated automatically '
            'from Loan Amount ÷ Annual Income — no need to set it manually.</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            previous_loan_defaults_on_file = st.selectbox(
                "Previous Loan Defaults on File", ["No", "Yes"]
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="form-block">', unsafe_allow_html=True)
        st.markdown("#### 🕰️ Credit History")
        c1, c2 = st.columns(2)
        with c1:
            cb_person_cred_hist_length = st.number_input(
                "Credit History Length (years)", min_value=0.0, max_value=60.0, value=6.0, step=1.0
            )
        with c2:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Predict Loan Approval")

    if submitted:
        if model is None:
            st.error("Cannot run prediction — model file was not found. See the sidebar for instructions.")
        else:
            loan_percent_income = (loan_amnt / person_income) if person_income > 0 else 0.0

            input_df = pd.DataFrame(
                [{
                    "person_age": float(person_age),
                    "person_gender": person_gender,
                    "person_education": person_education,
                    "person_income": float(person_income),
                    "person_emp_exp": int(person_emp_exp),
                    "person_home_ownership": person_home_ownership,
                    "loan_amnt": float(loan_amnt),
                    "loan_intent": loan_intent,
                    "loan_int_rate": float(loan_int_rate),
                    "loan_percent_income": float(loan_percent_income),
                    "cb_person_cred_hist_length": float(cb_person_cred_hist_length),
                    "credit_score": int(credit_score),
                    "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
                }]
            )

            try:
                prediction, approve_prob = resolve_prediction(model, input_df)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Make sure the column names/order above match the columns the model "
                    "was trained on, and that categorical values match the categories "
                    "used during training."
                )
                st.stop()

            st.write("")
            res_col, gauge_col = st.columns([1, 1])

            with res_col:
                if prediction == 1:
                    st.balloons()
                    st.markdown(
                        f"""
                        <div class="result-approved">
                            <div style="font-size:2.6rem;">✅</div>
                            <div class="result-title" style="color:#ffd447;">Loan Approved</div>
                            <div class="result-sub">Approval confidence: {approve_prob*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-rejected">
                            <div style="font-size:2.6rem;">❌</div>
                            <div class="result-title" style="color:#ff8b8b;">Loan Rejected</div>
                            <div class="result-sub">Approval confidence: {approve_prob*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with gauge_col:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=approve_prob * 100,
                        number={"suffix": "%", "font": {"color": "#ffffff", "size": 36}},
                        title={"text": "Approval Probability", "font": {"color": "#d9d3c2", "size": 16}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#a49f8f"},
                            "bar": {"color": "#d4af37"},
                            "bgcolor": "rgba(0,0,0,0)",
                            "borderwidth": 1,
                            "bordercolor": "rgba(255,255,255,0.15)",
                            "steps": [
                                {"range": [0, 40], "color": "rgba(224,82,82,0.30)"},
                                {"range": [40, 70], "color": "rgba(255,212,71,0.22)"},
                                {"range": [70, 100], "color": "rgba(212,175,55,0.35)"},
                            ],
                        },
                    )
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f4f1e8"},
                    height=280,
                    margin=dict(l=20, r=20, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Probability breakdown — donut chart
            donut_fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Approved", "Rejected"],
                        values=[approve_prob, 1 - approve_prob],
                        hole=0.62,
                        marker=dict(colors=["#d4af37", "#e05252"], line=dict(color="#000000", width=2)),
                        textinfo="label+percent",
                        textfont=dict(color="#050505", size=13),
                        sort=False,
                    )
                ]
            )
            donut_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f4f1e8"},
                height=280,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(color="#d9d3c2")),
                margin=dict(l=10, r=10, t=20, b=10),
                annotations=[dict(
                    text=f"{approve_prob*100:.0f}%",
                    x=0.5, y=0.5,
                    font=dict(size=26, color="#ffffff", family="Poppins"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(donut_fig, use_container_width=True)

            # Key ratios recap
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(
                    f"""<div class="glass-card"><h3 style="font-size:1.5rem;">{loan_percent_income*100:.1f}%</h3><p>Loan-to-Income Ratio</p></div>""",
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f"""<div class="glass-card"><h3 style="font-size:1.5rem;">{loan_int_rate:.1f}%</h3><p>Interest Rate</p></div>""",
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    f"""<div class="glass-card"><h3 style="font-size:1.5rem;">{int(credit_score)}</h3><p>Credit Score</p></div>""",
                    unsafe_allow_html=True,
                )

            # Applicant summary
            display_df = input_df.copy()
            display_df["loan_percent_income"] = display_df["loan_percent_income"].apply(lambda v: f"{v*100:.1f}%")
            with st.expander("📋 View submitted applicant profile"):
                st.dataframe(display_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ======================================================================================
# PAGE: MODEL INSIGHTS
# ======================================================================================
elif page == "📊 Model Insights":

    st.markdown('<div class="section-title">📊 Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Understand which factors most influence the model\'s loan decisions.</div>', unsafe_allow_html=True)

    gauge_col, img_col = st.columns([1, 1])
    with gauge_col:
        acc_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=94,
                number={"suffix": "%", "font": {"color": "#ffffff", "size": 34}},
                title={"text": "Held-out Test Accuracy", "font": {"color": "#d9d3c2", "size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#a49f8f"},
                    "bar": {"color": "#ffd447"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255,255,255,0.15)",
                    "steps": [
                        {"range": [0, 60], "color": "rgba(224,82,82,0.25)"},
                        {"range": [60, 85], "color": "rgba(255,212,71,0.22)"},
                        {"range": [85, 100], "color": "rgba(212,175,55,0.30)"},
                    ],
                },
            )
        )
        acc_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f4f1e8"},
            height=260,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        st.plotly_chart(acc_fig, use_container_width=True)
    with img_col:
        st.markdown(
            """
            <div class="photo-tile" style="height:260px; background-image:url('https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=800&auto=format&fit=crop');">
                <div class="photo-tile-icon">📈</div>
                <div class="photo-tile-label">Data-driven risk scoring</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    if model is None:
        st.warning("Load a model to view feature importance. Place the model file next to `app.py`.")
    else:
        try:
            feature_names = [
                "person_age", "person_gender", "person_education", "person_income",
                "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
                "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
                "credit_score", "previous_loan_defaults_on_file",
            ]
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({"Feature": feature_names[:len(importances)], "Importance": importances})
            imp_df_sorted = imp_df.sort_values("Importance", ascending=True)

            bar_col, radar_col = st.columns([1.3, 1])
            with bar_col:
                st.markdown("##### 🏆 Full Feature Importance")
                fig = px.bar(
                    imp_df_sorted,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=["#1a1a1a", "#d4af37"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f4f1e8"},
                    height=460,
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            with radar_col:
                st.markdown("##### 🎯 Top 5 Drivers")
                top5 = imp_df.sort_values("Importance", ascending=False).head(5)
                radar_fig = go.Figure()
                radar_fig.add_trace(
                    go.Scatterpolar(
                        r=top5["Importance"].tolist() + [top5["Importance"].iloc[0]],
                        theta=top5["Feature"].tolist() + [top5["Feature"].iloc[0]],
                        fill="toself",
                        line_color="#d4af37",
                        fillcolor="rgba(212,175,55,0.25)",
                    )
                )
                radar_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f4f1e8", "size": 10},
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(showticklabels=False, gridcolor="rgba(255,255,255,0.12)"),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.12)"),
                    ),
                    showlegend=False,
                    height=460,
                    margin=dict(l=30, r=30, t=20, b=20),
                )
                st.plotly_chart(radar_fig, use_container_width=True)
        except Exception as e:
            st.info(f"Feature importance is unavailable for this model object ({e}).")

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <p>Algorithm</p>
                <h3 style="font-size:1.4rem;">CatBoost Classifier</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <p>Tuning Method</p>
                <h3 style="font-size:1.4rem;">Randomized Search CV</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="glass-card">
                <p>Test Accuracy</p>
                <h3 style="font-size:1.4rem; color:#ffd447;">94%</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ======================================================================================
# PAGE: ABOUT
# ======================================================================================
else:
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-caption">
            <b>VaultIQ</b> is a machine-learning-powered loan approval predictor concept built
            for a data science / machine learning portfolio. It is built with a
            <b>CatBoost</b> gradient boosting classifier, reaching <b>94% accuracy</b> on the
            held-out test set. It evaluates applicant demographics, income, employment history,
            loan characteristics, and credit history to estimate the likelihood that a loan
            application will be approved.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📝 Feature Reference")
        feature_table = pd.DataFrame(
            [
                ["👤 person_age", "Applicant's age", "Float"],
                ["🚻 person_gender", "Applicant's gender", "Categorical"],
                ["🎓 person_education", "Applicant's highest level of education", "Categorical"],
                ["💰 person_income", "Applicant's annual income", "Float"],
                ["📅 person_emp_exp", "Years of employment experience", "Integer"],
                ["🏠 person_home_ownership", "Home ownership status", "Categorical"],
                ["🏦 loan_amnt", "Amount of loan requested", "Float"],
                ["🎯 loan_intent", "Intended purpose of the loan", "Categorical"],
                ["📈 loan_int_rate", "Interest rate applicable to the loan", "Float"],
                ["📊 loan_percent_income", "Loan amount as % of annual income (auto-calculated)", "Float"],
                ["🕰️ cb_person_cred_hist_length", "Years of credit history", "Float"],
                ["💳 credit_score", "Applicant's credit score", "Integer"],
                ["❗ previous_loan_defaults_on_file", "Indicator of previous loan defaults", "Categorical"],
            ],
            columns=["Feature", "Description", "Type"],
        )
        st.dataframe(feature_table, hide_index=True, use_container_width=True)

        st.write("")
        st.markdown("#### 🧮 Feature Type Breakdown")
        type_counts = feature_table["Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        donut_fig2 = px.pie(
            type_counts,
            names="Type",
            values="Count",
            hole=0.55,
            color="Type",
            color_discrete_map={"Categorical": "#ffd447", "Float": "#d4af37", "Integer": "#f4d67a"},
        )
        donut_fig2.update_traces(textfont_color="#050505", textinfo="label+value")
        donut_fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f4f1e8"},
            height=320,
            showlegend=True,
            legend=dict(font=dict(color="#d9d3c2")),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(donut_fig2, use_container_width=True)

    with right:
        st.image(
            "https://images.unsplash.com/photo-1580519542036-c47de6196ba5?q=80&w=1000&auto=format&fit=crop",
            caption="Data-driven credit decisions",
            use_container_width=True,
        )

        mrow1, mrow2 = st.columns(2)
        with mrow1:
            st.markdown(
                """<div class="mosaic-img"><img src="https://images.unsplash.com/photo-1560472354-b33ff0c44a43?q=80&w=500&auto=format&fit=crop" style="width:100%; display:block;"></div>""",
                unsafe_allow_html=True,
            )
        with mrow2:
            st.markdown(
                """<div class="mosaic-img"><img src="https://images.unsplash.com/photo-1579621970563-ebec7560ff3e?q=80&w=500&auto=format&fit=crop" style="width:100%; display:block;"></div>""",
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="footer-note">VaultIQ — concept / portfolio project · CatBoost Loan Approval Model · Built with Streamlit & Plotly</div>',
    unsafe_allow_html=True,
)