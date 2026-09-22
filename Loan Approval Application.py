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
    page_title="NBK Smart Loan Predictor | Academic Project",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================================
# CUSTOM CSS — BANKING / FINTECH THEME
# ======================================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, .hero-title {
        font-family: 'Poppins', sans-serif !important;
    }

    html, body {
        background: #0b1120;
    }

    /* App background */
    .stApp {
        background: transparent;
        color: #eef2f9;
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
        background: rgba(11, 17, 32, 0.4) !important;
        backdrop-filter: blur(4px);
    }

    /* Hide default hamburger footer branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* HERO BANNER */
    .hero-banner {
        position: relative;
        border-radius: 22px;
        overflow: hidden;
        padding: 3.2rem 2.5rem;
        margin-bottom: 2rem;
        background-image:
            linear-gradient(120deg, rgba(6,12,28,0.88) 10%, rgba(11,42,74,0.75) 55%, rgba(4,120,120,0.55) 100%),
            url('https://images.unsplash.com/photo-1554224155-6726b3ff858f?q=80&w=1600&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.45);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-eyebrow {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 30px;
        background: rgba(56, 224, 173, 0.15);
        border: 1px solid rgba(56, 224, 173, 0.4);
        color: #f0c869;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.6rem 0;
        line-height: 1.15;
    }
    .hero-title span {
        background: linear-gradient(90deg, #f0c869, #6aa9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cdd8ee;
        max-width: 680px;
        line-height: 1.6;
        font-weight: 300;
    }

    /* METRIC / GLASS CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 1.4rem 1.4rem;
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        height: 100%;
    }
    .glass-card h3 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff;
    }
    .glass-card p {
        margin: 0.2rem 0 0 0;
        color: #9db0cf;
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
        color: #93a4c2;
        font-size: 0.9rem;
        margin-bottom: 1.1rem;
        font-weight: 300;
    }

    /* FORM CONTAINER */
    .form-block {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.6rem 1.7rem 1rem 1.7rem;
        margin-bottom: 1.3rem;
    }

    /* BUTTON */
    .stButton>button {
        background: linear-gradient(90deg, #d4af37, #2f8bff);
        color: #051019;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.6rem;
        font-weight: 700;
        font-size: 1.02rem;
        letter-spacing: 0.02em;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 10px 25px rgba(212,175,55, 0.25);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 32px rgba(47, 139, 255, 0.35);
        color: #051019;
    }

    /* RESULT CARDS */
    .result-approved {
        background: linear-gradient(135deg, rgba(212,175,55,0.18), rgba(212,175,55,0.04));
        border: 1px solid rgba(212,175,55,0.5);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-rejected {
        background: linear-gradient(135deg, rgba(255,90,90,0.18), rgba(255,90,90,0.04));
        border: 1px solid rgba(255,90,90,0.5);
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
        color: #cdd8ee;
        font-size: 0.95rem;
        font-weight: 300;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060c18 0%, #0c1830 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.98rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.1rem;
        color: #b9c6e0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(212,175,55,0.16) !important;
        color: #f0c869 !important;
        font-weight: 700;
    }

    /* GLASS CARD HOVER */
    .glass-card {
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(212,175,55,0.4);
        box-shadow: 0 14px 34px rgba(212,175,55,0.15);
    }

    /* PHOTO STRIP TILES */
    .photo-tile {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        height: 150px;
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .photo-tile:hover {
        transform: scale(1.03);
        box-shadow: 0 16px 36px rgba(212,175,55,0.25);
    }
    .photo-tile::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(0deg, rgba(4,10,22,0.92) 5%, rgba(4,10,22,0.15) 60%, rgba(4,10,22,0.05) 100%);
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
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        transition: transform 0.25s ease;
    }
    .mosaic-img:hover {
        transform: translateY(-4px);
    }

    /* NBK-STYLE MONOGRAM */
    .nbk-monogram {
        width: 64px;
        height: 64px;
        margin: 0 auto 0.6rem auto;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #0b1e3d, #123166);
        border: 1.5px solid #d4af37;
        box-shadow: 0 6px 18px rgba(212,175,55,0.25);
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        letter-spacing: 0.03em;
        color: #f0c869;
    }

    /* DISCLAIMER BANNER */
    .disclaimer-banner {
        background: rgba(212,175,55,0.08);
        border: 1px solid rgba(212,175,55,0.35);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-size: 0.8rem;
        color: #cdd8ee;
        text-align: center;
        margin-bottom: 1.2rem;
    }

    /* Dataframe / misc text */
    .footer-note {
        text-align:center;
        color:#5e6f8f;
        font-size:0.8rem;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255,255,255,0.06);
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
            radial-gradient(ellipse at 20% 20%, rgba(212,175,55,0.10), transparent 55%),
            radial-gradient(ellipse at 80% 75%, rgba(47,139,255,0.12), transparent 55%),
            linear-gradient(180deg, #0b1120 0%, #0f1c33 45%, #101d33 100%);">
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

        const NUM_NODES = 70;
        const MAX_DIST = 150;
        const nodes = [];
        for (let i = 0; i < NUM_NODES; i++) {
            nodes.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.35,
                vy: (Math.random() - 0.5) * 0.35,
                r: Math.random() * 1.8 + 1.0,
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
                        const alpha = (1 - dist / MAX_DIST) * 0.35;
                        ctx.strokeStyle = `rgba(240,200,105, ${alpha})`;
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
                grad.addColorStop(0, 'rgba(106, 169, 255, 0.9)');
                grad.addColorStop(1, 'rgba(106, 169, 255, 0)');
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

# ======================================================================================
# SIDEBAR
# ======================================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.4rem 0 1.2rem 0;">
            <div class="nbk-monogram">NBK</div>
            <div style="font-family:'Poppins',sans-serif; font-weight:800; font-size:1.15rem; color:#fff;">
                Smart Loan <span style="color:#f0c869;">Predictor</span>
            </div>
            <div style="color:#7f90b3; font-size:0.78rem;">Academic Project · Inspired by NBK</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["🔮 Predict", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <p style="margin-bottom:0.3rem;">Model Accuracy</p>
            <h3 style="color:#f0c869;">94%</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Powered by CatBoost Gradient Boosting")

    st.markdown(
        """
        <div style="font-size:0.72rem; color:#5e6f8f; text-align:center; margin-top:0.8rem; line-height:1.5;">
        This is an independent academic/portfolio project inspired by
        National Bank of Kuwait's branding. It is not affiliated with,
        endorsed by, or operated by NBK.
        </div>
        """,
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

# ======================================================================================
# HERO
# ======================================================================================
st.markdown(
    '<div class="disclaimer-banner">🎓 Academic / portfolio project inspired by National Bank of Kuwait\'s brand style — not an official NBK product or affiliated with NBK in any way.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-eyebrow">🔒 AI-Powered Credit Risk Engine</div>
        <div class="hero-title">Instant, explainable <span>loan approval</span><br>decisions in seconds</div>
        <div class="hero-subtitle">
            A CatBoost gradient-boosting model trained on applicant demographics,
            income, employment, and credit-history data predicts loan approval outcomes with
            94% accuracy — styled here as a concept banking experience for NBK, built for a
            data science / machine learning portfolio.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
            person_income = st.number_input("Annual Income ($)", min_value=0, value=55000, step=1000)
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

        c1, c2 = st.columns(2)
        with c1:
            loan_percent_income = st.slider(
                "Loan Amount as % of Income", min_value=0.0, max_value=1.0, value=0.18, step=0.01
            )
        with c2:
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
                proba = model.predict_proba(input_df)[0]
                prediction = int(np.argmax(proba))
                approve_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
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
                            <div class="result-title" style="color:#f0c869;">Loan Approved</div>
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
                            <div class="result-title" style="color:#ff8080;">Loan Rejected</div>
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
                        title={"text": "Approval Probability", "font": {"color": "#cdd8ee", "size": 16}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#7f90b3"},
                            "bar": {"color": "#d4af37"},
                            "bgcolor": "rgba(0,0,0,0)",
                            "borderwidth": 1,
                            "bordercolor": "rgba(255,255,255,0.15)",
                            "steps": [
                                {"range": [0, 40], "color": "rgba(255,90,90,0.35)"},
                                {"range": [40, 70], "color": "rgba(255,200,80,0.30)"},
                                {"range": [70, 100], "color": "rgba(212,175,55,0.35)"},
                            ],
                        },
                    )
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#eef2f9"},
                    height=280,
                    margin=dict(l=20, r=20, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Probability breakdown bar chart
            prob_df = pd.DataFrame(
                {
                    "Outcome": ["Rejected", "Approved"],
                    "Probability": [float(proba[0]), float(proba[1])] if len(proba) > 1 else [1 - approve_prob, approve_prob],
                }
            )
            bar_fig = px.bar(
                prob_df,
                x="Probability",
                y="Outcome",
                orientation="h",
                color="Outcome",
                color_discrete_map={"Rejected": "#ff5a5a", "Approved": "#d4af37"},
                text=prob_df["Probability"].apply(lambda v: f"{v*100:.1f}%"),
            )
            bar_fig.update_traces(textposition="outside")
            bar_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#eef2f9"},
                showlegend=False,
                height=220,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.08)"),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # Applicant summary
            with st.expander("📋 View submitted applicant profile"):
                st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

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
                title={"text": "Held-out Test Accuracy", "font": {"color": "#cdd8ee", "size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#7f90b3"},
                    "bar": {"color": "#f0c869"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255,255,255,0.15)",
                    "steps": [
                        {"range": [0, 60], "color": "rgba(255,90,90,0.25)"},
                        {"range": [60, 85], "color": "rgba(255,200,80,0.25)"},
                        {"range": [85, 100], "color": "rgba(212,175,55,0.30)"},
                    ],
                },
            )
        )
        acc_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#eef2f9"},
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
                    color_continuous_scale=["#0f1c33", "#d4af37"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#eef2f9"},
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
                    font={"color": "#eef2f9", "size": 10},
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
                <h3 style="font-size:1.4rem; color:#f0c869;">94%</h3>
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
            This <b>NBK Smart Loan Predictor</b> is a machine-learning-powered loan approval
            predictor concept, styled after NBK's brand colors as a portfolio project. It is
            built with a
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
                ["📊 loan_percent_income", "Loan amount as % of annual income", "Float"],
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
        donut_fig = px.pie(
            type_counts,
            names="Type",
            values="Count",
            hole=0.55,
            color="Type",
            color_discrete_map={"Categorical": "#2f8bff", "Float": "#d4af37", "Integer": "#f0c869"},
        )
        donut_fig.update_traces(textfont_color="#0b1120", textinfo="label+value")
        donut_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#eef2f9"},
            height=320,
            showlegend=True,
            legend=dict(font=dict(color="#cdd8ee")),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(donut_fig, use_container_width=True)

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

        st.write("")
        st.markdown(
            """
            <div class="glass-card" style="margin-top:1rem;">
                <p>How to run</p>
                <h3 style="font-size:1.05rem; font-weight:500; color:#cdd8ee; line-height:1.6;">
                1. Place <code>catboost_loan_approval.cbm</code> next to <code>app.py</code><br>
                2. <code>pip install -r requirements.txt</code><br>
                3. <code>python -m streamlit run app.py</code>
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer-note">Academic project inspired by National Bank of Kuwait (NBK) branding — not affiliated with NBK · CatBoost Loan Approval Model · Built with Streamlit & Plotly</div>',
    unsafe_allow_html=True,
)