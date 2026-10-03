import pandas as pd
import streamlit as st

from predict import predict_article


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
BASE_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(99,102,241,0.12), transparent 28%),
        radial-gradient(circle at 90% 5%, rgba(20,184,166,0.10), transparent 25%),
        #0b1020;
}

/* Remove Streamlit top spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(148,163,184,0.12);
}

section[data-testid="stSidebar"] h3 {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Main title */
.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.1rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    margin-bottom: 0.15rem;
    background: linear-gradient(90deg, #ffffff, #a5b4fc, #5eead4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.tagline {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 760px;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* Top badge */
.ai-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(129,140,248,0.28);
    color: #c7d2fe;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
}

/* Info cards */
.info-card {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 18px;
    padding: 1.15rem 1.25rem;
    min-height: 105px;
}

.info-label {
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-weight: 700;
}

.info-value {
    color: #f8fafc;
    font-size: 1.15rem;
    font-weight: 600;
    margin-top: 0.35rem;
}

/* Section titles */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 0.6rem;
}

/* Input card */
.input-card {
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 20px;
    padding: 1.35rem;
}

/* Text area */
textarea {
    background: rgba(2,6,23,0.65) !important;
    border: 1px solid rgba(148,163,184,0.20) !important;
    border-radius: 14px !important;
}

/* Buttons */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* Verdict */
.verdict {
    border-radius: 20px;
    padding: 1.5rem;
    margin-top: 1.3rem;
    border: 1px solid var(--border);
    background: var(--background);
}

.verdict-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    color: #94a3b8;
}

.verdict-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 0.3rem 0;
    color: #f8fafc;
}

.verdict-confidence {
    color: #cbd5e1;
    font-size: 0.95rem;
}

/* Small text */
.muted {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* Footer */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(148,163,184,0.08);
}

</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:

    st.markdown(
        """
        <div style="
            font-family:'Space Grotesk';
            font-size:1.35rem;
            font-weight:700;
            margin-bottom:1.2rem;
        ">
        📰 TruthLens AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How it works")

    st.markdown(
        """
        **01 — Paste**

        Add the headline and body of a news article.

        **02 — Analyze**

        NewsGaurd analyzes the text using its learned language patterns.

        **03 — Review**

        The system returns a predicted classification and confidence.
        """
    )

    st.divider()

    st.markdown("### Model")

    st.markdown(
        """
        **NewsGaurd**

        BERT-based fake-news classification model.

        **Benchmark accuracy:** 61.95%

        Evaluated on 2,000 WELFake samples.
        """
    )

    st.divider()

    st.markdown("### Important")

    st.markdown(
        """
        This tool estimates whether text resembles
        **fake or real news according to the model**.

        It does **not independently verify facts,
        sources, or events.**
        """
    )


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------
st.markdown(
    '<div class="ai-badge">✦ AI-POWERED NEWS CLASSIFICATION</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brand">TruthLens AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="tagline">
        Analyze the language of a news article and see whether
        the model classifies it as <b>real</b> or <b>fake</b>.
        Built for education, experimentation, and understanding
        how machine-learning classifiers detect patterns in news.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# TOP INFO CARDS
# ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Model</div>
            <div class="info-value">NewsGaurd</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Architecture</div>
            <div class="info-value">BERT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Benchmark</div>
            <div class="info-value">61.95%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Purpose</div>
            <div class="info-value">Education</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# ARTICLE INPUT
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">Analyze a news article</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="muted" style="margin-bottom:0.8rem;">
        For the best result, paste the complete article including its headline.
    </div>
    """,
    unsafe_allow_html=True,
)

text = st.text_area(
    "Article text",
    label_visibility="collapsed",
    height=300,
    placeholder=(
        "Paste the headline and article text here...\n\n"
        "Example:\n"
        "Scientists have announced..."
    ),
)

analyze = st.button(
    "🔍  Analyze Article",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------
if analyze:

    with st.spinner("Analyzing article with NewsGaurd..."):
        out = predict_article(text)

    if out["label"] == "UNKNOWN":

        st.warning(out["headline"])

    elif out["label"] == "ERROR":

        st.error(out["headline"])

    else:

        if out["label"] == "REAL NEWS":
            border = "#34d399"
            background = "rgba(16,185,129,0.10)"
        else:
            border = "#fb7185"
            background = "rgba(244,63,94,0.10)"

        confidence = (
            out["p_real"]
            if out["label"] == "REAL NEWS"
            else out["p_fake"]
        )

        st.markdown(
            f"""
            <div class="verdict"
                 style="
                    --border:{border};
                    --background:{background};
                    border-color:{border};
                    background:{background};
                 ">
                <div class="verdict-label">MODEL VERDICT</div>

                <div class="verdict-title">
                    {out["label"]}
                </div>

                <div class="verdict-confidence">
                    Confidence: <b>{confidence:.1%}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "Real probability",
                f"{out['p_real']:.1%}"
            )

        with m2:
            st.metric(
                "Fake probability",
                f"{out['p_fake']:.1%}"
            )

        with m3:
            st.metric(
                "Words analyzed",
                f"{out['word_count']}"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability chart
        st.markdown(
            '<div class="section-title">Prediction breakdown</div>',
            unsafe_allow_html=True,
        )

        chart = pd.DataFrame(
            {
                "Probability": [
                    out["p_real"],
                    out["p_fake"],
                ]
            },
            index=["Real", "Fake"],
        )

        st.bar_chart(
            chart,
            height=250,
        )

        st.markdown(
            """
            <div class="muted">
                The probabilities represent the model's classification
                confidence. They should not be interpreted as proof that
                an article is factually true or false.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if out["word_count"] < 40:

            st.warning(
                f"""
                Only **{out['word_count']} words** were analyzed.
                Very short text may produce less reliable predictions.
                """
            )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        TruthLens AI · Powered by NewsGaurd · Educational project ·
        Not a substitute for professional fact-checking
    </div>
    """,
    unsafe_allow_html=True,
)
