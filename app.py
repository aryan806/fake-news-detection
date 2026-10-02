import pandas as pd
import streamlit as st

from predict import predict_article

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


_BASE_CSS = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap" rel="stylesheet">
    <style>
      html, body, [class*="css"]  {
        font-family: 'DM Sans', system-ui, sans-serif !important;
      }
      .hero-wrap {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.18) 0%, rgba(99, 102, 241, 0.15) 50%, rgba(244, 63, 94, 0.12) 100%);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 20px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.5rem;
      }
      .hero-wrap h1 {
        font-family: 'Fraunces', Georgia, serif !important;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.5rem 0;
        font-size: clamp(1.75rem, 4vw, 2.35rem);
        line-height: 1.15;
      }
      .hero-sub {
        color: rgba(226, 232, 240, 0.82);
        font-size: 1.05rem;
        line-height: 1.55;
        max-width: 52rem;
        margin: 0;
      }
      .step-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.25rem; }
      .step-pill {
        flex: 1;
        min-width: 140px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 0.75rem 1rem;
      }
      .step-num {
        color: #2dd4bf;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
      }
      .step-txt { font-size: 0.9rem; margin-top: 0.25rem; color: #e2e8f0; }
      .verdict-card {
        border-radius: 16px;
        padding: 1.35rem 1.5rem;
        margin: 0.5rem 0 1rem 0;
        border: 2px solid var(--vc-border);
        background: var(--vc-bg);
      }
      .verdict-kicker {
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        font-weight: 600;
        opacity: 0.85;
        margin-bottom: 0.35rem;
      }
      .verdict-title {
        font-family: 'Fraunces', Georgia, serif !important;
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
      }
      .verdict-sub { font-size: 0.95rem; opacity: 0.9; margin: 0; }
    </style>
    """


st.markdown(_BASE_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### How to use")
    st.markdown(
        "1. **Paste** a full news article (headline + body is best).\n"
        "2. Click **Analyze article**.\n"
        "3. Read the **probabilities**—this is a statistical guess, not fact-checking."
    )
    st.divider()
    st.markdown("### Limits")
    st.markdown(
        "- Trained on a **specific dataset** (Reuters-style “real” vs. web “fake”).\n"
        "- **Short or casual text** often looks “fake” to the model.\n"
        "- Use for **learning / demos**, not legal or editorial decisions."
    )
    st.divider()
    st.caption("Stack: TF‑IDF + Random Forest · Streamlit")

hero = """
<div class="hero-wrap">
  <h1>Fake News Detector</h1>
  <p class="hero-sub">
    Paste an article to see how a simple ML model classifies it against patterns it learned
    from labeled news data—not whether the story is objectively true.
  </p>
  <div class="step-row">
    <div class="step-pill"><div class="step-num">STEP 1</div><div class="step-txt">Paste full article text</div></div>
    <div class="step-pill"><div class="step-num">STEP 2</div><div class="step-txt">Run the model</div></div>
    <div class="step-pill"><div class="step-num">STEP 3</div><div class="step-txt">Interpret probabilities</div></div>
  </div>
</div>
"""
st.markdown(hero, unsafe_allow_html=True)

left, right = st.columns((1.15, 1), gap="large")

with left:
    st.markdown("##### Article")
    text = st.text_area(
        "article",
        label_visibility="collapsed",
        height=280,
        placeholder="Paste several paragraphs from one article for more reliable scores…",
    )
    analyze = st.button("Analyze article", type="primary", use_container_width=True)

with right:
    st.markdown("##### Tips for better results")
    st.info(
        "Prefer **neutral wire or newspaper** prose and **enough length** (dozens of content words). "
        "One-liners and tweets are usually misclassified because they do not match training data."
    )
    st.markdown("##### What the numbers mean")
    st.markdown(
        "- **P(real)** / **P(fake)** come from the model’s `predict_proba`.\n"
        "- The bar chart shows the same split visually.\n"
        "- Low word count after stopword removal triggers a reliability warning."
    )

st.divider()

if analyze:
    with st.spinner("Scoring text…"):
        out = predict_article(text)

    if out["label"] == "UNKNOWN":
        st.warning(out["headline"])
    else:
        if out["label"] == "REAL NEWS":
            v_border, v_bg = "#10b981", "rgba(16, 185, 129, 0.14)"
        else:
            v_border, v_bg = "#fb7185", "rgba(251, 113, 133, 0.12)"

        st.markdown(
            f"""
            <div class="verdict-card" style="--vc-border: {v_border}; --vc-bg: {v_bg};">
              <div class="verdict-kicker">MODEL VERDICT</div>
              <div class="verdict-title">{out["label"]}</div>
              <p class="verdict-sub">Confidence on this label: <strong>{(out["p_fake"] if out["label"] == "FAKE NEWS" else out["p_real"]):.1%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("P (real)", f"{out['p_real']:.1%}")
        m2.metric("P (fake)", f"{out['p_fake']:.1%}")
        m3.metric("Content words", f"{out['word_count']}")

        chart = pd.DataFrame(
            {"probability": [out["p_real"], out["p_fake"]]},
            index=["Real", "Fake"],
        )
        st.markdown("##### Probability split")
        st.bar_chart(chart, height=220)

        if out["word_count"] < 40:
            st.info(
                f"**{out['word_count']}** content words after cleaning—scores are **unreliable** until you paste a longer article."
            )

st.markdown(
    '<p style="text-align:center;opacity:0.55;font-size:0.85rem;margin-top:2.5rem;">'
    "Educational demo · Not a substitute for professional fact-checking"
    "</p>",
    unsafe_allow_html=True,
)
