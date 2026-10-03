"""
TruthLens AI - Streamlit frontend.

Only the user interface lives here. The prediction logic stays in predict.py
(predict_article), which this file imports and calls without changing it.
"""

import streamlit as st

from predict import predict_article  # backend: do not change

# ---------------------------------------------------------------------------
# Page setup and constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TruthLens AI - Fake News Detection",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_NAME = "NewsGaurd"
DISCLAIMER = (
    "This tool provides a machine-learning classification. It does not "
    "independently verify the factual accuracy of an article."
)


def html(markup: str) -> None:
    """Render raw HTML. Lines are joined so Markdown never treats them as code."""
    flat = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 1. CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
:root {
  color-scheme: light;
  --bg: #f7f6f3;
  --panel: #ffffff;
  --border: #d9d6cf;
  --text: #1f2328;
  --muted: #5f6670;
  --accent: #1f4e79;
  --accent-dark: #173d5f;
  --green: #2e7d4f;
  --red: #b3382c;
  --track: #e6e3dc;
}

/* Page */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
#MainMenu, footer { display: none !important; }
.block-container { max-width: 880px; padding: 1.5rem 1.25rem 3rem; }
html { scroll-behavior: smooth; }
[data-testid="stMarkdownContainer"] { color: var(--text); }
[data-testid="stSpinner"], [data-testid="stSpinner"] * { color: var(--muted) !important; }

/* Header */
.header {
  display: flex; justify-content: space-between; align-items: baseline;
  flex-wrap: wrap; gap: 0.5rem 1.5rem;
  padding-bottom: 0.9rem; border-bottom: 1px solid var(--border);
}
.brand { font-size: 1.2rem; font-weight: 700; }
.brand-sub { font-size: 0.9rem; color: var(--muted); margin-top: 0.1rem; }
.nav { display: flex; gap: 1.2rem; font-size: 0.95rem; }
.nav a { color: var(--muted) !important; text-decoration: none !important; }
.nav a:hover { color: var(--accent) !important; text-decoration: underline !important; }

/* Intro */
.intro { margin: 1.6rem 0 1.3rem; max-width: 620px; }
.intro-title { font-size: 1.5rem; font-weight: 600; line-height: 1.3; }
.intro-text { color: var(--muted); margin-top: 0.4rem; line-height: 1.6; }

/* Analyzer panel (a bordered Streamlit container holding the textarea) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(textarea) {
  background: var(--panel);
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 0.5rem 0.9rem 0.6rem;
}
.panel-title { font-size: 1.1rem; font-weight: 600; margin-top: 0.4rem; }
.panel-text { color: var(--muted); font-size: 0.93rem; margin: 0.2rem 0 0.7rem; }
.stTextArea [data-baseweb="textarea"], .stTextArea [data-baseweb="base-input"] {
  background: #fff !important; border-radius: 6px !important;
  border: 1px solid #bdb9b0 !important;
}
.stTextArea [data-baseweb="textarea"]:focus-within { border-color: var(--accent) !important; }
.stTextArea textarea {
  background: #fff !important; color: var(--text) !important;
  font-size: 0.97rem !important; line-height: 1.55 !important;
  font-family: inherit !important;
}
.stTextArea textarea::placeholder { color: #8a8f98 !important; }
.counter { color: var(--muted); font-size: 0.82rem; margin: 0.1rem 0 0.7rem; }

/* Button */
.stButton > button {
  background: var(--accent) !important; color: #fff !important;
  border: 1px solid var(--accent) !important; border-radius: 6px;
  padding: 0.5rem 1.4rem; font-weight: 600; font-size: 0.97rem; box-shadow: none;
}
.stButton > button:hover { background: var(--accent-dark) !important; border-color: var(--accent-dark) !important; }
.stButton > button p { color: #fff !important; }

/* Notices and result */
.notice {
  margin-top: 1rem; padding: 0.7rem 0.9rem; border-radius: 6px; font-size: 0.93rem;
  background: var(--panel); border: 1px solid var(--border);
}
.notice.warn { border-color: #c9a227; }
.notice.err { border-color: var(--red); }
.result {
  margin-top: 1.25rem; padding: 1.2rem 1.3rem; border-radius: 8px;
  background: var(--panel); border: 1px solid var(--border);
}
.result-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.verdict { font-size: 1.9rem; font-weight: 700; margin-top: 0.15rem; }
.verdict.real { color: var(--green); }
.verdict.fake { color: var(--red); }
.conf { margin: 0.1rem 0 1.1rem; }
.bar-row { display: flex; justify-content: space-between; font-size: 0.93rem; margin-top: 0.7rem; }
.bar-row b { font-weight: 600; }
.track { height: 8px; border-radius: 3px; background: var(--track); margin-top: 0.3rem; overflow: hidden; }
.fill { height: 100%; }
.fill.real { background: var(--green); }
.fill.fake { background: var(--red); }
.meta { margin-top: 1.1rem; color: var(--muted); font-size: 0.9rem; }
.short-note { margin-top: 0.4rem; color: var(--muted); font-size: 0.85rem; }

/* Disclaimer */
.disclaimer {
  margin-top: 1.25rem; padding: 0.7rem 0.9rem; border-radius: 6px; font-size: 0.88rem;
  color: #3d4a57; background: #eaeff3; border: 1px solid #d3dbe3;
}

/* Sections */
.section { margin-top: 2.6rem; scroll-margin-top: 1rem; }
.section-title { font-size: 1.25rem; font-weight: 600; margin-bottom: 0.9rem; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; }
.step { border-top: 2px solid var(--text); padding-top: 0.6rem; }
.step-num { font-size: 0.85rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.step-name { font-weight: 600; margin-top: 0.1rem; }
.step-text { color: var(--muted); font-size: 0.9rem; margin-top: 0.25rem; line-height: 1.5; }

.kv { display: grid; grid-template-columns: 11rem 1fr; padding: 0.5rem 0; border-bottom: 1px solid var(--border); font-size: 0.95rem; }
.kv:first-child { border-top: 1px solid var(--border); }
.kv .k { color: var(--muted); }
.model-note { margin-top: 0.8rem; color: var(--muted); font-size: 0.88rem; line-height: 1.6; max-width: 640px; }

/* Footer */
.footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.88rem; color: var(--muted); }
.footer b { color: var(--text); font-weight: 600; }

/* Small screens */
@media (max-width: 640px) {
  .block-container { padding: 1rem 0.9rem 2.5rem; }
  .steps { grid-template-columns: 1fr; gap: 1rem; }
  .kv { grid-template-columns: 8.5rem 1fr; }
  .verdict { font-size: 1.6rem; }
  .stButton > button { width: 100%; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Header and intro
# ---------------------------------------------------------------------------
def render_header() -> None:
    html("""
    <div class="header">
      <div>
        <div class="brand">TruthLens AI</div>
        <div class="brand-sub">Fake News Detection</div>
      </div>
      <div class="nav">
        <a href="#analyzer">Analyzer</a>
        <a href="#how-it-works">How it works</a>
        <a href="#about">About</a>
      </div>
    </div>
    """)


def render_intro() -> None:
    html("""
    <div class="intro">
      <div class="intro-title">Analyze a news article using a pretrained text-classification model.</div>
      <div class="intro-text">Paste an article below to see how the model classifies its language as real or fake.</div>
    </div>
    """)


# ---------------------------------------------------------------------------
# 3. Analyzer (input panel)
# ---------------------------------------------------------------------------
def render_analyzer():
    """Draw the input panel. Returns (text, analyze_clicked)."""
    html('<div id="analyzer"></div>')
    try:
        panel = st.container(border=True)
    except TypeError:  # very old Streamlit without bordered containers
        panel = st.container()
    with panel:
        html("""
        <div class="panel-title">Analyze article</div>
        <div class="panel-text">Paste the headline and article text below.</div>
        """)
        text = st.text_area(
            "Article text",
            key="article_text",
            height=260,
            placeholder="Headline\n\nArticle text...",
            label_visibility="collapsed",
        )
        html(f'<div class="counter">{len(text.split()):,} words · {len(text):,} characters</div>')
        clicked = st.button("Analyze Article", type="primary")
    return text, clicked


# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------
def _to_fraction(value) -> float:
    """Accept 0-1 or 0-100 values and return a number between 0 and 1."""
    value = float(value)
    if value > 1.0:
        value /= 100.0
    return min(max(value, 0.0), 1.0)


def normalize_result(raw, text: str) -> dict:
    """
    Read predict_article() output (a dict with label, p_real, p_fake,
    word_count; a 4-item tuple/list in that order also works).
    """
    if isinstance(raw, dict):
        label, p_real, p_fake = raw.get("label"), raw["p_real"], raw["p_fake"]
        word_count = raw.get("word_count")
    else:
        label, p_real, p_fake, word_count = raw
    p_real, p_fake = _to_fraction(p_real), _to_fraction(p_fake)

    name = str(label).upper()
    if "FAKE" in name:
        is_fake = True
    elif "REAL" in name:
        is_fake = False
    else:  # label is not a clear word, so trust the probabilities
        is_fake = p_fake > p_real

    return {
        "is_fake": is_fake,
        "p_real": p_real,
        "p_fake": p_fake,
        "word_count": int(word_count) if word_count is not None else len(text.split()),
    }


def render_result(res: dict) -> None:
    tone = "fake" if res["is_fake"] else "real"
    verdict = "FAKE NEWS" if res["is_fake"] else "REAL NEWS"
    conf = res["p_fake"] if res["is_fake"] else res["p_real"]
    real_pct, fake_pct = res["p_real"] * 100, res["p_fake"] * 100

    short_note = ""
    if res["word_count"] < 30:
        short_note = ('<div class="short-note">This text is quite short, so the model has '
                      'little to work with. Treat the result with extra care.</div>')

    html(f"""
    <div class="result">
      <div class="result-label">Prediction</div>
      <div class="verdict {tone}">{verdict}</div>
      <div class="conf">Confidence: {conf * 100:.1f}%</div>

      <div class="bar-row"><span>Real probability</span><b>{real_pct:.1f}%</b></div>
      <div class="track"><div class="fill real" style="width:{real_pct:.1f}%"></div></div>

      <div class="bar-row"><span>Fake probability</span><b>{fake_pct:.1f}%</b></div>
      <div class="track"><div class="fill fake" style="width:{fake_pct:.1f}%"></div></div>

      <div class="meta">Words analyzed: {res["word_count"]:,}</div>
      {short_note}
    </div>
    """)


def render_notice(kind: str, message: str) -> None:
    html(f'<div class="notice {kind}">{message}</div>')


def render_disclaimer() -> None:
    html(f'<div class="disclaimer">{DISCLAIMER}</div>')


# ---------------------------------------------------------------------------
# 5. How it works and model information
# ---------------------------------------------------------------------------
def render_how_it_works() -> None:
    html(f"""
    <div class="section" id="how-it-works">
      <div class="section-title">How it works</div>
      <div class="steps">
        <div class="step"><div class="step-num">01</div><div class="step-name">Paste article</div>
          <div class="step-text">Copy the headline and article text into the analyzer.</div></div>
        <div class="step"><div class="step-num">02</div><div class="step-name">Run {MODEL_NAME}</div>
          <div class="step-text">The model reads the text and scores it as real or fake.</div></div>
        <div class="step"><div class="step-num">03</div><div class="step-name">Review prediction</div>
          <div class="step-text">Check the label, the confidence and both probabilities.</div></div>
      </div>
    </div>
    """)


def render_model_info() -> None:
    rows = [
        ("Model", MODEL_NAME),
        ("Architecture", "BERT-based text classification"),
        ("Evaluation", "61.95% accuracy"),
        ("Evaluation sample", "2,000 WELFake articles"),
        ("Precision", "72.25%"),
        ("Recall", "38.80%"),
        ("F1 Score", "50.49%"),
    ]
    body = "".join(f'<div class="kv"><div class="k">{k}</div><div>{v}</div></div>' for k, v in rows)
    html(f"""
    <div class="section" id="about">
      <div class="section-title">Model information</div>
      {body}
      <div class="model-note">These figures describe performance on the evaluation sample.
      They do not guarantee that any single prediction is correct.</div>
    </div>
    """)


# ---------------------------------------------------------------------------
# 6. Footer
# ---------------------------------------------------------------------------
def render_footer() -> None:
    html(f"""
    <div class="footer">
      <b>TruthLens AI</b><br>
      Educational project · Powered by {MODEL_NAME}
    </div>
    """)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css()
    render_header()
    render_intro()

    text, clicked = render_analyzer()

    # Handle a click on "Analyze Article". Results are kept in session_state
    # so they stay on screen if the page reruns.
    if clicked:
        st.session_state["result"], st.session_state["notice"] = None, None
        if not text.strip():
            st.session_state["notice"] = ("warn", "Please paste a news article first, then press Analyze.")
        else:
            try:
                with st.spinner(f"Analyzing with {MODEL_NAME}..."):
                    raw = predict_article(text)
                st.session_state["result"] = normalize_result(raw, text)
            except Exception:
                st.session_state["notice"] = (
                    "err",
                    "Something went wrong while analyzing the article. "
                    "Please try again in a moment or try a different text.",
                )

    if st.session_state.get("notice"):
        render_notice(*st.session_state["notice"])
    if st.session_state.get("result"):
        render_result(st.session_state["result"])

    render_disclaimer()
    render_how_it_works()
    render_model_info()
    render_footer()


main()
