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
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_NAME = "NewsGaurd"
DISCLAIMER = (
    "TruthLens AI provides a machine-learning classification, "
    "not independent fact verification."
)


def html(markup: str) -> None:
    """Render raw HTML. Lines are joined so Markdown never treats them as code."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 1. CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  color-scheme: light;
  --bg: #f6f5f2;
  --panel: #ffffff;
  --border: #dcd9d2;
  --text: #1b2433;
  --muted: #5b6678;
  --accent: #2456a6;
  --accent-dark: #1b447f;
  --green: #2d7a4d;
  --green-tint: #f1f7f3;
  --red: #b23a30;
  --red-tint: #fbf3f2;
  --track: #e8e5de;
  --shadow: 0 1px 2px rgba(27,36,51,0.06), 0 6px 18px rgba(27,36,51,0.05);
}

/* Page */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
#MainMenu, footer { display: none !important; }
.block-container {
  max-width: 1120px !important; margin: 0 auto;
  padding: 0.6rem 1.5rem 3rem !important;
}
html { scroll-behavior: smooth; }
[data-testid="stVerticalBlock"] { gap: 0.5rem; }
[data-testid="stMarkdownContainer"] { color: var(--text); }
[data-testid="stSpinner"], [data-testid="stSpinner"] * { color: var(--muted) !important; }

/* Header */
.header {
  display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: 0.4rem 1.5rem;
  padding: 0.35rem 0 0.8rem; border-bottom: 1px solid var(--border);
}
.brand { display: flex; align-items: baseline; gap: 0.8rem; flex-wrap: wrap; }
.brand-name { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; }
.brand-sub { font-size: 0.85rem; color: var(--muted); padding-left: 0.8rem; border-left: 1px solid var(--border); }
.nav { display: flex; gap: 1.4rem; font-size: 0.92rem; font-weight: 500; }
.nav a { color: var(--muted) !important; text-decoration: none !important; }
.nav a:hover { color: var(--accent) !important; }

/* Hero: headline on the left, short text on the right */
.hero {
  display: grid; grid-template-columns: 1.1fr 1fr; gap: 1rem 3rem;
  align-items: end; margin: 1rem 0 1rem;
}
.eyebrow { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: var(--accent); }
.hero-title { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; margin-top: 0.35rem; }
.hero-sub { font-size: 1.02rem; color: var(--text); margin-top: 0.4rem; line-height: 1.5; }
.hero-text { color: var(--muted); font-size: 0.95rem; line-height: 1.6; }

/* Information strip (one bar, thin dividers) */
.strip {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
  background: var(--border); border: 1px solid var(--border);
  border-radius: 10px; overflow: hidden; margin-bottom: 0.9rem;
}
.strip > div { background: var(--panel); padding: 0.65rem 1rem; }
.strip .k { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 600; }
.strip .v { font-weight: 600; margin-top: 0.1rem; font-size: 0.97rem; }

/* Analyzer panel (a bordered Streamlit container holding the textarea) */
.st-key-analyzer-panel {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: var(--shadow) !important;
  padding: 0.6rem 1rem 0.8rem !important;
}
.panel-title { font-size: 1.15rem; font-weight: 650; margin-top: 0.3rem; }
.panel-text { color: var(--muted); font-size: 0.92rem; margin: 0.15rem 0 0.5rem; }
.stTextArea [data-testid="stTextAreaRootElement"] {
  background: #fff !important; border-radius: 8px !important;
  border: 1px solid #b9b5ab !important; overflow: hidden;
}
.stTextArea [data-testid="stTextAreaRootElement"]:focus-within {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(36,86,166,0.13) !important;
}
.stTextArea textarea {
  background: #fff !important; color: var(--text) !important;
  font-size: 0.98rem !important; line-height: 1.6 !important;
  padding: 0.8rem 1rem !important; font-family: inherit !important;
  border: 0 !important; border-radius: 0 !important; outline: none !important; box-shadow: none !important;
}
.stTextArea textarea::placeholder { color: #8b919c !important; }
.counter { color: var(--muted); font-size: 0.85rem; }

/* Button */
div.stButton, [data-testid="stButton"] { display: flex; justify-content: flex-end; }
.st-key-analyzer-panel [data-testid="stColumn"]:last-of-type [data-testid="stVerticalBlock"] { align-items: flex-end; }
.stButton > button {
  min-width: 230px; padding: 0.65rem 1.6rem; border-radius: 8px;
  background: var(--accent) !important; color: #fff !important;
  border: 1px solid var(--accent) !important;
  font-weight: 600; font-size: 1rem; box-shadow: 0 1px 2px rgba(27,36,51,0.15);
}
.stButton > button:hover { background: var(--accent-dark) !important; border-color: var(--accent-dark) !important; }
.stButton > button p { color: #fff !important; font-weight: 600; font-size: 1rem; }

/* Notices and result */
.notice {
  margin-top: 1rem; padding: 0.7rem 0.95rem; border-radius: 8px; font-size: 0.93rem;
  background: var(--panel); border: 1px solid var(--border);
}
.notice.warn { border-color: #c9a227; }
.notice.err { border-color: var(--red); }
.result {
  margin-top: 1.2rem; padding: 1.3rem 1.5rem 1.2rem; border-radius: 12px;
  background: var(--panel); border: 1px solid var(--border); box-shadow: var(--shadow);
}
.result.real { background: var(--green-tint); border-top: 3px solid var(--green); }
.result.fake { background: var(--red-tint); border-top: 3px solid var(--red); }
.result-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.verdict { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; margin-top: 0.15rem; line-height: 1.15; }
.result.real .verdict { color: var(--green); }
.result.fake .verdict { color: var(--red); }
.conf { color: var(--muted); margin-top: 0.15rem; }
.probs { margin-top: 1.1rem; padding-top: 1rem; border-top: 1px solid rgba(27,36,51,0.12); }
.prob-row { display: grid; grid-template-columns: 3.4rem 4.2rem 1fr; align-items: center; gap: 0.8rem; padding: 0.3rem 0; }
.prob-row .name { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.06em; color: var(--muted); }
.prob-row .pct { font-weight: 700; font-variant-numeric: tabular-nums; }
.track { height: 10px; border-radius: 5px; background: var(--track); overflow: hidden; }
.fill { height: 100%; border-radius: 5px; }
.fill.real { background: var(--green); }
.fill.fake { background: var(--red); }
.meta { margin-top: 0.9rem; color: var(--muted); font-size: 0.9rem; }
.short-note { margin-top: 0.3rem; color: var(--muted); font-size: 0.85rem; }

/* Disclaimer */
.disclaimer {
  margin-top: 1.2rem; padding: 0.7rem 1rem; border-radius: 8px; font-size: 0.9rem;
  color: #2f3f55; background: #edf2f9; border: 1px solid #d3deee;
}

/* Sections */
.section { margin-top: 2.6rem; padding-top: 1.4rem; border-top: 1px solid var(--border); scroll-margin-top: 1rem; }
.section-title { font-size: 1.3rem; font-weight: 650; letter-spacing: -0.01em; margin-bottom: 1rem; }

.steps { display: grid; grid-template-columns: repeat(3, 1fr); }
.step { padding: 0 1.5rem; border-left: 1px solid var(--border); }
.step:first-child { padding-left: 0; border-left: 0; }
.step-num { color: var(--accent); font-weight: 700; font-size: 0.95rem; font-variant-numeric: tabular-nums; }
.step-name { font-weight: 600; margin-top: 0.15rem; }
.step-text { color: var(--muted); font-size: 0.92rem; margin-top: 0.2rem; line-height: 1.5; }

.about { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem 3rem; align-items: start; }
.about-text { color: var(--muted); line-height: 1.7; font-size: 0.95rem; }
.about-text div + div { margin-top: 0.7rem; }
.about-text b { color: var(--text); font-weight: 600; }
.kv { display: flex; justify-content: space-between; padding: 0.55rem 0; border-bottom: 1px solid var(--border); font-size: 0.97rem; }
.kv:first-child { border-top: 1px solid var(--border); }
.kv .k { color: var(--muted); }
.kv .v { font-weight: 600; font-variant-numeric: tabular-nums; }
.eval-set { margin-top: 0.9rem; font-size: 0.93rem; }
.eval-set span { color: var(--muted); }
.model-note { margin-top: 0.5rem; color: var(--muted); font-size: 0.85rem; line-height: 1.55; }

/* Footer */
.footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.88rem; color: var(--muted); line-height: 1.6; }
.footer b { color: var(--text); font-weight: 600; }

/* Tablet and mobile */
@media (max-width: 860px) {
  .hero { grid-template-columns: 1fr; margin-top: 1.2rem; }
  .about { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .block-container { padding: 0.7rem 0.9rem 2.5rem !important; }
  .brand-sub { border-left: 0; padding-left: 0; }
  .strip { grid-template-columns: repeat(2, 1fr); }
  .steps { grid-template-columns: 1fr; }
  .step { padding: 0.9rem 0; border-left: 0; border-top: 1px solid var(--border); }
  .step:first-child { border-top: 0; padding-top: 0; }
  .hero-title { font-size: 1.7rem; }
  .verdict { font-size: 1.8rem; }
  .result { padding: 1.1rem; }
  div.stButton, [data-testid="stButton"] { justify-content: stretch; }
  .st-key-analyzer-panel [data-testid="stColumn"]:last-of-type [data-testid="stVerticalBlock"] { align-items: stretch; }
  .st-key-analyzer-panel [data-testid="stElementContainer"]:has([data-testid="stButton"]) { width: 100% !important; }
  .stButton > button { width: 100%; min-width: 0; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Header, hero and information strip
# ---------------------------------------------------------------------------
def render_header() -> None:
    html("""
    <div class="header">
      <div class="brand">
        <span class="brand-name">TruthLens AI</span>
        <span class="brand-sub">Fake News Detection</span>
      </div>
      <div class="nav">
        <a href="#analyzer">Analyzer</a>
        <a href="#how-it-works">How it works</a>
        <a href="#about">About</a>
      </div>
    </div>
    """)


def render_hero() -> None:
    html(f"""
    <div class="hero">
      <div>
        <div class="eyebrow">Powered by {MODEL_NAME}</div>
        <div class="hero-title">Fake News Detection</div>
        <div class="hero-sub">Analyze a news article using a pretrained text-classification model.</div>
      </div>
      <div class="hero-text">Paste the headline and article body below to see how
      {MODEL_NAME} classifies the text.</div>
    </div>
    """)


def render_info_strip() -> None:
    html(f"""
    <div class="strip">
      <div><div class="k">Model</div><div class="v">{MODEL_NAME}</div></div>
      <div><div class="k">Architecture</div><div class="v">BERT</div></div>
      <div><div class="k">Accuracy</div><div class="v">61.95%</div></div>
      <div><div class="k">Evaluation</div><div class="v">2,000 articles</div></div>
    </div>
    """)


# ---------------------------------------------------------------------------
# 3. Analyzer (input panel)
# ---------------------------------------------------------------------------
def render_analyzer():
    """Draw the input panel. Returns (text, analyze_clicked)."""
    html('<div id="analyzer"></div>')
    try:
        panel = st.container(border=True, key="analyzer-panel")
    except TypeError:  # older Streamlit: plain bordered container, default look
        try:
            panel = st.container(border=True)
        except TypeError:
            panel = st.container()
    with panel:
        html("""
        <div class="panel-title">Analyze article</div>
        <div class="panel-text">Paste the headline and article text below.</div>
        """)
        text = st.text_area(
            "Article text",
            key="article_text",
            height=250,
            placeholder="Headline\n\nPaste the full article text here...",
            label_visibility="collapsed",
        )
        try:
            left, right = st.columns([3, 2], vertical_alignment="center")
        except TypeError:  # older Streamlit without vertical_alignment
            left, right = st.columns([3, 2])
        with left:
            html(f'<div class="counter">{len(text.split()):,} words · {len(text):,} characters</div>')
        with right:
            clicked = st.button("Analyze Article →", type="primary")
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
    <div class="result {tone}">
      <div class="result-label">Prediction</div>
      <div class="verdict">{verdict}</div>
      <div class="conf">{conf * 100:.1f}% confidence</div>

      <div class="probs">
        <div class="prob-row">
          <span class="name">REAL</span><span class="pct">{real_pct:.1f}%</span>
          <div class="track"><div class="fill real" style="width:{real_pct:.1f}%"></div></div>
        </div>
        <div class="prob-row">
          <span class="name">FAKE</span><span class="pct">{fake_pct:.1f}%</span>
          <div class="track"><div class="fill fake" style="width:{fake_pct:.1f}%"></div></div>
        </div>
      </div>

      <div class="meta">Words analyzed: {res["word_count"]:,}</div>
      {short_note}
    </div>
    """)


def render_notice(kind: str, message: str) -> None:
    html(f'<div class="notice {kind}">{message}</div>')


def render_disclaimer() -> None:
    html(f'<div class="disclaimer">{DISCLAIMER}</div>')


# ---------------------------------------------------------------------------
# 5. How it works and model section
# ---------------------------------------------------------------------------
def render_how_it_works() -> None:
    html(f"""
    <div class="section" id="how-it-works">
      <div class="section-title">How it works</div>
      <div class="steps">
        <div class="step"><div class="step-num">01</div><div class="step-name">Paste article</div>
          <div class="step-text">Add the headline and article body.</div></div>
        <div class="step"><div class="step-num">02</div><div class="step-name">Analyze</div>
          <div class="step-text">{MODEL_NAME} processes the text.</div></div>
        <div class="step"><div class="step-num">03</div><div class="step-name">Review</div>
          <div class="step-text">Examine the prediction and probabilities.</div></div>
      </div>
    </div>
    """)


def render_model_info() -> None:
    rows = [("Accuracy", "61.95%"), ("Precision", "72.25%"), ("Recall", "38.80%"), ("F1 Score", "50.49%")]
    table = "".join(f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in rows)
    html(f"""
    <div class="section" id="about">
      <div class="section-title">About the model</div>
      <div class="about">
        <div class="about-text">
          <div><b>{MODEL_NAME}</b> is a BERT-based text-classification model. TruthLens AI
          uses the pretrained model <b>nallarahul/{MODEL_NAME}</b> as it is, without any
          changes.</div>
          <div>The model reads the wording of an article and returns a probability for
          real and for fake. It picks up patterns in language, so it is a classifier and
          not a fact-checker.</div>
        </div>
        <div>
          {table}
          <div class="eval-set"><span>Evaluation set:</span> 2,000 WELFake samples</div>
          <div class="model-note">These figures come from our external evaluation sample and do
          not guarantee the correctness of individual predictions.</div>
        </div>
      </div>
    </div>
    """)


# ---------------------------------------------------------------------------
# 6. Footer
# ---------------------------------------------------------------------------
def render_footer() -> None:
    html(f"""
    <div class="footer">
      <b>TruthLens AI</b><br>
      Educational AI Project · Powered by {MODEL_NAME}
    </div>
    """)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css()
    render_header()
    render_hero()
    render_info_strip()

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
