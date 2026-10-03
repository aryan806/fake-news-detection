"""
TruthLens AI - Streamlit frontend.

Only the user interface lives here. The prediction logic stays in predict.py
(predict_article), which this file imports and calls without changing it.
"""

import streamlit as st

from predict import predict_article  # backend: do not change

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_NAME = "NewsGaurd"
BENCH_ACCURACY = "61.95%"
BENCH_SAMPLES = "2,000"
DISCLAIMER = (
    "TruthLens AI provides a machine-learning classification, not independent "
    "fact verification. A prediction should not be treated as proof that a "
    "story is true or false."
)


def html(markup: str) -> None:
    """Render raw HTML. Lines are joined so Markdown never treats them as code."""
    flat = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 1. CSS (all styling lives here)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #070b14;
  --card: rgba(255,255,255,0.045);
  --card-solid: #0e1424;
  --border: rgba(255,255,255,0.09);
  --text: #e8ecf5;
  --muted: #8f9ab3;
  --accent: #7c8cff;
  --accent2: #22d3ee;
  --green: #34d399;
  --red: #fb7185;
}

/* Base page */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  color: var(--text);
}
.stApp {
  background:
    radial-gradient(900px 500px at 12% -5%, rgba(99,102,241,0.22), transparent 60%),
    radial-gradient(800px 450px at 95% 5%, rgba(34,211,238,0.14), transparent 60%),
    var(--bg);
}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer,
[data-testid="stDecoration"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
.block-container { max-width: 1120px; padding: 1.2rem 1.4rem 3rem; }
html { scroll-behavior: smooth; }

/* Header */
.nav {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; padding: 0.8rem 1.1rem; border-radius: 18px;
  background: var(--card); border: 1px solid var(--border);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
.brand { display: flex; align-items: center; gap: 0.7rem; }
.logo {
  width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center;
  background: linear-gradient(135deg, var(--accent), var(--accent2)); font-size: 1.1rem;
  box-shadow: 0 6px 20px rgba(99,102,241,0.4);
}
.wordmark { font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; }
.wordmark span { color: var(--accent2); }
.tag {
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); border: 1px solid var(--border); border-radius: 999px; padding: 0.2rem 0.6rem;
}
.nav-links { display: flex; gap: 1.3rem; }
.nav-links a { color: var(--muted) !important; text-decoration: none !important; font-size: 0.9rem; font-weight: 500; }
.nav-links a:hover { color: var(--text) !important; }

/* Hero */
.hero { text-align: center; padding: 3.6rem 0.5rem 1.8rem; }
.badge {
  display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em;
  padding: 0.4rem 0.9rem; border-radius: 999px; color: #c7d0ff;
  background: rgba(124,140,255,0.12); border: 1px solid rgba(124,140,255,0.35);
}
.hero h1 {
  font-size: clamp(2.1rem, 6vw, 4rem); line-height: 1.05; font-weight: 800;
  letter-spacing: -0.035em; margin: 1.1rem 0 0.9rem; padding: 0;
  background: linear-gradient(120deg, #ffffff 20%, #9fb0ff 60%, #22d3ee);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { max-width: 640px; margin: 0 auto; color: var(--muted); font-size: clamp(0.95rem, 2.2vw, 1.1rem); line-height: 1.65; }
.tagline { margin-top: 1rem; font-size: 0.85rem; color: var(--accent2); font-weight: 600; letter-spacing: 0.04em; }

/* Stat cards */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.9rem; margin: 1.2rem 0 2.2rem; }
.stat { background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 1.1rem 1.2rem; }
.stat .k { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 600; }
.stat .v { font-size: 1.45rem; font-weight: 700; margin-top: 0.35rem; letter-spacing: -0.02em; }
.stat .v.grad { background: linear-gradient(120deg, var(--accent), var(--accent2)); -webkit-background-clip: text; background-clip: text; color: transparent; }

/* Analyzer card (a bordered Streamlit container that holds a textarea) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(textarea) {
  background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid var(--border) !important; border-radius: 24px !important;
  padding: 0.6rem 0.8rem 0.4rem; box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}
.card-title { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin: 0.6rem 0 0.3rem; }
.card-sub { color: var(--muted); font-size: 0.95rem; line-height: 1.6; margin-bottom: 0.8rem; }
.stTextArea textarea {
  background: rgba(7,11,20,0.7) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 16px !important;
  font-size: 1rem !important; line-height: 1.6 !important; padding: 1rem 1.1rem !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}
.stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(124,140,255,0.2) !important; }
.stTextArea textarea::placeholder { color: #5d6784 !important; }
.counter { color: var(--muted); font-size: 0.8rem; margin: -0.2rem 0 0.8rem 0.2rem; }

/* Primary button */
.stButton { width: 100%; }
.stButton > button {
  width: 100%; border: 0; border-radius: 14px; padding: 0.85rem 1.2rem; font-weight: 700;
  font-size: 1.05rem; color: #fff !important;
  background: linear-gradient(120deg, #6366f1, #22a6ee) !important;
  box-shadow: 0 10px 30px rgba(99,102,241,0.35); transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 14px 36px rgba(99,102,241,0.5); }
.stButton > button:active { transform: translateY(0); }

/* Result card */
.result { margin-top: 1.6rem; border-radius: 24px; padding: 1.6rem; border: 1px solid var(--border); animation: rise .5s ease both; }
.result.real { background: linear-gradient(160deg, rgba(52,211,153,0.13), rgba(52,211,153,0.03)); border-color: rgba(52,211,153,0.35); }
.result.fake { background: linear-gradient(160deg, rgba(251,113,133,0.14), rgba(251,113,133,0.03)); border-color: rgba(251,113,133,0.38); }
.kicker { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.14em; color: var(--muted); font-weight: 600; }
.verdict { font-size: clamp(2rem, 6vw, 3.2rem); font-weight: 800; letter-spacing: -0.03em; margin: 0.3rem 0; line-height: 1.1; }
.result.real .verdict { color: var(--green); }
.result.fake .verdict { color: var(--red); }
.conf { font-size: 1.05rem; color: var(--text); font-weight: 500; }
.prob-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.9rem; margin: 1.4rem 0 1rem; }
.prob { background: rgba(7,11,20,0.55); border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.1rem; }
.prob .row { display: flex; justify-content: space-between; align-items: baseline; }
.prob .name { color: var(--muted); font-size: 0.85rem; font-weight: 600; }
.prob .pct { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }
.track { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.08); margin-top: 0.7rem; overflow: hidden; }
.fill { height: 100%; border-radius: 999px; transform-origin: left; animation: grow .9s ease both; }
.fill.g { background: linear-gradient(90deg, #10b981, var(--green)); }
.fill.r { background: linear-gradient(90deg, #f43f5e, var(--red)); }
.split-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 600; margin: 1.2rem 0 0.5rem; }
.split { display: flex; height: 16px; border-radius: 999px; overflow: hidden; background: rgba(255,255,255,0.06); }
.split .g { background: linear-gradient(90deg, #10b981, var(--green)); }
.split .r { background: linear-gradient(90deg, #f43f5e, var(--red)); }
.split > div { transform-origin: left; animation: grow .9s ease both; }
.legend { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--muted); margin-top: 0.45rem; gap: 1rem; flex-wrap: wrap; }
.chip { display: inline-block; margin-top: 1.1rem; font-size: 0.85rem; color: var(--muted); border: 1px solid var(--border); border-radius: 999px; padding: 0.3rem 0.85rem; }
.chip b { color: var(--text); }
.note { margin-top: 0.8rem; font-size: 0.82rem; color: var(--muted); }
.notice { margin-top: 1.2rem; padding: 0.9rem 1.1rem; border-radius: 14px; font-size: 0.92rem; border: 1px solid var(--border); background: var(--card); color: var(--text); }
.notice.warn { border-color: rgba(250,204,21,0.4); background: rgba(250,204,21,0.07); }
.notice.err { border-color: rgba(251,113,133,0.4); background: rgba(251,113,133,0.07); }
.disclaimer { margin-top: 1rem; font-size: 0.78rem; line-height: 1.6; color: #6f7a94; }

/* Sections */
.section { margin-top: 3.2rem; scroll-margin-top: 1rem; }
.section h2 { font-size: clamp(1.4rem, 3.6vw, 1.9rem); font-weight: 700; letter-spacing: -0.02em; margin: 0 0 0.4rem; padding: 0; }
.section .lead { color: var(--muted); margin-bottom: 1.2rem; font-size: 0.97rem; line-height: 1.6; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.9rem; }
.step { background: var(--card); border: 1px solid var(--border); border-radius: 20px; padding: 1.3rem; }
.num { width: 42px; height: 42px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; font-size: 0.9rem;
  background: linear-gradient(135deg, rgba(124,140,255,0.25), rgba(34,211,238,0.2)); border: 1px solid rgba(124,140,255,0.4); }
.step h3 { font-size: 1.05rem; font-weight: 600; margin: 0.9rem 0 0.3rem; padding: 0; }
.step p { color: var(--muted); font-size: 0.9rem; line-height: 1.6; margin: 0; }
.about { background: var(--card); border: 1px solid var(--border); border-radius: 22px; padding: 1.5rem; display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.6rem; }
.about p { color: var(--muted); line-height: 1.7; font-size: 0.93rem; margin: 0 0 0.7rem; }
.about ul { margin: 0; padding-left: 1.1rem; color: var(--muted); line-height: 1.9; font-size: 0.93rem; }
.bench { background: rgba(7,11,20,0.55); border: 1px solid var(--border); border-radius: 16px; padding: 1.2rem; }
.bench .big { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em;
  background: linear-gradient(120deg, var(--accent), var(--accent2)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.bench .cap { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.8rem; }
.mini { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-top: 0.4rem; }
.mini div { border: 1px solid var(--border); border-radius: 12px; padding: 0.6rem; text-align: center; }
.mini b { display: block; font-size: 1rem; }
.mini span { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.bench .warn { margin-top: 0.9rem; font-size: 0.8rem; line-height: 1.6; color: #a4aec6; }

/* Footer */
.footer { margin-top: 3.5rem; padding-top: 1.4rem; border-top: 1px solid var(--border); display: flex; justify-content: space-between;
  flex-wrap: wrap; gap: 0.5rem; color: var(--muted); font-size: 0.85rem; }

/* Animations */
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* Mobile and tablet */
@media (max-width: 760px) {
  .nav-links { display: none; }
  .tag { display: none; }
  .hero { padding-top: 2.4rem; }
  .block-container { padding: 0.8rem 0.9rem 2.5rem; }
  .result { padding: 1.2rem; }
  .footer { flex-direction: column; text-align: center; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Header and hero
# ---------------------------------------------------------------------------
def render_header() -> None:
    html("""
    <div class="nav">
      <div class="brand">
        <div class="logo">🔍</div>
        <div class="wordmark">TruthLens<span> AI</span></div>
        <div class="tag">AI News Classifier</div>
      </div>
      <div class="nav-links">
        <a href="#analyzer">Analyzer</a>
        <a href="#how-it-works">How it works</a>
        <a href="#about">About</a>
      </div>
    </div>
    """)


def render_hero() -> None:
    html(f"""
    <div class="hero">
      <span class="badge">POWERED BY NEWSGAURD</span>
      <h1>Detect patterns. Question the story.</h1>
      <p>TruthLens AI analyzes news text using a pretrained language model and
      predicts whether it resembles real or fake news.</p>
      <div class="tagline">AI-powered news classification</div>
    </div>
    """)


def render_stats() -> None:
    html(f"""
    <div class="stats">
      <div class="stat"><div class="k">Model</div><div class="v">{MODEL_NAME}</div></div>
      <div class="stat"><div class="k">Architecture</div><div class="v">BERT</div></div>
      <div class="stat"><div class="k">Benchmark accuracy</div><div class="v grad">{BENCH_ACCURACY}</div></div>
      <div class="stat"><div class="k">Dataset evaluation</div><div class="v">{BENCH_SAMPLES} WELFake samples</div></div>
    </div>
    """)


# ---------------------------------------------------------------------------
# 3. Analyzer (input card)
# ---------------------------------------------------------------------------
def render_analyzer():
    """Draw the input card. Returns (text, analyze_clicked)."""
    html('<div id="analyzer"></div>')
    try:
        card = st.container(border=True)
    except TypeError:  # very old Streamlit without bordered containers
        card = st.container()
    with card:
        html("""
        <div class="card-title">Analyze a news article</div>
        <div class="card-sub">Paste the headline and the full article text below.
        Longer, complete articles give the model more to work with.</div>
        """)
        text = st.text_area(
            "Article text",
            key="article_text",
            height=280,
            placeholder="Paste the headline, then the article body here...",
            label_visibility="collapsed",
        )
        words = len(text.split())
        html(f'<div class="counter">{words:,} words · {len(text):,} characters</div>')
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
        short_note = ('<div class="notice warn">This text is quite short. '
                      'The model has little to work with, so treat the result with extra care.</div>')

    html(f"""
    <div class="result {tone}">
      <div class="kicker">Prediction</div>
      <div class="verdict">{verdict}</div>
      <div class="conf">Model confidence: {conf * 100:.1f}%</div>

      <div class="prob-grid">
        <div class="prob">
          <div class="row"><span class="name">Real probability</span><span class="pct">{real_pct:.1f}%</span></div>
          <div class="track"><div class="fill g" style="width:{real_pct:.1f}%"></div></div>
        </div>
        <div class="prob">
          <div class="row"><span class="name">Fake probability</span><span class="pct">{fake_pct:.1f}%</span></div>
          <div class="track"><div class="fill r" style="width:{fake_pct:.1f}%"></div></div>
        </div>
      </div>

      <div class="split-label">Probability split</div>
      <div class="split">
        <div class="g" style="width:{real_pct:.1f}%"></div>
        <div class="r" style="width:{fake_pct:.1f}%"></div>
      </div>
      <div class="legend"><span>Real {real_pct:.1f}%</span><span>Fake {fake_pct:.1f}%</span></div>

      <span class="chip">Words analyzed: <b>{res["word_count"]:,}</b></span>
      <div class="disclaimer">{DISCLAIMER}</div>
    </div>
    {short_note}
    """)


def render_notice(kind: str, message: str) -> None:
    html(f'<div class="notice {kind}">{message}</div>')


# ---------------------------------------------------------------------------
# 5. How it works and model info
# ---------------------------------------------------------------------------
def render_how_it_works() -> None:
    html(f"""
    <div class="section" id="how-it-works">
      <h2>How it works</h2>
      <div class="lead">Three simple steps from article to prediction.</div>
      <div class="steps">
        <div class="step"><div class="num">01</div><h3>📋 Paste article</h3>
          <p>Copy a headline and the article body into the analyzer.</p></div>
        <div class="step"><div class="num">02</div><h3>🧠 Analyze with {MODEL_NAME}</h3>
          <p>A pretrained BERT-based language model reads the text and scores it.</p></div>
        <div class="step"><div class="num">03</div><h3>✅ Review prediction</h3>
          <p>See the label, the model confidence and the real and fake probabilities.</p></div>
      </div>
    </div>
    """)


def render_model_info() -> None:
    html(f"""
    <div class="section" id="about">
      <h2>About the model</h2>
      <div class="lead">What runs behind TruthLens AI.</div>
      <div class="about">
        <div>
          <p><b style="color:#e8ecf5">{MODEL_NAME}</b> is a BERT-based text classification
          model used for fake and real news classification.</p>
          <ul>
            <li>Model: {MODEL_NAME}</li>
            <li>Type: BERT-based text classification</li>
            <li>Task: fake / real news classification</li>
          </ul>
        </div>
        <div class="bench">
          <div class="big">{BENCH_ACCURACY}</div>
          <div class="cap">accuracy on an external evaluation of {BENCH_SAMPLES} WELFake samples</div>
          <div class="mini">
            <div><b>72.25%</b><span>Precision</span></div>
            <div><b>38.80%</b><span>Recall</span></div>
            <div><b>50.49%</b><span>F1 score</span></div>
          </div>
          <div class="warn">This benchmark measures performance on the evaluation sample and
          does not guarantee the correctness of individual predictions.</div>
        </div>
      </div>
      <div class="disclaimer">{DISCLAIMER}</div>
    </div>
    """)


# ---------------------------------------------------------------------------
# 6. Footer
# ---------------------------------------------------------------------------
def render_footer() -> None:
    html(f"""
    <div class="footer">
      <span>TruthLens AI · Educational AI Project</span>
      <span>Powered by {MODEL_NAME}</span>
    </div>
    """)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css()
    render_header()
    render_hero()
    render_stats()

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

    render_how_it_works()
    render_model_info()
    render_footer()


main()
