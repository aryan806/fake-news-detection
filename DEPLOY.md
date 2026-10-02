# Deploying Fake News Detector

The app is a **Streamlit** UI plus `predict.py`, two pickle files, and NLTK stopwords (downloaded at build or first run).

Example source repo: [mdraj-hunter/fake-news-detector](https://github.com/mdraj-hunter/fake-news-detector).

## What to put on GitHub

Track at least:

- `app.py`, `predict.py`
- `fake_news_model.pkl`, `tfidf_vectorizer.pkl`
- `requirements.txt`, `.streamlit/config.toml`
- `Dockerfile`, `.dockerignore` (if you use Docker)

You can omit `Fake.csv`, `True.csv`, and the notebook if you only deploy the app.

### Large model file (~59 MB)

GitHub warns above **50 MB** per file. Options:

1. **Git LFS** — `git lfs track "*.pkl"` then commit (good for teams).
2. **Host the model elsewhere** — e.g. cloud storage with a signed URL, then download in `predict.py` on startup (more work).
3. **Smaller model** — retrain with fewer trees / smaller vectorizer and replace the pickles.

---

## Option A — Streamlit Community Cloud (easiest public URL)

1. Push the repo to **GitHub** (with the files above).
2. Sign in at [https://share.streamlit.io](https://share.streamlit.io) with GitHub.
3. **New app** → pick repo, branch, and main file **`app.py`**.
4. Set **Python version** to **3.11** in app settings if needed.
5. Deploy. Streamlit installs `requirements.txt` and runs `streamlit run app.py`.

**NLTK:** `predict.py` calls `nltk.download('stopwords', quiet=True)` on import, so the first request may be slightly slower while stopwords download into the container cache.

**Secrets:** This project does not need API keys. Do not commit `.streamlit/secrets.toml`.

---

## Option B — Docker (any VPS, Fly.io, Railway, etc.)

On a machine with Docker **running**:

```bash
docker build -t fake-news-detector .
docker run -p 8501:8501 fake-news-detector
```

Open `http://localhost:8501` (or your host’s IP and port **8501**).

For production, put a **reverse proxy** (Caddy, nginx) in front with HTTPS and optional auth.

---

## Option C — Run on a server without Docker

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Bind `0.0.0.0` only on trusted networks, or use SSH tunneling / a proxy.

---

## Checklist before sharing widely

- [ ] Test with a **long** paste from a real article and confirm probabilities look reasonable.
- [ ] Add a **privacy note** if users paste sensitive text (you are not logging by default, but your host might).
- [ ] Confirm **Streamlit / host** terms allow your use case.

This detector is a **demo**; label quality depends entirely on the training data and must not be sold as ground truth.
