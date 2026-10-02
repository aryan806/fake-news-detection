# Fake News Detector

**Live app:** [Streamlit public demo](https://fake-news-detector-xcgxfcivdd3quciygjnpzy.streamlit.app/) (`https://fake-news-detector-xcgxfcivdd3quciygjnpzy.streamlit.app/`) · **Repo:** [github.com/mdraj-hunter/fake-news-detector](https://github.com/mdraj-hunter/fake-news-detector)

Classifies pasted news text as **real** or **fake** using **TF‑IDF + Random Forest** (labels from a paired fake/real article dataset). Built with Python, pandas, scikit-learn, NLTK, and Streamlit—runs locally, in Docker, or on [Streamlit Community Cloud](https://streamlit.io/cloud).

> Educational / portfolio use only. Scores reflect patterns in the training data, not independent fact-checking.

## Live app

**Try it:** [https://fake-news-detector-xcgxfcivdd3quciygjnpzy.streamlit.app/](https://fake-news-detector-xcgxfcivdd3quciygjnpzy.streamlit.app/) (Streamlit Community Cloud).

## Quick start (local)

```bash
git clone https://github.com/mdraj-hunter/fake-news-detector.git
cd fake-news-detector
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
streamlit run app.py
```

Place **`fake_news_model.pkl`** and **`tfidf_vectorizer.pkl`** in the project root (or train your own with `fake_news_detector.ipynb`). Training CSVs are optional locally; they are gitignored by default because of size—see **Data** below.

## Docker

```bash
docker build -t fake-news-detector .
docker run -p 8501:8501 fake-news-detector
```

## Deploy for a public link

Step-by-step (Streamlit Cloud, Docker, server): **[DEPLOY.md](DEPLOY.md)**.

## Data

The original Kaggle-style dataset is described in the repo summary (~23k fake + ~21k real articles). To retrain, download **`Fake.csv`** / **`True.csv`** into this folder (same names as in the notebook). They are listed in `.gitignore` so they are not pushed to GitHub by default.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI |
| `predict.py` | Load pickles, clean text, `predict_proba` |
| `fake_news_detector.ipynb` | Train / evaluate / export pickles |
| `requirements.txt` | Runtime dependencies |
| `.streamlit/config.toml` | Theme & server defaults |

## License

Use and modify for learning and portfolio use; respect the license terms of the dataset you train on.
