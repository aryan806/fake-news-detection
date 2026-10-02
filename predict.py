
import os
import urllib.request
import joblib
import re
import nltk
from nltk.corpus import stopwords

MODEL_PATH = "fake_news_model.pkl"
MODEL_URL = "https://github.com/aryan806/fake-news-detection/releases/download/v1.0/fake_news_model.pkl"

# Download the model if it is not already available
if not os.path.exists(MODEL_PATH):
    print("Downloading the trained model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def predict_article(article_text):
    """Labels: 0 = real, 1 = fake."""
    text = (article_text or "").lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(
        word for word in text.split()
        if word not in stop_words
    )

    if not text.strip():
        return {
            "label": "UNKNOWN",
            "headline": "No usable words after cleaning.",
            "p_real": 0.0,
            "p_fake": 0.0,
            "word_count": 0,
        }

    vec = vectorizer.transform([text])
    pred = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    idx = {int(c): i for i, c in enumerate(model.classes_)}

    p_real = float(proba[idx[0]])
    p_fake = float(proba[idx[1]])

    label = "FAKE NEWS" if pred == 1 else "REAL NEWS"
    confidence = p_fake if pred == 1 else p_real
    word_count = len(text.split())

    return {
        "label": label,
        "headline": f"{label} ({confidence:.0%} confidence)",
        "p_real": p_real,
        "p_fake": p_fake,
        "word_count": word_count,
    }


if __name__ == "__main__":
    article = input("Paste a news article here:\n")
    result = predict_article(article)
    print(result.get("headline", result))
