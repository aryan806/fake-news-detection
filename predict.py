import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_article(article_text):
    """Labels match training: 0 = real, 1 = fake (see fake_news_detector.ipynb)."""
    text = (article_text or "").lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])

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
    conf = p_fake if pred == 1 else p_real
    print(f"\nPrediction: {label} ({conf:.1%} confidence)")

    wc = len(text.split())
    return {
        "label": label,
        "headline": f"{label} ({conf:.0%} confidence)",
        "p_real": p_real,
        "p_fake": p_fake,
        "word_count": wc,
    }


if __name__ == "__main__":
    article = input("Paste a news article here:\n")
    out = predict_article(article)
    print(out.get("headline", out))