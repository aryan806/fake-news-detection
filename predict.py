import streamlit as st
from transformers import pipeline

MODEL_NAME = "himel05/fake-news-roberta"


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
        max_length=512
    )


classifier = load_model()


def predict_article(article_text):
    if not article_text or not article_text.strip():
        return {
            "label": "UNKNOWN",
            "headline": "Please enter a news article.",
            "p_real": 0.0,
            "p_fake": 0.0,
            "word_count": 0,
        }

    text = article_text.strip()

    results = classifier(text, top_k=2)

    # Convert the model output into a simple label -> score dictionary
    scores = {
        result["label"]: float(result["score"])
        for result in results
    }

    # This model uses:
    # LABEL_0 = REAL
    # LABEL_1 = FAKE
    p_real = scores.get("LABEL_0", 0.0)
    p_fake = scores.get("LABEL_1", 0.0)

    if p_fake >= p_real:
        final_label = "FAKE NEWS"
        confidence = p_fake
    else:
        final_label = "REAL NEWS"
        confidence = p_real

    return {
        "label": final_label,
        "headline": f"{final_label} ({confidence:.0%} confidence)",
        "p_real": p_real,
        "p_fake": p_fake,
        "word_count": len(text.split()),
    }
