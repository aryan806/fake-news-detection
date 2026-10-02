import streamlit as st
from transformers import pipeline

MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"


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

    result = classifier(text)[0]

    label = result["label"]
    score = float(result["score"])

    if label == "LABEL_1":
        final_label = "REAL NEWS"
        p_real = score
        p_fake = 1.0 - score
    else:
        final_label = "FAKE NEWS"
        p_fake = score
        p_real = 1.0 - score

    return {
        "label": final_label,
        "headline": f"{final_label} ({score:.0%} confidence)",
        "p_real": p_real,
        "p_fake": p_fake,
        "word_count": len(text.split()),
    }
