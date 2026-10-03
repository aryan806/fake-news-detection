import streamlit as st
from transformers import pipeline


MODEL_NAME = "nallarahul/NewsGaurd"


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
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

    try:
        result = classifier(
            text,
            truncation=True,
            max_length=512
        )[0]

        # NewsGaurd mapping:
        # LABEL_0 = FAKE
        # LABEL_1 = REAL
        raw_label = result["label"]
        score = float(result["score"])

        if raw_label in ["LABEL_0", "0", "FAKE"]:
            label = "FAKE NEWS"
            p_fake = score
            p_real = 1.0 - score
        else:
            label = "REAL NEWS"
            p_real = score
            p_fake = 1.0 - score

        confidence = max(p_real, p_fake)

        return {
            "label": label,
            "headline": f"{label} ({confidence:.0%} confidence)",
            "p_real": p_real,
            "p_fake": p_fake,
            "word_count": len(text.split()),
        }

    except Exception as e:
        st.error(
            f"Model error: {type(e).__name__}: {e}"
        )

        return {
            "label": "ERROR",
            "headline": "Unable to analyze the article.",
            "p_real": 0.0,
            "p_fake": 0.0,
            "word_count": len(text.split()),
        }
