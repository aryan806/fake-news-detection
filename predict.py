import streamlit as st
from google import genai


MODEL_NAME = "gemini-3.8-flash"


@st.cache_resource
def get_client():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


client = get_client()


def predict_article(article_text):
    """
    Analyze a news article using Gemini + Google Search.

    The result is an evidence-based assessment:
    SUPPORTED
    CONTRADICTED
    MIXED
    INSUFFICIENT EVIDENCE
    """

    if not article_text or not article_text.strip():
        return {
            "label": "UNKNOWN",
            "headline": "Please enter a news article.",
            "p_real": 0.0,
            "p_fake": 0.0,
            "word_count": 0,
            "analysis": "",
            "sources": [],
        }

    text = article_text.strip()

    prompt = f"""
You are a careful news verification assistant.

Investigate the following news article using Google Search.

ARTICLE:
{text}

Your job is NOT to judge the article based on writing style, wording,
publisher style, or words such as "Reuters".

Instead:

1. Identify the main factual claims in the article.
2. Search the web for evidence about those claims.
3. Prefer:
   - official government sources
   - official organizations
   - scientific institutions
   - primary sources
   - reputable news organizations
4. Compare the article's claims against the evidence.
5. Do not call something false merely because you could not find evidence.
6. Mention conflicting evidence when it exists.
7. Do not claim certainty when the evidence is incomplete.

Choose exactly ONE assessment:

SUPPORTED
CONTRADICTED
MIXED
INSUFFICIENT EVIDENCE

Return your response in this format:

ASSESSMENT: <one label>

SUMMARY:
<2-4 concise sentences explaining the result>

IMPORTANT CLAIMS:
- <claim 1>
- <claim 2>
- <claim 3>

EVIDENCE:
<explain what the web evidence shows>

LIMITATIONS:
<explain uncertainty, missing evidence, or conflicting sources>
"""

    try:
        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
            tools=[
                {
                    "type": "google_search"
                }
            ],
        )

        analysis = interaction.output_text or "No analysis was returned."

        # Detect Gemini's assessment.
        assessment = "INSUFFICIENT EVIDENCE"

        for possible_label in [
            "SUPPORTED",
            "CONTRADICTED",
            "MIXED",
            "INSUFFICIENT EVIDENCE",
        ]:
            if f"ASSESSMENT: {possible_label}" in analysis:
                assessment = possible_label
                break

        # Collect citations returned by Google Search grounding.
        sources = []
        seen_urls = set()

        for step in interaction.steps:

            if step.type != "model_output":
                continue

            for content_block in step.content:

                if not hasattr(content_block, "annotations"):
                    continue

                if not content_block.annotations:
                    continue

                for annotation in content_block.annotations:

                    if getattr(annotation, "type", None) != "url_citation":
                        continue

                    url = getattr(annotation, "uri", None)
                    title = getattr(annotation, "title", None)

                    if url and url not in seen_urls:
                        seen_urls.add(url)

                        sources.append(
                            {
                                "title": title or url,
                                "url": url,
                            }
                        )

        return {
            "label": assessment,
            "headline": f"Assessment: {assessment}",

            # These are kept only so the current app.py
            # continues to run.
            # They are NOT Gemini probabilities.
            "p_real": 0.0,
            "p_fake": 0.0,

            "word_count": len(text.split()),
            "analysis": analysis,
            "sources": sources,
        }

    except Exception as e:

        st.error(
            f"Gemini error: {type(e).__name__}: {e}"
        )

        return {
            "label": "ERROR",
            "headline": "Gemini analysis failed.",
            "p_real": 0.0,
            "p_fake": 0.0,
            "word_count": len(text.split()),
            "analysis": f"{type(e).__name__}: {e}",
            "sources": [],
        }
