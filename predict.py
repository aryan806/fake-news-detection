import streamlit as st
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


client = get_client()


def predict_article(article_text):
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

Analyze the following news article using Google Search to check its important factual claims.

ARTICLE:
{text}

Instructions:
1. Identify the main factual claims.
2. Search the web for reliable evidence about those claims.
3. Prefer authoritative sources, official organizations, reputable news organizations,
   scientific institutions, and primary sources.
4. Compare the claims with the evidence you find.
5. Do not decide that an article is false merely because evidence is not found.
6. Do not use writing style or words such as "Reuters" as proof.
7. Give exactly one assessment:
   SUPPORTED
   CONTRADICTED
   MIXED
   INSUFFICIENT EVIDENCE
8. Explain the assessment briefly.
9. Mention important uncertainty or conflicting evidence.

Return:

ASSESSMENT: <one label>

SUMMARY:
<2-4 sentences>

IMPORTANT CLAIMS:
- <claim 1>
- <claim 2>
- <claim 3>

EVIDENCE:
<brief explanation>

LIMITATIONS:
<brief explanation>
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        google_search=types.GoogleSearch()
                    )
                ]
            ),
        )

        analysis = response.text or "No analysis was returned."

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

        if assessment == "SUPPORTED":
            label = "SUPPORTED"
        elif assessment == "CONTRADICTED":
            label = "CONTRADICTED"
        elif assessment == "MIXED":
            label = "MIXED"
        else:
            label = "INSUFFICIENT EVIDENCE"

        sources = []

        if response.candidates:
            metadata = response.candidates[0].grounding_metadata

            if metadata and metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if chunk.web:
                        sources.append(
                            {
                                "title": chunk.web.title,
                                "url": chunk.web.uri,
                            }
                        )

        unique_sources = []
        seen_urls = set()

        for source in sources:
            url = source["url"]

            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)

        return {
            "label": label,
            "headline": f"Assessment: {label}",
            "p_real": 0.0,
            "p_fake": 0.0,
            "word_count": len(text.split()),
            "analysis": analysis,
            "sources": unique_sources,
        }

    except Exception as e:
        st.error(f"Gemini error: {type(e).__name__}: {e}")

        return {
            "label": "ERROR",
            "headline": "Gemini analysis failed.",
            "p_real": 0.0,
            "p_fake": 0.0,
            "word_count": len(text.split()),
            "analysis": f"{type(e).__name__}: {e}",
            "sources": [],
        }
