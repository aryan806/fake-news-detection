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
1. Identify the main factual claims in the article.
2. Search the web for reliable evidence about those claims.
3. Prefer authoritative sources, official organizations, reputable news organizations,
   scientific institutions, and primary sources when available.
4. Compare the article's claims with the evidence you find.
5. Do NOT decide that an article is false merely because you cannot find evidence.
6. Do NOT rely on the writing style, wording, or presence of words such as Reuters as proof.
7. Give one overall assessment using exactly one of:
   - SUPPORTED
   - CONTRADICTED
   - MIXED
   - INSUFFICIENT EVIDENCE
8. Explain the assessment briefly.
9. Mention important uncertainty or conflicting evidence.
10. Do not claim certainty when the evidence is incomplete.

Return your answer in this format:

ASSESSMENT: <one of the four labels>

SUMMARY:
<2-4 sentence explanation>

IMPORTANT CLAIMS:
- <claim 1>
- <claim 2>
- <claim 3>

EVIDENCE:
<brief explanation of what reliable sources indicate>

LIMITATIONS:
<brief explanation of uncertainty or limitations>
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )

        analysis = response.text or "No analysis was returned."

        # Extract the assessment from Gemini's response.
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

        # Keep these fields for compatibility with the current app.
        # They are categorical placeholders, NOT probabilities.
        if assessment == "SUPPORTED":
            p_real = 1.0
            p_fake = 0.0
            label = "SUPPORTED"
        elif assessment == "CONTRADICTED":
            p_real = 0.0
            p_fake = 1.0
            label = "CONTRADICTED"
        elif assessment == "MIXED":
            p_real = 0.5
            p_fake = 0.5
            label = "MIXED"
        else:
            p_real = 0.5
            p_fake = 0.5
            label = "INSUFFICIENT EVIDENCE"

        # Extract grounded web sources from Gemini's response.
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

        # Remove duplicate sources.
        unique_sources = []
        seen_urls = set()

        for source in sources:
            if source["url"] not in seen_urls:
                seen_urls.add(source["url"])
                unique_sources.append(source)

        return {
            "label": label,
            "headline": f"Assessment: {label}",
            "p_real": p_real,
            "p_fake": p_fake,
            "word_count": len(text.split()),
            "analysis": analysis,
            "sources": unique_sources,
        }

    except Exception as e:
        return {
            "label": "ERROR",
            "headline": "Unable to analyze the article.",
            "p_real": 0.0,
            "p_fake": 0.0,
            "word_count": len(text.split()),
            "analysis": f"Error: {str(e)}",
            "sources": [],
        }
