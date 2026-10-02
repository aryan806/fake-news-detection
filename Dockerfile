FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import nltk; nltk.download('stopwords')"

COPY app.py predict.py fake_news_model.pkl tfidf_vectorizer.pkl ./
COPY .streamlit/config.toml .streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
