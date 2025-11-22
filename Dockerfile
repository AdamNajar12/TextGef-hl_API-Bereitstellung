# Use lightweight Python 3.11
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY app.py .

# ✅ Pre-download HuggingFace model to cache inside container
ENV TRANSFORMERS_CACHE=/tmp/.cache
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

# Expose Flask port
EXPOSE 8080

# Run the app
CMD ["gunicorn", "app:application", "-b", "0.0.0.0:8080"]
