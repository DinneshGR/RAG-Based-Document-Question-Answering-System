FROM python:3.11-slim

# System dependencies required by PyMuPDF and sentence-transformers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend ./frontend
COPY evaluation ./evaluation
COPY .env.example ./.env.example

RUN mkdir -p data/uploads data/chroma

EXPOSE 8000 8501

# Default command runs the FastAPI backend; docker-compose overrides
# the frontend service's command to run Streamlit instead.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
