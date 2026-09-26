# Chess commentary web app: FastAPI + Stockfish + Groq-hosted commentator.
FROM python:3.11-slim

# Debian's stockfish package (a recent release; the exact version depends on the base image).
RUN apt-get update && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

COPY chess_annotator ./chess_annotator
COPY webapp ./webapp

ENV STOCKFISH_PATH=/usr/games/stockfish \
    DB_PATH=/data/webapp_data.sqlite3 \
    TRUST_PROXY_HEADERS=true \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn webapp.server:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
