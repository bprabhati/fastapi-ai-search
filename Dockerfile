FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY . /app

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry lock --no-interaction && \
    poetry install --no-interaction --no-ansi


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
