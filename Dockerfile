FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE CITATION.cff ./
COPY asds_omega/ ./asds_omega/
COPY app/ ./app/
COPY tests/ ./tests/
COPY paper/ ./paper/

RUN pip install --no-cache-dir -e ".[dev,benchmarks]"

EXPOSE 8000

CMD ["asds-omega", "serve", "--host", "0.0.0.0", "--port", "8000"]
