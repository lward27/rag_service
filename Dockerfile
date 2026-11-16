FROM python:3.11-slim
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/lward27/rag_service" \
      org.opencontainers.image.revision="$VCS_REF"
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN opentelemetry-bootstrap -a install

COPY app/ ./app/

EXPOSE 8000

ENTRYPOINT ["opentelemetry-instrument", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
