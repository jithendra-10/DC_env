FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY dataclean/ ./dataclean/
COPY baseline/  ./baseline/
COPY server.py  .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Gradio UI enabled by default — judges can demo at /web
ENV ENABLE_WEB_INTERFACE=true
ENV PORT=7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
