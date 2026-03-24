# ── Base image ────────────────────────────────────────────────────
FROM python:3.11

# ── Environment ───────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Working directory ─────────────────────────────────────────────
WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install langchain-anthropic langchain-openai

# ── Application code ──────────────────────────────────────────────
COPY . .

# ── Streamlit config ──────────────────────────────────────────────
# Disable telemetry and set headless mode for container environments
RUN mkdir -p /root/.streamlit && \
    echo '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml && \
    echo '[server]\nheadless = true\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n' \
    > /root/.streamlit/config.toml

# ── Port ──────────────────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ───────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py"]
