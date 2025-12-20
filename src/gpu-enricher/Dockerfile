# Pin to specific digest for reproducible builds and supply chain security
# To update: docker pull python:3.12-slim && docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim
FROM python:3.12-slim@sha256:a866731a6b71c4a194a845d86e06568725e430ed21821d0c52e102c3d34c5622

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application modules
COPY main.py auth.py anomaly.py billing.py notifications.py reports.py rightsizing.py config.py ./

# Copy configuration if present
COPY config.yaml* ./

# Switch to non-root user
USER appuser

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "main.py"]