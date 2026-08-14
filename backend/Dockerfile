# Backend container for the Personal Finance Analyzer.
# Note: this container does NOT include Ollama or the model itself --
# Ollama runs as its own service (see docker-compose.yml) and this
# container talks to it over the Docker network.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching -- only re-installs
# when requirements.txt actually changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# SQLite file lives here inside the container; mounted as a volume
# in docker-compose.yml so data survives container restarts/rebuilds.
VOLUME ["/app/data"]

EXPOSE 7864

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7864"]
