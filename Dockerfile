FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install OpenEnv core from source
RUN git clone --depth=1 https://github.com/meta-pytorch/OpenEnv.git /openenv
RUN pip install --no-cache-dir -e "/openenv[core]"

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure server is importable
ENV PYTHONPATH=/app:/openenv/src:/openenv

# Expose port used by HF Spaces
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "2", "--timeout-keep-alive", "75"]
