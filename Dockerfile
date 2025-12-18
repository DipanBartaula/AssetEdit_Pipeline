# Dockerfile for Image to 3D Pipeline with Lightning AI
# Minimal setup - 3D generation handled by Lightning AI

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/root/.cache/torch \
    HF_HOME=/root/.cache/huggingface

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models/ ./models/
COPY utils/ ./utils/
COPY pipeline.py .
COPY app.py .

# Create necessary directories
RUN mkdir -p /outputs /assets /logs

# Expose port for Gradio
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import gradio; print('Gradio OK')" || exit 1

# Run Gradio app
CMD ["python", "app.py"]
