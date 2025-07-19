FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies, including NeMo toolkit and Gradio
RUN python3 -m pip install --no-cache-dir \
    "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git" \
    gradio \
    librosa \
    soundfile

# Create a non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Set HuggingFace cache directory and pre-download the model as the app user
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/canary-qwen-2.5b', cache_dir=f'{HF_HOME}')"

# Copy application code
COPY --chown=appuser:appuser app.py .

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Expose port
EXPOSE 7860

# Run the application
CMD ["python3", "app.py"]
