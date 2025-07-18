FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update pip to latest version
RUN pip3 install --upgrade pip

# Create app directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directories for models and data (will be mounted from host)
RUN mkdir -p /models /data

# Set environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SHARE=false

# Expose port
EXPOSE 7860

# Run the application
CMD ["python3", "app.py"]