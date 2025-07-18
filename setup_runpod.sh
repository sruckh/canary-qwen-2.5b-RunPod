#!/bin/bash

# RunPod Setup Script for Canary-Qwen-2.5B
# This script should be run on the RunPod host to prepare the environment

set -e

echo "🚀 Setting up RunPod environment for Canary-Qwen-2.5B..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODEL_DIR="/workspace/models"
CACHE_DIR="/workspace/cache"
VENV_DIR="/workspace/venv"
LOGS_DIR="/workspace/logs"

# Create directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p $MODEL_DIR $CACHE_DIR $VENV_DIR $LOGS_DIR

# Check for GPU
echo -e "${YELLOW}🔍 Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo -e "${GREEN}✅ GPU detected${NC}"
else
    echo -e "${RED}❌ No GPU detected - this may cause issues${NC}"
fi

# Update system packages
echo -e "${YELLOW}📦 Updating system packages...${NC}"
apt-get update
apt-get install -y python3.9 python3.9-pip python3.9-venv python3.9-dev \
    ffmpeg libsndfile1 wget curl git build-essential

# Create virtual environment
echo -e "${YELLOW}🐍 Creating Python virtual environment...${NC}"
python3.9 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Upgrade pip and install basic tools
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo -e "${YELLOW}🔥 Installing PyTorch with CUDA support...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install NeMo toolkit from source (required for latest features)
echo -e "${YELLOW}📚 Installing NeMo toolkit...${NC}"
pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"

# Install other dependencies
echo -e "${YELLOW}📋 Installing other dependencies...${NC}"
pip install gradio>=4.0.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.1 \
    audiofile>=1.3.0 \
    numpy>=1.24.0 \
    transformers>=4.35.0 \
    huggingface-hub>=0.17.0 \
    accelerate>=0.24.0 \
    requests>=2.31.0 \
    tqdm>=4.66.0 \
    pydantic>=2.0.0

# Download the model
echo -e "${YELLOW}📥 Downloading Canary-Qwen-2.5B model...${NC}"
python3 -c "
import os
os.environ['HF_HOME'] = '$CACHE_DIR'
os.environ['TRANSFORMERS_CACHE'] = '$CACHE_DIR'
os.environ['HF_DATASETS_CACHE'] = '$CACHE_DIR'

try:
    from nemo.collections.speechlm2.models import SALM
    print('Loading model...')
    model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    model.save_to('$MODEL_DIR/canary-qwen-2.5b')
    print('Model downloaded and saved successfully!')
except Exception as e:
    print(f'Error downloading model: {e}')
    exit(1)
"

# Set up environment variables
echo -e "${YELLOW}🔧 Setting up environment variables...${NC}"
cat > /workspace/env_vars.sh << 'EOF'
export MODEL_PATH="/workspace/models/canary-qwen-2.5b"
export HF_HOME="/workspace/cache"
export TRANSFORMERS_CACHE="/workspace/cache"
export HF_DATASETS_CACHE="/workspace/cache"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/workspace:$PYTHONPATH"
export PATH="/workspace/venv/bin:$PATH"
EOF

# Create activation script
echo -e "${YELLOW}📝 Creating activation script...${NC}"
cat > /workspace/activate.sh << 'EOF'
#!/bin/bash
source /workspace/venv/bin/activate
source /workspace/env_vars.sh
echo "🎯 Canary-Qwen environment activated!"
echo "Model path: $MODEL_PATH"
echo "Python: $(which python)"
echo "GPU: $(nvidia-smi -L | head -1)"
EOF

chmod +x /workspace/activate.sh

# Create launch script
echo -e "${YELLOW}🚀 Creating launch script...${NC}"
cat > /workspace/launch.sh << 'EOF'
#!/bin/bash
cd /workspace
source /workspace/activate.sh

echo "🎵 Starting Canary-Qwen Gradio Interface..."
python app.py 2>&1 | tee /workspace/logs/app.log
EOF

chmod +x /workspace/launch.sh

# Test the installation
echo -e "${YELLOW}🧪 Testing installation...${NC}"
source $VENV_DIR/bin/activate
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')

try:
    from nemo.collections.speechlm2.models import SALM
    print('✅ NeMo import successful')
except Exception as e:
    print(f'❌ NeMo import failed: {e}')

try:
    import gradio as gr
    print('✅ Gradio import successful')
except Exception as e:
    print(f'❌ Gradio import failed: {e}')
"

# Create systemd service for auto-start (optional)
echo -e "${YELLOW}⚙️ Creating systemd service...${NC}"
cat > /etc/systemd/system/canary-qwen.service << EOF
[Unit]
Description=Canary-Qwen Gradio Interface
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
ExecStart=/workspace/launch.sh
Restart=always
RestartSec=10
Environment=PATH=/workspace/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=PYTHONPATH=/workspace

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable canary-qwen

echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "${GREEN}📋 Summary:${NC}"
echo "  - Model installed at: $MODEL_DIR/canary-qwen-2.5b"
echo "  - Virtual environment: $VENV_DIR"
echo "  - Cache directory: $CACHE_DIR"
echo "  - Logs directory: $LOGS_DIR"
echo ""
echo -e "${GREEN}🚀 To run the application:${NC}"
echo "  1. Activate environment: source /workspace/activate.sh"
echo "  2. Launch application: /workspace/launch.sh"
echo "  3. Or start service: systemctl start canary-qwen"
echo ""
echo -e "${GREEN}🌐 The application will be available at: http://localhost:7860${NC}"
echo -e "${YELLOW}💡 Remember to configure your environment variables in RunPod!${NC}"