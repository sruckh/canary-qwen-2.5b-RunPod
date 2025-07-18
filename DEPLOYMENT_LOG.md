# Canary-Qwen-2.5B RunPod Deployment Log

## Project Overview
Created a dockerized deployment of NVIDIA's Canary-Qwen-2.5B speech recognition model for RunPod with Gradio interface.

## Repository Details
- **GitHub**: https://github.com/sruckh/canary-qwen-2.5b-RunPod
- **Docker Hub**: gemneye/canary-qwen-2.5b-runpod:latest
- **GitHub Actions**: Automated Docker builds on push to main

## Files Created
1. **Dockerfile** - Slim container with runtime dependencies
2. **requirements.txt** - Python dependencies for NeMo and Gradio
3. **app.py** - Full-featured Gradio interface with dual-mode functionality
4. **setup_runpod.sh** - Host setup script for RunPod environment
5. **docker-compose.yml** - Container orchestration
6. **README.md** - Complete documentation
7. **.github/workflows/docker-build.yml** - GitHub Actions workflow

## Key Features Implemented
- **Dual Mode Operation**: ASR (transcription) + LLM (text analysis)
- **Gradio Interface**: Professional UI with audio upload, transcription, and Q&A
- **RunPod Optimized**: Slim container, host-based models and dependencies
- **Environment Variables**: GRADIO_SHARE, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT
- **GPU Support**: CUDA 12.6.3 with cuDNN runtime
- **Automated Builds**: GitHub Actions → Docker Hub

## Issues Resolved

### 1. Initial Base Image Issue
- **Problem**: `nvidia/cuda:12.1-runtime-ubuntu20.04` doesn't exist
- **Solution**: Updated to `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`
- **Commit**: fb5ef0c

### 2. Python Version Compatibility
- **Problem**: Python 3.9 packages not available in Ubuntu 22.04
- **Solution**: Updated to use Python 3.10 (default in Ubuntu 22.04)
- **Changes**: `python3`, `python3-pip`, `python3-dev`
- **Commit**: 79e8452

### 3. SoX System Dependencies
- **Problem**: `sox` Python package couldn't find system SoX command
- **Solution**: Added `sox`, `libsox-dev`, `build-essential` packages
- **Commit**: 2a1add9

### 4. Python Build Dependencies
- **Problem**: `sox` package needed `typing_extensions` during setup
- **Solution**: Pre-install `numpy`, `typing_extensions`, `setuptools`, `wheel`, `cython`
- **Commit**: db48c74

## Current Status
- **Last Build**: In progress (db48c74)
- **Build Status**: Resolving sox package dependencies
- **Next Steps**: Monitor build completion, test deployment

## GitHub Actions Configuration
- **Secrets**: DOCKER_USERNAME, DOCKER_PASSWORD (configured)
- **Registry**: docker.io
- **Image**: gemneye/canary-qwen-2.5b-runpod
- **Triggers**: Push to main, tags, PRs

## Docker Hub Details
- **Repository**: gemneye/canary-qwen-2.5b-runpod
- **Auto-build**: Enabled via GitHub Actions
- **Description**: Auto-updated from README.md

## Deployment Command
```bash
docker run -d --name canary-qwen --gpus all -p 7860:7860 \
  -v /workspace/models:/models:ro \
  -v /workspace/cache:/root/.cache:rw \
  -v /workspace/data:/data:rw \
  -e GRADIO_SHARE=true \
  gemneye/canary-qwen-2.5b-runpod:latest
```

## Host Setup
```bash
# Run on RunPod host first
chmod +x setup_runpod.sh
./setup_runpod.sh
```

## Architecture
- **Container**: Slim runtime with app code only
- **Host**: Models (2.5B params), cache, dependencies via setup script
- **Gradio Interface**: Port 7860, audio upload, transcription, LLM Q&A
- **GPU**: NVIDIA runtime with CUDA 12.6.3 support

## Environment Variables
- `GRADIO_SHARE`: Enable/disable share links
- `GRADIO_SERVER_NAME`: Server bind address (0.0.0.0)
- `GRADIO_SERVER_PORT`: Server port (7860)
- `MODEL_PATH`: Model location (/models/canary-qwen-2.5b)

## Outstanding Items
- [ ] Verify final Docker build completion
- [ ] Test container deployment
- [ ] Validate Gradio interface functionality
- [ ] Test audio transcription and LLM inference
- [ ] Update documentation with final deployment instructions

## Technical Notes
- Using Ubuntu 22.04 base with Python 3.10
- NeMo toolkit installed from git (latest version)
- Audio processing: 16kHz mono, <40s optimal
- Model: English-only, 2.5B parameters, 418 RTFx
- Performance: 5.63% WER average on benchmarks

---
*Last updated: 2025-07-18*
*Status: Docker build in progress, troubleshooting sox dependencies*