# Canary-Qwen-2.5B RunPod Deployment

A dockerized deployment of NVIDIA's Canary-Qwen-2.5B speech recognition model for RunPod, featuring a Gradio interface for audio transcription and LLM-powered text analysis.

## 🎯 Overview

This container provides:
- **ASR Mode**: Speech-to-text transcription with punctuation and capitalization
- **LLM Mode**: Question-answering and analysis of transcribed text
- **Gradio Interface**: User-friendly web interface for audio upload and processing
- **RunPod Optimized**: Slim container with host-based model and dependency management

## 🚀 Quick Start on RunPod

### 1. Deploy Container
```bash
# Use the built container image
docker run -d \
  --name canary-qwen \
  --gpus all \
  -p 7860:7860 \
  -v /workspace/models:/models:ro \
  -v /workspace/cache:/root/.cache:rw \
  -v /workspace/data:/data:rw \
  -e GRADIO_SHARE=true \
  your-registry/canary-qwen-2.5b:latest
```

### 2. Environment Variables
Configure these in your RunPod environment:
```bash
GRADIO_SHARE=true          # Enable/disable Gradio share links
GRADIO_SERVER_NAME=0.0.0.0 # Server bind address
GRADIO_SERVER_PORT=7860    # Server port
MODEL_PATH=/models/canary-qwen-2.5b  # Model location
```

### 3. Setup RunPod Host
First, prepare the RunPod host with models and dependencies:
```bash
# Upload and run the setup script
chmod +x setup_runpod.sh
./setup_runpod.sh
```

This will:
- Install system dependencies
- Create Python virtual environment
- Download the Canary-Qwen-2.5B model
- Set up directory structure
- Configure environment variables

## 📁 Directory Structure

```
/workspace/
├── models/                 # Model files (2.5B parameters)
│   └── canary-qwen-2.5b/
├── cache/                  # HuggingFace cache
├── data/                   # Temporary audio files
├── logs/                   # Application logs
├── venv/                   # Python virtual environment
├── activate.sh             # Environment activation script
├── launch.sh               # Application launch script
└── env_vars.sh             # Environment variables
```

## 🎵 Using the Interface

### Audio Requirements
- **Format**: WAV, FLAC, MP3, or other common audio formats
- **Sample Rate**: Automatically resampled to 16kHz
- **Channels**: Automatically converted to mono
- **Duration**: Optimal performance with <40 seconds
- **Language**: English only

### Interface Features

#### 1. Audio Upload
- Drag and drop audio files
- Record directly using microphone
- Automatic transcription on upload

#### 2. Transcription (ASR Mode)
- High-accuracy speech-to-text
- Automatic punctuation and capitalization
- Real-time processing

#### 3. LLM Analysis
- Ask questions about the transcribed content
- Summarization and analysis
- Content extraction and insights

### Example Prompts
```
"Summarize the main points discussed in the audio."
"What is the speaker's tone and emotion?"
"Extract any important dates, names, or numbers mentioned."
"What questions does the speaker ask?"
"Identify the key topics covered in this audio."
```

## 🛠️ Development and Customization

### Building the Container
```bash
# Clone and build
git clone <your-repo>
cd canary-qwen-2.5b
docker build -t canary-qwen-2.5b .
```

### Local Development (No GPU)
For development without GPU access:
```bash
# Edit docker-compose.yml to comment out GPU sections
# Then run
docker-compose up --build
```

### Customizing the Interface
Edit `app.py` to modify:
- UI layout and styling
- Processing parameters
- Model configuration
- Response formatting

## 📊 Performance Specifications

### Model Performance
- **Parameters**: 2.5 billion
- **RTFx**: 418 (Real-time factor)
- **WER**: 5.63% average on benchmarks
- **Languages**: English only

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support
- **Memory**: 8GB+ GPU memory recommended
- **Storage**: 10GB+ for model and cache
- **Network**: For model download and Gradio share

### Benchmarks
| Dataset | WER |
|---------|-----|
| AMI | 10.18% |
| GigaSpeech | 9.41% |
| LibriSpeech Clean | 1.60% |
| LibriSpeech Other | 3.10% |
| Earnings22 | 10.42% |

## 🔧 Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_SHARE` | `false` | Enable Gradio share links |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Server bind address |
| `GRADIO_SERVER_PORT` | `7860` | Server port |
| `MODEL_PATH` | `/models/canary-qwen-2.5b` | Model directory |
| `HF_HOME` | `/root/.cache` | HuggingFace cache |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection |

### Model Configuration
The model supports two modes:
1. **ASR Mode**: Direct speech-to-text transcription
2. **LLM Mode**: Text analysis using the underlying language model

## 🐛 Troubleshooting

### Common Issues

#### Model Not Loading
```bash
# Check model path
ls -la /models/canary-qwen-2.5b/

# Check GPU availability
nvidia-smi

# Check logs
docker logs canary-qwen
```

#### Audio Processing Errors
- Ensure audio file is valid format
- Check file size (<100MB recommended)
- Verify audio duration (<40 seconds optimal)

#### Memory Issues
- Reduce batch size in model configuration
- Ensure sufficient GPU memory (8GB+)
- Monitor memory usage with `nvidia-smi`

### Performance Optimization
- Use 16kHz mono audio for best performance
- Keep audio segments under 40 seconds
- Enable GPU acceleration
- Use SSD storage for model files

## 📄 License

This project uses the NVIDIA Canary-Qwen-2.5B model under the CC-BY-4.0 license.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review application logs
- Verify RunPod environment configuration
- Ensure proper model installation

---

**Note**: This container is optimized for RunPod deployment with GPU acceleration. Local development without GPU is possible but will have limited functionality.