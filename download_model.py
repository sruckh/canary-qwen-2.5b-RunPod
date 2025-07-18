
import os
from huggingface_hub import snapshot_download
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_id = "nvidia/canary-qwen-2.5b"
local_dir = "/models/canary-qwen-2.5b"

logger.info(f"Downloading model {model_id} to {local_dir}...")

if not os.path.exists(local_dir):
    os.makedirs(local_dir)

try:
    snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
    logger.info("Model downloaded successfully!")
except Exception as e:
    logger.error(f"Failed to download model: {str(e)}")
    # Exit with a non-zero status code to fail the Docker build
    exit(1)
