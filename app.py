import os
import logging
import gradio as gr
import torch
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import librosa
import soundfile as sf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
MODEL_PATH = os.getenv("MODEL_PATH", "/models/canary-qwen-2.5b")

class CanaryQwenInterface:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the Canary-Qwen model"""
        try:
            from nemo.collections.speechlm2.models import SALM
            
            logger.info("Loading Canary-Qwen-2.5B model...")
            # Use the model ID directly - it will automatically use cached version
            self.model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
            
            self.model.eval()
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio file to match model requirements"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Ensure 16kHz sample rate
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Ensure mono
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            
            # Save preprocessed audio
            temp_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_path, audio, 16000)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using ASR mode"""
        try:
            if self.model is None:
                return "❌ Model not loaded. Please check logs."
            
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_path)
            
            # Transcribe using ASR mode
            prompt = [{"role": "user", "content": f"Transcribe the following: {self.model.audio_locator_tag}", "audio": [processed_audio]}]
            
            answer_ids = self.model.generate(
                prompts=[prompt],
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9
            )
            
            # Decode the response
            transcript = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
            
            # Clean up temp file
            if os.path.exists(processed_audio):
                os.remove(processed_audio)
            
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return f"❌ Transcription failed: {str(e)}"
    
    def llm_inference(self, transcript: str, prompt: str) -> str:
        """Perform LLM inference on the transcript"""
        try:
            if self.model is None:
                return "❌ Model not loaded. Please check logs."
            
            if not transcript or not prompt:
                return "❌ Please provide both transcript and prompt."
            
            # Use LLM mode (disable audio adapter)
            with self.model.llm.disable_adapter():
                full_prompt = f"{prompt}\n\nTranscript: {transcript}"
                
                answer_ids = self.model.generate(
                    prompts=[[{"role": "user", "content": full_prompt}]],
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9
                )
                
                response = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
                return response.strip()
                
        except Exception as e:
            logger.error(f"LLM inference failed: {str(e)}")
            return f"❌ LLM inference failed: {str(e)}"

# Initialize the interface
canary_interface = CanaryQwenInterface()

def transcribe_and_analyze(audio_file, llm_prompt: str) -> Tuple[str, str]:
    """Main function to handle both transcription and LLM analysis"""
    if audio_file is None:
        return "❌ Please upload an audio file.", ""
    
    try:
        # Step 1: Transcribe audio
        transcript = canary_interface.transcribe_audio(audio_file)
        
        if transcript.startswith("❌"):
            return transcript, ""
        
        # Step 2: LLM analysis if prompt provided
        llm_response = ""
        if llm_prompt and llm_prompt.strip():
            llm_response = canary_interface.llm_inference(transcript, llm_prompt)
        
        return transcript, llm_response
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return f"❌ Processing failed: {str(e)}", ""

def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #2d5a87;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 2em;
    }
    """
    
    with gr.Blocks(css=css, title="Canary-Qwen Audio Transcription") as demo:
        gr.HTML("""
        <div class="title">🎵 Canary-Qwen Audio Transcription</div>
        <div class="subtitle">Upload audio → Get transcription → Ask questions about the content</div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                llm_prompt = gr.Textbox(
                    label="LLM Prompt (Optional)",
                    placeholder="Ask a question about the transcript (e.g., 'Summarize the main points', 'What is the speaker's opinion about...?')",
                    lines=3
                )
                
                submit_btn = gr.Button("🎯 Transcribe & Analyze", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                transcript_output = gr.Textbox(
                    label="📝 Transcript",
                    lines=8,
                    max_lines=15,
                    show_copy_button=True
                )
                
                llm_output = gr.Textbox(
                    label="🤖 LLM Analysis",
                    lines=8,
                    max_lines=15,
                    show_copy_button=True
                )
        
        # Examples section
        gr.Examples(
            examples=[
                [None, "Summarize the main points discussed in the audio."],
                [None, "What is the speaker's tone and emotion?"],
                [None, "Extract any important dates, names, or numbers mentioned."],
                [None, "What questions does the speaker ask?"],
                [None, "Identify the key topics covered in this audio."]
            ],
            inputs=[audio_input, llm_prompt],
            label="Example Prompts"
        )
        
        # Event handlers
        submit_btn.click(
            fn=transcribe_and_analyze,
            inputs=[audio_input, llm_prompt],
            outputs=[transcript_output, llm_output]
        )
        
        # Auto-submit on audio upload
        audio_input.change(
            fn=transcribe_and_analyze,
            inputs=[audio_input, llm_prompt],
            outputs=[transcript_output, llm_output]
        )
        
        # Model info
        gr.HTML("""
        <div style="margin-top: 2em; padding: 1em; background-color: #f8f9fa; border-radius: 8px;">
            <h3>ℹ️ Model Information</h3>
            <p><strong>Model:</strong> NVIDIA Canary-Qwen-2.5B</p>
            <p><strong>Capabilities:</strong> English speech recognition with punctuation and capitalization</p>
            <p><strong>Audio Requirements:</strong> 16kHz mono audio, max 40 seconds duration</p>
            <p><strong>Modes:</strong> ASR (transcription) + LLM (text analysis)</p>
        </div>
        """)
    
    return demo

def main():
    """Main function to run the application"""
    logger.info("Starting Canary-Qwen Gradio Interface...")
    
    # Load model
    if not canary_interface.load_model():
        logger.error("Failed to load model. The interface will still start but won't work properly.")
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    logger.info(f"Launching interface on {GRADIO_SERVER_NAME}:{GRADIO_SERVER_PORT}")
    logger.info(f"Share link: {GRADIO_SHARE}")
    
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        show_error=True,
        show_api=False
    )

if __name__ == "__main__":
    main()