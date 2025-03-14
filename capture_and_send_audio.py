import pyaudio
import wave
import io
import requests
import time
from ollama import Client
import logging
import random
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress PyAudio debug logs
pyaudio_log = logging.getLogger('pyaudio')
pyaudio_log.setLevel(logging.ERROR)

# Configuration
CHUNK = 8192 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 # Replace with your microphone's sample rate if different
RECORD_SECONDS = 3
WHISPER_URL = "http://10.69.10.253:8000/v1/audio/transcriptions"  # Replace with your server IP and port
OLLAMA_BASE_URL = "http://10.69.10.253:11434"  # Replace with your Ollama server IP and port
STABLE_DIFFUSION_URL = "http://10.69.10.253:7860/sdapi/v1/txt2img"  # Replace with your Stable-Diffusion-WebUI-Forge server IP and port
WAIT_TIME = 30  # Time to wait before recording again

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    logger.info("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    logger.info("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # Create a WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    return wav_buffer.getvalue()

def send_audio_to_server(audio_data):
    # Set the filename and MIME type
    filename = 'audio_file.wav'
    mime_type = 'audio/wav'
    # Create a tuple for the files parameter
    files = {'file': (filename, audio_data, mime_type)}
    # form data
    form_data = {
        "model": "Systran/faster-whisper-large-v2"
    }
    response = requests.post(WHISPER_URL, data=form_data, files=files)
    if response.status_code == 200:
        logger.info("Transcription successful.")
        return response.json()
    else:
        logger.error(f"Error during transcription: {response.text}")
        return None

def enhance_prompt_with_ollama(prompt):
    client = Client(host=OLLAMA_BASE_URL)  # Initialize the Ollama client
    unique_identifier = random.randint(1, 1000000)  # Generate a random identifier
    # Instructions for Ollama to generate a prompt suitable for Stable-Diffusion-WebUI-Forge
    instruction = (
        f"Based on the following transcribed text, understand the context of the conversation "
        f"and generate a detailed and descriptive prompt for image generation using Stable-Diffusion-WebUI-Forge. "
        f"The prompt should be formatted as follows:\n"
        f"<Positive Prompt>\n\n"
        f"Do not include any other text.\n"
        f"If there is no transcribed text provided, then pick a random topic of your choice and generate the prompt.\n\n"
        f"Transcribed Text:\n\n{prompt}\n\n"
        f"Unique Identifier: {unique_identifier}"
    )
    try:
        response = client.generate(
            model="gemma3:27b",
            prompt=instruction
        )
        if hasattr(response, 'response'):
            logger.info("Ollama request successful.")
            return response.response.strip()
        else:
            logger.error(f"Error during Ollama request: {response}")
            return prompt
    except Exception as e:
        logger.exception("Exception during Ollama request")
        return prompt

def generate_image_with_stable_diffusion(prompt):
    payload = {
        "prompt": prompt.strip(),
        "batch_size": 1,
        "steps": 50,  # Increased steps for better quality
        "width": 1920,
        "height": 1080,
        "distilled_cfg_scale": 3.5,  # Adjusted distilled CFG scale
        "cfg_scale": 7.5,  # Adjusted CFG scale
        "sampler_name": "Euler",  # Changed sampler
        "seed": -1,  # Random seed for variation
        "denoising_strength": 0.75,  # Denoising strength can affect quality
        "scheduler": "Simple"
    }
    
    response = requests.post(STABLE_DIFFUSION_URL, json=payload)
    if response.status_code == 200:
        logger.info("Image generation successful.")
        data = response.json()
        # Assuming the image is returned in base64 format
        if "images" in data and len(data["images"]) > 0:
            return data["images"][0]
        else:
            logger.error(f"No images received from Stable-Diffusion-WebUI-Forge: {data}")
            return None
    else:
        logger.error(f"Error during image generation: {response.text}")
        return None

if __name__ == "__main__":
    while True:
        audio_data = record_audio()
        transcription_response = send_audio_to_server(audio_data)
        
        if transcription_response and "text" in transcription_response:
            transcribed_text = transcription_response["text"]
            logger.info(f"Transcribed Text: {transcribed_text}")
            
            # Enhance the prompt using Ollama
            enhanced_prompt = enhance_prompt_with_ollama(transcribed_text)
            logger.info(f"Enhanced Prompt for Stable-Diffusion-WebUI-Forge: {enhanced_prompt}")
        else:
            logger.info("No text received from the transcription service. Generating a random topic.")
            # Generate a random prompt without transcribed text
            enhanced_prompt = enhance_prompt_with_ollama("")
            logger.info(f"Random Topic Enhanced Prompt for Stable-Diffusion-WebUI-Forge: {enhanced_prompt}")
        
        if enhanced_prompt:
            # Generate image using the enhanced prompt
            image_data = generate_image_with_stable_diffusion(enhanced_prompt)
            if image_data:
                # Save or display the generated image
                image_path = f"generated_image_{int(time.time())}.png"
                with open(image_path, "wb") as img_file:
                    img_file.write(base64.b64decode(image_data))
                logger.info(f"Image saved to {image_path}")
            else:
                logger.error("Failed to generate image.")
        
        time.sleep(WAIT_TIME)