import pyaudio
import wave
import io
import requests
import time
from ollama import Client
import logging

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
RATE = 44100 # Replace with your microphone's sample rate
RECORD_SECONDS = 3
WHISPER_URL = "http://10.69.10.253:8000/v1/audio/transcriptions"  # Replace with your server IP and port
OLLAMA_BASE_URL = "http://10.69.10.253:11434"  # Replace with your Ollama server IP and port
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
    # Instructions for Ollama to generate a prompt suitable for ComfyUI
    instruction = f"Based on the following transcribed text, extract the context, generate a detailed and descriptive prompt for image generation using ComfyUI. Give me only the prompt and nothßing else. \n\nPrompt: {prompt}"
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

if __name__ == "__main__":
    while True:
        audio_data = record_audio()
        transcription_response = send_audio_to_server(audio_data)
        
        if transcription_response and "text" in transcription_response:
            transcribed_text = transcription_response["text"]
            logger.info(f"Transcribed Text: {transcribed_text}")
            
            # Enhance the prompt using Ollama
            enhanced_prompt = enhance_prompt_with_ollama(transcribed_text)
            logger.info(f"Enhanced Prompt for ComfyUI: {enhanced_prompt}")
        else:
            logger.error("No text received from the transcription service.")
        
        time.sleep(WAIT_TIME)