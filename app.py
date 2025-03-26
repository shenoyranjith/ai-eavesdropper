import pyaudio
import wave
import io
import requests
import time
import threading
from ollama import Client
import logging
import base64
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
WHISPER_URL = "http://10.69.10.253:8000/v1/audio/transcriptions"
OLLAMA_BASE_URL = "http://10.69.10.253:11434"
STABLE_DIFFUSION_URL = "http://10.69.10.253:7860/sdapi/v1/txt2img"
WAIT_TIME = 1800  # 30 minutes between recordings

class AppGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Display")
        self.attributes('-fullscreen', True)
        self.configure(bg='black')
        logger.debug("Initializing GUI components")
        
        # Main image display canvas
        self.image_canvas = tk.Canvas(self, bg='black', highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Overlay setup
        self.overlay_visible = False
        self.overlay_window = tk.Toplevel(self)
        self.overlay_window.overrideredirect(True)
        self.overlay_window.config(bg='black')
        self.overlay_window.attributes('-topmost', True)
        self.overlay_window.withdraw()
        logger.debug("Overlay window initialized")
        
        # Position and size overlay
        self.overlay_width = int(self.winfo_screenwidth() * 0.95)
        self.overlay_height = int(self.winfo_screenheight() * 0.8)
        self.overlay_window.geometry(
            f"{self.overlay_width}x{self.overlay_height}+"
            f"{int(self.winfo_screenwidth()*0.02)}+{int(self.winfo_screenheight()*0.02)}"
        )
        
        # Text display components
        self.text_frame = tk.Frame(self.overlay_window, bg='black')
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'Vertical.TScrollbar',
            background='darkgray',
            troughcolor='black',
            bordercolor='gray',
            arrowsize=15,
            arrowcolor='white'
        )
        
        # Text widget setup
        self.text_widget = tk.Text(
            self.text_frame,
            wrap=tk.WORD,
            bg='black',
            fg='white',
            font=('Arial', 12),
            highlightthickness=0,
            borderwidth=0,
            spacing1=4,
            spacing2=4,
            state='disabled',
            exportselection=False
        )
        logger.debug("Text widget configured with anti-selection bindings")
        
        # Event bindings for text widget
        self.text_widget.bind('<Button-1>', lambda e: 'break')
        self.text_widget.bind('<B1-Motion>', lambda e: 'break')
        self.text_widget.bind('<Double-1>', lambda e: 'break')
        self.text_widget.bind('<Control-c>', lambda e: 'break')
        self.text_widget.bind('<Button-3>', lambda e: 'break')
        
        # Scrollbar implementation
        self.scrollbar = ttk.Scrollbar(
            self.text_frame,
            orient=tk.VERTICAL,
            command=self.text_widget.yview,
            style='Vertical.TScrollbar'
        )
        self.text_widget['yscrollcommand'] = self.scrollbar.set
        
        # Grid layout
        self.text_widget.grid(row=0, column=0, sticky='nsew')
        self.scrollbar.grid(row=0, column=1, sticky='ns')
        self.text_frame.grid_rowconfigure(0, weight=1)
        self.text_frame.grid_columnconfigure(0, weight=1)
        
        # Toggle button
        self.toggle_button = tk.Button(
            self,
            text="Toggle Overlay",
            bg='white',
            fg='black',
            command=self.toggle_overlay,
            bd=0,
            highlightthickness=0,
            font=('Arial', 12)
        )
        self.toggle_button.place(
            relx=0.9,
            rely=0.9,
            anchor=tk.CENTER
        )
        logger.debug("GUI initialization complete")
    
    def toggle_overlay(self):
        logger.info(f"Toggle overlay requested (current state: {self.overlay_visible})")
        self.overlay_visible = not self.overlay_visible
        if self.overlay_visible:
            self.overlay_window.deiconify()
            logger.debug("Overlay activated")
        else:
            self.overlay_window.withdraw()
            logger.debug("Overlay deactivated")
    
    def set_texts(self, transcription, context, prompt):
        logger.debug("Updating UI text components")
        text_content = (
            f"Transcription:\n{transcription}\n\n"
            f"Context:\n{context}\n\n"
            f"Prompt:\n{prompt}"
        )
        self.text_widget.config(state='normal')
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, text_content)
        self.text_widget.yview_moveto(0)
        self.text_widget.config(state='disabled')
        logger.info("Text components updated successfully")
    
    def update_image(self, image_data):
        logger.debug("Attempting to update image")
        if not image_data:
            logger.error("No image data received")
            return
        try:
            img_data = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_data))
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            img = img.resize((screen_width, screen_height), Image.LANCZOS)
            self.image_canvas.delete("all")
            photo = ImageTk.PhotoImage(img)
            self.image_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.image = photo  # Keep reference
            logger.info("Image updated successfully")
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}", exc_info=True)

class AudioProcessor:
    def __init__(self, app):
        self.app = app
        self.is_recording = False
        self.lock = threading.Lock()
        logger.debug("Audio processor initialized")
    
    def start(self):
        logger.info("Starting audio processing thread")
        threading.Thread(target=self.run, daemon=True).start()
    
    def run(self):
        logger.info("Audio processing loop started")
        while True:
            try:
                logger.debug("Starting conversation processing")
                self.process_conversation()
                logger.debug("Processing complete, waiting %d seconds", WAIT_TIME)
                time.sleep(WAIT_TIME)
            except Exception as e:
                logger.error("Critical error in processing loop: %s", str(e), exc_info=True)
                time.sleep(60)
    
    def record_audio(self):
        logger.info("Starting audio recording")
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        frames = []
        try:
            for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            logger.debug("Recording completed successfully")
        except Exception as e:
            logger.error("Recording error: %s", str(e))
            return None
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        logger.debug("Audio data prepared for transcription")
        return wav_buffer.getvalue()
    
    def transcribe_audio(self, audio_data):
        logger.info("Starting transcription")
        filename = 'audio_file.wav'
        mime_type = 'audio/wav'
        files = {'file': (filename, audio_data, mime_type)}
        form_data = {"model": "Systran/faster-whisper-large-v2"}
        
        try:
            response = requests.post(WHISPER_URL, data=form_data, files=files)
            logger.debug("Transcription response received: %s", response.status_code)
            if response.status_code == 200:
                transcription = response.json().get('text', '')
                logger.info(f"Transcription result: {transcription[:100]}...")
                return transcription
            else:
                logger.error("Transcription failed: %s", response.text)
                return ""
        except requests.exceptions.RequestException as e:
            logger.error("Network error during transcription: %s", str(e))
            return ""
    
    def extract_context(self, prompt):
        logger.info("Extracting conversation context")
        client = Client(host=OLLAMA_BASE_URL)
        instruction = (
            f"Based on transcribed text: {prompt[:50]}..., generate context"
        )
        
        try:
            response = client.generate(model="gemma3:27b", prompt=instruction)
            context = getattr(response, 'response', 'No context')
            logger.info(f"Extracted context: {context[:100]}...")
            return context
        except Exception as e:
            logger.exception("Context extraction failed")
            return "Context extraction failed"
    
    def generate_image_prompt(self, context):
        logger.info("Generating image prompt")
        client = Client(host=OLLAMA_BASE_URL)
        instruction = (
            f"Context: {context[:50]}..., generate image prompt"
        )
        
        try:
            response = client.generate(model="gemma3:27b", prompt=instruction)
            prompt = getattr(response, 'response', 'No prompt')
            logger.info(f"Generated prompt: {prompt[:100]}...")
            return prompt
        except Exception as e:
            logger.exception("Prompt generation failed")
            return "Prompt generation failed"
    
    def generate_image(self, prompt):
        logger.info("Starting image generation")
        payload = {
            "prompt": prompt.strip(),
            "batch_size": 1,
            "steps": 50,
            "width": 800,
            "height": 480,
            "distilled_cfg_scale": 3.5,
            "cfg_scale": 7.5,
            "sampler_name": "Euler",
            "seed": -1,
            "denoising_strength": 0.75,
            "scheduler": "Simple"
        }
        logger.debug("Sending payload: %s", str(payload)[:200])
        
        try:
            response = requests.post(
                STABLE_DIFFUSION_URL,
                json=payload,
                timeout=900
            )
            logger.debug("Image API response: %s", response.status_code)
            if response.status_code == 200:
                images = response.json().get("images", [])
                if images:
                    logger.info("Image generation successful")
                    return images[0]
                else:
                    logger.error("No images returned in response")
            else:
                logger.error("Image API error: %s - %s", response.status_code, response.text[:200])
        except requests.exceptions.RequestException as e:
            logger.error("Image generation network error: %s", str(e))
        return None
    
    def process_conversation(self):
        logger.info("Processing new conversation cycle")
        try:
            logger.debug("Starting audio capture")
            audio_data = self.record_audio()
            if not audio_data:
                logger.error("Audio recording failed, skipping processing")
                return
            
            logger.debug("Processing transcription")
            transcription = self.transcribe_audio(audio_data)
            logger.debug("Extracting context")
            context = self.extract_context(transcription)
            logger.debug("Generating image prompt")
            prompt = self.generate_image_prompt(context)
            logger.debug("Generating image")
            image_data = self.generate_image(prompt)
            
            # Update UI components
            self.app.set_texts(transcription, context, prompt)
            if image_data:
                self.app.after(0, self.app.update_image, image_data)
            else:
                logger.error("Image generation failed, no image to display")
        except Exception as e:
            logger.error("Uncaught exception in processing: %s", str(e), exc_info=True)

if __name__ == "__main__":
    logger.info("Starting application")
    app = AppGUI()
    processor = AudioProcessor(app)
    processor.start()
    app.mainloop()