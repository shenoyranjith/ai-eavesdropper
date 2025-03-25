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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
        
        # Main image display canvas
        self.image_canvas = tk.Canvas(self, bg='black', highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Overlay setup
        self.overlay_visible = False
        self.overlay_window = tk.Toplevel(self)
        self.overlay_window.overrideredirect(True)  # No window borders
        self.overlay_window.config(bg='black')
        self.overlay_window.attributes('-topmost', True)
        self.overlay_window.withdraw()  # Start hidden
        
        # Position and size the overlay window (2% padding on all sides)
        self.overlay_width = int(self.winfo_screenwidth() * 0.95)
        self.overlay_height = int(self.winfo_screenheight() * 0.8)
        self.overlay_window.geometry(
            f"{self.overlay_width}x{self.overlay_height}+"
            f"{int(self.winfo_screenwidth()*0.02)}+{int(self.winfo_screenheight()*0.02)}"
        )
        
        # Text display frame
        self.text_frame = tk.Frame(self.overlay_window, bg='black')
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure styled scrollbar
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
        
        # Text widget with strict no-interaction settings
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
        
        # Block all selection interactions
        self.text_widget.bind('<Button-1>', lambda e: 'break')
        self.text_widget.bind('<B1-Motion>', lambda e: 'break')
        self.text_widget.bind('<Double-1>', lambda e: 'break')
        self.text_widget.bind('<Control-c>', lambda e: 'break')
        self.text_widget.bind('<Button-3>', lambda e: 'break')
        
        # Create styled scrollbar
        self.scrollbar = ttk.Scrollbar(
            self.text_frame,
            orient=tk.VERTICAL,
            command=self.text_widget.yview,
            style='Vertical.TScrollbar'
        )
        self.text_widget['yscrollcommand'] = self.scrollbar.set
        
        # Grid layout for text and scrollbar
        self.text_widget.grid(row=0, column=0, sticky='nsew')
        self.scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Configure grid weights
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

    def toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible
        if self.overlay_visible:
            self.overlay_window.deiconify()
        else:
            self.overlay_window.withdraw()

    def set_texts(self, transcription, context, prompt):
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

    def update_image(self, image_data):
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
            self.image = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")

class AudioProcessor:
    def __init__(self, app):
        self.app = app
        self.is_recording = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            try:
                self.process_conversation()
                time.sleep(WAIT_TIME)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}. Retrying in 60 seconds.")
                time.sleep(60)

    def record_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        logger.info("Recording...")
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        logger.info("Finished recording.")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        return wav_buffer.getvalue()

    def transcribe_audio(self, audio_data):
        filename = 'audio_file.wav'
        mime_type = 'audio/wav'
        files = {'file': (filename, audio_data, mime_type)}
        form_data = {"model": "Systran/faster-whisper-large-v2"}
        response = requests.post(WHISPER_URL, data=form_data, files=files)
        if response.status_code == 200:
            logger.info("Transcription successful.")
            return response.json().get('text', '')
        else:
            logger.error(f"Error during transcription: {response.text}")
            return ""

    def extract_context(self, prompt):
        client = Client(host=OLLAMA_BASE_URL)
        instruction = (
            f"Based on the following transcribed text, understand the context of the conversation "
            f"and provide a detailed summary and key points about the topic.\n"
            f"If the transcribed text isn't clear or is empty, pick a random topic and use that.\n"
            f"Transcribed Text:\n\n{prompt}\n\n"
        )
        try:
            response = client.generate(model="gemma3:27b", prompt=instruction)
            return getattr(response, 'response', 'No context available')
        except Exception as e:
            logger.exception("Exception during context extraction")
            return "Context extraction failed"

    def generate_image_prompt(self, context):
        client = Client(host=OLLAMA_BASE_URL)
        instruction = (
            f"Based on the following context, generate a detailed and descriptive prompt "
            f"for image generation using Stable-Diffusion-WebUI-Forge. The prompt should be formatted as follows:\n"
            f"<Positive Prompt>\n\n"
            f"Do not include any other text.\n"
            f"Context:\n\n{context}\n\n"
        )
        try:
            response = client.generate(model="gemma3:27b", prompt=instruction)
            return getattr(response, 'response', 'No prompt generated')
        except Exception as e:
            logger.exception("Exception during prompt generation")
            return "Prompt generation failed"

    def generate_image(self, prompt):
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
        response = requests.post(STABLE_DIFFUSION_URL, json=payload)
        if response.status_code == 200 and "images" in response.json():
            return response.json()["images"][0]
        else:
            logger.error(f"Error during image generation: {response.text}")
            return None

    def process_conversation(self):
        audio_data = self.record_audio()
        if not audio_data:
            logger.warning("Recording failed. Skipping transcription.")
            return
        try:
            transcription = self.transcribe_audio(audio_data)
            context = self.extract_context(transcription)
            prompt = self.generate_image_prompt(context)
            image_data = self.generate_image(prompt)
            
            # Update UI elements
            self.app.set_texts(transcription, context, prompt)
            if image_data:
                self.app.after(0, self.app.update_image, image_data)
            else:
                logger.error("Image generation failed - no data to display")
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    app = AppGUI()
    processor = AudioProcessor(app)
    processor.start()
    app.mainloop()