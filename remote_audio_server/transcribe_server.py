from flask import Flask, request, jsonify
import io
import torch
from faster_whisper.whisper import WhisperModel

app = Flask(__name__)

# Initialize the Whisper model
model_name = "base"  # You can change this to another model if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(model_name, device=device)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_data = request.data

    # Convert bytes to an in-memory file-like object
    audio_file = io.BytesIO(audio_data)

    # Transcribe the audio
    result = model.transcribe(audio_file, language="en")  # You can change the language if needed

    return jsonify({"transcription": result["text"]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)