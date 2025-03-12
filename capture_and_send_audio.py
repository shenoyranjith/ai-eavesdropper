import pyaudio
import wave
import io
import requests
import time

# Configuration
CHUNK = 8192 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 # Replace with your microphone's sample rate
RECORD_SECONDS = 3
SERVER_URL = "http://<REMOTE_SERVER>/v1/audio/transcriptions"  # Replace with your server IP and port
WAIT_TIME = 30  # Time to wait before recording again

def record_audio():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

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

    response = requests.post(SERVER_URL, data=form_data, files=files)
    if response.status_code == 200:
        print("Transcription:", response.json())
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    while True:
        audio_data = record_audio()
        send_audio_to_server(audio_data)
        time.sleep(WAIT_TIME)