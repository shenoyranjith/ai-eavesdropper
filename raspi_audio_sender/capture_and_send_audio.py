import pyaudio
import wave
import io
import requests
import time

# Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
SERVER_URL = "http://<REMOTE_SERVER_IP>:5000/transcribe"  # Replace with your server IP and port

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

    return b''.join(frames)

def send_audio_to_server(audio_data):
    headers = {'Content-Type': 'audio/wav'}
    response = requests.post(SERVER_URL, data=audio_data, headers=headers)
    if response.status_code == 200:
        print("Transcription:", response.json().get("transcription"))
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    while True:
        audio_data = record_audio()
        send_audio_to_server(audio_data)
        time.sleep(1)  # Wait for a second before recording again