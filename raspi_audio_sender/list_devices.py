import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        print(f"Device ID {int(device_info['index'])} - {device_info['name']}")
        print(f"  Max input channels: {int(device_info['maxInputChannels'])}")
        print(f"  Default sample rate: {int(device_info['defaultSampleRate'])}")

    p.terminate()

if __name__ == "__main__":
    list_audio_devices()