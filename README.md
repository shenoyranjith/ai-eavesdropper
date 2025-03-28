# AI-Eavesdropper

## Description

AI-Eavesdropper listens to what you are talking and generates related images using a combination of: [Speaches](https://github.com/speaches-ai/speaches), [Ollama](https://ollama.com/), and [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge).

### Features

- **Real-time Speech Recognition**: Uses Speaches to transcribe your speech.
- **Text-to-Image Generation**: Leverages Ollama for generating high-quality text prompts, which are then used by ComfyUI to create corresponding images.

### How It Works

1. **Speech Transcription**: AI-Eavesdropper listens to your speech using Speaches and transcribes it into text.
2. **Text Processing**: The transcribed text is then processed by Ollama to generate refined prompts that describe the visual content you want.
3. **Image Generation**: Stable Diffusion WebUI Forge takes these prompts and generates high-quality images based on the descriptions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

### Steps

1. **Clone** this repository on the device which is listening to you:
    ```bash
    git clone https://github.com/shenoyranjith/ai-eavesdropper.git
    ```

2. **Navigate** to the project directory:
    ```bash
    cd ai-eavesdropper
    ```

3. **Setup the virtual enviroment** (if any):
    ```bash
    sudo apt-get install -y python3-venv portaudio19-dev libportaudiocpp0 build-essential swig python3-dev
    python3 -m venv venv
    source venv/bin/activate
    pip install pyaudio wave requests numpy opencv-python pillow ollama
    ```

4. **Set up Ollama**:
    - Download and install [Ollama](https://ollama.com/download)
    - This project uses gemma3:27b model to generate text prompts. But you use any model you want by modifying the python script.

5. **Set up Speaches**:
    ```bash
    docker run --detach --network host --publish 8000:8000 --restart always --name speaches --volume hf-hub-cache:/home/ubuntu/.cache/huggingface/hub --gpus=all ghcr.io/speaches-ai/speaches:latest-cuda
    ```
    For more information check out their [docs](https://speaches.ai/)

6. **Set up Stable Diffusion WebUI Forge**:
    - Follow the instructions in the Stable Diffusion WebUI Forge [documentation](https://github.com/lllyasviel/stable-diffusion-webui-forge?tab=readme-ov-file#installing-forge).
    - Before running the `run.bat`, update the following
      - Add `--api --listen` to the `set COMMANDLINE_ARGS=` in the `webui-user.bat`
      - Create a virtual environment using venv inside the folder cloned by `update.bat`.
      - Set `SKIP_VENV` to 0 in `environment.bat`

## Usage

### Basic Usage

1. Start the AI-Eavesdropper system by running the main script:
    ```bash
    python3 app.py
    ```

2. Speak into your microphone, and AI-Eavesdropper will transcribe your speech, generate a text prompt, and create an image based on the prompt.

### Running as a service

1. Create the Service File:
    ```
    sudo nano /etc/systemd/system/ai-eavesdropper.service
    ```

2. Add the Following Content to the Service File:
    ```
    [Unit]
    Description=AI Eavesdropper Script
    After=network.target

    [Service]
    User=user
    Group=user
    WorkingDirectory=/home/user/ai-eavesdropper
    Environment="DISPLAY=:0"
    ExecStart=/home/user/ai-eavesdropper/venv/bin/python3 /home/user/ai-eavesdropper/app.py
    Restart=always
    RestartSec=30

    [Install]
    WantedBy=multi-user.target
    ```

3. Reload Systemd Daemon:
    ```
    sudo systemctl daemon-reload
    ```

4. Enable and Start the Service:
    ```
    sudo systemctl enable ai-eavesdropper.service
    sudo systemctl start ai-eavesdropper.service
    ```

5. View Logs to Debug:
    ```
    journalctl -u ai-eavesdropper.service -f
    ```

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)