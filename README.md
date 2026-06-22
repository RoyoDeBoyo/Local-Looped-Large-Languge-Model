# Local Looped Large Languge Model
 A looped version of modern ai models that can run locally on your systems. This is an extension of the AI agent architecture. This project also supports third party API's but is primarly designed to be run on your own device.

# Local Looped LLM (LLLLM)
## Features

This LLLLM features multiple AI agents working in tandem. A large multimodal model acts as the "brain" with a summary model and a grading model working to reduce context length.

## Prerequisites

Make sure ollama is installed and running.
For the current `multimodal_inference.py` file, `gemma4:e4b` is being used as the main brain, but feel free to change this to another multimodal model. `qwen2.5:0.5b` is being used as the summary model, this model should be smaller than the main brain and run as fast as possible to give more CPU and GPU time to the larger multimodal model. `nomic-embed-text` is being used as the comarison model. 

## Installation

```
cd looped-llm
python -m venv .venv
pip install -r requirements.txt
```

Run `ollama pull <model>` to get your models. To use default models run these commands

```
ollama pull gemma4:e4b
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
```

A camera is required for this project. I am using [DroidCam](https://droidcam.app/) on a mobile phone. Make sure you are on the same local network as your computer and then put the ip address displayed in the `camera_source` variable with this format. `"http://IP_ADDRESS:PORT/video"`

## Usage

For stronger computers variables such as `MAX_FRAMES_TO_SEND`, `MIN_OBSERVATION_WINDOW`, `MAX_BUFFER_SIZE` can be changed to get better results. The text input stream is provided through a tk inter text box and runs on a seperate thread.
Logging can be turned on and off globaly or module by module. Images are stored in black and white to reduce overall file size.

## License

MIT