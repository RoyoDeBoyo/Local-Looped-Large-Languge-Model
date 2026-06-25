# Local Looped Large Language Model (LLLLM)

A looped architecture for modern AI models that can run locally on your system or integrate with external third-party APIs. This project extends standard AI agent architectures by creating a persistent, looping multimodal observer. While it supports external providers, it is primarily designed to run locally on your own device for maximum privacy and control.

## Features

- **Multi-Agent Architecture**: Features multiple AI agents working in tandem. A large multimodal model acts as the "Main Brain," while supporting models ("Summary Brain", "Comparison Brain", and "Vision Brain") work continuously in the background to reduce context length, handle memory, and pre-process observations.
- **Provider Agnostic**: Supports local inference via Ollama, as well as external cloud providers including OpenAI, Anthropic, Google Gemini, Groq, Together, OpenRouter, X.AI, and DeepSeek.
- **Vision & Memory Management**: Features real-time camera tracking, destructive image compression for token optimization, and intelligent memory culling to keep the context window performant over long sessions.
- **Interactive Test Suite**: Includes a comprehensive suite of interactive tests with synthetic data to evaluate compression algorithms, agent prompts, and swappable model performance.

## Prerequisites

- **Python 3.8+**
- **Ollama**: Required if you plan to run models locally. Make sure Ollama is installed and running on your system.
- **Camera**: A camera is required for the main multimodal loop. You can use a local webcam or [DroidCam](https://droidcam.app/) on a mobile device.

## Installation

```bash
git clone <repository_url>
cd Local-Looped-Large-Languge-Model
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### Local Models (Ollama)
If using local models, pull your desired models before running. For example, using the default configurations:
```bash
ollama pull llama3.2:1b
ollama pull minicpm-v4.6:latest
ollama pull nomic-embed-text:latest
```

## Configuration

All system configurations, model variables, camera sources, and API keys are managed in `sys-config.json`. 
- **API Keys**: Add your external provider keys (e.g., `openai_api_key`) to `sys-config.json` to enable them.
- **Camera Settings**: Update `camera_variables` in `sys-config.json`. For DroidCam, use the format `"http://IP_ADDRESS:PORT/video"`.

## Usage

Start the main looped agent:
```bash
python main.py
```
Upon startup, the system will interactively prompt you to select the AI provider and specific model you wish to use for the Main Brain and other supporting components.

For stronger computers, variables in `sys-config.json` such as `max_frames_to_send`, `min_observation_window`, and `max_buffer_size` can be adjusted to get better results.

## Testing

This project includes a comprehensive interactive test suite to validate functionality and fine-tune model configurations. The tests use synthetic data stored in the `synthetic_data/` folder.

To run the interactive tests, simply execute them from the terminal:
- **Vision Compression**: `python tests/test_vision_compression.py`
- **Vision Analysis (Swappable Models)**: `python tests/test_vision_analysis.py`
- **Agent Prompts**: `python tests/test_agents.py`
- **Memory Culling**: `python tests/test_memory_static.py`
- **NLP Similarity**: `python tests/test_nlp.py`

When running the tests, follow the interactive command-line prompts. You can leave inputs blank to automatically run the tests using the provided synthetic datasets.

## License

MIT