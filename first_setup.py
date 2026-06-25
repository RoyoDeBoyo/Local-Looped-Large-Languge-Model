import json
import os
import shutil
import torch

from dependancies.external_api import get_local_models, get_provider_models, display_and_select_model

def main():
    print("\033[96m[System] Welcome to the Local-Looped-Large-Language-Model First Setup!\033[0m\n")

    print("\n\033[96m[System] Checking system resources...\033[0m")

    # scan system to see available VRAM
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        vram_gb = total_vram / (1024**3)
        print(f"\033[97m[System] Detected {vram_gb:.1f}/24.0 GB of recommended GPU VRAM.\033[0m")
        if vram_gb < 24.0:
            print(f"\033[93m[Warning] You only have {vram_gb:.1f}GB of the recommended 24.0GB VRAM.\033[0m")
            print("\033[93mRunning multiple multimodal AI models simultaneously (Main Brain + Vision Brain) requires significant compute.\033[0m")
            print("\033[93mWe highly recommend using external APIs for at least one of these brains to prevent out-of-memory errors or severe lag.\033[0m")
        else:
            print(f"\033[92m[System] Excellent! Your system has {vram_gb:.1f}GB of VRAM. This will allow you to run fully locally without performance degradation.\033[0m")
    else:
        print("\033[91m[CRITICAL WARNING] No CUDA-compatible GPU detected on your system!\033[0m")
        print("\033[93mThis application relies heavily on GPU acceleration to process images and run Large Language Models in real-time.\033[0m")
        print("\033[93mAttempting to run this fully locally on a CPU will result in extremely slow performance and potential crashes.\033[0m")
        print("\033[93mYou MUST configure and use external APIs (like OpenAI, Anthropic, or Gemini) to ensure the system functions correctly.\033[0m")

    models_list = get_local_models()

    default_choice = models_list[0] if models_list else ""
    local_model_default = display_and_select_model(models_list, default_model=default_choice, prompt_text="Select a default local (Ollama) model for the main brain:", allow_skip=True)
    summary_model_default = display_and_select_model(models_list, default_model=default_choice, prompt_text="Select a default local (Ollama) model for the text summary:", allow_skip=True)
    comparison_model_default = display_and_select_model(models_list, default_model=default_choice, prompt_text="Select a default local (Ollama) model for the text comparisons:", allow_skip=True)
    vision_model_default = display_and_select_model(models_list, default_model=default_choice, prompt_text="Select a default local (Ollama) model for the vision summary:", allow_skip=True)

    providers = [
        "openai",
        "anthropic",
        "gemini",
        "groq",
        "together",
        "openrouter",
        "xai",
        "deepseek"
    ]

    api_keys = {}
    default_models = {}

    print("\n\033[96m[System] Now let's configure external API providers.\033[0m")
    print("Press Enter to skip a provider if you do not want to configure it.\n")

    for provider in providers:
        key = input(f"\033[95mEnter API key for {provider} (leave blank to skip): \033[0m").strip()
        api_keys[f"{provider}_api_key"] = key
        
        if key:
            print(f"\033[96mFetching available models for {provider}...\033[0m")
            available_models = get_provider_models(provider, key)
            if available_models:
                default_model = display_and_select_model(available_models, prompt_text=f"Select a default model for {provider}:")
            else:
                print(f"\033[93mCould not automatically fetch models. Please enter manually.\033[0m")
                default_model = input(f"\033[94mEnter default model name for {provider} (e.g. claude-3-opus-20240229, gpt-4o): \033[0m").strip()
            
            default_models[f"{provider}_default_model"] = default_model

        else:
            default_models[f"{provider}_default_model"] = ""

    config = {
        "analytics_and_logging": {
            "compare_responses": True,
            "log_transcript": True,
            "log_comparisons": True,
            "log_inference_times": True,
            "log_images": True,
            "log_token_usage": True
        },
        "model_variables": {
            "main_brain": local_model_default,
            "max_buffer_size": 3600,
            "min_observation_window": 3.0,
            "camera_fps": 30,
            "motion_threshold": 7.5,
            "max_context": 131072,
            "summary_brain": summary_model_default if summary_model_default else "qwen2.5:0.5b",
            "comparison_brain": comparison_model_default if comparison_model_default else "nomic-embed-text",
            "vision_brain": vision_model_default if vision_model_default else "minicpm-v4.6:1b",
            "vision_downscale_factor": 2,
            "vision_color_bits": 4,
            "summary_model_max_tokens": 512,
            "max_frames_to_send": 20
        },
        "camera_variables": {
            "droid_cam_source": "",
            "local_webcam_source": 0,
            "external_webcam_source": 1,
            "active_camera_source": 0
        },
        "system_prompt": {
            "role": "system",
            "content": "You are an autonomous multimodal agent observing an indoor environment through a camera.\nTime context is provided in brackets, e.g., [14:05:00].\nDirect human interactions will be flagged as [User Question].\nAutonomous visual updates will be flagged as [Camera Sequence].\n[Vision Memory] tags indicate previously seen events that were compressed to save context. Pay attention to them.\n\nYour primary goal is to be a helpful assistant by using all context available to you. This includes both the text inputs and the visual inputs if relevant.\nRefrain from using deep reasoning as this will impact your context. If a question is simple and does not require deep reasoning, answer it directly. If the problem does require deep reasoning, you should reason about the problem before answering. This will give you a better understanding of the situation and allow you to provide a more accurate response.\nMention specific objects only if they are directly involved in the action or relevant to the context. Avoid describing static background items.\nIf the frames show no meaningful action or change, respond concisely that nothing significant is happening.\nYou have a partial view of the environment, however you can make analysis of previously seen areas based on memory.\nMake sure all responses only contain relevant information.\nRefrain from giving long explinations with examples and only output what is necessary."
        },
        "system_directive": "[SYSTEM DIRECTIVE: If you have a problem that you are working on then use this prompt as context to help, otherwise answer the question directly based on the attached single frame. No conversational filler]",
        "image_summary_system_prompt": {
            "role": "system", 
            "content": "You are a visual analysis assistant. Describe the image and any movement that may have happened in a way that you could understand what is happening without the image present. Do not use any conversational filler."
        }
    }

    # Add api keys and default models to model_variables
    for k, v in api_keys.items():
        config["model_variables"][k] = v
    for k, v in default_models.items():
        config["model_variables"][k] = v

    if os.path.exists("sys-config.json"):
        shutil.copy("sys-config.json", "sys-config.json.bak")
        print("\n\033[93m[System] Backed up existing sys-config.json to sys-config.json.bak\033[0m")

    with open("sys-config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n\033[92m[Success] Setup complete! sys-config.json has been generated.\033[0m")
    print("\n\033[93m========================================================================\033[0m")
    print("\033[93m                                DISCLAIMER                                \033[0m")
    print("\033[93m========================================================================\033[0m")
    print("\033[97mPlease review the generated sys-config.json and adjust any advanced parameters.\033[0m")
    print("\033[97mIn particular, max_buffer_sie and max_frames_to_send default to 3600 and 20 respectively.\033[0m")
    print("\033[97mChange these defaults for better results depending on your hardware.\033[0m")
    print("\033[93m========================================================================\033[0m\n")

if __name__ == "__main__":
    main()
