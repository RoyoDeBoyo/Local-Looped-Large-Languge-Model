import json
import os
import shutil
import sys
import urllib.request
import urllib.error

def get_provider_models(provider, api_key):
    url = ""
    headers = {}
    if provider == "openai":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/models"
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    elif provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    elif provider == "groq":
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
    elif provider == "together":
        url = "https://api.together.xyz/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
    elif provider == "openrouter":
        url = "https://openrouter.ai/api/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
    elif provider == "xai":
        url = "https://api.x.ai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
    elif provider == "deepseek":
        url = "https://api.deepseek.com/models"
        headers = {"Authorization": f"Bearer {api_key}"}
    else:
        return []

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if provider == "gemini":
                models = []
                for m in data.get("models", []):
                    name = m.get("name", "")
                    if name.startswith("models/"):
                        name = name[7:]
                    models.append(name)
                return models
            elif isinstance(data, list):
                return [m.get("id") for m in data if "id" in m]
            else:
                return [m.get("id") for m in data.get("data", []) if "id" in m]
    except Exception:
        return []

def main():
    print("\033[96m[System] Welcome to the Local-Looped-Large-Language-Model First Setup!\033[0m\n")

    models_list = []
    try:
        import ollama
        try:
            ollama_models = ollama.list()
            if hasattr(ollama_models, 'models'):
                models_list = [getattr(m, 'model', getattr(m, 'name', '')) for m in ollama_models.models]
            else:
                models_list = [m.get('name', '') for m in ollama_models.get('models', [])]
        except Exception as e:
            print(f"\033[93m[Warning] Could not connect to Ollama: {e}\033[0m")
    except ImportError:
        print("\033[93m[Warning] Ollama Python package not found.\033[0m")

    def select_local_model(prompt_text, default_choice="1"):
        if not models_list:
            return ""
        print(f"\n\033[96m{prompt_text}\033[0m")
        print("  0: Skip local model (Use external API / None)")
        for idx, model_name in enumerate(models_list, 1):
            print(f"  {idx}: {model_name}")
        
        selected_idx = -1
        while selected_idx < 0 or selected_idx > len(models_list):
            try:
                user_input = input(f"\033[96mEnter the number of your choice (default {default_choice}): \033[0m")
                if user_input.strip() == "":
                    selected_idx = int(default_choice)
                else:
                    selected_idx = int(user_input)
                    if selected_idx < 0 or selected_idx > len(models_list):
                        print("\033[91mInvalid choice. Please enter a valid number.\033[0m")
            except ValueError:
                print("\033[91mInvalid input. Please enter a number.\033[0m")
        
        if selected_idx == 0:
            return ""
        return models_list[selected_idx - 1]

    local_model_default = select_local_model("Select a default local (Ollama) model for the main brain:", "1" if models_list else "0")
    summary_model_default = select_local_model("Select a default local (Ollama) model for the summary brain:", "0")
    comparison_model_default = select_local_model("Select a default local (Ollama) model for the comparison brain:", "0")

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
                print(f"\033[96mSelect a default model for {provider}:\033[0m")
                for idx, model_name in enumerate(available_models, 1):
                    print(f"  {idx}: {model_name}")
                
                selected_idx = -1
                while selected_idx < 1 or selected_idx > len(available_models):
                    try:
                        user_input = input(f"\033[96mEnter the number of your choice (default 1): \033[0m")
                        if user_input.strip() == "":
                            selected_idx = 1
                        else:
                            selected_idx = int(user_input)
                            if selected_idx < 1 or selected_idx > len(available_models):
                                print("\033[91mInvalid choice. Please enter a valid number.\033[0m")
                    except ValueError:
                        print("\033[91mInvalid input. Please enter a number.\033[0m")
                default_model = available_models[selected_idx - 1]
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
            "content": "You are an autonomous multimodal agent observing an indoor environment through a camera.\n    - Time context is provided in brackets, e.g., [14:05:00].\n    - Direct human interactions will be flagged as [User Question].\n    - Autonomous visual updates will be flagged as [Camera Sequence].\n\n    Your primary goal is to analyze visual input for significant actions or changes.\n    - Describe the movement or action across the image sequence you receive.\n    - Mention specific objects only if they are directly involved in the action or relevant to the context. Avoid describing static background items.\n    - If the frames show no meaningful action or change, respond concisely that nothing significant is happening.\n    - You have a partial view of the environment, however you can make analysis of previously seen areas based on memory.\n    - Keep all responses concise."
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
