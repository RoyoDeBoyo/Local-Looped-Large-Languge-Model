import requests
import ollama

def get_local_models():
    models_list = []
    try:
        ollama_models = ollama.list()
        if hasattr(ollama_models, 'models'):
            models_list = [getattr(m, 'model', getattr(m, 'name', '')) for m in ollama_models.models]
        else:
            models_list = [m.get('name', '') for m in ollama_models.get('models', [])]
    except Exception as e:
        print(f"\033[93m[Warning] Could not connect to Ollama: {e}\033[0m")
    return models_list

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
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()

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

def display_and_select_model(available_models, default_model=None, prompt_text=None, allow_skip=False, skip_text="Skip local model (Use external API / None)"):
    if not available_models:
        return "" if allow_skip else None
        
    if prompt_text:
        if prompt_text.startswith("\n"):
            print(f"\n\033[96m{prompt_text[1:]}\033[0m")
        else:
            print(f"\033[96m{prompt_text}\033[0m")

    if allow_skip:
        print(f"  0: {skip_text}")

    default_idx = 1
    if allow_skip and (default_model == "" or default_model == "0"):
        default_idx = 0
    elif default_model and default_model in available_models:
        default_idx = available_models.index(default_model) + 1
        
    for idx, model_name in enumerate(available_models, 1):
        print(f"  {idx}: {model_name}")
        
    selected_idx = -1
    min_idx = 0 if allow_skip else 1
    while selected_idx < min_idx or selected_idx > len(available_models):
        try:
            display_default = default_model if default_model and default_model in available_models else ("0" if allow_skip and default_idx == 0 else available_models[default_idx-1])
            if default_model is not None or allow_skip:
                user_input = input(f"\033[96mEnter the number of your choice (default {default_idx} - {display_default}): \033[0m").strip()
            else:
                user_input = input(f"\033[96mEnter the number of your choice (default 1): \033[0m").strip()
                
            if user_input == "":
                selected_idx = default_idx
            else:
                selected_idx = int(user_input)
                if selected_idx < min_idx or selected_idx > len(available_models):
                    print("\033[91mInvalid choice. Please enter a valid number.\033[0m")
        except ValueError:
            print("\033[91mInvalid input. Please enter a number.\033[0m")
            
    if allow_skip and selected_idx == 0:
        return ""
    return available_models[selected_idx - 1]
