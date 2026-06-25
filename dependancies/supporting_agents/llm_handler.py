import ollama
import anthropic
import openai
from google import genai
from google.genai import types
from dependancies.external_api import get_provider_models, display_and_select_model, get_local_models

class LLMHandler:
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.active_provider = None
        self.api_client = None
        self.model_name = None
        
        self.supported_providers = {
            'openai': {'key': model_cfg.get('openai_api_key', ""), 'base_url': None},
            'anthropic': {'key': model_cfg.get('anthropic_api_key', "")},
            'gemini': {'key': model_cfg.get('gemini_api_key', "")},
            'groq': {'key': model_cfg.get('groq_api_key', ""), 'base_url': 'https://api.groq.com/openai/v1'},
            'together': {'key': model_cfg.get('together_api_key', ""), 'base_url': 'https://api.together.xyz/v1'},
            'openrouter': {'key': model_cfg.get('openrouter_api_key', ""), 'base_url': 'https://openrouter.ai/api/v1'},
            'xai': {'key': model_cfg.get('xai_api_key', ""), 'base_url': 'https://api.x.ai/v1'},
            'deepseek': {'key': model_cfg.get('deepseek_api_key', ""), 'base_url': 'https://api.deepseek.com/v1'}
        }

    def setup(self, brain_name="Main Brain", model_key="main_brain"):
        active_providers = [name for name, info in self.supported_providers.items() if info.get('key')]

        print(f"\n\033[96m[System] Select an AI Provider for the {brain_name}:\033[0m")
        print("  0: local (ollama)")

        for idx, provider_name in enumerate(active_providers, 1):
            print(f"  {idx}: {provider_name}")

        selected_idx = -1
        while selected_idx < 0 or selected_idx > len(active_providers):
            try:
                user_input = input("\033[96mEnter the number of your choice (default 0): \033[0m")
                if user_input.strip() == "":
                    selected_idx = 0
                else:
                    selected_idx = int(user_input)
                    if selected_idx < 0 or selected_idx > len(active_providers):
                        print("\033[91mInvalid choice. Please enter a valid number.\033[0m")
            except ValueError:
                print("\033[91mInvalid input. Please enter a number.\033[0m")

        self.active_provider = active_providers[selected_idx - 1] if selected_idx > 0 else None

        if self.active_provider:
            default_model = self.model_cfg.get(f"{self.active_provider}_default_model", "")
            api_key = self.supported_providers[self.active_provider]['key']
            
            print(f"\n\033[96mFetching available models for {self.active_provider}...\033[0m")
            available_models = get_provider_models(self.active_provider, api_key)
            
            if available_models:
                self.model_name = display_and_select_model(available_models, default_model=default_model, prompt_text=f"Select a model for {self.active_provider}:")
            else:
                model_input = input(f"\033[96mEnter model name for {self.active_provider} (default '{default_model}'): \033[0m").strip()
                self.model_name = model_input if model_input else default_model
        else:
            default_model = self.model_cfg.get(model_key, 'CHANGE ME')
            
            available_models = get_local_models()
                
            if available_models:
                self.model_name = display_and_select_model(available_models, default_model=default_model, prompt_text="\nSelect a local Ollama model:")
            else:
                model_input = input(f"\033[96mEnter local Ollama model name (default '{default_model}'): \033[0m").strip()
                self.model_name = model_input if model_input else default_model

        print(f"\033[92m[System] Initializing with {model_key.upper()} = {self.model_name}\033[0m")

        if self.active_provider:
            provider_info = self.supported_providers[self.active_provider]
            if self.active_provider == 'anthropic':
                self.api_client = anthropic.Anthropic(api_key=provider_info['key'])
            elif self.active_provider == 'gemini':
                self.api_client = genai.Client(api_key=provider_info['key'])
            else:
                self.api_client = openai.OpenAI(api_key=provider_info['key'], base_url=provider_info.get('base_url'))
                
        return self

    def execute_model_inference(self, messages, max_context):
        if self.active_provider == 'anthropic':
            anthropic_messages = []
            system_prompt_content = ""
            for msg in messages:
                if msg['role'] == 'system':
                    system_prompt_content = msg['content']
                elif msg['role'] == 'user':
                    content = [{"type": "text", "text": msg['content']}]
                    if 'images' in msg:
                        for img in msg['images']:
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img
                                }
                            })
                    anthropic_messages.append({"role": "user", "content": content})
                elif msg['role'] == 'assistant':
                    anthropic_messages.append({"role": "assistant", "content": msg['content']})
            
            response = self.api_client.messages.create(
                model=self.model_name,
                system=system_prompt_content,
                messages=anthropic_messages,
                max_tokens=8192
            )
            agent_reply_original = response.content[0].text.strip()
            prompt_tokens = response.usage.input_tokens
            generation_tokens = response.usage.output_tokens
            
        elif self.active_provider == 'gemini':
            gemini_messages = []
            system_instruction = ""
            for msg in messages:
                if msg['role'] == 'system':
                    system_instruction = msg['content']
                elif msg['role'] == 'user':
                    parts = [types.Part.from_text(text=msg['content'])]
                    if 'images' in msg:
                        import base64
                        for img in msg['images']:
                            img_bytes = base64.b64decode(img)
                            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                    gemini_messages.append(types.Content(role="user", parts=parts))
                elif msg['role'] == 'assistant':
                    gemini_messages.append(types.Content(role="model", parts=[types.Part.from_text(text=msg['content'])]))
                    
            config = types.GenerateContentConfig(system_instruction=system_instruction)
            response = self.api_client.models.generate_content(
                model=self.model_name, 
                contents=gemini_messages,
                config=config
            )
            agent_reply_original = response.text
            prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0
            generation_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0
            
        elif self.api_client is not None: # All OpenAI-compatible providers
            openai_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    openai_messages.append({"role": "system", "content": msg['content']})
                elif msg['role'] == 'user':
                    content = [{"type": "text", "text": msg['content']}]
                    if 'images' in msg:
                        for img in msg['images']:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                            })
                    openai_messages.append({"role": "user", "content": content})
                elif msg['role'] == 'assistant':
                    openai_messages.append({"role": "assistant", "content": msg['content']})
            
            response = self.api_client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages
            )
            agent_reply_original = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0
            generation_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
            
        else:
            response = ollama.chat(
                model=self.model_name, 
                messages=messages,
                options={
                    'num_ctx': max_context, 
                    'num_gpu': 99,       
                    'num_thread': 8      
                }
            )
            agent_reply_original = response['message']['content'].strip()
            prompt_tokens = response.get('prompt_eval_count', 0)
            generation_tokens = response.get('eval_count', 0)
            
        return agent_reply_original, prompt_tokens, generation_tokens
        
    def process_audio_input(self, audio_data):
        """
        Placeholder for handling incoming audio data.
        To be implemented later.
        """
        pass
        
    def generate_audio_output(self, text_response):
        """
        Placeholder for generating audio output from text.
        To be implemented later.
        """
        pass
