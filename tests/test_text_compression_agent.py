import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dependancies.supporting_agents.llm_handler import LLMHandler

def main():
    print("\033[96m=== Text Compression Agent Prompt Test ===\033[0m")
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'sys-config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            model_cfg = config.get('model_variables', {})
    except Exception as e:
        print(f"\033[91mError loading config: {e}\033[0m")
        return

    handler = LLMHandler(model_cfg)
    handler.setup(brain_name="Text Compression Agent Testing Brain", model_key="summary_brain")

    print("\n\033[93m--- Testing Text Compression Agent Prompt ---\033[0m")
    print("\033[90mScenario: Two highly similar observations need merging.\033[0m")
    obs1 = "I see a red car driving down the street."
    obs2 = "A red vehicle is moving along the road."
    print(obs1)
    print(obs2)
    
    memory_prompt = f"Action 1: {obs1}\nAction 2: {obs2}"
    messages_memory = [
        {"role": "system", "content": "You are a condenser agent. Your job is to take two semantically similar observations and merge them into a single, brief sentence. Do not hallucinate or add outside details"},
        {"role": "user", "content": memory_prompt}
    ]
    
    try:
        response, _, _ = handler.execute_model_inference(messages_memory, max_context=4096)
        print(f"\033[92m[Merged Memory Response]:\033[0m {response}")
    except Exception as e:
        print(f"\033[91mError: {e}\033[0m")

    print("\n\033[92mMemory Agent Prompt Test Complete.\033[0m")

if __name__ == "__main__":
    main()
