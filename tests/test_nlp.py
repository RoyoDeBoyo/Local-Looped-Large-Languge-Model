import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dependancies.nlp import calculate_semantic_similarity

def main():
    print("\033[96m=== NLP Semantic Similarity Test ===\033[0m")
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'sys-config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            model_cfg = config.get('model_variables', {})
            default_model = model_cfg.get('comparison_brain', 'nomic-embed-text:latest')
    except Exception as e:
        print(f"\033[91mError loading config: {e}\033[0m")
        default_model = 'nomic-embed-text:latest'

    model_name = input(f"\033[96mEnter Ollama embedding model (default '{default_model}'): \033[0m").strip()
    if not model_name:
        model_name = default_model
        
    text1 = input("\033[96mEnter first text (default: 'The quick brown fox'): \033[0m").strip()
    if not text1:
        text1 = "The quick brown fox"
        
    text2 = input("\033[96mEnter second text (default: 'A fast brown fox'): \033[0m").strip()
    if not text2:
        text2 = "A fast brown fox"
        
    print(f"\n\033[93mComparing:\n1: '{text1}'\n2: '{text2}'\nModel: {model_name}\033[0m")
    
    score = calculate_semantic_similarity(text1, text2, model_name)
    
    print(f"\n\033[92mSimilarity Score: {score:.4f}\033[0m")
    if score > 0.8:
        print("\033[96mResult: Highly Similar (>0.8)\033[0m")
    elif score > 0.5:
        print("\033[96mResult: Moderately Similar\033[0m")
    else:
        print("\033[96mResult: Not Very Similar\033[0m")
        
    print("\n\033[92mNLP Semantic Similarity Test Complete.\033[0m")

if __name__ == "__main__":
    main()
