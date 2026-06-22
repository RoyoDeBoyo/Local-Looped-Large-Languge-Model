import ollama
import numpy as np

def calculate_semantic_similarity(text1, text2, model_name):
    if not text1 or not text2:
        return 0.0
    
    try:
        # Generate mathematical vector representations of the text
        emb1 = ollama.embeddings(model=model_name, prompt=text1)['embedding']
        emb2 = ollama.embeddings(model=model_name, prompt=text2)['embedding']
        
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        
        # Calculate cosine similarity (1.0 means identical meaning)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return similarity
    except Exception as e:
        print(f"\033[91m[System] Embedding Error: {e}\033[0m")
        return 0.0
