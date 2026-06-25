import sys
import os
import json
import base64
import glob
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dependancies.supporting_agents.llm_handler import LLMHandler

def main():
    print("\033[96m=== Vision Analysis Test (Swappable Model) ===\033[0m")
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'sys-config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            model_cfg = config.get('model_variables', {})
    except Exception as e:
        print(f"\033[91mError loading config: {e}\033[0m")
        return

    # Initialize the LLM handler to let user pick the model
    handler = LLMHandler(model_cfg)
    handler.setup(brain_name="Vision Analysis Brain", model_key="vision_brain")

    # Ask for custom path or use synthetic
    image_path = input("\n\033[96mEnter image path (leave blank to run on 5 synthetic images): \033[0m").strip()
    
    images_to_test = []
    if not image_path:
        synthetic_dir = os.path.join(os.path.dirname(__file__), 'synthetic_data', 'test_vision_analysis')
        if not os.path.exists(synthetic_dir):
            print(f"\033[91mError: Synthetic data directory '{synthetic_dir}' not found.\033[0m")
            return
        images_to_test = glob.glob(os.path.join(synthetic_dir, '*.[pj][np][g]'))
        if not images_to_test:
             images_to_test = glob.glob(os.path.join(synthetic_dir, '*.png'))
        print(f"\033[92mFound {len(images_to_test)} synthetic images.\033[0m")
    else:
        if os.path.exists(image_path):
            if os.path.isdir(image_path):
                images_to_test = glob.glob(os.path.join(image_path, '*.[pj][np][g]'))
                print(f"\033[92mFound {len(images_to_test)} images in directory.\033[0m")
            else:
                images_to_test.append(image_path)
        else:
            print(f"\033[91mError: Path '{image_path}' not found.\033[0m")
            return

    for idx, img_path in enumerate(images_to_test, 1):
        print(f"\n\033[93m--- Testing Vision Model on Image {idx}/{len(images_to_test)}: {os.path.basename(img_path)} ---\033[0m")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"\033[91mCould not read {img_path}\033[0m")
            continue
            
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        messages = [
            {
                "role": "system", 
                "content": "You are a visual analysis assistant. Describe the image in detail."
            },
            {
                "role": "user",
                "content": "What is in this image?",
                "images": [img_b64]
            }
        ]
        
        print("\033[96mSending request to model...\033[0m")
        try:
            response, prompt_tok, gen_tok = handler.execute_model_inference(messages, max_context=8192)
            print(f"\n\033[92m[Response]\033[0m\n{response}")
            print(f"\n\033[90mTokens - Prompt: {prompt_tok}, Gen: {gen_tok}\033[0m")
        except Exception as e:
            print(f"\033[91mError during inference: {e}\033[0m")
            
        print("\n\033[96mPress Enter to continue to the next image...\033[0m")
        input()

    print("\033[92mVision Analysis Test Complete.\033[0m")

if __name__ == "__main__":
    main()
