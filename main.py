import cv2
import base64
import ollama
import time
import threading
import queue
import datetime
import numpy as np
import collections
import os
import json
import openai
import anthropic
from google import genai
from google.genai import types

# suppress Qt font warnings
os.environ["QT_LOGGING_RULES"] = "*=false"

# Load config from json
with open('sys-config.json', 'r') as f:
    config = json.load(f)

model_cfg = config['model_variables']
cam_cfg = config['camera_variables']
SYSTEM_PROMPT = config['system_prompt']


MAX_BUFFER_SIZE = model_cfg['max_buffer_size']
MIN_OBSERVATION_WINDOW = model_cfg['min_observation_window']
CAMERA_FPS = model_cfg['camera_fps']
MOTION_THRESHOLD = model_cfg['motion_threshold']
MAX_CONTEXT = model_cfg['max_context']
SUMMARY_BRAIN = model_cfg['summary_brain']
COMPARISON_BRAIN = model_cfg['comparison_brain']
SUMMARY_MODEL_MAX_TOKENS = model_cfg['summary_model_max_tokens']
MAX_FRAMES_TO_SEND = model_cfg['max_frames_to_send']
ACTIVE_CAMERA_SOURCE = cam_cfg['active_camera_source']

SUPPORTED_PROVIDERS = {
    'openai': {'key': model_cfg.get('openai_api_key', ""), 'base_url': None},
    'anthropic': {'key': model_cfg.get('anthropic_api_key', "")},
    'gemini': {'key': model_cfg.get('gemini_api_key', "")},
    'groq': {'key': model_cfg.get('groq_api_key', ""), 'base_url': 'https://api.groq.com/openai/v1'},
    'together': {'key': model_cfg.get('together_api_key', ""), 'base_url': 'https://api.together.xyz/v1'},
    'openrouter': {'key': model_cfg.get('openrouter_api_key', ""), 'base_url': 'https://openrouter.ai/api/v1'},
    'xai': {'key': model_cfg.get('xai_api_key', ""), 'base_url': 'https://api.x.ai/v1'},
    'deepseek': {'key': model_cfg.get('deepseek_api_key', ""), 'base_url': 'https://api.deepseek.com/v1'}
}

active_providers = [name for name, info in SUPPORTED_PROVIDERS.items() if info.get('key')]

print("\n\033[96m[System] Select an AI Provider for the Main Brain:\033[0m")
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

active_provider = active_providers[selected_idx - 1] if selected_idx > 0 else None

if active_provider:
    default_model = model_cfg.get(f"{active_provider}_default_model", "")
    api_key = SUPPORTED_PROVIDERS[active_provider]['key']
    
    from first_setup import get_provider_models
    print(f"\n\033[96mFetching available models for {active_provider}...\033[0m")
    available_models = get_provider_models(active_provider, api_key)
    
    if available_models:
        print(f"\033[96mSelect a model for {active_provider}:\033[0m")
        default_idx = 1
        if default_model in available_models:
            default_idx = available_models.index(default_model) + 1
            
        for idx, model_name in enumerate(available_models, 1):
            print(f"  {idx}: {model_name}")
            
        selected_idx = -1
        while selected_idx < 1 or selected_idx > len(available_models):
            try:
                display_default = default_model if default_model else available_models[default_idx-1]
                user_input = input(f"\033[96mEnter the number of your choice (default {default_idx} - {display_default}): \033[0m").strip()
                if user_input == "":
                    selected_idx = default_idx
                else:
                    selected_idx = int(user_input)
                    if selected_idx < 1 or selected_idx > len(available_models):
                        print("\033[91mInvalid choice. Please enter a valid number.\033[0m")
            except ValueError:
                print("\033[91mInvalid input. Please enter a number.\033[0m")
        
        MAIN_BRAIN = available_models[selected_idx - 1]
    else:
        model_input = input(f"\033[96mEnter model name for {active_provider} (default '{default_model}'): \033[0m").strip()
        MAIN_BRAIN = model_input if model_input else default_model
else:
    default_model = model_cfg.get('main_brain', 'CHANGE ME')
    try:
        import ollama
        ollama_models = ollama.list()
        if hasattr(ollama_models, 'models'):
            available_models = [getattr(m, 'model', getattr(m, 'name', '')) for m in ollama_models.models]
        else:
            available_models = [m.get('name', '') for m in ollama_models.get('models', [])]
            
        if available_models:
            print(f"\n\033[96mSelect a local Ollama model:\033[0m")
            default_idx = 1
            if default_model in available_models:
                default_idx = available_models.index(default_model) + 1
                
            for idx, model_name in enumerate(available_models, 1):
                print(f"  {idx}: {model_name}")
                
            selected_idx = -1
            while selected_idx < 1 or selected_idx > len(available_models):
                try:
                    display_default = default_model if default_model else available_models[default_idx-1]
                    user_input = input(f"\033[96mEnter the number of your choice (default {default_idx} - {display_default}): \033[0m").strip()
                    if user_input == "":
                        selected_idx = default_idx
                    else:
                        selected_idx = int(user_input)
                        if selected_idx < 1 or selected_idx > len(available_models):
                            print("\033[91mInvalid choice. Please enter a valid number.\033[0m")
                except ValueError:
                    print("\033[91mInvalid input. Please enter a number.\033[0m")
            MAIN_BRAIN = available_models[selected_idx - 1]
        else:
            model_input = input(f"\033[96mEnter local Ollama model name (default '{default_model}'): \033[0m").strip()
            MAIN_BRAIN = model_input if model_input else default_model
    except Exception:
        model_input = input(f"\033[96mEnter local Ollama model name (default '{default_model}'): \033[0m").strip()
        MAIN_BRAIN = model_input if model_input else default_model

print(f"\033[92m[System] Initializing with MAIN_BRAIN = {MAIN_BRAIN}\033[0m")

api_client = None

if active_provider:
    provider_info = SUPPORTED_PROVIDERS[active_provider]
    if active_provider == 'anthropic':
        api_client = anthropic.Anthropic(api_key=provider_info['key'])
    elif active_provider == 'gemini':
        api_client = genai.Client(api_key=provider_info['key'])
    else:
        api_client = openai.OpenAI(api_key=provider_info['key'], base_url=provider_info.get('base_url'))

from dependancies.logger import SessionLogger
from dependancies.gui import launch_input_gui
from dependancies.vision import calculate_visual_difference
from dependancies.memory import cull_old_memory
from dependancies.nlp import calculate_semantic_similarity

from dependancies.networked_camera import networked_camera_worker
from dependancies.physical_camera import physical_camera_worker

input_queue = queue.Queue()
session_start_dt = datetime.datetime.now()
session_logger = SessionLogger(config, session_start_dt)

# --- MAIN AI EXECUTION ---
gui_thread = threading.Thread(target=launch_input_gui, args=(input_queue,), daemon=True)
gui_thread.start()

frame_buffer = collections.deque(maxlen=MAX_BUFFER_SIZE)
camera_stop_event = threading.Event()
camera_source = ACTIVE_CAMERA_SOURCE

if camera_source == "":
    camera_source = 0
elif isinstance(camera_source, str) and camera_source.isdigit():
    camera_source = int(camera_source)

if isinstance(camera_source, str) and (camera_source.startswith("http") or camera_source.startswith("rtsp")):
    cam_thread = threading.Thread(target=networked_camera_worker, args=(camera_source, frame_buffer, camera_stop_event), daemon=True)
else:
    cam_thread = threading.Thread(target=physical_camera_worker, args=(camera_source, frame_buffer, camera_stop_event), daemon=True)

cam_thread.start()

print("\033[97m[System] Connecting to camera stream and filling initial buffer...\033[0m")
initial_required_frames = int(MIN_OBSERVATION_WINDOW * CAMERA_FPS)
while len(frame_buffer) < initial_required_frames:
    time.sleep(0.1)

messages = [SYSTEM_PROMPT]
last_processed_frame = None

print("\033[97m[System] Dynamic-Timeline Agent Loop Started. Gatekeeper Active.\033[0m")

last_eval_time = time.time()

try:
    while True:
        current_time = time.time()
        elapsed_since_last_eval = current_time - last_eval_time
        
        if elapsed_since_last_eval < MIN_OBSERVATION_WINDOW:
            time.sleep(MIN_OBSERVATION_WINDOW - elapsed_since_last_eval)
            current_time = time.time()
            elapsed_since_last_eval = current_time - last_eval_time

        all_buffered_frames = list(frame_buffer)
        if not all_buffered_frames:
            continue
            
        current_frame = all_buffered_frames[-1]

        cv2.imshow("Agent Eyes", current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        user_prompt = None
        try:
            user_prompt = input_queue.get_nowait()
        except queue.Empty:
            pass

        diff_percent = calculate_visual_difference(last_processed_frame, current_frame)
        
        if diff_percent < MOTION_THRESHOLD and user_prompt is None:
            time.sleep(0.2) 
            continue
            
        history_seconds = min(elapsed_since_last_eval, MAX_BUFFER_SIZE / CAMERA_FPS)
        last_eval_time = time.time()
        
        # --- THE SMART SNAPSHOT (Text Branching) ---
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        if history_seconds <= 3.0:
            FRAMES_TO_SEND = 6
        else:
            FRAMES_TO_SEND = min(int(history_seconds * 2), MAX_FRAMES_TO_SEND)
            
        frames_needed_count = int(history_seconds * CAMERA_FPS)
        frames_needed_count = min(frames_needed_count, len(all_buffered_frames))
        current_timeline_window = all_buffered_frames[-frames_needed_count:]
        
        indices = np.linspace(0, len(current_timeline_window) - 1, FRAMES_TO_SEND, dtype=int)
        
        debug_frames = [] 
        for idx in indices:
            f = cv2.resize(current_timeline_window[idx], (336, 336))
            debug_frames.append(f)
            
        stitched_timeline_view = np.concatenate(debug_frames, axis=1)
        
        if stitched_timeline_view.shape[1] > 1600:
            scale_ratio = 1600 / stitched_timeline_view.shape[1]
            display_view = cv2.resize(stitched_timeline_view, (int(stitched_timeline_view.shape[1] * scale_ratio), int(stitched_timeline_view.shape[0] * scale_ratio)))
            cv2.imshow("Frames Sent to LLM (Sequence)", display_view)
        else:
            cv2.imshow("Frames Sent to LLM (Sequence)", stitched_timeline_view)
        
        _, buffer = cv2.imencode('.jpg', stitched_timeline_view)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        if user_prompt:
            print(f"\n\033[97m[System] Intercepted User Input: '{user_prompt}'\033[0m")
            
            _, buffer = cv2.imencode('.jpg', cv2.resize(current_frame, (512, 512)))
            single_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                'role': 'user',
                'content': f"[{timestamp}] [User Question]: {user_prompt}. The first image attached is a high-resolution view of the scene. The rest is a sequence of frames showing the action over time.\n[SYSTEM DIRECTIVE: Answer the user's question directly based on the attached single frame. No conversational filler.]",
                'images': [single_frame_base64, img_base64]
            }
            
        else:
            print(f"\n\033[97m[System] Motion Detected ({diff_percent:.1f}%). Sending {FRAMES_TO_SEND}-frame sequence over {history_seconds:.2f}s...\033[0m")
            strict_suffix = "\n\n[SYSTEM DIRECTIVE: Respond directly to the prompt. Focus on describing the chronological movement across the frames from left to right. No conversational filler.]"
            payload = {
                'role': 'user',
                'content': f"[{timestamp}] [Camera Sequence]: Attached is a single image containing {FRAMES_TO_SEND} chronological frames reading left to right, spanning the last {history_seconds:.1f} seconds. What action or movement occurred?{strict_suffix}",
                'images': [img_base64]
            }

        messages.append(payload)

        if not user_prompt:
            session_logger.log_session_metrics(
                step_type="pre_inference",
                image_frame=stitched_timeline_view,
                timestamp_str=timestamp
            )

        # Execute Inference
        inference_start_time = time.time()
        
        if active_provider == 'anthropic':
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
            
            response = api_client.messages.create(
                model=MAIN_BRAIN,
                system=system_prompt_content,
                messages=anthropic_messages,
                max_tokens=4096
            )
            agent_reply_original = response.content[0].text.strip()
            prompt_tokens = response.usage.input_tokens
            generation_tokens = response.usage.output_tokens
            
        elif active_provider == 'gemini':
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
            response = api_client.models.generate_content(
                model=MAIN_BRAIN, 
                contents=gemini_messages,
                config=config
            )
            agent_reply_original = response.text
            prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0
            generation_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0
            
        elif api_client is not None: # All OpenAI-compatible providers
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
            
            response = api_client.chat.completions.create(
                model=MAIN_BRAIN,
                messages=openai_messages
            )
            agent_reply_original = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0
            generation_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
            
        else:
            response = ollama.chat(
                model=MAIN_BRAIN, 
                messages=messages,
                options={
                    'num_ctx': MAX_CONTEXT, 
                    'num_gpu': 99,       
                    'num_thread': 8      
                }
            )
            agent_reply_original = response['message']['content'].strip()
            prompt_tokens = response.get('prompt_eval_count', 0)
            generation_tokens = response.get('eval_count', 0)
        
        inference_duration = time.time() - inference_start_time
        print(f"\033[93m[Debug] Inference Step Time: {inference_duration:.2f}s\033[0m")

        if 'images' in messages[-1]:
            del messages[-1]['images']

        agent_reply_final = agent_reply_original
        
        total_tokens = prompt_tokens + generation_tokens
        print(f"\033[94m[Score] Token Usage: {total_tokens}/{MAX_CONTEXT}\033[0m")

        # --- GENERATIVE SUMMARIZATION --- 
        summary_triggered = False
        last_agent_message = None
        last_agent_index = -1

        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'assistant':
                last_agent_message = messages[i]['content']
                last_agent_index = i
                break

        # Check semantic similarity
        similarity_score = calculate_semantic_similarity(agent_reply_original, last_agent_message, COMPARISON_BRAIN)
        print(f"\033[94m[Score] Similarity: {similarity_score:.2f}\033[0m")

        comparison_payload = None

        if similarity_score > 0.8:
            print(f"\033[96m[System] High semantic overlap detected ({similarity_score:.2f}). Triggering memory consolidation...\033[0m")
            
            combined_text = f"Action 1: {last_agent_message}\nAction 2: {agent_reply_original}\n\n[SYSTEM DIRECTIVE: Merge these two observations into a single, brief sentence. Do not hallucinate or add outside details]"
            
            try:
                summary_response = ollama.generate(
                    model=SUMMARY_BRAIN, 
                    prompt=combined_text,
                    options={
                        'num_predict': SUMMARY_MODEL_MAX_TOKENS,
                        'num_thread': 4,
                        'num_gpu': 99
                    }
                )

                summary_triggered = True

                consolidated_reply = summary_response['response'].strip()
                print(f"\033[95m[Summary Agent]: {consolidated_reply}\033[0m")
                
                comparison_payload = {
                    "last_msg": last_agent_message,
                    "current_msg": agent_reply_original,
                    "score": f"{similarity_score:.2f}",
                    "summary_msg": consolidated_reply
                }

                messages.pop(last_agent_index)
                if last_agent_index - 1 > 0 and messages[last_agent_index - 1].get('role') == 'user':
                     messages.pop(last_agent_index - 1)
                     
                agent_reply_final = consolidated_reply
                
            except Exception as e:
                print(f"\033[91m[System] Summarization Error: {e}. Falling back to default deduplication.\033[0m")
                messages.pop(last_agent_index)
                if last_agent_index - 1 > 0 and messages[last_agent_index - 1].get('role') == 'user':
                     messages.pop(last_agent_index - 1)

        if summary_triggered == False:
            print(f"\033[97m[Agent]: {agent_reply_final}\033[0m")

        messages.append({
            'role': 'assistant',
            'content': agent_reply_final
        })

        # --- POST-INFERENCE METRIC LOGGING ---
        session_logger.log_session_metrics(
            step_type="inference",
            inference_duration=inference_duration,
            total_tokens=total_tokens,
            messages_history=messages,
            comparison_data=comparison_payload
        )
        
        last_processed_frame = current_frame.copy()
        messages = cull_old_memory(messages, max_age_hours=4.0)

        if len(messages) > 40: 
            messages.pop(1)
            messages.pop(1)


except KeyboardInterrupt:
    print("\n\033[97m[System] Loop terminated by user. Initiating safe shutdown...\033[0m")

session_logger.log_session_metrics(step_type="runtime_end")

camera_stop_event.set()
print("\033[97m[System] Waiting for camera hardware to release...\033[0m")
if cam_thread.is_alive():
    cam_thread.join(timeout=3.0)

cv2.destroyAllWindows()
print("\033[97m[System] Shutdown complete. Goodbye!\033[0m")