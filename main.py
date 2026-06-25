import cv2
import time
import threading
import queue
import datetime
import collections
import json
import concurrent.futures

# Dependancy imports
from dependancies.logger import SessionLogger
from dependancies.gui import launch_input_gui
from dependancies.networked_camera import networked_camera_worker
from dependancies.physical_camera import physical_camera_worker
from dependancies.vision import calculate_visual_difference, compress_image_destructively
from dependancies.supporting_agents.llm_handler import LLMHandler
from dependancies.supporting_agents.snapshot_agent import build_smart_snapshot
from dependancies.supporting_agents.memory_agent import consolidate_memory, enforce_memory_limits

# Load config from json
with open('sys-config.json', 'r') as f:
    config = json.load(f)

model_cfg = config['model_variables']
cam_cfg = config['camera_variables']
SYSTEM_PROMPT = config['system_prompt']
SYSTEM_DIRECTIVE = config['system_directive']
VISION_SYSTEM_PROMPT = config['image_summary_system_prompt']


MAX_BUFFER_SIZE = model_cfg['max_buffer_size']
MIN_OBSERVATION_WINDOW = model_cfg['min_observation_window']
CAMERA_FPS = model_cfg['camera_fps']
MOTION_THRESHOLD = model_cfg['motion_threshold']
MAX_CONTEXT = model_cfg['max_context']
SUMMARY_BRAIN = model_cfg['summary_brain']
COMPARISON_BRAIN = model_cfg['comparison_brain']
SUMMARY_MODEL_MAX_TOKENS = model_cfg['summary_model_max_tokens']
MAX_FRAMES_TO_SEND = model_cfg['max_frames_to_send']
VISION_DOWNSCALE_FACTOR = model_cfg.get('vision_downscale_factor', 2)
VISION_COLOR_BITS = model_cfg.get('vision_color_bits', 4)
ACTIVE_CAMERA_SOURCE = cam_cfg['active_camera_source']

llm_handler = LLMHandler(model_cfg)
llm_handler.setup(brain_name="Main Brain", model_key="main_brain")

vision_handler = LLMHandler(model_cfg)
vision_handler.setup(brain_name="Vision Brain", model_key="vision_brain")




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
        payload, stitched_timeline_view, timestamp, history_seconds = build_smart_snapshot(current_frame, all_buffered_frames, user_prompt, diff_percent, elapsed_since_last_eval, MAX_BUFFER_SIZE, CAMERA_FPS, MAX_FRAMES_TO_SEND, SYSTEM_DIRECTIVE)

        messages.append(payload)

        if not user_prompt:
            session_logger.log_session_metrics(step_type="pre_inference",image_frame=stitched_timeline_view,timestamp_str=timestamp)

        # Execute Inference
        inference_start_time = time.time()
        
        has_images = 'images' in messages[-1]
        vision_summary = "No vision data processed."
        
        if has_images:
            vision_messages = [VISION_SYSTEM_PROMPT, messages[-1]]
            
            def run_vision():
                reply, _, _ = vision_handler.execute_model_inference(vision_messages, 4096)
                return reply

            with concurrent.futures.ThreadPoolExecutor() as executor:
                vision_future = executor.submit(run_vision)
                
                agent_reply_original, prompt_tokens, generation_tokens = llm_handler.execute_model_inference(
                    messages, MAX_CONTEXT
                )
                
                vision_summary = vision_future.result()
        else:
            agent_reply_original, prompt_tokens, generation_tokens = llm_handler.execute_model_inference(
                messages, MAX_CONTEXT
            )
        
        inference_duration = time.time() - inference_start_time
        print(f"\033[93m[Debug] Inference Step Time: {inference_duration:.2f}s\033[0m")
        print(f"\033[95m[Vision Model Summary]: {vision_summary}\033[0m")

        if has_images:
            compressed_images = []
            for img in messages[-1]['images']:
                compressed_images.append(compress_image_destructively(img, VISION_DOWNSCALE_FACTOR, VISION_COLOR_BITS))
            messages[-1]['images'] = compressed_images
            messages[-1]['content'] += f"\n[Vision Memory]: {vision_summary}"

        agent_reply_final = agent_reply_original
        
        total_tokens = prompt_tokens + generation_tokens
        print(f"\033[94m[Score] Token Usage: {total_tokens}/{MAX_CONTEXT}\033[0m")

        # --- GENERATIVE SUMMARIZATION --- 
        comparison_payload, messages = consolidate_memory(
            messages, agent_reply_original, COMPARISON_BRAIN, SUMMARY_BRAIN, SUMMARY_MODEL_MAX_TOKENS
        )

        # --- POST-INFERENCE METRIC LOGGING ---
        session_logger.log_session_metrics(
            step_type="inference",
            inference_duration=inference_duration,
            total_tokens=total_tokens,
            messages_history=messages,
            comparison_data=comparison_payload
        )
        
        last_processed_frame = current_frame.copy()
        messages = enforce_memory_limits(messages, max_age_hours=4.0, max_items=40)


except KeyboardInterrupt:
    print("\n\033[97m[System] Loop terminated by user. Initiating safe shutdown...\033[0m")

session_logger.log_session_metrics(step_type="runtime_end")

camera_stop_event.set()
print("\033[97m[System] Waiting for camera hardware to release...\033[0m")
if cam_thread.is_alive():
    cam_thread.join(timeout=3.0)

cv2.destroyAllWindows()
print("\033[97m[System] Shutdown complete. Goodbye!\033[0m")