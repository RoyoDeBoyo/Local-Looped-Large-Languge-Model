import datetime
import cv2
import numpy as np
import base64

def build_smart_snapshot(
    current_frame, 
    all_buffered_frames, 
    user_prompt, 
    diff_percent, 
    elapsed_since_last_eval, 
    MAX_BUFFER_SIZE, 
    CAMERA_FPS, 
    MAX_FRAMES_TO_SEND, 
    SYSTEM_DIRECTIVE
):
    history_seconds = min(elapsed_since_last_eval, MAX_BUFFER_SIZE / CAMERA_FPS)
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
            'content': f"[{timestamp}] [User Question]: {user_prompt}. The first image attached is a high-resolution view of the scene. The rest is a sequence of frames showing the action over time.\n{SYSTEM_DIRECTIVE}",
            'images': [single_frame_base64, img_base64]
        }
        
    else:
        print(f"\n\033[97m[System] Motion Detected ({diff_percent:.1f}%). Sending {FRAMES_TO_SEND}-frame sequence over {history_seconds:.2f}s...\033[0m")
        strict_suffix = f"\n\n{SYSTEM_DIRECTIVE}"
        payload = {
            'role': 'user',
            'content': f"[{timestamp}] [Camera Sequence]: Attached is a single image containing {FRAMES_TO_SEND} chronological frames reading left to right, spanning the last {history_seconds:.1f} seconds.{strict_suffix}",
            'images': [img_base64]
        }

    return payload, stitched_timeline_view, timestamp, history_seconds
