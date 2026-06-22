import cv2
import time

def networked_camera_worker(source, frame_buffer, stop_event):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            small_frame = cv2.resize(frame, (400, 400))
            frame_buffer.append(small_frame)
        else:
            time.sleep(0.01) 
            
    cap.release()
