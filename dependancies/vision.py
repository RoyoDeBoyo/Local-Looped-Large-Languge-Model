import cv2
import numpy as np
import base64

def calculate_visual_difference(frame1, frame2):
    if frame1 is None or frame2 is None:
        return 100.0
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    return (changed_pixels / total_pixels) * 100.0

def compress_image_destructively(base64_str, downscale_factor=8, bits=2):
    """
    Destructively compress a base64 image by reducing resolution and color depth.
    """
    if not base64_str:
        return base64_str

    try:
        # Decode base64
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return base64_str

        # Reduce resolution
        if downscale_factor > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // downscale_factor, h // downscale_factor), interpolation=cv2.INTER_AREA)

        # Reduce color depth (e.g., 4 bits per channel)
        if bits < 8:
            shift = 8 - bits
            # Shift bits to reduce color space, then shift back to maintain 0-255 range
            img = (img >> shift) << shift

        # Re-encode to base64
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"\033[91m[Vision Error] Failed to compress image destructively: {e}\033[0m")
        return base64_str
