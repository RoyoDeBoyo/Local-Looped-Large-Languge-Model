import sys
import os
import cv2
import base64
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dependancies.vision import compress_image_destructively
import numpy as np

def main():
    print("\033[96m=== Vision Compression Test ===\033[0m")
    image_path = input("\033[96mEnter image path (leave blank to run on 5 synthetic images): \033[0m").strip()
    
    downscale_input = input("\033[96mEnter downscale factor (default 2): \033[0m").strip()
    downscale = int(downscale_input) if downscale_input else 2
    
    bits_input = input("\033[96mEnter color bits per channel (default 4): \033[0m").strip()
    bits = int(bits_input) if bits_input else 4

    images_to_test = []

    if not image_path:
        synthetic_dir = os.path.join(os.path.dirname(__file__),'synthetic_data', 'test_vision_compression')
        if not os.path.exists(synthetic_dir):
            print(f"\033[91mError: Synthetic data directory '{synthetic_dir}' not found.\033[0m")
            return
        # Load all png/jpg images in the directory
        images_to_test = glob.glob(os.path.join(synthetic_dir, '*.[pj][np][g]'))
        if not images_to_test:
             images_to_test = glob.glob(os.path.join(synthetic_dir, '*.png'))
        print(f"\033[92mFound {len(images_to_test)} synthetic images.\033[0m")
    else:
        if os.path.exists(image_path):
            images_to_test.append(image_path)
        elif os.path.isdir(image_path):
            images_to_test = glob.glob(os.path.join(image_path, '*.[pj][np][g]'))
            print(f"\033[92mFound {len(images_to_test)} images in directory.\033[0m")
        else:
            print(f"\033[91mError: Path '{image_path}' not found.\033[0m")
            return

    for idx, img_path in enumerate(images_to_test, 1):
        print(f"\n\033[93m--- Testing Image {idx}/{len(images_to_test)}: {os.path.basename(img_path)} ---\033[0m")
        img = cv2.imread(img_path)
        if img is None:
            print(f"\033[91mError: Could not read image '{img_path}'.\033[0m")
            continue
            
        _, buffer = cv2.imencode('.jpg', img)
        original_b64 = base64.b64encode(buffer).decode('utf-8')

        print(f"Original image shape: {img.shape}")
        print(f"Original base64 length: {len(original_b64)}")

        # Compress
        print(f"Compressing with downscale_factor={downscale}, bits={bits}...")
        compressed_b64 = compress_image_destructively(original_b64, downscale, bits)
        
        print(f"Compressed base64 length: {len(compressed_b64)}")
        reduction = 100 * (1 - len(compressed_b64) / len(original_b64))
        print(f"Size reduction: {reduction:.1f}%")

        # Decode and show result
        img_data = base64.b64decode(compressed_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        compressed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print(f"Compressed image shape: {compressed_img.shape}")

        # Display side-by-side if dimensions match, else just show both
        cv2.imshow(f"Original - {os.path.basename(img_path)}", img)
        
        # Scale compressed image up for visual comparison of quality loss
        upscaled = cv2.resize(compressed_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"Compressed {bits}bit - {os.path.basename(img_path)}", upscaled)
        
        print("\033[96mPress any key in the image windows to continue to the next image (or close test)...\033[0m")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n\033[92mTest Complete.\033[0m")

if __name__ == "__main__":
    main()
