import cv2
import numpy as np
import os

def generate_static_background(width, height, type='color', color=(255,255,255)):
    if type == 'color':
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = color
        return img
    elif type == 'road':
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (80, 80, 80) # asphalt gray
        # Draw lane lines
        for y in range(0, height, 40):
            cv2.line(img, (width//2, y), (width//2, y+20), (255, 255, 255), 4)
        return img
    elif type == 'stars':
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Add random stars
        num_stars = 100
        for _ in range(num_stars):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            r = np.random.randint(1, 3)
            c = np.random.randint(150, 256)
            cv2.circle(img, (x,y), r, (c,c,c), -1)
        return img
    elif type == 'brick':
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (50, 50, 150) # brick red in BGR
        for y in range(0, height, 50):
            offset = 0 if (y//50)%2 == 0 else 50
            cv2.line(img, (0, y), (width, y), (200, 200, 200), 2)
            for x in range(offset, width, 100):
                cv2.line(img, (x, y), (x, y+50), (200, 200, 200), 2)
        return img
    return np.zeros((height, width, 3), dtype=np.uint8)

def stitch_frames(frames):
    return np.concatenate(frames, axis=1)

def main():
    print("\033[96mGenerating Dynamic Synthetic Strips...\033[0m")
    out_dir = os.path.join(os.path.dirname(__file__), 'synthetic_data', 'test_vision_analysis_dynamic')
    os.makedirs(out_dir, exist_ok=True)
    
    frames_count = 10
    w, h = 500, 500

    # Sequence 1: Simple 1 (Red rectangle moving left to right on blue background)
    print("Generating Sequence 1 (Simple)...")
    frames1 = []
    bg1 = generate_static_background(w, h, 'color', (200, 100, 50)) # Blueish BGR
    for i in range(frames_count):
        f = bg1.copy()
        x = int(50 + i * (400 / frames_count))
        y = 250
        cv2.rectangle(f, (x-30, y-20), (x+30, y+20), (0, 0, 255), -1) # Red rectangle
        frames1.append(f)
    cv2.imwrite(os.path.join(out_dir, 'seq1_simple_rect.png'), stitch_frames(frames1))

    # Sequence 2: Simple 2 (Yellow circle moving top to bottom on green background)
    print("Generating Sequence 2 (Simple)...")
    frames2 = []
    bg2 = generate_static_background(w, h, 'color', (100, 200, 100)) # Green BGR
    for i in range(frames_count):
        f = bg2.copy()
        x = 250
        y = int(50 + i * (400 / frames_count))
        cv2.circle(f, (x, y), 25, (0, 255, 255), -1) # Yellow circle
        frames2.append(f)
    cv2.imwrite(os.path.join(out_dir, 'seq2_simple_circle.png'), stitch_frames(frames2))

    # Sequence 3: Complex 1 (White triangle moving diagonally on stars)
    print("Generating Sequence 3 (Complex)...")
    frames3 = []
    bg3 = generate_static_background(w, h, 'stars') 
    for i in range(frames_count):
        f = bg3.copy()
        x = int(50 + i * (400 / frames_count))
        y = int(450 - i * (400 / frames_count))
        
        # Draw triangle
        pt1 = (x, y-20)
        pt2 = (x-15, y+15)
        pt3 = (x+15, y+15)
        triangle_cnt = np.array( [pt1, pt2, pt3] )
        cv2.drawContours(f, [triangle_cnt], 0, (255,255,255), -1)
        # Add exhaust
        cv2.circle(f, (x, y+20), 5, (0, 100, 255), -1)
        frames3.append(f)
    cv2.imwrite(os.path.join(out_dir, 'seq3_complex_space.png'), stitch_frames(frames3))

    # Sequence 4: Complex 2 (Two cars moving on road in opposite directions)
    print("Generating Sequence 4 (Complex)...")
    frames4 = []
    bg4 = generate_static_background(w, h, 'road') 
    for i in range(frames_count):
        f = bg4.copy()
        
        # Car 1 (Blue, moving down on left lane)
        x1 = 125
        y1 = int(50 + i * (400 / frames_count))
        cv2.rectangle(f, (x1-20, y1-30), (x1+20, y1+30), (255, 0, 0), -1)
        cv2.rectangle(f, (x1-15, y1-15), (x1+15, y1+15), (200, 200, 200), -1) # roof
        
        # Car 2 (Red, moving up on right lane)
        x2 = 375
        y2 = int(450 - i * (400 / frames_count))
        cv2.rectangle(f, (x2-20, y2-30), (x2+20, y2+30), (0, 0, 255), -1)
        cv2.rectangle(f, (x2-15, y2-15), (x2+15, y2+15), (200, 200, 200), -1) # roof
        
        frames4.append(f)
    cv2.imwrite(os.path.join(out_dir, 'seq4_complex_cars.png'), stitch_frames(frames4))

    # Sequence 5: Completely static (Brick wall)
    print("Generating Sequence 5 (Static)...")
    frames5 = []
    bg5 = generate_static_background(w, h, 'brick') 
    for i in range(frames_count):
        # We append exact copies, no movement
        frames5.append(bg5.copy())
    cv2.imwrite(os.path.join(out_dir, 'seq5_static_wall.png'), stitch_frames(frames5))

    print(f"\033[92mSuccess! Generated 5 dynamic strips (10x500x500 each) in {out_dir}\033[0m")

if __name__ == "__main__":
    main()
