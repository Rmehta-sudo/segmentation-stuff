import cv2
import numpy as np

# Create a simple test video with a moving circle
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/test_circle.mp4', fourcc, 10.0, (640, 480))

# Generate 50 frames (5 seconds at 10 fps)
for i in range(50):
    # Create black background
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Moving circle - moves from left to right
    x = int(50 + (i * 11))  # Move 11 pixels per frame
    y = 240  # Center vertically
    
    # Draw a filled circle
    cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
    
    # Add some background texture
    cv2.rectangle(frame, (0, 400), (640, 480), (50, 50, 50), -1)
    cv2.rectangle(frame, (200, 0), (220, 480), (100, 100, 100), -1)
    cv2.rectangle(frame, (420, 0), (440, 480), (100, 100, 100), -1)
    
    out.write(frame)

out.release()
print("Test video created: videos/test_circle.mp4")