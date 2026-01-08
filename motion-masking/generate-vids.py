import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

TEXT = "testup"
# ----------------------------
# Parameters
# ----------------------------
W, H = 512, 512
FRAMES = 120
DOT_DENSITY = 0.5
SHIFT_SPEED = 3  # Pixels to move per frame

# ----------------------------
# Create text mask
# ----------------------------
mask_img = Image.new("L", (W, H), 0)
draw = ImageDraw.Draw(mask_img)

# Try to load a font, fallback to default if not found
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 150)
except:
    font = ImageFont.load_default()

draw.text((80, 160), TEXT, fill=255, font=font)
text_mask = np.array(mask_img) > 0

# ----------------------------
# Initialize Video Writer
# ----------------------------
import os
os.makedirs("illusion-videos", exist_ok=True)

# Try multiple codec options in order of preference
codecs_to_try = [
    cv2.VideoWriter_fourcc(*"mp4v"),  # MPEG-4 (most compatible)
    cv2.VideoWriter_fourcc(*"XVID"),  # Xvid
    cv2.VideoWriter_fourcc(*"MJPG"),  # Motion JPEG
]

out = None
for codec in codecs_to_try:
    try:
        out = cv2.VideoWriter(
            "illusion-videos/motion_illusion_"+TEXT+".mp4",
            codec,
            30,
            (W, H),
            isColor=False
        )
        if out.isOpened():
            print(f"Using codec: {codec}")
            break
        else:
            out.release()
    except:
        continue

if out is None or not out.isOpened():
    raise RuntimeError("Could not initialize video writer with any codec")

# Initial random noise
noise_layer = (np.random.rand(H, W) < DOT_DENSITY).astype(np.uint8) * 255

for i in range(FRAMES):
    # Create two versions of the noise shifted in opposite directions
    # np.roll moves pixels and wraps them around the edge
    inside_motion = np.roll(noise_layer, i * SHIFT_SPEED, axis=0)      # Moves Down
    outside_motion = np.roll(noise_layer, -i * SHIFT_SPEED, axis=0)    # Moves Up

    # Combine them using the mask
    final_frame = np.where(text_mask, inside_motion, outside_motion)

    out.write(final_frame.astype(np.uint8))

out.release()
print("Saved motion_illusion",TEXT,".mp4")