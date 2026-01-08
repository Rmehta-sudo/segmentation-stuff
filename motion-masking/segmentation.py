import cv2
import numpy as np
from collections import deque
from pathlib import Path

# ==========================
# USER PARAMETERS
# ==========================

VIDEO_PATH = "illusion-videos/motion_illusion_RACHIT.mp4"
OUTPUT_DIR = "output"
X_FRAMES = 5

PATCH_H = 8
PATCH_W = 8
STRIDE_Y = 4
STRIDE_X = 4

MAX_DISPLACEMENT = 5
CORR_THRESHOLD = 0.2
EPS = 1e-6

# ==========================
# SETUP
# ==========================

Path(OUTPUT_DIR).mkdir(exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

frame_buffer = deque(maxlen=X_FRAMES + 1)
frame_idx = 0

print("Starting streaming processing...")

# ==========================
# MAIN STREAMING LOOP
# ==========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame_buffer.append(gray)

    # Wait until we have enough future frames
    if len(frame_buffer) < X_FRAMES + 1:
        frame_idx += 1
        continue

    ref_frame = frame_buffer[0]
    future_frames = list(frame_buffer)[1:]

    H, W = ref_frame.shape

    pixel_vel_sum = np.zeros((H, W), dtype=np.float32)
    pixel_vel_count = np.zeros((H, W), dtype=np.int32)

    # ==========================
    # PATCH VELOCITY ESTIMATION
    # ==========================

    for y in range(0, H - PATCH_H + 1, STRIDE_Y):
        for x in range(0, W - PATCH_W + 1, STRIDE_X):

            ref_patch = ref_frame[y:y+PATCH_H, x:x+PATCH_W]
            ref_patch = ref_patch - ref_patch.mean()

            best_corr = -1.0
            best_disp = 0

            for disp in range(-MAX_DISPLACEMENT, MAX_DISPLACEMENT + 1):
                corr_vals = []

                for k in range(X_FRAMES):
                    y_shift = y + disp
                    if y_shift < 0 or y_shift + PATCH_H > H:
                        continue

                    tgt_patch = future_frames[k][
                        y_shift:y_shift+PATCH_H,
                        x:x+PATCH_W
                    ]
                    tgt_patch = tgt_patch - tgt_patch.mean()

                    num = np.sum(ref_patch * tgt_patch)
                    den = np.sqrt(
                        np.sum(ref_patch**2) *
                        np.sum(tgt_patch**2)
                    ) + EPS

                    corr_vals.append(num / den)

                if not corr_vals:
                    continue

                mean_corr = np.mean(corr_vals)

                if mean_corr > best_corr:
                    best_corr = mean_corr
                    best_disp = disp

            if best_corr < CORR_THRESHOLD:
                continue

            velocity = best_disp / X_FRAMES

            pixel_vel_sum[y:y+PATCH_H, x:x+PATCH_W] += velocity
            pixel_vel_count[y:y+PATCH_H, x:x+PATCH_W] += 1

    # ==========================
    # PIXEL-WISE RECONSTRUCTION
    # ==========================

    valid_mask = pixel_vel_count > 0
    pixel_velocity = np.zeros((H, W), dtype=np.float32)
    pixel_velocity[valid_mask] = (
        pixel_vel_sum[valid_mask] / pixel_vel_count[valid_mask]
    )

    vmax = np.percentile(np.abs(pixel_velocity[valid_mask]), 99)
    velocity_norm = np.clip(pixel_velocity / (vmax + EPS), -1, 1)

    # ==========================
    # VISUALIZATION
    # ==========================

    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    color_img[..., 0] = np.clip(255 * velocity_norm, 0, 255)     # Red (up)
    color_img[..., 2] = np.clip(255 * -velocity_norm, 0, 255)    # Blue (down)

    out_file = Path(OUTPUT_DIR) / f"vel_{frame_idx:05d}.png"
    cv2.imwrite(str(out_file), color_img)

    print(f"[Frame {frame_idx}] Saved {out_file}")

    frame_idx += 1

cap.release()
print("Processing complete.")
