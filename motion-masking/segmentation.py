import cv2
import numpy as np
from collections import deque
from pathlib import Path

# ==========================
# USER PARAMETERS
# ==========================

VIDEO_PATH = "illusion-videos/test_circle.mp4"
OUTPUT_DIR = "output"+VIDEO_PATH[15:]
X_FRAMES = 4

PATCH_H = 8
PATCH_W = 8
STRIDE_Y = 8
STRIDE_X = 8

MAX_DISPLACEMENT = 3
CORR_THRESHOLD = 0.25
EPS = 1e-6

SKIP_N = 3   # <<< USE ONLY EVERY 3RD FRAME

# ==========================
# SETUP
# ==========================

Path(OUTPUT_DIR).mkdir(exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

frame_buffer = deque(maxlen=X_FRAMES + 1)
frame_idx = 0        # index of processed frames
raw_idx = 0          # index of all decoded frames

print("Starting streaming processing...")

# ==========================
# MAIN STREAMING LOOP
# ==========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ==========================
    # USE ONLY EVERY 3RD FRAME
    # ==========================
    if raw_idx % SKIP_N != 0:
        raw_idx += 1
        continue

    raw_idx += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame_buffer.append(gray)

    if len(frame_buffer) < X_FRAMES + 1:
        frame_idx += 1
        continue

    ref_frame = frame_buffer[0]
    future_frames = list(frame_buffer)[1:]

    H, W = ref_frame.shape

    pixel_vel_sum_x = np.zeros((H, W), dtype=np.float32)
    pixel_vel_sum_y = np.zeros((H, W), dtype=np.float32)
    pixel_vel_count = np.zeros((H, W), dtype=np.int32)

    # ==========================
    # PATCH VELOCITY ESTIMATION
    # ==========================

    for y in range(0, H - PATCH_H + 1, STRIDE_Y):
        for x in range(0, W - PATCH_W + 1, STRIDE_X):

            ref_patch = ref_frame[y:y+PATCH_H, x:x+PATCH_W]
            ref_patch -= ref_patch.mean()

            best_corr = -1.0
            best_dx = 0
            best_dy = 0

            for dy in range(-MAX_DISPLACEMENT, MAX_DISPLACEMENT + 1):
                for dx in range(-MAX_DISPLACEMENT, MAX_DISPLACEMENT + 1):

                    corr_vals = []

                    for k in range(X_FRAMES):
                        y_shift = y + dy
                        x_shift = x + dx

                        if (
                            y_shift < 0 or y_shift + PATCH_H > H or
                            x_shift < 0 or x_shift + PATCH_W > W
                        ):
                            continue

                        tgt_patch = future_frames[k][
                            y_shift:y_shift+PATCH_H,
                            x_shift:x_shift+PATCH_W
                        ]
                        tgt_patch -= tgt_patch.mean()

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
                        best_dx = dx
                        best_dy = dy

            if best_corr < CORR_THRESHOLD:
                continue

            vx = best_dx / X_FRAMES
            vy = best_dy / X_FRAMES

            pixel_vel_sum_x[y:y+PATCH_H, x:x+PATCH_W] += vx
            pixel_vel_sum_y[y:y+PATCH_H, x:x+PATCH_W] += vy
            pixel_vel_count[y:y+PATCH_H, x:x+PATCH_W] += 1

    # ==========================
    # PIXEL-WISE RECONSTRUCTION
    # ==========================

    valid = pixel_vel_count > 0

    vx = np.zeros((H, W), dtype=np.float32)
    vy = np.zeros((H, W), dtype=np.float32)

    vx[valid] = pixel_vel_sum_x[valid] / pixel_vel_count[valid]
    vy[valid] = pixel_vel_sum_y[valid] / pixel_vel_count[valid]

    mag = np.sqrt(vx**2 + vy**2)
    vmax = np.percentile(mag[valid], 99)

    mag_norm = np.clip(mag / (vmax + EPS), 0, 1)
    angle = np.arctan2(vy, vx)

    # ==========================
    # VISUALIZATION (HSV FLOW)
    # ==========================

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (255 * mag_norm).astype(np.uint8)

    color_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out_file = Path(OUTPUT_DIR) / f"vel_{frame_idx:05d}.png"
    cv2.imwrite(str(out_file), color_img)

    print(f"[Frame {frame_idx}] Saved {out_file}")
    frame_idx += 1

cap.release()
print("Processing complete.")
