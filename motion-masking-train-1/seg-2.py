import cv2
import numpy as np
from collections import deque
from pathlib import Path

# ==========================
# USER PARAMETERS
# ==========================

VIDEO_PATH = "illusion-videos/rollingball.mp4"
OUTPUT_DIR = "output/" + VIDEO_PATH[16:-4] + "_algo2_energy"

# Save per-frame tensors for training
SAVE_TENSORS = True
SAVE_SPEED = True

X_FRAMES = 4

PATCH_H = 8
PATCH_W = 8
STRIDE_Y = 4
STRIDE_X = 4

MAX_DISPLACEMENT = 5
CORR_THRESHOLD = 0.2
EPS = 1e-6

# ==========================
# MOTION AXES (illusion-safe)
# ==========================

MOTION_AXES = [
    (0,  1),   # down
    (0, -1),   # up
    (1,  0),   # right
    (-1, 0),   # left
]

# ==========================
# SETUP
# ==========================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

frame_buffer = deque(maxlen=X_FRAMES + 1)
frame_idx = 0

print("Starting illusion-aware motion processing...")

# ==========================
# MAIN LOOP
# ==========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame_buffer.append(gray)

    if len(frame_buffer) < X_FRAMES + 1:
        frame_idx += 1
        continue

    ref_frame = frame_buffer[0]
    future_frames = list(frame_buffer)[1:]

    H, W = ref_frame.shape

    # --- pixel accumulators ---
    pixel_vx_sum = np.zeros((H, W), dtype=np.float32)
    pixel_vy_sum = np.zeros((H, W), dtype=np.float32)
    pixel_energy_sum = np.zeros((H, W), dtype=np.float32)
    pixel_count = np.zeros((H, W), dtype=np.int32)

    # ==========================
    # PATCH MOTION ENERGY
    # ==========================

    for y in range(0, H - PATCH_H + 1, STRIDE_Y):
        for x in range(0, W - PATCH_W + 1, STRIDE_X):

            ref_patch = ref_frame[y:y+PATCH_H, x:x+PATCH_W]
            ref_patch = ref_patch - ref_patch.mean()

            for ax, ay in MOTION_AXES:

                best_corr = -1.0
                best_disp = 0

                for disp in range(-MAX_DISPLACEMENT, MAX_DISPLACEMENT + 1):

                    corr_vals = []

                    for k in range(X_FRAMES):
                        y2 = y + disp * ay
                        x2 = x + disp * ax

                        if y2 < 0 or y2 + PATCH_H > H:
                            continue
                        if x2 < 0 or x2 + PATCH_W > W:
                            continue

                        tgt_patch = future_frames[k][
                            y2:y2+PATCH_H,
                            x2:x2+PATCH_W
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

                # -------- MOTION ENERGY --------
                velocity = best_disp / X_FRAMES
                energy = best_corr ** 2   # confidence weighting

                vx = ax * velocity * energy
                vy = ay * velocity * energy

                pixel_vx_sum[y:y+PATCH_H, x:x+PATCH_W] += vx
                pixel_vy_sum[y:y+PATCH_H, x:x+PATCH_W] += vy
                pixel_energy_sum[y:y+PATCH_H, x:x+PATCH_W] += energy
                pixel_count[y:y+PATCH_H, x:x+PATCH_W] += 1

    # ==========================
    # PIXEL AGGREGATION
    # ==========================

    valid = pixel_energy_sum > 0

    vx = np.zeros((H, W), dtype=np.float32)
    vy = np.zeros((H, W), dtype=np.float32)

    vx[valid] = pixel_vx_sum[valid] / (pixel_energy_sum[valid] + EPS)
    vy[valid] = pixel_vy_sum[valid] / (pixel_energy_sum[valid] + EPS)

    speed = np.sqrt(vx**2 + vy**2)

    # ==========================
    # VISUALIZATION (illusion-friendly)
    # ==========================

    vmax = np.percentile(speed[valid], 99) + EPS

    # --- segmentation-like vertical motion map ---
    norm_v = np.clip(vy / vmax, -1, 1)

    color = np.zeros((H, W, 3), dtype=np.uint8)
    color[..., 0] = np.clip(255 * norm_v, 0, 255)      # red = down
    color[..., 2] = np.clip(255 * -norm_v, 0, 255)     # blue = up

    # ==========================
    # SAVE (per-frame sample dir)
    # ==========================

    frame_dir = Path(OUTPUT_DIR) / f"frame_{frame_idx:05d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    vis_file = frame_dir / "vis.png"
    cv2.imwrite(str(vis_file), color)

    if SAVE_TENSORS:
        # Store float32 arrays; consumers can normalize/scale later.
        np.save(str(frame_dir / "vx.npy"), vx.astype(np.float32))
        np.save(str(frame_dir / "vy.npy"), vy.astype(np.float32))
        np.save(str(frame_dir / "energy.npy"), pixel_energy_sum.astype(np.float32))
        if SAVE_SPEED:
            np.save(str(frame_dir / "speed.npy"), speed.astype(np.float32))

    print(f"[Frame {frame_idx}] Saved {frame_dir}")
    frame_idx += 1

cap.release()
print("Processing complete.")
