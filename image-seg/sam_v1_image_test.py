import sys
sys.path.append("../segment-anything")

import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# =======================
# CONFIG
# =======================
INPUT_IMAGE = "images/cars.jpg"
OUTPUT_DIR = "output"
CHECKPOINT = "../sam-checkpoints/sam_vit_b_01ec64.pth"

assert os.path.exists(CHECKPOINT), "Checkpoint missing"

# =======================
# UTILS
# =======================
def save_mask_overlay(image, mask, out_path, color=(255, 0, 0), alpha=0.4):
    overlay = image.copy()
    color_mask = image.copy()
    color_mask[mask > 0] = color
    overlay = (overlay * (1 - alpha) + color_mask * alpha).astype("uint8")
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# =======================
# SETUP
# =======================
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT).to(device)
predictor = SamPredictor(sam)

# =======================
# IMAGE
# =======================
image = cv2.imread(INPUT_IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w = image.shape[:2]
if max(h, w) > 1024:
    scale = 1024 / max(h, w)
    image = cv2.resize(image, (int(w * scale), int(h * scale)))

predictor.set_image(image)

h, w = image.shape[:2]
point_coords = np.array([[0.5 * w, 0.5 * h]])
point_labels = np.array([1])

# =======================
# PREDICT
# =======================
masks, scores, _ = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True
)

print("SAM v1 scores:", scores)

# =======================
# SAVE TOP-3
# =======================
# =======================
# SAVE TOP-3 MASKS (SAFE)
# =======================
os.makedirs("output", exist_ok=True)

base = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]
order = np.argsort(scores)[::-1]

for rank, idx in enumerate(order[:3], start=1):
    mask = masks[idx]

    # force boolean mask (IMPORTANT)
    mask = mask.astype(np.uint8)

    out_path = f"output/{base}_sam_v1_rank{rank}_score{scores[idx]:.3f}.png"
    save_mask_overlay(
        image=image,
        mask=mask,
        out_path=out_path,
        color=(255, 0, 0),
    )

    print("Saved:", out_path)

