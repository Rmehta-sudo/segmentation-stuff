import sys
import os
import cv2
import torch
import numpy as np

sys.path.append("../sam2")

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# =======================
# CONFIG
# =======================
INPUT_IMAGE = "images/cars.jpg"
OUTPUT_DIR = "output"
CHECKPOINT = "../checkpoints/sam2/sam2.1_hiera_base_plus.pt"

# =======================
# UTILS
# =======================
def save_mask_overlay(image, mask, out_path, color=(0, 255, 0), alpha=0.4):
    overlay = image.copy()
    color_mask = image.copy()
    color_mask[mask > 0] = color
    overlay = (overlay * (1 - alpha) + color_mask * alpha).astype("uint8")
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# =======================
# SETUP
# =======================
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

GlobalHydra.instance().clear()

with initialize_config_dir(
    config_dir=os.path.abspath("../sam2/sam2/configs"),
    job_name="sam2_image_test",
    version_base=None,
):
    sam = build_sam2(
        config_file="sam2.1/sam2.1_hiera_b+",
        checkpoint=CHECKPOINT,
        device=device,
    )

predictor = SAM2ImagePredictor(sam)

# =======================
# IMAGE
# =======================
image = cv2.imread(INPUT_IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    multimask_output=True,
)

print("SAM-2 scores:", scores)
print("Masks shape:", masks.shape)

# =======================
# SAVE TOP-3
# =======================
os.makedirs(OUTPUT_DIR, exist_ok=True)
base = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]

order = np.argsort(scores)[::-1]
for rank, idx in enumerate(order[:3], start=1):
    out_name = f"{OUTPUT_DIR}/{base}_sam2_rank{rank}_score{scores[idx]:.3f}.png"
    save_mask_overlay(image, masks[idx], out_name, color=(0, 255, 0))
