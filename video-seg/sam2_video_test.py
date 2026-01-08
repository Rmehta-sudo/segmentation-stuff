import sys, os, cv2, torch, numpy as np
sys.path.append("../sam2")

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2_video_predictor

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------- Hydra fix ----------------
GlobalHydra.instance().clear()
with initialize_config_dir(
    config_dir=os.path.abspath("../sam2/sam2/configs"),
    job_name="sam2_video",
    version_base=None,
):
    predictor = build_sam2_video_predictor(
        config_file="sam2.1/sam2.1_hiera_b+",
        ckpt_path="../checkpoints/sam2/sam2.1_hiera_base_plus.pt",
        device=device,
    )
# ------------------------------------------

print("Building SAM-2 video predictor...")
# predictor is already built above

frames_dir = "frames-motion_illusion_RACHIT"
out_dir = "output/sam2/"+frames_dir[7:]
os.makedirs(out_dir, exist_ok=True)

print("Loading frames...")
frame_files = sorted(os.listdir(frames_dir))
print(f"Found {len(frame_files)} frames")

h, w = cv2.imread(os.path.join(frames_dir, frame_files[0])).shape[:2]
print(f"Frame dimensions: {w}x{h}")

# Initialize video tracking
print("Initializing video tracking...")
inference_state = predictor.init_state(video_path=frames_dir)

# Add the first frame with the predicted mask
obj_id = 1
point_coords = np.array([[w//2, h//2]])
point_labels = np.array([1])

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=obj_id,
    points=point_coords,
    labels=point_labels,
)

print("Propagating masks through video...")
# Propagate through all frames
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Save visualizations
print("Saving results...")
for frame_idx, fname in enumerate(frame_files):
    frame = cv2.imread(os.path.join(frames_dir, fname))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get mask for this frame
    if frame_idx in video_segments and obj_id in video_segments[frame_idx]:
        mask = video_segments[frame_idx][obj_id]
        if mask.ndim > 2:
            mask = mask.squeeze()
    else:
        mask = np.zeros((h, w), dtype=bool)
    
    # Create overlay
    overlay = frame.copy()
    overlay[mask] = [0, 255, 0]  # Green overlay
    out_frame = (0.6 * frame + 0.4 * overlay).astype("uint8")
    
    # Save result
    out_path = f"{out_dir}/{frame_idx:05d}.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

print(f"SAM-2 results saved to {out_dir}/")
print("SAM-2 video segmentation complete!")