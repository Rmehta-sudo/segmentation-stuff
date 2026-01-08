import sys, os, cv2, torch, numpy as np
sys.path.append("../segment-anything")

from segment_anything import sam_model_registry, SamPredictor

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading SAM-1 model...")
sam = sam_model_registry["vit_b"](
    checkpoint="../sam-checkpoints/sam_vit_b_01ec64.pth"
).to(device)
predictor = SamPredictor(sam)

frames_dir = "frames"
out_dir = "output/sam1"
os.makedirs(out_dir, exist_ok=True)

frame_files = sorted(os.listdir(frames_dir))
print(f"Processing {len(frame_files)} frames with SAM-1...")

for i, fname in enumerate(frame_files):
    print(f"Processing frame {i+1}/{len(frame_files)}: {fname}")
    
    frame = cv2.imread(os.path.join(frames_dir, fname))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    predictor.set_image(frame)

    h, w = frame.shape[:2]
    # Use same click point as SAM-2 for fair comparison
    point_coords = np.array([[w//2, h//2]])
    point_labels = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )

    mask = masks[0]
    overlay = frame.copy()
    overlay[mask > 0] = [255, 0, 0]  # Red overlay
    out_frame = (0.6 * frame + 0.4 * overlay).astype("uint8")

    out_path = f"{out_dir}/{i:05d}.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

print(f"SAM-1 results saved to {out_dir}/")
print("SAM-1 frame-by-frame segmentation complete!")