"""Batch teacher extraction for a directory of videos.

Goal: for each *.mp4 in a directory, run the motion-energy teacher and save
per-frame supervision tensors (vx/vy/energy[/speed]) + vis.png into:
  output/<video_stem>_algo2_energy/frame_00004/...

We stop after a small number of frames (default: 4 frames, starting at frame 4).

Run from this directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _make_motion_axes(num_axes: int) -> list[tuple[float, float]]:
    """Return a set of (ax, ay) directions.

    For 12 axes we use directions at 0°,30°,...,330° (image coordinates: +x right, +y down).
    """
    if num_axes == 4:
        return [(0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0)]
    if num_axes == 8:
        return [
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ]
    if num_axes == 12:
        axes: list[tuple[float, float]] = []
        for deg in range(0, 360, 30):
            rad = np.deg2rad(deg)
            axes.append((float(np.cos(rad)), float(np.sin(rad))))
        return axes
    raise ValueError("num_axes must be one of: 4, 8, 12")


def process_video(
    video_path: Path,
    output_dir: Path,
    x_frames: int,
    patch_h: int,
    patch_w: int,
    stride_y: int,
    stride_x: int,
    max_displacement: int,
    corr_threshold: float,
    start_frame: int,
    num_frames: int,
    save_speed: bool,
    num_axes: int,
) -> None:
    motion_axes = _make_motion_axes(num_axes)
    eps = 1e-6

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # We only need frames up to the last reference frame we plan to save,
    # plus the lookahead window required to compute it.
    last_ref_idx = start_frame + num_frames - 1
    max_needed = last_ref_idx + x_frames

    from collections import deque

    buf = deque(maxlen=x_frames + 1)
    frame_idx = 0
    saved = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    while frame_idx <= max_needed:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        buf.append(gray)

        if len(buf) < x_frames + 1:
            frame_idx += 1
            continue

        # The reference frame index corresponding to buf[0].
        # When we've read up to frame_idx (0-based), buf contains frames:
        #   [frame_idx-x_frames, ..., frame_idx]
        ref_idx = frame_idx - x_frames

        # Save supervision based on reference frame index.
        if ref_idx < start_frame:
            frame_idx += 1
            continue

        if saved >= num_frames:
            break

        ref_frame = buf[0]
        future_frames = list(buf)[1:]
        H, W = ref_frame.shape

        pixel_vx_sum = np.zeros((H, W), dtype=np.float32)
        pixel_vy_sum = np.zeros((H, W), dtype=np.float32)
        pixel_energy_sum = np.zeros((H, W), dtype=np.float32)

        for y in range(0, H - patch_h + 1, stride_y):
            for x in range(0, W - patch_w + 1, stride_x):
                ref_patch = ref_frame[y : y + patch_h, x : x + patch_w]
                ref_patch = ref_patch - ref_patch.mean()

                for ax, ay in motion_axes:
                    best_corr = -1.0
                    best_disp = 0

                    for disp in range(-max_displacement, max_displacement + 1):
                        corr_vals = []
                        for k in range(x_frames):
                            # Allow fractional direction vectors; index with rounded pixel shifts.
                            y2 = int(round(y + disp * ay))
                            x2 = int(round(x + disp * ax))
                            if y2 < 0 or y2 + patch_h > H:
                                continue
                            if x2 < 0 or x2 + patch_w > W:
                                continue

                            tgt_patch = future_frames[k][y2 : y2 + patch_h, x2 : x2 + patch_w]
                            tgt_patch = tgt_patch - tgt_patch.mean()

                            num = np.sum(ref_patch * tgt_patch)
                            den = (
                                np.sqrt(np.sum(ref_patch ** 2) * np.sum(tgt_patch ** 2)) + eps
                            )
                            corr_vals.append(num / den)

                        if not corr_vals:
                            continue

                        mean_corr = float(np.mean(corr_vals))
                        if mean_corr > best_corr:
                            best_corr = mean_corr
                            best_disp = disp

                    if best_corr < corr_threshold:
                        continue

                    velocity = best_disp / x_frames
                    energy = best_corr**2

                    vx = ax * velocity * energy
                    vy = ay * velocity * energy

                    pixel_vx_sum[y : y + patch_h, x : x + patch_w] += vx
                    pixel_vy_sum[y : y + patch_h, x : x + patch_w] += vy
                    pixel_energy_sum[y : y + patch_h, x : x + patch_w] += energy

        valid = pixel_energy_sum > 0
        vx = np.zeros((H, W), dtype=np.float32)
        vy = np.zeros((H, W), dtype=np.float32)
        vx[valid] = pixel_vx_sum[valid] / (pixel_energy_sum[valid] + eps)
        vy[valid] = pixel_vy_sum[valid] / (pixel_energy_sum[valid] + eps)
        speed = np.sqrt(vx**2 + vy**2)

        vmax = np.percentile(speed[valid], 99) + eps
        norm_v = np.clip(vy / vmax, -1, 1)

        color = np.zeros((H, W, 3), dtype=np.uint8)
        color[..., 0] = np.clip(255 * norm_v, 0, 255)
        color[..., 2] = np.clip(255 * -norm_v, 0, 255)

        frame_dir = output_dir / f"frame_{ref_idx:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(frame_dir / "vis.png"), color)
        np.save(str(frame_dir / "vx.npy"), vx.astype(np.float32))
        np.save(str(frame_dir / "vy.npy"), vy.astype(np.float32))
        np.save(str(frame_dir / "energy.npy"), pixel_energy_sum.astype(np.float32))
        if save_speed:
            np.save(str(frame_dir / "speed.npy"), speed.astype(np.float32))

        saved += 1
        if saved % 5 == 0:
            print(f"  saved {saved}/{num_frames} frames for {video_path.name}")

        frame_idx += 1

    cap.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--video-dir",
        type=str,
        default="illusion-videos/dataset_100",
        help="Directory containing generated mp4s (dataset_100).",
    )
    p.add_argument(
        "--extra-videos",
        type=str,
        nargs="*",
        default=[
            "illusion-videos/motion_illusion_IlluSION.mp4",
            "illusion-videos/motion_illusion_RACHIT.mp4",
            "illusion-videos/motion_illusion_Random.mp4",
            "illusion-videos/motion_illusion_testup.mp4",
        ],
        help="Additional mp4s to include (e.g., the 4 original illusion videos).",
    )
    p.add_argument("--out-root", type=str, default="output")
    p.add_argument("--x-frames", type=int, default=4)

    p.add_argument(
        "--num-axes",
        type=int,
        default=12,
        choices=[4, 8, 12],
        help="How many motion directions to search (default: 12).",
    )

    p.add_argument("--patch-h", type=int, default=8)
    p.add_argument("--patch-w", type=int, default=8)
    p.add_argument("--stride-y", type=int, default=4)
    p.add_argument("--stride-x", type=int, default=4)
    p.add_argument("--max-displacement", type=int, default=5)
    p.add_argument("--corr-threshold", type=float, default=0.2)

    p.add_argument("--start-frame", type=int, default=4)
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--save-speed", action="store_true")
    p.add_argument("--limit-videos", type=int, default=None)
    p.add_argument(
        "--no-extra",
        action="store_true",
        help="Disable including --extra-videos (only process --video-dir).",
    )
    p.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4"],
        help=(
            "Run a staged subset in this order: "
            "(1) RACHIT only, (2) first 10 dataset videos, (3) other 3 manual illusion videos, "
            "(4) remaining dataset videos. Use --stage all to run all stages in one go."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    video_dir = Path(args.video_dir)
    out_root = Path(args.out_root)

    dataset_videos = sorted(video_dir.glob("*.mp4"))
    if args.limit_videos is not None:
        dataset_videos = dataset_videos[: int(args.limit_videos)]

    manual_videos: list[Path] = []
    if not args.no_extra:
        for p in args.extra_videos:
            vp = Path(p)
            if vp.exists():
                manual_videos.append(vp)
            else:
                print(f"[warn] extra video not found, skipping: {vp}")

    rachit = [v for v in manual_videos if v.stem == "motion_illusion_RACHIT"]
    others = [v for v in manual_videos if v.stem != "motion_illusion_RACHIT"]
    first10 = dataset_videos[:10]
    rest90 = dataset_videos[10:]

    # Stage selection
    videos: list[Path]
    if args.stage == "all":
        # Run all stages in the requested order.
        videos = rachit + first10 + others + rest90
    elif args.stage == "1":
        videos = rachit
    elif args.stage == "2":
        videos = first10
    elif args.stage == "3":
        videos = others
    else:
        videos = rest90

    # De-dup by absolute path (preserve order)
    seen: set[str] = set()
    uniq_videos: list[Path] = []
    for vp in videos:
        key = str(vp.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq_videos.append(vp)
    videos = uniq_videos

    if not videos:
        raise RuntimeError(
            f"No videos found. Looked in {video_dir} and extra-videos list."
        )

    for i, vp in enumerate(videos, start=1):
        # Requested naming: number_text (stable, simple)
        out_dir = out_root / f"{i:03d}_{vp.stem}_algo2_energy"
        print(f"[{i:03d}/{len(videos)}] {vp} -> {out_dir}")
        process_video(
            video_path=vp,
            output_dir=out_dir,
            x_frames=args.x_frames,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            stride_y=args.stride_y,
            stride_x=args.stride_x,
            max_displacement=args.max_displacement,
            corr_threshold=args.corr_threshold,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            save_speed=args.save_speed,
            num_axes=args.num_axes,
        )

    print("Done.")


if __name__ == "__main__":
    main()
