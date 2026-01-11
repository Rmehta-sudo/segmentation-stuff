"""Extract teacher tensors for the 4 existing illusion videos.

This keeps your initial bootstrap dataset (the 4 hand-made illusion videos)
consistent with the dataset_100 extraction format.

Outputs per video:
  output/<video_stem>_algo2_energy/frame_00004 .. frame_00018/

Run from this directory.
"""

from __future__ import annotations

from pathlib import Path

from extract_teacher_batch import process_video


VIDEOS = [
    Path("illusion-videos/motion_illusion_IlluSION.mp4"),
    Path("illusion-videos/motion_illusion_RACHIT.mp4"),
    Path("illusion-videos/motion_illusion_Random.mp4"),
    Path("illusion-videos/motion_illusion_testup.mp4"),
]


def main() -> None:
    out_root = Path("output")

    for i, vp in enumerate(VIDEOS, start=1):
        if not vp.exists():
            raise RuntimeError(f"Missing video: {vp}")
        out_dir = out_root / f"{vp.stem}_algo2_energy"
        print(f"[{i}/4] {vp} -> {out_dir}")

        process_video(
            video_path=vp,
            output_dir=out_dir,
            x_frames=4,
            patch_h=8,
            patch_w=8,
            stride_y=4,
            stride_x=4,
            max_displacement=5,
            corr_threshold=0.2,
            start_frame=4,
            num_frames=3,
            save_speed=True,
        )

    print("Done.")


if __name__ == "__main__":
    main()
