from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Sample:
    """One supervised sample.

    clip: float32 array shaped (T, 1, H, W) in range [0, 1]
    vx/vy/energy: float32 arrays shaped (H, W)
    """

    clip: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    energy: np.ndarray
    meta: Dict[str, object]


def _sorted_frame_dirs(video_output_dir: Path) -> List[Path]:
    return sorted([p for p in video_output_dir.glob("frame_*") if p.is_dir()])


def _load_teacher_arrays(frame_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vx = np.load(frame_dir / "vx.npy").astype(np.float32)
    vy = np.load(frame_dir / "vy.npy").astype(np.float32)
    energy = np.load(frame_dir / "energy.npy").astype(np.float32)
    return vx, vy, energy


def _read_gray_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        raise EOFError
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


class MotionTeacherDataset:
    """Loads clips from videos and targets from per-frame teacher dirs.

    Layout expected per video:
      output/<video>_algo2_energy/frame_00004/{vx.npy,vy.npy,energy.npy}

    For a target frame index i, we return clip frames [i-X_FRAMES, ..., i] (T=X_FRAMES+1)
    and targets from that frame_dir.

    Note: We intentionally keep this dependency-light (numpy + opencv).
    """

    def __init__(
        self,
        root_dir: str | Path,
        video_paths: List[str | Path],
        teacher_output_dirs: List[str | Path],
        x_frames: int = 4,
        max_samples_per_video: int | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.video_paths = [Path(p) for p in video_paths]
        self.teacher_output_dirs = [Path(p) for p in teacher_output_dirs]
        if len(self.video_paths) != len(self.teacher_output_dirs):
            raise ValueError("video_paths and teacher_output_dirs must have same length")

        self.x_frames = int(x_frames)
        if self.x_frames < 1:
            raise ValueError("x_frames must be >= 1")

        self._index: List[Tuple[int, Path, int]] = []
        # (video_idx, frame_dir, frame_idx)

        for vid_i, out_dir in enumerate(self.teacher_output_dirs):
            frame_dirs = _sorted_frame_dirs(out_dir)
            if max_samples_per_video is not None:
                frame_dirs = frame_dirs[: int(max_samples_per_video)]

            for fd in frame_dirs:
                # frame_00004 -> 4
                frame_idx = int(fd.name.split("_")[1])
                self._index.append((vid_i, fd, frame_idx))

        if not self._index:
            raise RuntimeError(
                "No samples found. Did you run the teacher to generate output/<video>_algo2_energy/frame_*/ ?"
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Sample:
        vid_i, frame_dir, frame_idx = self._index[idx]
        video_path = self.root_dir / self.video_paths[vid_i]

        vx, vy, energy = _load_teacher_arrays(frame_dir)
        H, W = vx.shape

        # Load clip frames [frame_idx - x_frames ... frame_idx] inclusive.
        start = frame_idx - self.x_frames
        end = frame_idx
        if start < 0:
            raise IndexError(
                f"Not enough history for frame_idx={frame_idx} with x_frames={self.x_frames}. "
                "Regenerate teacher outputs starting later or increase warmup."
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames: List[np.ndarray] = []
        try:
            for _ in range(start, end + 1):
                frames.append(_read_gray_frame(cap))
        finally:
            cap.release()

        clip = np.stack(frames, axis=0)  # (T, H, W)
        if clip.shape[1] != H or clip.shape[2] != W:
            raise RuntimeError(
                f"Video frame size {clip.shape[1:]} does not match teacher tensors {(H, W)} in {frame_dir}"
            )

        clip = clip[:, None, :, :]  # (T, 1, H, W)

        return Sample(
            clip=clip.astype(np.float32),
            vx=vx,
            vy=vy,
            energy=energy,
            meta={
                "video": str(video_path),
                "frame_dir": str(frame_dir),
                "frame_idx": frame_idx,
            },
        )
