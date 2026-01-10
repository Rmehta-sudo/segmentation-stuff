from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import MotionTeacherDataset, Sample
from model import Tiny3DMotionNet


class TorchWrapper(Dataset):
    """Wraps MotionTeacherDataset to output torch tensors."""

    def __init__(self, ds: MotionTeacherDataset):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        s: Sample = self.ds[idx]
        clip = torch.from_numpy(s.clip)  # (T,1,H,W)
        target = torch.from_numpy(np.stack([s.vx, s.vy], axis=0))  # (2,H,W)
        energy = torch.from_numpy(s.energy)  # (H,W)
        return clip, target, energy, s.meta


def collate(batch):
    clips, targets, energies, metas = zip(*batch)
    clips = torch.stack(clips, dim=0)  # (B,T,1,H,W)
    targets = torch.stack(targets, dim=0)  # (B,2,H,W)
    energies = torch.stack(energies, dim=0)  # (B,H,W)
    return clips, targets, energies, metas


@dataclass
class TrainConfig:
    x_frames: int = 4
    batch_size: int = 2
    lr: float = 1e-3
    epochs: int = 1
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def energy_weighted_l1(pred: torch.Tensor, target: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    """pred/target: (B,2,H,W), energy: (B,H,W)."""
    # Normalize weights per-batch so loss scale is stable.
    w = energy.clamp_min(0.0)
    w = w / (w.mean(dim=(1, 2), keepdim=True) + 1e-6)
    w = w.unsqueeze(1)  # (B,1,H,W)
    return (w * (pred - target).abs()).mean()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Tiny3DMotionNet on teacher-extracted motion tensors.")

    p.add_argument(
        "--type",
        type=str,
        default="auto",
        choices=["auto", "illusion"],
        help=(
            "Dataset preset. 'auto' discovers output/*_algo2_energy and infers videos; "
            "'illusion' uses the original 4 videos + their teacher dirs."
        ),
    )

    p.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory for video paths (default: current directory).",
    )
    p.add_argument(
        "--videos",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of videos (if not using auto-discovery).",
    )
    p.add_argument(
        "--teacher-dirs",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of teacher dirs (if not using auto-discovery).",
    )
    p.add_argument("--x-frames", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--max-samples-per-video",
        type=int,
        default=3,
        help="How many teacher frames to use per video (default: 3).",
    )
    p.add_argument("--out", type=str, default="runs/motion_teacher")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root)

    def infer_video_from_teacher_dir(td: Path) -> str | None:
        # td like output/003_text_motion_000_algo2_energy
        name = td.name
        if name.endswith("_algo2_energy"):
            stem = name[: -len("_algo2_energy")]
        else:
            stem = name

        # Drop numeric prefix if present: 003_foo -> foo
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            stem2 = parts[1]
        else:
            stem2 = stem

        # Search common locations.
        candidates = [
            root / "illusion-videos" / f"{stem2}.mp4",
            root / "illusion-videos/dataset_100" / f"{stem2}.mp4",
            root / "illusion-videos/dataset_60" / f"{stem2}.mp4",
        ]
        for c in candidates:
            if c.exists():
                return str(c.relative_to(root))
        return None

    if args.type == "auto":
        # Discover all numbered teacher dirs, sorted.
        all_teacher_dirs = sorted((root / "output").glob("*_algo2_energy"))
        if not all_teacher_dirs:
            raise RuntimeError(
                "No teacher dirs found under output/*_algo2_energy. Run extract_teacher_batch.py first."
            )

        videos: list[str] = []
        teacher_dirs: list[str] = []
        missing_videos: list[str] = []

        for td in all_teacher_dirs:
            v = infer_video_from_teacher_dir(td)
            if v is None:
                missing_videos.append(str(td))
                continue
            videos.append(v)
            teacher_dirs.append(str(td.relative_to(root)))

        if missing_videos:
            raise RuntimeError(
                "Found teacher dirs but couldn't infer video path for:\n" + "\n".join(missing_videos)
            )

        args.videos = videos
        args.teacher_dirs = teacher_dirs

    elif args.type == "illusion":
        args.videos = [
            "illusion-videos/motion_illusion_IlluSION.mp4",
            "illusion-videos/motion_illusion_RACHIT.mp4",
            "illusion-videos/motion_illusion_Random.mp4",
            "illusion-videos/motion_illusion_testup.mp4",
        ]
        args.teacher_dirs = [
            "output/motion_illusion_IlluSION_algo2_energy",
            "output/motion_illusion_RACHIT_algo2_energy",
            "output/motion_illusion_Random_algo2_energy",
            "output/motion_illusion_testup_algo2_energy",
        ]

    # User explicit override (must be complete lists)
    if args.videos is None or args.teacher_dirs is None:
        raise RuntimeError("Internal: args.videos/teacher_dirs were not set")
    if len(args.videos) != len(args.teacher_dirs):
        raise ValueError("videos and teacher-dirs must have same length")

    cfg = TrainConfig(
        x_frames=args.x_frames,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_workers=args.num_workers,
    )

    teacher_dirs = [root / d for d in args.teacher_dirs]

    # basic existence check early
    missing = [str(d) for d in teacher_dirs if not d.exists()]
    if missing:
        raise RuntimeError(
            "Missing teacher dirs (run seg-2.py on these videos first):\n" + "\n".join(missing)
        )

    ds = MotionTeacherDataset(
        root_dir=root,
        video_paths=args.videos,
        teacher_output_dirs=args.teacher_dirs,
        x_frames=cfg.x_frames,
        max_samples_per_video=args.max_samples_per_video,
    )
    tds = TorchWrapper(ds)

    dl = DataLoader(
        tds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        collate_fn=collate,
    )

    model = Tiny3DMotionNet(x_frames=cfg.x_frames).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    model.train()
    for epoch in range(cfg.epochs):
        for clips, targets, energies, _metas in dl:
            clips = clips.to(cfg.device)
            targets = targets.to(cfg.device)
            energies = energies.to(cfg.device)

            pred = model(clips)
            loss = energy_weighted_l1(pred, targets, energies)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 10 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.6f}")

            step += 1

        ckpt = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "x_frames": cfg.x_frames,
        }
        torch.save(ckpt, out_dir / f"ckpt_epoch_{epoch:03d}.pt")

    print(f"Done. Checkpoints in: {out_dir}")


if __name__ == "__main__":
    main()
