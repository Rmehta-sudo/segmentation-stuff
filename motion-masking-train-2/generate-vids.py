"""Generate a synthetic motion-illusion dataset.

Creates N grayscale videos where pixels inside a text mask drift in one direction
and pixels outside drift in another direction. The two directions are separated
by at least MIN_ANGLE_DIFF_DEG so the motion boundary is learnable but not
hard-coded (varies per video).

Outputs:
  illusion-videos/dataset_60/<name>.mp4
  illusion-videos/dataset_60/manifest.jsonl

Run from this directory.
"""

from __future__ import annotations

import json
import math
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Dataset parameters
# ----------------------------

W, H = 512, 512
FPS = 30
FRAMES = 120

# Randomized per video
DOT_DENSITY_RANGE = (0.25, 0.60)
# NOTE: The patch-correlation teacher struggles when true motion exceeds its
# search radius (max_displacement / x_frames) and/or becomes too aliased.
# We cap speeds to keep the generated dataset within the teacher's reliable
# regime. If you want to push higher speeds, increase the teacher's
# --max-displacement and/or use a multi-scale/coarse-to-fine teacher.
SPEED_RANGE_PX_PER_FRAME = (1.6, 3.5)

N_VIDEOS = 100
MIN_ANGLE_DIFF_DEG = 80.0

OUT_DIR = Path("illusion-videos/dataset_100")
MANIFEST_PATH = OUT_DIR / "manifest.jsonl"

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _pick_font(font_size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _angle_to_unit_vec(deg: float) -> Tuple[float, float]:
    rad = math.radians(deg)
    # x right, y down (image coordinates)
    return float(math.cos(rad)), float(math.sin(rad))


def _wrap_angle_deg(a: float) -> float:
    return a % 360.0


def _angular_distance_deg(a: float, b: float) -> float:
    d = abs(_wrap_angle_deg(a) - _wrap_angle_deg(b))
    return min(d, 360.0 - d)


def _sample_two_separated_angles(min_sep_deg: float) -> Tuple[float, float]:
    a1 = random.uniform(0.0, 360.0)
    for _ in range(1000):
        a2 = random.uniform(0.0, 360.0)
        if _angular_distance_deg(a1, a2) >= min_sep_deg:
            return a1, a2
    return a1, _wrap_angle_deg(a1 + min_sep_deg)


def _random_text() -> str:
    # Mix random tokens and short word-like chunks.
    if random.random() < 0.6:
        n = random.randint(3, 10)
        alphabet = string.ascii_uppercase + string.digits
        return "".join(random.choice(alphabet) for _ in range(n))

    syllables = [
        "RA",
        "CHI",
        "ME",
        "TA",
        "ILLU",
        "SION",
        "MOT",
        "ION",
        "FLOW",
        "MASK",
        "SEG",
        "PIX",
    ]
    k = random.randint(2, 4)
    return "".join(random.choice(syllables) for _ in range(k))


@dataclass
class VideoSpec:
    name: str
    text: str
    angle_text_deg: float
    angle_bg_deg: float
    speed: float
    dot_density: float
    text_rotation_deg: float
    text_pos_xy: Tuple[int, int]
    font_size: int
    w: int
    h: int
    frames: int
    fps: int
    seed: int


def _make_text_mask(text: str, font_size: int, rotation_deg: float, pos_xy: Tuple[int, int]) -> np.ndarray:
    # Render -> rotate -> paste into full-resolution mask.
    font = _pick_font(font_size)

    tmp = Image.new("L", (W, H), 0)
    dtmp = ImageDraw.Draw(tmp)
    bbox = dtmp.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad = 30
    cw = min(W, tw + 2 * pad)
    ch = min(H, th + 2 * pad)
    text_canvas = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(text_canvas)
    draw.text((pad, pad), text, fill=255, font=font)

    rotated = text_canvas.rotate(rotation_deg, resample=Image.BILINEAR, expand=True)

    mask_img = Image.new("L", (W, H), 0)
    x, y = pos_xy
    x = max(0, min(W - rotated.size[0], x))
    y = max(0, min(H - rotated.size[1], y))
    mask_img.paste(rotated, (x, y))

    return (np.array(mask_img) > 0)


def _init_writer(path: Path) -> cv2.VideoWriter:
    codecs_to_try = [
        cv2.VideoWriter_fourcc(*"mp4v"),
        cv2.VideoWriter_fourcc(*"XVID"),
        cv2.VideoWriter_fourcc(*"MJPG"),
    ]
    for codec in codecs_to_try:
        out = cv2.VideoWriter(str(path), codec, FPS, (W, H), isColor=False)
        if out.isOpened():
            return out
        out.release()
    raise RuntimeError(f"Could not initialize video writer for {path}")


def _shift_noise(noise: np.ndarray, dx: float, dy: float, i: int) -> np.ndarray:
    sx = int(round(i * dx))
    sy = int(round(i * dy))
    out = np.roll(noise, shift=sy, axis=0)
    out = np.roll(out, shift=sx, axis=1)
    return out


def generate_one(spec: VideoSpec) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(spec.seed)
    np.random.seed(spec.seed)

    text_mask = _make_text_mask(
        spec.text,
        font_size=spec.font_size,
        rotation_deg=spec.text_rotation_deg,
        pos_xy=spec.text_pos_xy,
    )

    noise = (np.random.rand(H, W) < spec.dot_density).astype(np.uint8) * 255

    ux_t, uy_t = _angle_to_unit_vec(spec.angle_text_deg)
    ux_b, uy_b = _angle_to_unit_vec(spec.angle_bg_deg)
    dx_t, dy_t = ux_t * spec.speed, uy_t * spec.speed
    dx_b, dy_b = ux_b * spec.speed, uy_b * spec.speed

    out_path = OUT_DIR / f"{spec.name}.mp4"
    out = _init_writer(out_path)
    try:
        for i in range(spec.frames):
            inside = _shift_noise(noise, dx_t, dy_t, i)
            outside = _shift_noise(noise, dx_b, dy_b, i)
            frame = np.where(text_mask, inside, outside).astype(np.uint8)
            out.write(frame)
    finally:
        out.release()

    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(spec.__dict__) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    base_seed = 1337
    random.seed(base_seed)

    for i in range(N_VIDEOS):
        text = _random_text()
        a_text, a_bg = _sample_two_separated_angles(MIN_ANGLE_DIFF_DEG)

        dot_density = random.uniform(*DOT_DENSITY_RANGE)
        speed = random.uniform(*SPEED_RANGE_PX_PER_FRAME)

        font_size = random.randint(70, 220)
        text_rot = random.uniform(-180.0, 180.0)
        pos_x = random.randint(0, W - 1)
        pos_y = random.randint(0, H - 1)

        spec = VideoSpec(
            name=f"text_motion_{i:03d}",
            text=text,
            angle_text_deg=float(a_text),
            angle_bg_deg=float(a_bg),
            speed=float(speed),
            dot_density=float(dot_density),
            text_rotation_deg=float(text_rot),
            text_pos_xy=(int(pos_x), int(pos_y)),
            font_size=int(font_size),
            w=W,
            h=H,
            frames=FRAMES,
            fps=FPS,
            seed=base_seed + i,
        )
        print(
            f"[{i+1:03d}/{N_VIDEOS}] {spec.name} text='{spec.text}' "
            f"angles(text/bg)=({spec.angle_text_deg:.1f}/{spec.angle_bg_deg:.1f}) "
            f"speed={spec.speed:.2f} dens={spec.dot_density:.2f} rot={spec.text_rotation_deg:.1f}"
        )
        generate_one(spec)

    print(f"Done. Videos in {OUT_DIR} and manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()