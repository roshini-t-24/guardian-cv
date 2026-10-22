"""Module E — Deterministic synthetic clinical MP4 generator.

Used ONLY when /videos is empty. These clips are *real rendered video*:
pixel contours, motion fields and illumination genuinely exist in the
MP4 bitstream, so every downstream metric (MOG2 masks, Kalman states,
Farneback flow, telemetry confidence) is still measured from pixels.
Any genuine ward footage dropped into videos/ takes precedence — the
generator never overwrites an existing file.

    clip1_asymmetric_gait.avi   right-knee stiffness / antalgic limp
    clip2_symmetric_baseline.avi  healthy symmetric corridor walk
    clip3_postural_sway.avi     acute mid-corridor balance instability
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FRAME_W, FRAME_H = 480, 360
FPS = 15
N_FRAMES = 135                 # 9.0 s per clip
FOURCC = "mp4v"
FLOOR_Y = int(FRAME_H * 0.72)  # wall / floor boundary
HIP_Y = FLOOR_Y - 96           # thigh(46) + shank(44) + clearance(6)

CLIPS = {
    "clip1_asymmetric_gait.avi": {
        "label": "Asymmetric gait — right knee stiffness",
        "seed": 11,
        "profile": "asymmetric",
    },
    "clip2_symmetric_baseline.avi": {
        "label": "Symmetric baseline corridor walk",
        "seed": 22,
        "profile": "symmetric",
    },
    "clip3_postural_sway.avi": {
        "label": "Acute mid-corridor postural sway",
        "seed": 33,
        "profile": "sway",
    },
}

# Muted clinical clothing tones (BGR) — dark enough to segment on the lit wall.
SKIN_BGR = (96, 128, 168)
CLOTH_NEAR_BGR = (58, 46, 38)   # near-side limbs / torso shirt
CLOTH_FAR_BGR = (78, 62, 50)    # far-side limbs (slightly lighter = depth cue)


def _ip(point) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))


def _corridor_background(rng: np.random.Generator) -> np.ndarray:
    """Static lit hospital corridor with deterministic sensor grain."""
    bg = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)

    # Wall: cool plaster gradient.
    for y in range(FLOOR_Y):
        shade = 208 - int(42 * y / FLOOR_Y)
        bg[y, :] = (shade - 18, shade - 8, shade)

    # Floor: polished vinyl gradient brightening toward the camera.
    for y in range(FLOOR_Y, FRAME_H):
        t = (y - FLOOR_Y) / max(FRAME_H - FLOOR_Y, 1)
        shade = int(92 + 62 * t)
        bg[y, :] = (shade - 26, shade - 12, shade)

    # Skirting + handrail.
    cv2.rectangle(bg, (0, FLOOR_Y - 70), (FRAME_W, FLOOR_Y - 64), (150, 158, 166), -1)
    cv2.line(bg, (0, FLOOR_Y - 6), (FRAME_W, FLOOR_Y - 6), (118, 128, 136), 3)

    # Ward doors.
    for dx in (36, 306):
        cv2.rectangle(bg, (dx, FLOOR_Y - 152), (dx + 76, FLOOR_Y - 6), (126, 110, 94), -1)
        cv2.rectangle(bg, (dx + 6, FLOOR_Y - 146), (dx + 70, FLOOR_Y - 12), (140, 124, 106), -1)
        cv2.circle(bg, (dx + 64, FLOOR_Y - 82), 3, (72, 72, 74), -1)

    # Ceiling light pools.
    for lx in (120, 360):
        cv2.ellipse(bg, (lx, 24), (56, 12), 0, 0, 360, (236, 240, 244), -1)

    # Deterministic sensor grain (fixed for the clip, not per-frame).
    grain = rng.integers(-7, 8, size=(FRAME_H, FRAME_W, 1), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    return bg


def _leg_points(hip, thigh_len: float, shank_len: float,
                thigh_deg: float, knee_flex_deg: float):
    """hip -> knee -> ankle in image coords (y down).

    thigh_deg is measured from vertical, positive = forward (rightward);
    the shank trails the thigh by knee_flex_deg.
    """
    th = np.radians(thigh_deg)
    knee = (hip[0] + thigh_len * np.sin(th), hip[1] + thigh_len * np.cos(th))
    sh = np.radians(thigh_deg - knee_flex_deg)
    ankle = (knee[0] + shank_len * np.sin(sh), knee[1] + shank_len * np.cos(sh))
    return knee, ankle


def _draw_walker(img: np.ndarray, hip, phase: float,
                 right_gain: float = 1.0, arms_up: float = 0.0) -> None:
    """Sagittal-view walking figure. right_gain < 1 stiffens the right knee."""
    thigh_len, shank_len, torso_len = 46.0, 44.0, 52.0
    bob = 3.0 * np.sin(2.0 * phase)
    hip = (hip[0], hip[1] + bob)
    shoulder = (hip[0] + 2.0, hip[1] - torso_len)
    head_c = (shoulder[0] + 3.0, shoulder[1] - 18.0)

    # Legs: far leg first (phase offset 0), near leg offset by pi.
    for offset, gain, colour in (
        (0.0, 1.0, CLOTH_FAR_BGR),
        (np.pi, right_gain, CLOTH_NEAR_BGR),
    ):
        thigh_deg = 26.0 * np.sin(phase + offset)
        knee_flex = 62.0 * gain * max(0.0, np.sin(phase + offset + 1.05)) ** 1.4
        knee, ankle = _leg_points(hip, thigh_len, shank_len, thigh_deg, knee_flex)
        cv2.line(img, _ip(hip), _ip(knee), colour, 11, cv2.LINE_AA)
        cv2.line(img, _ip(knee), _ip(ankle), colour, 9, cv2.LINE_AA)
        foot = (ankle[0] + 10.0, ankle[1] + 4.0)
        cv2.line(img, _ip(ankle), _ip(foot), colour, 7, cv2.LINE_AA)

    # Torso.
    cv2.line(img, _ip(hip), _ip(shoulder), CLOTH_NEAR_BGR, 15, cv2.LINE_AA)

    # Arms: gentle counter-swing; arms_up raises them for the sway clip.
    arm_swing = 20.0 * np.sin(phase)
    for sgn in (-1.0, 1.0):
        hand_low = (
            shoulder[0] + sgn * 7.0 + 14.0 * np.sin(np.radians(sgn * arm_swing)),
            shoulder[1] + 38.0,
        )
        hand_high = (shoulder[0] + sgn * 36.0, shoulder[1] + 10.0)
        hand = (
            hand_low[0] + (hand_high[0] - hand_low[0]) * arms_up,
            hand_low[1] + (hand_high[1] - hand_low[1]) * arms_up,
        )
        cv2.line(img, _ip(shoulder), _ip(hand), SKIN_BGR, 8, cv2.LINE_AA)

    # Head (redacted downstream by the Zero-PII shield).
    cv2.circle(img, _ip(head_c), 15, SKIN_BGR, -1, lineType=cv2.LINE_AA)


def _render_clip(out_path: Path, seed: int, profile: str) -> None:
    """Render one deterministic 9 s clinical clip at 480x360 @ 15 fps."""
    rng = np.random.default_rng(seed)
    bg = _corridor_background(rng)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*("FFV1" if out_path.suffix == ".avi" else FOURCC)), FPS, (FRAME_W, FRAME_H)
    )

    for t in range(N_FRAMES):
        frame = bg.copy()

        if profile == "sway":
            # Standing mid-corridor: sway amplitude ramps in, one stumble
            # transient around t=78, arms rise as instability grows.
            ramp = min(1.0, t / 45.0)
            stumble = 14.0 * np.exp(-(((t - 78) / 4.0) ** 2))
            hip_x = 236.0 + 26.0 * ramp * np.sin(t * 0.11) + stumble
            hip_y = HIP_Y + 3.0 * ramp * np.sin(t * 0.23)
            arms_up = float(np.clip(1.6 * ramp * abs(np.sin(t * 0.11)), 0.0, 1.0))
            _draw_walker(frame, (hip_x, hip_y), phase=0.35, arms_up=arms_up)
        else:
            # Transit left -> right across the corridor camera's field.
            progress = (t + 8) / (N_FRAMES + 16)
            hip_x = -40.0 + progress * (FRAME_W + 80.0)
            cadence = 0.28 if profile == "asymmetric" else 0.30
            right_gain = 0.30 if profile == "asymmetric" else 1.0
            _draw_walker(frame, (hip_x, float(HIP_Y)),
                         phase=t * cadence, right_gain=right_gain)

        writer.write(frame)
    writer.release()


def ensure_clips(video_dir: str | Path = "videos") -> list[Path]:
    """Create any missing clinical clips. Never overwrites existing footage."""
    out_dir = Path(video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for name, meta in CLIPS.items():
        target = out_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        _render_clip(target, seed=meta["seed"], profile=meta["profile"])
        created.append(target)
    return created


def list_clips(video_dir: str | Path = "videos") -> list[Path]:
    """All .avi clips currently in the video directory, sorted by name."""
    directory = Path(video_dir)
    if not directory.exists():
        return []
    return sorted([*directory.glob("*.mp4"), *directory.glob("*.avi")])