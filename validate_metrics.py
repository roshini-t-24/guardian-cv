"""Proof harness for the gait / stability metrics.

    python validate_metrics.py

1. Ground-truth check. The three procedural clips are rendered from a known
   kinematic model (utils/video_generator.py), so the TRUE per-leg knee
   flexion of every frame can be recomputed here and compared with what the
   analyser recovered from pixels alone. The analyser never sees this model.
2. Ordering check. Every clip in videos/ is run through the full pipeline
   and its tier is compared with the label in EXPECTED. The labels live
   ONLY in this test harness; nothing in utils/ reads a filename.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.gait_analysis import GaitAnalyzer  # noqa: E402
from utils.video_generator import ensure_clips, list_clips  # noqa: E402
from utils.vision_pipeline import ClinicalVisionPipeline  # noqa: E402

# Test oracle only: stem -> tiers the metrics are allowed to produce.
EXPECTED = {
    "clip1_asymmetric_gait": {"RED"},
    "clip2_symmetric_baseline": {"GREEN"},
    "clip3_postural_sway": {"RED", "AMBER"},
    # Drawn-figure clips hide one leg behind the other. The honest outcomes are a
    # correct tier or INDETERMINATE; a false GREEN on a limp is the failure.
    "clip1_mixamo_limp_benchmark": {"RED", "AMBER", "INDETERMINATE"},
    "clip2_healthy_corridor_walk": {"GREEN", "INDETERMINATE"},
    "clip3_acute_stumble_event": {"RED", "AMBER"},
}
# True knee-flexion model of the procedural clips: (cadence rad/frame, near-leg gain)
TRUTH = {"clip1_asymmetric_gait": (0.28, 0.30), "clip2_symmetric_baseline": (0.30, 1.0)}


def analyse(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    pipe, gait = ClinicalVisionPipeline(), GaitAnalyzer(fps=fps)
    flex = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        joints, bbox = pipe.extract_joints(pipe.segment_foreground(frame))
        r = gait.update(frame, joints, bbox)
        flex.append((r["flex_a"], r["flex_b"]))
    cap.release()
    return gait.summary(), flex


def true_flexion(cadence: float, gain: float, n: int):
    def f(ph, g):
        return 62.0 * g * max(0.0, np.sin(ph + 1.05)) ** 1.4
    return [(f(t * cadence, 1.0), f(t * cadence + np.pi, gain)) for t in range(n)]


def main() -> int:
    ensure_clips(str(ROOT / "videos"))
    clips = list_clips(str(ROOT / "videos"))
    failures = 0

    print("\n[1] Recovered vs true swing-peak knee flexion (procedural clips)")
    print(f"{'clip':<28}{'true peaks':>16}{'recovered':>16}{'frame RMSE':>12}")
    for path in clips:
        if path.stem not in TRUTH:
            continue
        summary, flex = analyse(path)
        cadence, gain = TRUTH[path.stem]
        truth = true_flexion(cadence, gain, len(flex))
        pairs = [(e, t) for e, t in zip(flex, truth) if e[0] is not None]
        est = np.array([e for e, _ in pairs])
        tru = np.array([t for _, t in pairs])
        keep = np.mean((est - tru) ** 2) <= np.mean((est - tru[:, ::-1]) ** 2)
        tru = tru if keep else tru[:, ::-1]
        rmse = float(np.sqrt(np.mean((est - tru) ** 2)))
        tp = sorted([62.0 * gain, 62.0])
        rp = sorted([summary.get("peak_flexion_a", 0), summary.get("peak_flexion_b", 0)])
        print(f"{path.stem:<28}{tp[0]:>7.1f} /{tp[1]:>6.1f}{rp[0]:>8.1f} /{rp[1]:>6.1f}{rmse:>12.1f}")
        if abs((tp[1] - tp[0]) - (rp[1] - rp[0])) > 12.0:
            failures += 1
            print("   FAIL: recovered peak gap differs from truth by > 12 deg")

    print("\n[2] Tier per clip")
    print(f"{'clip':<30}{'SI %':>7}{'drop %':>8}{'sway %':>8}{'lean':>7}  {'tier':<14} expected")
    for path in clips:
        summary, _ = analyse(path)
        want = EXPECTED.get(path.stem)
        si = summary.get("symmetry_index_pct")
        status = ""
        if want is not None:
            status = "ok" if summary["risk_tier"] in want else "FAIL"
            failures += status == "FAIL"
        print(f"{path.stem:<30}{'-' if si is None else si:>7}"
              f"{summary.get('com_drop_pct', '-'):>8}{summary.get('sway_rms_pct', '-'):>8}"
              f"{summary.get('lean_excursion_deg', '-'):>7}  {str(summary['risk_tier']):<14} "
              f"{sorted(want) if want else '(unlabelled)'} {status}")
    print("\nRESULT:", "PASS" if failures == 0 else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())