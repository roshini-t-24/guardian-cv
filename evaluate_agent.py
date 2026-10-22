"""Agent effectiveness evaluation: every clip x stress level, agent OFF vs ON.

Same per-frame order as server.py. Reports mean SNR confidence, valid-leg-fit
rate and risk tier. tier_ok means the tier equals the unstressed, no-agent
reference tier for that clip.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from utils.agent_core import ActivePerceptionAgent
from utils.gait_analysis import GaitAnalyzer
from utils.telemetry import compute_visual_telemetry
from utils.video_generator import list_clips
from utils.vision_pipeline import ClinicalVisionPipeline

STRESS = {"nominal": 1.0, "dim": 0.5, "night": 0.32, "severe": 0.15}


def stress(frame: np.ndarray, gain: float, idx: int) -> np.ndarray:
    if gain >= 1.0:
        return frame
    out = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    noise = np.random.default_rng(idx).integers(0, 14, out.shape, np.uint8)
    return cv2.add(out, noise)


def run(path: Path, gain: float, agent_on: bool, max_frames: int) -> dict:
    pipe, agent, gait = ClinicalVisionPipeline(), ActivePerceptionAgent(), GaitAnalyzer()
    for o in (pipe, agent, gait):
        o.reset()
    cap = cv2.VideoCapture(str(path))
    prev_mask, conf, valid, n = None, [], 0, 0
    gains, policies = [], Counter()
    while n < max_frames:
        ok, raw = cap.read()
        if not ok:
            break
        raw = stress(raw, gain, n)
        committed = raw
        if agent_on:
            d, committed = agent.act(raw, mask=prev_mask)
            if d["triggered"]:
                gains.append(d["gain"])
                policies[d["committed_policy"] or "HOLD"] += 1
        mask = pipe.segment_foreground(committed)
        joints, bbox = pipe.extract_joints(mask)
        step = gait.update(committed, joints, bbox)
        tel = compute_visual_telemetry(cv2.cvtColor(committed, cv2.COLOR_BGR2GRAY), mask)
        conf.append(tel["snr_confidence"])
        valid += int(bool(step.get("valid")))
        prev_mask = mask
        n += 1
    cap.release()
    s = gait.summary()
    return {
        "frames": n,
        "mean_conf": round(float(np.mean(conf)), 2) if conf else 0.0,
        "valid_pct": round(100.0 * valid / max(n, 1), 1),
        "tier": s.get("risk_tier"),
        "interventions": len(gains),
        "mean_gain": round(float(np.mean(gains)), 2) if gains else 0.0,
        "policies": dict(policies),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--out", default="eval_agent_report.json")
    args = ap.parse_args()

    clips, seen = [], set()
    for p in sorted(list_clips("videos"), key=lambda q: q.suffix != ".avi"):
        if p.stem not in seen:
            seen.add(p.stem)
            clips.append(p)

    rows = []
    for clip in clips:
        ref = run(clip, 1.0, False, args.max_frames)["tier"]
        for level, gain in STRESS.items():
            off = run(clip, gain, False, args.max_frames)
            on = run(clip, gain, True, args.max_frames)
            rows.append({
                "clip": clip.stem, "stress": level, "ref_tier": ref,
                "off": off, "on": on,
                "conf_gain": round(on["mean_conf"] - off["mean_conf"], 2),
                "valid_gain_pct": round(on["valid_pct"] - off["valid_pct"], 1),
                "tier_ok_off": off["tier"] == ref,
                "tier_ok_on": on["tier"] == ref,
            })
            print(f"{clip.stem:<30}{level:<8}"
                  f"conf {off['mean_conf']:5.1f}->{on['mean_conf']:5.1f}  "
                  f"valid {off['valid_pct']:5.1f}->{on['valid_pct']:5.1f}%  "
                  f"tier {off['tier']}->{on['tier']} (ref {ref})  "
                  f"interventions {on['interventions']}")

    print("\n| stress | clips | mean conf off | mean conf on | mean valid% off | mean valid% on "
          "| tier ok off | tier ok on |")
    print("|---|---|---|---|---|---|---|---|")
    for level in STRESS:
        r = [x for x in rows if x["stress"] == level]
        print(f"| {level} | {len(r)} "
              f"| {np.mean([x['off']['mean_conf'] for x in r]):.1f} "
              f"| {np.mean([x['on']['mean_conf'] for x in r]):.1f} "
              f"| {np.mean([x['off']['valid_pct'] for x in r]):.1f} "
              f"| {np.mean([x['on']['valid_pct'] for x in r]):.1f} "
              f"| {sum(x['tier_ok_off'] for x in r)}/{len(r)} "
              f"| {sum(x['tier_ok_on'] for x in r)}/{len(r)} |")

    Path(args.out).write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()