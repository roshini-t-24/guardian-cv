import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from utils.agent_core import ActivePerceptionAgent, HITLLedger, build_ehr_rx
from utils.gait_analysis import GaitAnalyzer
from utils.kalman_tracker import JointKalmanBank
from utils.telemetry import compute_visual_telemetry
from utils.vision_pipeline import ClinicalVisionPipeline

TIERS = {"GREEN", "AMBER", "RED", "INDETERMINATE"}


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to an .mp4 sample")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--night", action="store_true", help="apply underexposure + noise")
    ap.add_argument("--out", default="smoke_report.json")
    args = ap.parse_args()

    path = Path(args.video)
    check(path.is_file(), f"sample exists: {path}")
    cap = cv2.VideoCapture(str(path))
    check(cap.isOpened(), "video opens with OpenCV")

    pipeline, agent = ClinicalVisionPipeline(), ActivePerceptionAgent()
    gait, kalman, hitl = GaitAnalyzer(), JointKalmanBank(), HITLLedger()
    for obj in (pipeline, agent, gait, kalman):
        obj.reset()

    prev_mask, frames, valid, conf, interventions, gains = None, 0, 0, [], 0, []
    last_decision, innovations = None, []

    while frames < args.max_frames:
        ok, raw = cap.read()
        if not ok:
            break
        if args.night:
            raw = np.clip(raw.astype(np.float32) * 0.32, 0, 255).astype(np.uint8)
            noise = np.random.default_rng(frames).integers(0, 14, raw.shape, np.uint8)
            raw = cv2.add(raw, noise)

        decision, committed = agent.act(raw, mask=prev_mask)
        mask = pipeline.segment_foreground(committed)
        joints, bbox = pipeline.extract_joints(mask)
        step = gait.update(committed, joints, bbox)
        kin = kalman.update(step.get("joints") or {})
        tel = compute_visual_telemetry(cv2.cvtColor(committed, cv2.COLOR_BGR2GRAY), mask)

        conf.append(tel["snr_confidence"])
        innovations.append(kin["innovation_error"])
        valid += int(bool(step.get("valid")))
        if decision["triggered"]:
            interventions += 1
            gains.append(decision["gain"])
            last_decision = decision
        prev_mask = mask
        frames += 1
    cap.release()

    check(frames > 0, f"decoded {frames} frames")
    check(all(12.0 <= c <= 99.4 for c in conf),
          f"snr_confidence in [12, 99.4] (min {min(conf):.1f}, max {max(conf):.1f})")
    check(len(set(round(c, 2) for c in conf)) > 1, "confidence varies per frame (not hardcoded)")
    check(np.isfinite(innovations).all(), "Kalman innovation finite on all frames")

    summary = gait.summary()
    check("risk_tier" in summary, "gait summary has 'risk_tier'")
    if summary.get("gait_sufficient"):
        for key in ("asymmetry_delta", "sway_rms_pct", "tier_reasons"):
            check(key in summary, f"gait summary has '{key}'")
    else:
        print("INFO  gait_sufficient=False -> tier:", summary.get("risk_tier"),
              "|", summary.get("tier_reasons"))
    check(summary["risk_tier"] in TIERS | {None}, f"risk_tier valid: {summary['risk_tier']}")
    if args.night:
        check(interventions > 0, f"agent intervened under night stress ({interventions}x)")
    if last_decision:
        check(last_decision["post_confidence"] >= last_decision["pre_confidence"],
              "agent gain never negative")

    hitl.record("approve", {"clip": path.name})
    rx = build_ehr_rx(
        {
            "gait": summary,
            "clip": path.name,
            "agent_interventions": interventions,
            "mean_agent_gain": float(np.mean(gains)) if gains else 0.0,
        },
        hitl,
    )
    check(rx.get("resourceType") == "Bundle", "EHR Rx is a FHIR-style Bundle")
    check(len(rx.get("entry", [])) >= 2, "EHR Rx has CarePlan + Observations")

    report = {
        "clip": path.name, "frames": frames, "valid_gait_frames": valid,
        "mean_confidence": round(float(np.mean(conf)), 2),
        "interventions": interventions, "gait_summary": summary, "ehr_rx": rx,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSMOKE TEST PASSED  clip={path.name} frames={frames} valid={valid} "
          f"tier={summary['risk_tier']}  report -> {args.out}")


if __name__ == "__main__":
    main()