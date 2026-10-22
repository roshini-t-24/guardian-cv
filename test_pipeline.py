"""GuardianCV pytest suite: telemetry, Kalman, angles, PII shield, agent, gait, API.

Run:  python -m pytest test_pipeline.py -q
Every assertion is computed from real pixels or filter state.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from utils.agent_core import ActivePerceptionAgent, HITLLedger, build_ehr_rx
from utils.gait_analysis import GaitAnalyzer
from utils.kalman_tracker import JointKalmanBank
from utils.telemetry import compute_visual_telemetry, quality_gate, shannon_entropy
from utils.video_generator import ensure_clips
from utils.vision_pipeline import (
    ClinicalVisionPipeline,
    apply_privacy_shield,
    calculate_angle,
)

TIERS = {"GREEN", "AMBER", "RED", "INDETERMINATE"}
CLIP = "clip2_symmetric_baseline.mp4"


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(402)


@pytest.fixture(scope="module")
def clip_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("videos")
    ensure_clips(d)
    return d


def _lit_frame(rng):
    base = rng.integers(150, 230, size=(240, 320), dtype=np.uint8)
    frame = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame, (110, 40), (210, 210), (45, 60, 80), -1)
    mask = np.zeros((240, 320), np.uint8)
    cv2.rectangle(mask, (110, 40), (210, 210), 255, -1)
    return frame, mask


# ---------------- telemetry ----------------
def test_entropy_bounds(rng):
    assert shannon_entropy(np.full((64, 64), 128, np.uint8)) < 0.05
    assert shannon_entropy(rng.integers(0, 256, (64, 64), dtype=np.uint8)) > 7.0


def test_confidence_within_band(rng):
    for _ in range(12):
        t = compute_visual_telemetry(rng.integers(0, 256, (120, 160), dtype=np.uint8))
        assert 12.0 <= t["snr_confidence"] <= 99.4


def test_dark_frame_degrades_confidence_and_trips_gate(rng):
    lit, mask = _lit_frame(rng)
    lit_t = compute_visual_telemetry(cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY), mask)
    dark = np.clip(lit.astype(np.float32) * 0.18, 0, 255).astype(np.uint8)
    dark_t = compute_visual_telemetry(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY), mask)
    assert dark_t["luminance"] < 85.0
    assert dark_t["snr_confidence"] < lit_t["snr_confidence"]
    assert quality_gate(dark_t)[0] is True


# ---------------- Kalman ----------------
def test_kalman_tracks_constant_velocity():
    bank = JointKalmanBank()
    out = None
    for i in range(40):
        out = bank.update({"hip": (100.0 + 2.0 * i, 150.0)})
    assert abs(out["smoothed_joints"]["hip"][0] - 178.0) < 4.0
    assert abs(out["joint_velocities"]["hip"] - 2.0) < 0.5


def test_kalman_innovation_flags_jump():
    bank = JointKalmanBank()
    for i in range(30):
        bank.update({"hip": (100.0 + 2.0 * i, 150.0)})
    assert bank.update({"hip": (260.0, 150.0)})["innovation_error"] > 20.0


def test_kalman_coasts_through_dropout():
    bank = JointKalmanBank()
    for i in range(30):
        bank.update({"hip": (100.0 + 2.0 * i, 150.0)})
    out = bank.update({})
    assert "hip" in out["smoothed_joints"] and out["tracked_joints"] == []


# ---------------- geometry + privacy ----------------
def test_calculate_angle_known_values():
    assert calculate_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0, abs=0.1)
    assert calculate_angle((-1, 0), (0, 0), (1, 0)) == pytest.approx(180.0, abs=0.1)


def test_silhouette_shield_redacts_head(rng):
    lit, mask = _lit_frame(rng)
    gray = cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY)
    out = apply_privacy_shield(lit, gray, mask, {"head": (160.0, 100.0)}, mode="silhouette")
    assert np.all(np.abs(out[92:108, 152:168].astype(int) - 20) <= 1)
    assert not np.array_equal(out, lit)


def test_corridor_shield_runs_without_crash(rng):
    lit, mask = _lit_frame(rng)
    gray = cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY)
    for head in ({"head": (160.0, 60.0)}, {"head": (5.0, 5.0)}, None):
        out = apply_privacy_shield(lit, gray, mask, head, mode="corridor")
        assert out.shape == lit.shape


# ---------------- agent ----------------
def test_agent_restores_underexposed_frame(rng):
    lit, mask = _lit_frame(rng)
    dark = np.clip(lit.astype(np.float32) * 0.15, 0, 255).astype(np.uint8)
    decision, restored = ActivePerceptionAgent().act(dark, mask=mask)
    assert decision["triggered"]
    assert decision["post_confidence"] >= decision["pre_confidence"]
    assert len(decision["candidate_scores"]) == 3
    assert not np.array_equal(restored, dark)


def test_agent_holds_nominal_frame(rng):
    lit, mask = _lit_frame(rng)
    decision, restored = ActivePerceptionAgent().act(lit, mask=mask)
    assert not decision["triggered"]
    assert np.array_equal(restored, lit)


# ---------------- end-to-end on a real MP4 ----------------
def _run(path, frames=90):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    pipe, gait = ClinicalVisionPipeline(), GaitAnalyzer()
    pipe.reset(); gait.reset()
    steps = 0
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        mask = pipe.segment_foreground(frame)
        joints, bbox = pipe.extract_joints(mask)
        gait.update(frame, joints, bbox)
        steps += 1
    cap.release()
    return steps, gait.summary()


def test_end_to_end_gait_summary(clip_dir):
    steps, s = _run(clip_dir / CLIP)
    assert steps > 60
    assert s["risk_tier"] in TIERS | {None}
    assert isinstance(s.get("tier_reasons"), list)
    assert "gait_sufficient" in s and "asymmetry_delta" in s
    if not s["gait_sufficient"]:
        # never a confident GREEN when the legs are not independently resolved
        assert s["risk_tier"] != "GREEN"

def test_ehr_bundle_from_real_run(clip_dir):
    _, s = _run(clip_dir / CLIP)
    h = HITLLedger()
    h.record("approve", {"clip": CLIP})
    rx = build_ehr_rx({"gait": s, "clip": CLIP, "agent_interventions": 0,
                       "mean_agent_gain": 0.0}, h)
    assert rx["resourceType"] == "Bundle"
    assert rx["entry"][0]["resource"]["resourceType"] == "CarePlan"
    assert rx["governance"]["hitl"]["approvals"] == 1


# ---------------- FastAPI ----------------
@pytest.fixture()
def client(clip_dir, monkeypatch):
    monkeypatch.setattr(server, "VIDEO_DIR", clip_dir)
    server.STATE["last_run"] = None
    server.STATE["hitl"] = HITLLedger()
    return TestClient(server.app)


def test_api_health_and_clips(client):
    assert client.get("/api/health").json()["status"] == "ok"
    names = [c["name"] for c in client.get("/api/clips").json()]
    assert CLIP in names


def test_api_ehr_requires_run(client):
    assert client.get("/api/ehr").status_code == 409


def test_api_evaluate_then_ehr(client):
    r = client.post("/api/evaluate", json={"clip": CLIP, "max_frames": 60})
    assert r.status_code == 200
    run = r.json()
    assert run["frames"] > 0 and 12.0 <= run["mean_confidence"] <= 99.4
    assert client.get("/api/ehr").json()["resourceType"] == "Bundle"


def test_api_night_triggers_agent(client):
    r = client.post("/api/evaluate",
                    json={"clip": CLIP, "night": True, "agent": True, "max_frames": 40})
    assert r.json()["agent_interventions"] > 0


def test_api_unknown_clip_and_bad_hitl(client):
    assert client.post("/api/evaluate", json={"clip": "nope.mp4"}).status_code == 404
    assert client.post("/api/hitl", json={"action": "x"}).status_code == 400


def test_api_hitl_audit_mcp(client):
    client.post("/api/hitl", json={"action": "approve", "clip": CLIP})
    assert client.get("/api/audit").json()["hitl"]["approvals"] == 1
    assert len(client.get("/api/mcp").json()) >= 3


def test_websocket_stream_completes(client):
    with client.websocket_connect(f"/ws/stream?clip={CLIP}&agent=true&pii=true") as ws:
        first = ws.receive_json()
        assert "jpeg_b64" in first and first["frame"] == 0
        last = first
        while not last.get("done"):
            last = ws.receive_json()
        assert last["summary"]["frames"] > 0