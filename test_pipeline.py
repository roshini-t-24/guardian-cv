import cv2
import numpy as np
import pytest
from utils.vision_pipeline import (
    JointKalmanBank,
    calculate_angle,
    compute_visual_telemetry,
    extract_joints,
    run_pipeline,
)
from utils.agent_core import ActivePerceptionAgent, generate_prescription_text


def _create_mock_patient_frame(asymmetric=False):
    """Generates a deterministic 640x420 test frame with a high-contrast bipedal silhouette."""
    frame = np.full((420, 640, 3), 210, dtype=np.uint8)
    fg_mask = np.zeros((420, 640), dtype=np.uint8)

    cx, cy = 320, 200
    head = (cx, cy - 85)
    hip = (cx, cy)

    # Left leg (extended)
    l_knee = (cx - 25, cy + 65)
    l_ankle = (cx - 30, cy + 135)

    # Right leg (either symmetric or flexed/asymmetric)
    if asymmetric:
        r_knee = (cx + 35, cy + 55)
        r_ankle = (cx + 5, cy + 125)
    else:
        r_knee = (cx + 25, cy + 65)
        r_ankle = (cx + 30, cy + 135)

    # Draw subject on both RGB frame and foreground mask
    for canvas, color in [(frame, (45, 65, 95)), (fg_mask, 255)]:
        cv2.circle(canvas, head, 22, color, -1)
        cv2.line(canvas, (head[0], head[1] + 20), hip, color, 18)
        cv2.line(canvas, hip, l_knee, color, 14)
        cv2.line(canvas, l_knee, l_ankle, color, 12)
        cv2.line(canvas, hip, r_knee, color, 14)
        cv2.line(canvas, r_knee, r_ankle, color, 12)

    return frame, fg_mask


def test_sagittal_angle_trigonometry():
    """Validates exact 2D vector angle mathematics across orthogonal and colinear joints."""
    # 180-degree straight leg (hip -> knee -> ankle colinear)
    straight_angle = calculate_angle((300, 100), (300, 200), (300, 300))
    assert np.isclose(straight_angle, 180.0, atol=1e-2)

    # 90-degree right-angle knee flexion
    right_angle = calculate_angle((300, 100), (300, 200), (400, 200))
    assert np.isclose(right_angle, 90.0, atol=1e-2)

    # 135-degree moderate flexion
    flex_angle = calculate_angle((300, 100), (300, 200), (370, 270))
    assert 134.0 <= flex_angle <= 136.0


def test_kalman_filter_bank_convergence_and_noise_rejection():
    """
    Verifies that the 6-joint 4D OpenCV Kalman Filter bank:
    1. Smooths high-frequency measurement jitter (reduces variance vs raw input).
    2. Estimates positive non-zero velocity when joints translate across frames.
    """
    bank = JointKalmanBank()
    np.random.seed(42)

    raw_x_errors = []
    kalman_x_errors = []
    velocities = {}

    for t in range(25):
        true_x = 150 + t * 6  # Constant horizontal velocity of 6 px/frame
        true_y = 200
        jitter_x = int(np.random.normal(0, 9.0))
        jitter_y = int(np.random.normal(0, 9.0))

        noisy_joints = {
            name: (true_x + jitter_x, true_y + jitter_y)
            for name in bank.joint_names
        }
        smoothed, velocities = bank.update(noisy_joints)

        if t >= 5:  # Evaluate after initial state warm-up
            raw_x_errors.append(abs(noisy_joints["l_knee"][0] - true_x))
            kalman_x_errors.append(abs(smoothed["l_knee"][0] - true_x))

    # Kalman filter must achieve lower mean absolute tracking error than raw noisy measurements
    assert np.mean(kalman_x_errors) < np.mean(raw_x_errors)
    # Velocity state must converge to positive motion (> 2.5 px/frame)
    assert velocities["l_knee"] > 2.5


def test_visual_telemetry_degradation_sensitivity():
    """Verifies that Shannon entropy, luminance, and SNR confidence drop under severe underexposure."""
    frame, fg_mask = _create_mock_patient_frame(asymmetric=False)
    gray_clean = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clean_tel = compute_visual_telemetry(gray_clean, fg_mask)

    # Degrade frame (simulate dark hospital corridor)
    dark_frame = cv2.convertScaleAbs(frame, alpha=0.18, beta=-15)
    gray_dark = cv2.cvtColor(dark_frame, cv2.COLOR_BGR2GRAY)
    dark_tel = compute_visual_telemetry(gray_dark, fg_mask)

    assert clean_tel["mean_lum"] > 150.0
    assert dark_tel["mean_lum"] < 40.0
    assert clean_tel["confidence"] > dark_tel["confidence"] + 20.0


def test_multi_policy_agent_snr_gain():
    """
    Validates that ActivePerceptionAgent evaluates all 3 OpenCV restoration policies
    on an underexposed + noisy frame and achieves a strictly positive SNR confidence gain.
    """
    frame, fg_mask = _create_mock_patient_frame(asymmetric=True)
    degraded = cv2.convertScaleAbs(frame, alpha=0.22, beta=-15)
    noise = np.random.normal(0, 14, degraded.shape).astype(np.int16)
    degraded = np.clip(degraded.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pre_gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
    pre_tel = compute_visual_telemetry(pre_gray, fg_mask)

    agent = ActivePerceptionAgent()
    restored_frame, post_tel, decision = agent.optimize_frame(degraded, fg_mask, pre_tel)

    assert decision["intervened"] is True
    assert len(decision["candidate_scores"]) == 3
    assert decision["post_confidence"] > decision["pre_confidence"]
    assert decision["gain"] > 10.0
    assert post_tel["mean_lum"] > pre_tel["mean_lum"]


def test_pixel_slice_joint_extraction_and_zero_pii_redaction():
    """
    Validates end-to-end run_pipeline execution:
    - Extracts anatomical joints from pixel contours without sine-wave mocks.
    - Enforces Zero-PII cranial black-out masking at the detected head centroid.
    """
    frame, fg_mask = _create_mock_patient_frame(asymmetric=True)
    kalman_bank = JointKalmanBank()
    agent = ActivePerceptionAgent()

    result = run_pipeline(
        frame=frame,
        prev_gray=None,
        fg_mask=fg_mask,
        kalman_bank=kalman_bank,
        agent_controller=agent,
        degrade=False,
        agent_active=True,
        privacy_on=True,
        draw_flow=False,
    )

    joints, _, _ = extract_joints(frame, fg_mask, kalman_bank=None)
    hx, hy = joints["head"]

    # Verify anatomical vertical ordering: Head Y < Hip Y < Knee Y < Ankle Y
    assert joints["head"][1] < joints["hip"][1] < joints["l_knee"][1] < joints["l_ankle"][1]

    # Verify Zero-PII cranial shield overwrites head center pixels with solid dark redaction (20, 20, 20)
    out_img = result["display_frame"]
    center_pixel = out_img[hy, hx]
    assert np.all(center_pixel == np.array([20, 20, 20]))


def test_clinical_ehr_prescription_generation():
    """Ensures clinical report generator formats asymmetry and active perception telemetry accurately."""
    report = generate_prescription_text(
        asym=24.6,
        mean_conf=91.2,
        flow_mag=0.52,
        selected_policy="Policy C: Hybrid Gamma + CLAHE (4.2) + Bilateral",
        therapist_status="Approved by Physical Therapist",
    )
    assert "High Fall Risk" in report
    assert "24.6°" in report
    assert "Policy C" in report