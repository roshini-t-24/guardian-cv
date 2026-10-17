import cv2
import numpy as np
import time
import platform
from utils.vision_pipeline import compute_visual_telemetry


class ActivePerceptionAgent:
    """
    Closed-Loop Multi-Policy Active Perception Controller.
    When camera degradation is detected, evaluates a candidate action space
    of OpenCV restoration kernels and selects the policy that maximizes
    composite visual confidence (Luminance + Shannon Entropy + Edge Density).
    """
    def __init__(self):
        self.last_selected_policy = "Pass-Through (Nominal Lighting)"
        self.evaluation_history = []

    @staticmethod
    def _apply_gamma_lut(image, gamma=0.45):
        inv_gamma = 1.0 / max(gamma, 0.05)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _policy_a_gamma_Lut(self, frame):
        """Policy A: Fast Non-Linear Gamma Look-Up Table (cv2.LUT, gamma=0.48)"""
        return self._apply_gamma_lut(frame, gamma=0.48)

    def _policy_b_lab_clahe(self, frame):
        """Policy B: CIELAB L-Channel CLAHE (clip=3.2) + Gaussian Smoothing"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
        cl = clahe.apply(l_ch)
        restored = cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        return cv2.GaussianBlur(restored, (3, 3), 0)

    def _policy_c_hybrid_night_recovery(self, frame):
        """Policy C: Multi-Stage Gamma Lift (0.42) + High-Clip CLAHE (4.2) + Bilateral Denoising"""
        lifted = self._apply_gamma_lut(frame, gamma=0.42)
        lab = cv2.cvtColor(lifted, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.2, tileGridSize=(8, 8))
        cl = clahe.apply(l_ch)
        merged = cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        return cv2.bilateralFilter(merged, d=5, sigmaColor=35, sigmaSpace=35)

    def optimize_frame(self, degraded_frame, fg_mask, pre_telemetry):
        candidates = [
            ("Policy A: OpenCV Gamma LUT (g=0.48)", self._policy_a_gamma_Lut),
            ("Policy B: LAB-CLAHE (clip=3.2) + Gaussian", self._policy_b_lab_clahe),
            ("Policy C: Hybrid Gamma + CLAHE (4.2) + Bilateral", self._policy_c_hybrid_night_recovery),
        ]

        best_frame = degraded_frame
        best_telemetry = pre_telemetry
        best_policy_name = "Pass-Through"
        best_score = pre_telemetry["confidence"]
        candidate_scores = {}

        for policy_name, fn in candidates:
            cand_frame = fn(degraded_frame)
            cand_gray = cv2.cvtColor(cand_frame, cv2.COLOR_BGR2GRAY)
            cand_tel = compute_visual_telemetry(cand_gray, fg_mask)
            candidate_scores[policy_name] = cand_tel["confidence"]

            if cand_tel["confidence"] > best_score:
                best_score = cand_tel["confidence"]
                best_frame = cand_frame
                best_telemetry = cand_tel
                best_policy_name = policy_name

        gain = round(best_score - pre_telemetry["confidence"], 1)
        self.last_selected_policy = best_policy_name
        decision = {
            "intervened": True,
            "selected_policy": best_policy_name,
            "pre_confidence": pre_telemetry["confidence"],
            "post_confidence": best_score,
            "gain": gain,
            "candidate_scores": candidate_scores,
        }
        return best_frame, best_telemetry, decision


def generate_prescription_text(asym, mean_conf, flow_mag, selected_policy, therapist_status):
    risk_level = "High Fall Risk (Requires Intervention)" if asym > 18.0 else "Normal Baseline Symmetry"
    sway_note = "Elevated postural sway velocity detected." if flow_mag > 0.45 else "Postural trunk velocity within normal limits."

    return f"""
**PATIENT ASSESSMENT REPORT — GUARDIANCV ACTIVE PERCEPTION ENGINE**
-------------------------------------------------------------------
* **Evaluation Timestamp:** {time.strftime("%Y-%m-%d %H:%M:%S")}
* **Compute Runtime:** OpenCV v{cv2.__version__} on `{platform.machine()}`
* **Active Perception Policy Applied:** `{selected_policy}` (Mean SNR Confidence: **{mean_conf:.1f}%**)
* **Kalman-Smoothed Kinematic Deviation:** Peak bilateral knee-flexion asymmetry measured at **{asym:.1f}°**.
* **Optical Flow Postural Velocity:** Mean torso/limb flow magnitude **{flow_mag:.3f} px/frame** ({sway_note})
* **Clinical Risk Classification:** **{risk_level}**

**EVIDENCE-BASED PHYSICAL THERAPY PROTOCOL:**
1. **Closed-Chain Eccentric Quadriceps Control:** 3 sets of 12 slow step-downs to address terminal knee extension asymmetry.
2. **Proprioceptive Stance Stabilization:** Single-leg balance with visual perturbation (3 x 30s) to reduce optical-flow trunk sway.
3. **Auditory-Cued Gait Symmetry Drill:** Metronome-paced ambulation (92 BPM) equalizing bilateral stance-to-swing phase duration.

**CLINICIAN GOVERNANCE (HITL):**
* Physical Therapist Sign-Off Status: `{therapist_status}`
"""