"""Module D — Closed-loop active-perception optimizer, HITL governance,
Model Context Protocol (MCP) tool schemas, and the clinical EHR
prescription generator.

The agent never trusts a degraded frame. When Module A's quality gate
trips (mu_L < 85.0 or composite C < 72.0 %) it:

    1. synthesises three OpenCV restoration candidates,
         A  Gamma_LUT_0.48                    non-linear cv2.LUT lift
         B  LAB_CLAHE_3.2_Gaussian            CIELAB CLAHE + 3x3 Gaussian
         C  Hybrid_Gamma_CLAHE_4.2_Bilateral  0.42 lift + CLAHE 4.2 + bilateral
    2. re-measures every candidate on the actual pixel matrix via
       compute_visual_telemetry(),
    3. commits  pi* = argmax_policy C(I_policy)
       — or keeps the raw frame when no policy improves objective
       telemetry (safe fallback, gain can never go negative).

Commit-and-hold
---------------
Scoring all three policies costs ~20 ms per frame and, applied
independently per frame, can swap the image domain under the background
model. So the agent evaluates once, then HOLDS the winning policy and only
pays for that one policy (~8 ms):

    nominal ──gate trips──► evaluate ──► hold ──┬─ window expired ─────► evaluate
       ▲                                        ├─ luminance drifted ──► evaluate
       │                                        ├─ policy stopped
       └── nominal for `release_after` frames ◄─┘   improving C ───────► evaluate
           (with exit margins)                   (never applies a policy that lowers C)

The release rule has hysteresis: the raw frame must clear the gate by a
margin for several consecutive frames, so a scene hovering at the
threshold does not chatter between raw and restored pixels.

Contract with the caller: `mask` is the foreground mask of the PREVIOUS
frame (used only for the hull-solidity term), so the background model in
Module C is fed once per time step, on the frame the pipeline tracks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.telemetry import (
    DEGRADATION_CONFIDENCE,
    DEGRADATION_LUMINANCE,
    compute_visual_telemetry,
    quality_gate,
)

POLICY_GAMMA = "Gamma_LUT_0.48"
POLICY_CLAHE = "LAB_CLAHE_3.2_Gaussian"
POLICY_HYBRID = "Hybrid_Gamma_CLAHE_4.2_Bilateral"

MODE_NOMINAL = "nominal"      # gate clear, raw frame passed through
MODE_EVALUATE = "evaluate"    # all policies scored, winner (or raw) committed
MODE_HOLD = "hold"            # committed policy re-applied without re-scoring
MODE_RELEASE = "release"      # scene recovered, back to raw pass-through
LOGGED_MODES = (MODE_EVALUATE, MODE_RELEASE)


# -- restoration policies (the candidate action space) ----------------------
def _gamma_lut(frame: np.ndarray, gamma: float) -> np.ndarray:
    """Non-linear illumination lift through a 256-entry cv2.LUT table."""
    table = ((np.arange(256) / 255.0) ** gamma * 255.0).astype(np.uint8)
    return cv2.LUT(frame, table)


def _denoise(frame: np.ndarray) -> np.ndarray:
    """5x5 Gaussian: sensor noise amplified by a brightness lift floods MOG2 otherwise."""
    return cv2.GaussianBlur(frame, (5, 5), 0)


def _policy_gamma(frame: np.ndarray) -> np.ndarray:
    """Policy A — pure gamma 0.48 LUT lift."""
    return _denoise(_gamma_lut(frame, gamma=0.48))


def _policy_lab_clahe(frame: np.ndarray) -> np.ndarray:
    """Policy B — CIELAB L-channel CLAHE(3.2, 8x8) + 3x3 Gaussian."""
    lab = cv2.cvtColor(_denoise(frame), cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(merged, (3, 3), 0)


def _policy_hybrid(frame: np.ndarray) -> np.ndarray:
    """Policy C — gamma 0.42 lift + CLAHE 4.2 + edge-preserving bilateral."""
    lifted = _denoise(_gamma_lut(frame, gamma=0.42))
    lab = cv2.cvtColor(lifted, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.2, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(merged, d=5, sigmaColor=35, sigmaSpace=35)


def _confidence(frame: np.ndarray, mask: Optional[np.ndarray]) -> float:
    return compute_visual_telemetry(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)[
        "snr_confidence"
    ]


class ActivePerceptionAgent:
    """Multi-policy ΔJ(I) maximiser with commit-and-hold and event logging."""

    def __init__(
        self,
        hold_frames: int = 15,
        release_after: int = 8,
        drift_ratio: float = 0.25,
        exit_margin_lum: float = 10.0,
        exit_margin_conf: float = 5.0,
    ) -> None:
        """
        hold_frames       frames a committed policy is reused before re-scoring
                          (15 = one second at the 15 fps ward cameras)
        release_after     consecutive margin-clear frames before going back to raw
        drift_ratio       relative luminance change since commit that forces a re-score
        exit_margin_*     how far past the gate a raw frame must be to count as recovered
        """
        self.policies: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            POLICY_GAMMA: _policy_gamma,
            POLICY_CLAHE: _policy_lab_clahe,
            POLICY_HYBRID: _policy_hybrid,
        }
        self.hold_frames = hold_frames
        self.release_after = release_after
        self.drift_ratio = drift_ratio
        self.exit_margin_lum = exit_margin_lum
        self.exit_margin_conf = exit_margin_conf

        # Event log: evaluations and releases only (not every frame).
        self.decision_log: List[dict] = []
        self.frames_seen = 0
        self.interventions = 0      # frames where the agent was actively handling a degraded scene
        self.evaluations = 0        # full 3-policy scorings
        self._gain_sum = 0.0

        self._held: Optional[dict] = None   # {"policy": name | None, "age": int, "lum": float}
        self._nominal_streak = 0

    # ------------------------------------------------------------------ api
    def reset(self) -> None:
        """Drop hold state — call when a new clip or stream starts."""
        self._held = None
        self._nominal_streak = 0

    @property
    def held_policy(self) -> Optional[str]:
        """Policy currently being held (None when nominal or holding raw)."""
        return self._held["policy"] if self._held else None

    @property
    def mean_gain(self) -> float:
        """Mean ΔJ over every frame the agent intervened on (confidence points)."""
        return self._gain_sum / self.interventions if self.interventions else 0.0

    def should_intervene(self, telemetry: dict) -> Tuple[bool, str]:
        """Trigger check: mu_L < 85.0 or composite C < 72.0 %."""
        return quality_gate(telemetry)

    def act(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        telemetry: Optional[dict] = None,
    ) -> Tuple[dict, np.ndarray]:
        """Handle one frame. Returns (decision_dict, committed_frame).

        decision["mode"] is one of nominal / evaluate / hold / release.
        decision["gain"] is never negative.
        """
        if telemetry is None:
            telemetry = compute_visual_telemetry(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask
            )
        self.frames_seen += 1
        degraded, reason = self.should_intervene(telemetry)
        pre = telemetry["snr_confidence"]
        luminance = telemetry["luminance"]

        if self._held is None:
            if not degraded:
                return self._decision(MODE_NOMINAL, False, reason, None, pre, pre, {}), frame
            return self._evaluate(frame, mask, luminance, pre, reason)

        # A policy is being held: decide whether the scene has recovered.
        if degraded:
            self._nominal_streak = 0
        else:
            cleared = (
                luminance >= DEGRADATION_LUMINANCE + self.exit_margin_lum
                and pre >= DEGRADATION_CONFIDENCE + self.exit_margin_conf
            )
            self._nominal_streak = self._nominal_streak + 1 if cleared else 0
            if self._nominal_streak >= self.release_after:
                self._held, self._nominal_streak = None, 0
                return (
                    self._decision(
                        MODE_RELEASE, False,
                        f"telemetry clear for {self.release_after} frames", None, pre, pre, {},
                    ),
                    frame,
                )
        return self._hold_or_reevaluate(frame, mask, luminance, pre, degraded, reason)

    # ------------------------------------------------------------ internals
    def _hold_or_reevaluate(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray],
        luminance: float,
        pre: float,
        degraded: bool,
        reason: str,
    ) -> Tuple[dict, np.ndarray]:
        held = self._held
        assert held is not None
        policy_name = held["policy"]

        if policy_name is None:
            restored, post = frame, pre          # holding the raw frame
        else:
            restored = self.policies[policy_name](frame)
            post = _confidence(restored, mask)
            if post <= pre:                      # never apply a policy that doesn't help
                return self._evaluate(
                    frame, mask, luminance, pre, "held policy no longer improves C"
                )

        drift = abs(luminance - held["lum"]) / max(held["lum"], 1.0)
        if held["age"] >= self.hold_frames:
            return self._evaluate(frame, mask, luminance, pre, "hold window expired")
        if drift > self.drift_ratio:
            return self._evaluate(
                frame, mask, luminance, pre, f"luminance drifted {drift:.0%} since commit"
            )

        held["age"] += 1
        note = reason if degraded else "holding through exit hysteresis"
        return self._decision(MODE_HOLD, True, note, policy_name, pre, post, {}), restored

    def _evaluate(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray],
        luminance: float,
        pre: float,
        reason: str,
    ) -> Tuple[dict, np.ndarray]:
        """Score every policy on the real pixels and commit the argmax."""
        candidate_scores: Dict[str, float] = {}
        candidates: Dict[str, np.ndarray] = {}
        for name, policy in self.policies.items():
            restored = policy(frame)
            candidate_scores[name] = round(_confidence(restored, mask), 2)
            candidates[name] = restored

        best_policy = max(candidate_scores, key=lambda k: candidate_scores[k])
        best_score = candidate_scores[best_policy]

        if best_score > pre:
            committed, post, policy = candidates[best_policy], best_score, best_policy
        else:
            # Safe fallback: no policy beat the raw frame -> hold it.
            committed, post, policy = frame, pre, None

        self.evaluations += 1
        self._held = {"policy": policy, "age": 0, "lum": luminance}
        self._nominal_streak = 0
        decision = self._decision(
            MODE_EVALUATE, True, reason, policy, pre, post, candidate_scores
        )
        return decision, committed

    def _decision(
        self,
        mode: str,
        triggered: bool,
        reason: str,
        policy: Optional[str],
        pre: float,
        post: float,
        scores: Dict[str, float],
    ) -> dict:
        gain = round(max(post - pre, 0.0), 2)
        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "mode": mode,
            "triggered": triggered,
            "reason": reason,
            "committed_policy": policy,
            "pre_confidence": round(pre, 2),
            "post_confidence": round(post, 2),
            "gain": gain,
            "candidate_scores": scores,
            "held_for": self._held["age"] if self._held else 0,
        }
        if triggered:
            self.interventions += 1
            self._gain_sum += gain
        if mode in LOGGED_MODES:
            self.decision_log.append(decision)
            if len(self.decision_log) > 128:
                self.decision_log = self.decision_log[-128:]
        return decision


# -- Human-in-the-loop governance -------------------------------------------
class HITLLedger:
    """Clinician override ledger — every approve/dismiss is audit-logged."""

    VALID_ACTIONS = ("approve", "dismiss")

    def __init__(self) -> None:
        self.approvals = 0
        self.dismissals = 0
        self.events: List[dict] = []

    def record(self, action: str, context: Optional[dict] = None) -> dict:
        action = action.lower()
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"unknown HITL action {action!r}; expected {self.VALID_ACTIONS}")
        if action == "approve":
            self.approvals += 1
        else:
            self.dismissals += 1
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "action": action,
            "context": context or {},
        }
        self.events.append(event)
        return event

    def snapshot(self) -> dict:
        return {
            "approvals": self.approvals,
            "dismissals": self.dismissals,
            "recent_events": self.events[-10:],
        }


# -- Model Context Protocol tool surface --------------------------------------
MCP_TOOL_SCHEMAS: List[dict] = [
    {
        "name": "guardian.assess_frame_quality",
        "description": (
            "Compute deterministic visual telemetry (mean luminance, Laplacian "
            "sharpness variance, Shannon entropy, Canny edge density, convex-hull "
            "solidity, composite SNR confidence) for the latest camera frame."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "frame_id": {"type": "integer", "description": "Monotonic frame index"},
                "include_mask": {"type": "boolean", "default": True},
            },
            "required": ["frame_id"],
        },
    },
    {
        "name": "guardian.run_active_perception",
        "description": (
            "Run the closed-loop restoration agent on a frame: score the 3-policy "
            "action space (Gamma LUT 0.48 / LAB-CLAHE 3.2 / Hybrid 0.42+CLAHE 4.2+"
            "Bilateral), commit the argmax delta-J(I) policy, then hold it for "
            "hold_frames frames before re-scoring."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "frame_id": {"type": "integer"},
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Bypass the quality gate and restore anyway",
                },
                "hold_frames": {
                    "type": "integer",
                    "default": 15,
                    "minimum": 1,
                    "description": "Frames a committed policy is reused before re-scoring",
                },
            },
            "required": ["frame_id"],
        },
    },
    {
        "name": "guardian.export_ehr_rx",
        "description": (
            "Generate the FHIR-style rehabilitation prescription bundle "
            "conditioned on measured peak gait asymmetry, optical-flow sway "
            "magnitude and active-perception telemetry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_ref": {"type": "string"},
                "risk_tier_override": {
                    "type": "string",
                    "enum": ["GREEN", "AMBER", "RED"],
                },
            },
            "required": ["patient_ref"],
        },
    },
]


# -- Clinical EHR prescription generator ----------------------------------------
RISK_TIERS: Dict[str, dict] = {
    "GREEN": {
        "summary": "Baseline mobility maintenance",
        "frequency": "3 sessions / week · 30 min",
        "focus": [
            "Symmetric gait reinforcement on level ground",
            "Hip extensor & ankle plantarflexor endurance",
            "Community ambulation with timed up-and-go re-checks",
        ],
        "precautions": ["Routine weekly re-screen with GuardianCV clip capture"],
    },
    "AMBER": {
        "summary": "Asymmetric gait retraining programme",
        "frequency": "5 sessions / week · 40 min + daily home-exercise programme",
        "focus": [
            "Step-length equalisation drills (visual + metronome cueing)",
            "Affected-limb single-leg stance loading progression",
            "Partial body-weight-supported treadmill intervals",
        ],
        "precautions": [
            "Therapist spotting within arm's reach",
            "Re-assess Δθ every 48 h; step down tier when Δθ < 6° for 3 captures",
        ],
    },
    "RED": {
        "summary": "Acute fall-risk — supervised mobility only",
        "frequency": "2× daily supervised sessions · 25 min + bed-level programme",
        "focus": [
            "Static & dynamic balance re-education (feet-together → tandem)",
            "Assistive-device fitting and transfer-technique review",
            "Orthostatic tolerance conditioning before corridor exposure",
        ],
        "precautions": [
            "No unsupervised corridor ambulation",
            "Bed/chair exit alarm active",
            "Medical-director notification if Δθ > 20° or sway > 3.5 px/frame",
        ],
    },
}


def stratify_risk(peak_asymmetry_deg: float, sway_px: float) -> str:
    """Tier the patient from measured biomechanics — never from a guess.

    RED   : Δθ >= 14°  or sway > 2.8 px/frame
    AMBER : Δθ >= 6°   or sway > 1.6 px/frame
    GREEN : otherwise
    """
    if peak_asymmetry_deg >= 14.0 or sway_px > 2.8:
        return "RED"
    if peak_asymmetry_deg >= 6.0 or sway_px > 1.6:
        return "AMBER"
    return "GREEN"


def build_ehr_rx(run_metrics: dict, hitl: Optional[HITLLedger] = None) -> dict:
    """FHIR-style CarePlan bundle conditioned on measured biomechanics.

    run_metrics keys: peak_asymmetry_deg, mean_sway_px, agent_interventions,
    mean_agent_gain (confidence points), optional risk_tier override, clip label.
    An unrecognised risk_tier is ignored and the tier is computed instead.
    """
    peak_asym = float(run_metrics.get("peak_asymmetry_deg", 0.0))
    sway = float(run_metrics.get("mean_sway_px", 0.0))
    requested = str(run_metrics.get("risk_tier") or "").upper()
    tier_name = requested if requested in RISK_TIERS else stratify_risk(peak_asym, sway)
    tier = RISK_TIERS[tier_name]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "resourceType": "Bundle",
        "type": "document",
        "timestamp": now,
        "meta": {"tag": [{"code": "guardian-cv-rx"}]},
        "entry": [
            {
                "resource": {
                    "resourceType": "CarePlan",
                    "status": "active",
                    "intent": "plan",
                    "title": f"GuardianCV Rehabilitation Prescription — {tier_name}",
                    "description": tier["summary"],
                    "activity": [
                        {"detail": {"description": item}} for item in tier["focus"]
                    ],
                    "note": [{"text": p} for p in tier["precautions"]],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "Peak bilateral knee-flexion asymmetry Δθ"},
                    "valueQuantity": {"value": round(peak_asym, 2), "unit": "deg"},
                    "interpretation": tier_name,
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "Mean torso optical-flow sway magnitude"},
                    "valueQuantity": {"value": round(sway, 3), "unit": "px/frame"},
                    "interpretation": tier_name,
                }
            },
        ],
        "governance": {
            "frequency": tier["frequency"],
            "source_clip": run_metrics.get("clip", "unknown"),
            "generated_by": "guardian-cv/1.0 · ActivePerceptionAgent",
            "active_perception": {
                "interventions": int(run_metrics.get("agent_interventions", 0)),
                "mean_gain_points": round(float(run_metrics.get("mean_agent_gain", 0.0)), 2),
            },
            "hitl": hitl.snapshot() if hitl else {
                "approvals": 0, "dismissals": 0, "recent_events": [],
            },
            "disclaimer": (
                "Decision-support artefact. Requires licensed clinician "
                "co-signature before execution. Tier thresholds have not been "
                "validated against clinical ground truth."
            ),
        },
    }