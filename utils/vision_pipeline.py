"""Module C — Pure-OpenCV 5 anatomical slicing & Zero-PII engine.

Per-frame pipeline
------------------
1. Hybrid foreground segmentation
       MOG2(history=80, varThreshold=16, detectShadows=False) fused with
       Canny(35, 115) structural edges, then elliptical morphological
       closing + opening — moving AND momentarily stationary limbs stay
       segmented.

       The background model must see exactly ONE frame per time step, and
       that frame must be the one the rest of the pipeline works on (the
       restored frame when the active-perception agent is on). Callers that
       only need a mask from some other image pass learn=False, which
       freezes the model (learningRate=0).
2. Pixel-driven anatomical band slicing (extract_joints)
       bbox of the largest contour -> four horizontal physiological zones
       -> cv2.moments centroids. Knee / ankle zones resolve dual limb
       contours, or bisect a merged mid-stance blob at its median column.
3. Sagittal knee-flexion trigonometry (calculate_angle)
       theta = arccos( (BA . BC) / (|BA||BC| + eps) ) in degrees.
4. Zero-PII privacy shield (apply_privacy_shield)
       "silhouette" — black canvas + neon Canny edges + teal morphological
                      contour + solid 32 px cranial redaction with a
                      "PII REDACTED" HUD tag.
       "corridor"   — room context kept, GaussianBlur((45,45), 25) over
                      the cranial ROI.
5. Dense optical-flow sway analysis
       calcOpticalFlowFarneback + cartToPolar -> directional vectors and
       a torso-ROI sway magnitude in px/frame, averaged over foreground
       pixels only so it does not depend on the bbox aspect ratio.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]

# Fractional vertical zones inside the subject bounding box (spec C.2).
ZONE_HEAD = (0.00, 0.18)     # cranial zone
ZONE_HIP = (0.42, 0.54)      # pelvic / hip centre of mass
ZONE_KNEE = (0.64, 0.78)     # tibiofemoral zone
ZONE_ANKLE = (0.82, 0.98)    # malleolar zone

HEAD_REDACTION_RADIUS = 32   # px — solid disc over the cranial ROI
BLUR_KERNEL = (45, 45)
BLUR_SIGMA = 25.0

# HUD palette (BGR) ---------------------------------------------------------
NEON_EDGE_BGR = (140, 255, 190)      # Canny edge tint on the shield canvas
SILHOUETTE_BGR = (190, 216, 62)      # teal morphological contour tint
REDACTION_BGR = (20, 20, 20)         # solid cranial redaction fill
REDACTION_TAG_BGR = (94, 110, 242)   # coral "PII REDACTED" HUD tag
FLOW_VECTOR_BGR = (255, 214, 90)     # optical-flow arrows


def _new_subtractor() -> cv2.BackgroundSubtractorMOG2:
    return cv2.createBackgroundSubtractorMOG2(
        history=80, varThreshold=16, detectShadows=False
    )


class ClinicalVisionPipeline:
    """Hybrid MOG2 + Canny segmentation with anatomical zone joint extraction."""

    def __init__(self, learning_rate: float = 0.005) -> None:
        # MOG2's automatic rate (1/history) absorbs a person who stands still
        # into the background within a few seconds — exactly the postural-sway
        # case this system exists to watch. A slow fixed rate keeps them
        # segmented (measured: joint detection on the standing clip at night
        # 78 % -> 100 %). Trade-off: a stale background clears more slowly.
        self.learning_rate = learning_rate
        self._subtractor = _new_subtractor()
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def reset(self) -> None:
        """Re-learn the background model — call when a new clip starts."""
        self._subtractor = _new_subtractor()

    # -- 1. hybrid segmentation ----------------------------------------------
    def segment_foreground(self, frame: np.ndarray, learn: bool = True) -> np.ndarray:
        """Fuse motion (MOG2) with structure (Canny) into a clean uint8 mask.

        learn=True  advance the background model by this frame (default,
                    at self.learning_rate).
        learn=False segment against the current model without updating it.
        Feed the model once per time step, on the frame you actually track.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = self._subtractor.apply(
            frame, learningRate=self.learning_rate if learn else 0.0
        )
        edges = cv2.Canny(gray, 35, 115)

        # Trust structural edges only near existing motion — this keeps
        # stationary limbs segmented without lighting up the whole corridor.
        motion_neighbourhood = cv2.dilate(motion, self._close_kernel)
        fused = cv2.bitwise_or(motion, cv2.bitwise_and(edges, motion_neighbourhood))
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, self._close_kernel)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, self._open_kernel)
        return fused

    # -- 2. anatomical band slicing --------------------------------------------
    def extract_joints(
        self, mask: np.ndarray
    ) -> Tuple[Optional[Dict[str, Point]], Optional[BBox]]:
        """Slice the silhouette into physiological bands and locate six joints."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        subject = max(contours, key=cv2.contourArea)
        if cv2.contourArea(subject) < 250.0:
            return None, None

        x, y, w, h = (int(v) for v in cv2.boundingRect(subject))
        bbox: BBox = (x, y, w, h)
        joints: Dict[str, Point] = {}

        joints["head"] = self._zone_centroid(mask, bbox, ZONE_HEAD) or (
            x + w / 2.0, y + h * 0.09,
        )
        joints["hip"] = self._zone_centroid(mask, bbox, ZONE_HIP) or (
            x + w / 2.0, y + h * 0.48,
        )

        for side, zone in (("knee", ZONE_KNEE), ("ankle", ZONE_ANKLE)):
            pair = self._zone_limb_pair(mask, bbox, zone)
            if pair is None:
                cx = x + w / 2.0
                cy = y + h * (zone[0] + zone[1]) / 2.0
                pair = ((cx - w * 0.10, cy), (cx + w * 0.10, cy))
            joints[f"l_{side}"], joints[f"r_{side}"] = pair

        return joints, bbox

    # -- slicing helpers ---------------------------------------------------------
    @staticmethod
    def _slice(
        mask: np.ndarray, bbox: BBox, zone: Tuple[float, float]
    ) -> Tuple[np.ndarray, int, int]:
        x, y, w, h = bbox
        y0 = y + int(h * zone[0])
        y1 = max(y0 + 1, y + int(h * zone[1]))
        return mask[y0:y1, x : x + w], x, y0

    @staticmethod
    def _centroid_from_moments(
        moments: dict, offset_x: int, offset_y: int
    ) -> Optional[Point]:
        m00 = moments["m00"]
        if m00 < 1e-3:
            return None
        return (moments["m10"] / m00 + offset_x, moments["m01"] / m00 + offset_y)

    def _zone_centroid(
        self, mask: np.ndarray, bbox: BBox, zone: Tuple[float, float]
    ) -> Optional[Point]:
        slab, ox, oy = self._slice(mask, bbox, zone)
        if slab.size == 0:
            return None
        return self._centroid_from_moments(cv2.moments(slab), ox, oy)

    def _zone_limb_pair(
        self, mask: np.ndarray, bbox: BBox, zone: Tuple[float, float]
    ) -> Optional[Tuple[Point, Point]]:
        """Two lateral centroids from one band — real limbs or a median split."""
        slab, ox, oy = self._slice(mask, bbox, zone)
        if slab.size == 0 or cv2.countNonZero(slab) < 20:
            return None

        contours, _ = cv2.findContours(slab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 12.0]

        if len(contours) >= 2:
            contours.sort(key=cv2.contourArea, reverse=True)
            pts = [
                self._centroid_from_moments(cv2.moments(c), ox, oy)
                for c in contours[:2]
            ]
            if all(p is not None for p in pts):
                return self._order_left_right(pts[0], pts[1])

        # Single merged mid-stance blob: split at the median foreground column.
        columns = np.where(slab.any(axis=0))[0]
        if columns.size < 4:
            return None
        split = int(np.median(columns))
        left_pt = self._centroid_from_moments(cv2.moments(slab[:, :split]), ox, oy)
        right_pt = self._centroid_from_moments(cv2.moments(slab[:, split:]), ox + split, oy)
        if left_pt is None or right_pt is None:
            return None
        return self._order_left_right(left_pt, right_pt)

    @staticmethod
    def _order_left_right(a: Point, b: Point) -> Tuple[Point, Point]:
        return (a, b) if a[0] <= b[0] else (b, a)


# -- 3. sagittal-plane trigonometry ----------------------------------------
def calculate_angle(a: Point, b: Point, c: Point) -> float:
    """Interior angle at b between segments b->a and b->c, in degrees.

    theta = arccos( (BA . BC) / (|BA| |BC| + eps) ) * 180 / pi
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cos_theta = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def knee_flexion(joints: Optional[Dict[str, Point]]) -> Optional[dict]:
    """Bilateral sagittal knee flexion + gait asymmetry delta Δθ."""
    required = ("hip", "l_knee", "r_knee", "l_ankle", "r_ankle")
    if joints is None or any(k not in joints for k in required):
        return None
    theta_left = calculate_angle(joints["hip"], joints["l_knee"], joints["l_ankle"])
    theta_right = calculate_angle(joints["hip"], joints["r_knee"], joints["r_ankle"])
    return {
        "theta_left": round(theta_left, 2),
        "theta_right": round(theta_right, 2),
        "asymmetry_delta": round(abs(theta_left - theta_right), 2),
    }


# -- 4. Zero-PII privacy shield ----------------------------------------------
def apply_privacy_shield(
    frame: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    joints: Optional[Dict[str, Point]],
    mode: str = "silhouette",
) -> np.ndarray:
    """Render the Zero-PII viewport.

    mode "silhouette" — black zero-PII canvas, neon Canny edges, teal
                        morphological contours, solid 32 px cranial
                        redaction + "PII REDACTED" HUD tag.
    mode "corridor"   — keep the room context, heavy Gaussian blur over
                        the tracked cranial ROI.
    """
    if mode == "corridor":
        out = frame.copy()
        if joints and "head" in joints:
            hx, hy = int(joints["head"][0]), int(joints["head"][1])
            height, width = out.shape[:2]
            x0, y0 = max(hx - 36, 0), max(hy - 40, 0)
            x1, y1 = min(hx + 36, width), min(hy + 32, height)
            if x1 > x0 and y1 > y0:
                out[y0:y1, x0:x1] = cv2.GaussianBlur(
                    out[y0:y1, x0:x1], BLUR_KERNEL, BLUR_SIGMA
                )
                cv2.putText(
                    out, "PII SHIELDED", (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, REDACTION_TAG_BGR, 1, cv2.LINE_AA,
                )
        return out

    # Full silhouette shield: raw RGB background & clothing never render.
    canvas = np.zeros_like(frame)
    edges = cv2.Canny(gray, 35, 115)
    canvas[edges > 0] = NEON_EDGE_BGR

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, SILHOUETTE_BGR, 2, cv2.LINE_AA)

    if joints and "head" in joints:
        hx, hy = int(joints["head"][0]), int(joints["head"][1])
        cv2.circle(canvas, (hx, hy), HEAD_REDACTION_RADIUS, REDACTION_BGR, -1)
        cv2.putText(
            canvas, "PII REDACTED", (hx - 52, max(hy - HEAD_REDACTION_RADIUS - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, REDACTION_TAG_BGR, 1, cv2.LINE_AA,
        )
    return canvas


# -- 5. dense optical flow -----------------------------------------------------
def compute_dense_flow(
    prev_gray: np.ndarray, gray: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Farneback dense flow between two frames -> (magnitude, angle) fields."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag, ang


def flow_sway_magnitude(
    mag: np.ndarray,
    bbox: Optional[BBox] = None,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Mean optical-flow magnitude (px/frame) on the subject.

    With a foreground mask the mean is taken over silhouette pixels inside
    the bbox only; without one it falls back to the whole bbox, then the
    whole frame. Masked averaging keeps the number from shrinking or
    growing with how much empty corridor the bbox happens to enclose.
    """
    if bbox is not None:
        x, y, w, h = bbox
        roi = mag[y : y + h, x : x + w]
        if roi.size:
            if mask is not None:
                fg = mask[y : y + h, x : x + w] > 0
                if np.count_nonzero(fg) >= 20:
                    return float(np.mean(roi[fg]))
            return float(np.mean(roi))
    return float(np.mean(mag))


def draw_flow_vectors(
    frame: np.ndarray,
    mag: np.ndarray,
    ang: np.ndarray,
    step: int = 28,
    min_mag: float = 1.2,
) -> None:
    """Overlay directional motion arrows where the flow is significant."""
    height, width = frame.shape[:2]
    for fy in range(step // 2, height, step):
        for fx in range(step // 2, width, step):
            if mag[fy, fx] < min_mag:
                continue
            dx = int(np.cos(ang[fy, fx]) * mag[fy, fx] * 4)
            dy = int(np.sin(ang[fy, fx]) * mag[fy, fx] * 4)
            cv2.arrowedLine(
                frame, (fx, fy), (fx + dx, fy + dy),
                FLOW_VECTOR_BGR, 1, cv2.LINE_AA, tipLength=0.35,
            )