"""Gait & stability analysis from silhouette pixels.

Why this module exists
----------------------
Reading knee angles off horizontal band centroids is unreliable in a side
view: the two legs overlap, "left" and "right" are just the leftmost and
rightmost blobs, and a knee can be paired with the other leg's ankle. That
made a healthy walk score as asymmetric as a limp.

Instead, a two-leg kinematic model (thigh + shank per leg, segment lengths
from the subject's own measured leg length) is fitted to the silhouette
every frame by maximising mask overlap (IoU), starting from the previous
frame's pose with a smoothness prior. Continuity of the fit — not screen
position — decides which leg is which, and the swing leg's flexion is
recovered even where the legs cross.

Everything is derived from pixels:

    silhouette   MOG2 background image  ->  |frame - background|  ->  Otsu
    leg pose     argmax IoU( rendered model , silhouette )
    COM / trunk  hip & head band centroids from the pipeline (cv2.moments)

Outputs (GaitAnalyzer.summary)
------------------------------
    peak_flexion_a / _b    90th-pct knee flexion per leg [deg] (robust with few strides)
    asymmetry_delta        |peak_a - peak_b| [deg]
    symmetry_index_pct     200 * |a - b| / (a + b)
    com_drop_pct           98th-pct downward hip excursion / body height
    lean_excursion_deg     98th-pct trunk-lean deviation from its median
    sway_rms_pct           detrended hip-x RMS / body height (standing only)
    risk_tier              GREEN / AMBER / RED, or INDETERMINATE when a walking
                           subject's legs cannot be observed independently

Limits, stated plainly
----------------------
* A side-view silhouette cannot tell left from right. Legs are reported as
  A and B; `stiffer_leg` names whichever one flexes less.
* A subject who is standing still is absorbed into the background model, so
  leg tracking is only meaningful while walking. Instability metrics use the
  pipeline's fused mask and still work for a standing subject.
* Tier thresholds were set on six clips (three procedural, three drawn
  figures). They are engineering calibration, NOT clinically validated.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# -- risk-tier thresholds (engineering calibration, not clinical) --------------
TIER_SI_AMBER = 10.0       # symmetry index %, peak knee flexion
TIER_SI_RED = 30.0
TIER_DROP_AMBER = 3.5      # COM drop, % of body height
TIER_DROP_RED = 6.0
TIER_SWAY_AMBER = 4.0      # detrended hip-x RMS, % of body height
TIER_SWAY_RED = 6.0

MIN_FIT_IOU = 0.5
MIN_STRIDES_PER_LEG = 2
LEG_COLLAPSE_R = 0.9        # fitted legs this correlated => one leg is hidden
WALKING_TRAVEL_BH = 0.5     # hip travel (body heights) that counts as walking
_TIER_RANK = {"GREEN": 0, "AMBER": 1, "RED": 2}


class LimbSilhouette:
    """Crisp subject silhouette from the MOG2 background image."""

    def __init__(self, learning_rate: float = 0.005) -> None:
        self.learning_rate = learning_rate
        self.reset()

    def reset(self) -> None:
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=80, varThreshold=16, detectShadows=False
        )

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        self._mog.apply(frame, learningRate=self.learning_rate)
        background = self._mog.getBackgroundImage()
        diff = cv2.cvtColor(cv2.absdiff(frame, background), cv2.COLOR_BGR2GRAY)
        diff = cv2.GaussianBlur(diff, (3, 3), 0)
        otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (diff > max(otsu * 0.7, 12.0)).astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K3)


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    return best if cv2.contourArea(best) > 250.0 else None


class LegTracker:
    """Fits [theta_a, flex_a, theta_b, flex_b] (deg) to the silhouette per frame.

    theta  thigh angle from the downward vertical, positive toward +x
    flex   knee flexion, the shank rotating opposite to the walking direction
    """

    _LO = np.array([-55.0, 0.0, -55.0, 0.0])
    _HI = np.array([55.0, 125.0, 55.0, 125.0])
    _SCALE = np.array([20.0, 40.0, 20.0, 40.0])

    def __init__(self, smoothness: float = 0.02) -> None:
        self.smoothness = smoothness
        self.reset()

    def reset(self) -> None:
        self.direction: Optional[int] = None
        self._prev: Optional[np.ndarray] = None
        self._prev2: Optional[np.ndarray] = None
        self._lengths: List[float] = []
        self._widths: List[float] = []
        self._bad_streak = 0

    def set_direction(self, direction: int) -> None:
        if direction != self.direction:
            self.direction = direction
            self._prev = self._prev2 = None

    # -- model rendering -----------------------------------------------------
    def _render(self, p, hip, th_len, sh_len, w_th, w_sh, origin, shape):
        canvas = np.zeros(shape, np.uint8)
        ox, oy = origin
        segments = []
        for theta, flex in ((p[0], p[1]), (p[2], p[3])):
            a = np.radians(theta)
            b = np.radians(theta - self.direction * flex)
            hx, hy = hip[0] - ox, hip[1] - oy
            kx, ky = hx + th_len * np.sin(a), hy + th_len * np.cos(a)
            ax, ay = kx + sh_len * np.sin(b), ky + sh_len * np.cos(b)
            cv2.line(canvas, (int(round(hx)), int(round(hy))),
                     (int(round(kx)), int(round(ky))), 255, w_th)
            cv2.line(canvas, (int(round(kx)), int(round(ky))),
                     (int(round(ax)), int(round(ay))), 255, w_sh)
            segments.append(((kx + ox, ky + oy), (ax + ox, ay + oy)))
        return canvas, segments

    # -- per-frame fit -----------------------------------------------------------
    def update(self, mask: np.ndarray) -> Optional[dict]:
        if self.direction is None:
            return None
        contour = _largest_contour(mask)
        if contour is None:
            return None
        x, y, w, h = cv2.boundingRect(contour)
        floor_y = y + h
        y_a, y_b = y + int(0.42 * h), y + int(0.54 * h) + 1
        mom = cv2.moments(mask[y_a:y_b, x : x + w])
        if mom["m00"] < 1.0:
            return None
        hip = (mom["m10"] / mom["m00"] + x, mom["m01"] / mom["m00"] + y_a)

        self._lengths.append(floor_y - hip[1])
        leg_len = float(np.median(self._lengths[-60:]))
        th_len, sh_len = 0.512 * leg_len, 0.488 * leg_len
        area = cv2.countNonZero(mask[int(hip[1]) : floor_y + 1, :])
        self._widths.append(area / (2.0 * max(leg_len, 1.0)))
        width = float(np.median(self._widths[-60:]))
        w_th, w_sh = max(int(round(width * 1.15)), 3), max(int(round(width * 0.95)), 3)

        x0 = max(int(hip[0] - 1.2 * leg_len), 0)
        y0 = max(int(hip[1] - 4), 0)
        x1 = min(int(hip[0] + 1.2 * leg_len), mask.shape[1])
        y1 = min(int(floor_y + 8), mask.shape[0])
        roi = mask[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        origin, shape = (x0, y0), roi.shape
        roi_area = cv2.countNonZero(roi)

        def cost(p, pred):
            model, _ = self._render(p, hip, th_len, sh_len, w_th, w_sh, origin, shape)
            inter = cv2.countNonZero(cv2.bitwise_and(model, roi))
            union = cv2.countNonZero(model) + roi_area - inter
            iou = inter / max(union, 1)
            prior = 0.0
            if pred is not None:
                prior = self.smoothness * float(np.sum(((p - pred) / self._SCALE) ** 2))
            return (1.0 - iou) + prior, iou

        if self._prev is None:
            best_cost, best_p = None, None
            for t1 in range(-40, 41, 10):
                for t2 in range(t1, 41, 10):
                    for f1 in (0, 30, 60):
                        for f2 in (0, 30, 60):
                            cand = np.array([t1, f1, t2, f2], float)
                            c, _ = cost(cand, None)
                            if best_cost is None or c < best_cost:
                                best_cost, best_p = c, cand
            p, pred = best_p, None
        else:
            drift = 0.6 * (self._prev - self._prev2) if self._prev2 is not None else 0.0
            pred = np.clip(self._prev + drift, self._LO, self._HI)
            p = pred.copy()

        current, iou = cost(p, pred)
        for step in (8.0, 4.0, 2.0, 1.0):
            for _ in range(4):
                improved = False
                for i in range(4):
                    for sign in (-1.0, 1.0):
                        q = p.copy()
                        q[i] = np.clip(q[i] + sign * step, self._LO[i], self._HI[i])
                        c, io = cost(q, pred)
                        if c < current - 1e-6:
                            p, current, iou, improved = q, c, io, True
                if not improved:
                    break

        if iou < MIN_FIT_IOU:
            self._bad_streak += 1
            if self._bad_streak >= 10:
                self._prev = self._prev2 = None      # re-initialise from scratch
            return {"valid": False, "iou": float(iou)}
        self._bad_streak = 0
        self._prev2, self._prev = self._prev, p.copy()
        _, segments = self._render(p, hip, th_len, sh_len, w_th, w_sh, origin, shape)
        return {
            "valid": True,
            "iou": float(iou),
            "flex": (float(p[1]), float(p[3])),
            "knees": (segments[0][0], segments[1][0]),
            "ankles": (segments[0][1], segments[1][1]),
        }


def _swing_peaks(series: np.ndarray, min_sep: int, min_prominence: float = 10.0) -> List[float]:
    """Swing-phase flexion maxima: local peaks with prominence, min_sep apart."""
    n = len(series)
    if n < 2 * min_sep + 1:
        return []
    padded = np.pad(series, 1, mode="edge")
    smooth = np.array([np.median(padded[i : i + 3]) for i in range(n)])
    peaks: List[int] = []
    for i in range(1, n - 1):
        if smooth[i] < smooth[i - 1] or smooth[i] <= smooth[i + 1]:
            continue
        left = smooth[max(0, i - min_sep) : i + 1].min()
        right = smooth[i : min(n, i + min_sep + 1)].min()
        if smooth[i] - max(left, right) >= min_prominence:
            peaks.append(i)
    kept: List[int] = []
    for i in peaks:
        if kept and i - kept[-1] < min_sep:
            if smooth[i] > smooth[kept[-1]]:
                kept[-1] = i
        else:
            kept.append(i)
    return [float(smooth[i]) for i in kept]


class GaitAnalyzer:
    """Streaming gait + stability analyser.

    Per frame call update(frame, joints, bbox) with the output of
    ClinicalVisionPipeline.extract_joints for that same frame.
    """

    MAX_HISTORY = 900

    def __init__(self, fps: float = 15.0) -> None:
        self.fps = fps
        self._silhouette = LimbSilhouette()
        self._legs = LegTracker()
        self.reset()

    def reset(self) -> None:
        self._silhouette.reset()
        self._legs.reset()
        self._flex_a: List[float] = []
        self._flex_b: List[float] = []
        self._com: List[Tuple[float, float, float, float, float]] = []  # hip x,y head x,y bbox_h
        self._hip_x_recent: List[float] = []
        self._frames = 0
        self._fit_attempts = 0
        self._fit_valid = 0

    # -- streaming update ------------------------------------------------------------
    def update(
        self,
        frame: np.ndarray,
        joints: Optional[Dict[str, Tuple[float, float]]],
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> dict:
        self._frames += 1
        limb_mask = self._silhouette(frame)
        result: dict = {"valid": False, "iou": 0.0, "flex_a": None, "flex_b": None,
                        "lean_deg": None, "joints": dict(joints) if joints else None}

        frame_w = frame.shape[1]
        inside = False
        if joints and bbox:
            bx, _, bw, bh = bbox
            # A subject cut by the frame edge, or a whole-frame blob while the
            # background model warms up, would corrupt every centroid.
            inside = bx > 1 and bx + bw < frame_w - 1 and bw < 0.5 * frame_w
        if inside:
            hip, head = joints["hip"], joints["head"]
            self._com.append((hip[0], hip[1], head[0], head[1], float(bh)))
            self._com = self._com[-self.MAX_HISTORY :]
            self._hip_x_recent = (self._hip_x_recent + [hip[0]])[-9:]
            if len(self._hip_x_recent) >= 9:
                travel = self._hip_x_recent[-1] - self._hip_x_recent[0]
                if abs(travel) > 0.1 * bh:
                    self._legs.set_direction(1 if travel > 0 else -1)
            elif self._legs.direction is None:
                self._legs.set_direction(1)
            result["lean_deg"] = float(
                np.degrees(np.arctan2(head[0] - hip[0], hip[1] - head[1]))
            )

        fit = self._legs.update(limb_mask) if inside else None
        if fit is not None:
            self._fit_attempts += 1
        if fit and fit["valid"]:
            self._fit_valid += 1
            self._flex_a.append(fit["flex"][0])
            self._flex_b.append(fit["flex"][1])
            self._flex_a, self._flex_b = (self._flex_a[-self.MAX_HISTORY :],
                                          self._flex_b[-self.MAX_HISTORY :])
            result.update(valid=True, iou=fit["iou"], flex_a=fit["flex"][0],
                          flex_b=fit["flex"][1])
            merged = dict(result["joints"] or {})
            merged["l_knee"], merged["r_knee"] = fit["knees"]
            merged["l_ankle"], merged["r_ankle"] = fit["ankles"]
            result["joints"] = merged
        elif fit:
            result["iou"] = fit["iou"]
        return result

    # -- clip-level metrics -------------------------------------------------------------
    def _usable_com(self) -> np.ndarray:
        """COM samples with warm-up / ghost frames removed (rows: hx hy gx gy h)."""
        if len(self._com) < 20:
            return np.zeros((0, 5))
        com = np.array(self._com)
        return com[com[:, 4] >= 0.85 * np.percentile(com[:, 4], 90)]

    def _gait_metrics(self, walking: bool) -> dict:
        a, b = np.array(self._flex_a), np.array(self._flex_b)
        out: dict = {"strides_a": 0, "strides_b": 0, "gait_sufficient": False,
                     "legs_resolved": None}
        if not walking or len(a) < 20:
            return out
        out["fit_rate"] = round(self._fit_valid / max(self._fit_attempts, 1), 2)
        if out["fit_rate"] < 0.5:
            return out
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0
        out["leg_correlation"] = round(corr, 2)
        # Healthy legs swing in anti-phase. Two fitted legs that move together
        # mean the silhouette hides one of them: asymmetry is not observable.
        out["legs_resolved"] = corr <= LEG_COLLAPSE_R
        if not out["legs_resolved"]:
            return out
        min_sep = max(4, int(round(0.35 * self.fps)))
        out["strides_a"] = len(_swing_peaks(a, min_sep))
        out["strides_b"] = len(_swing_peaks(b, min_sep))
        if min(out["strides_a"], out["strides_b"]) < MIN_STRIDES_PER_LEG:
            return out
        pa, pb = float(np.percentile(a, 90)), float(np.percentile(b, 90))
        out.update(
            gait_sufficient=True,
            peak_flexion_a=round(pa, 1),
            peak_flexion_b=round(pb, 1),
            asymmetry_delta=round(abs(pa - pb), 1),
            symmetry_index_pct=round(200.0 * abs(pa - pb) / max(pa + pb, 1e-6), 1),
            stiffer_leg="A" if pa < pb else "B",
        )
        return out

    def _stability_metrics(self, com: np.ndarray, walking: bool) -> dict:
        if len(com) < 20:
            return {"stability_sufficient": False, "walking": False}
        hip_x, hip_y, head_x, head_y, heights = com.T
        height = float(np.percentile(heights, 90))
        t = np.arange(len(com))
        drop = np.percentile(hip_y - np.median(hip_y), 98)
        lean = np.degrees(np.arctan2(head_x - hip_x, hip_y - head_y))
        out = {
            "stability_sufficient": True,
            "walking": walking,
            "com_drop_pct": round(float(100.0 * max(drop, 0.0) / height), 2),
            "lean_excursion_deg": round(float(np.percentile(np.abs(lean - np.median(lean)), 98)), 1),
            "body_height_px": round(height, 1),
        }
        if not walking:
            # Quiet-stance sway; for a walking subject hip-x is just gait speed.
            detrended = hip_x - np.polyval(np.polyfit(t, hip_x, 1), t)
            out["sway_rms_pct"] = round(float(100.0 * detrended.std() / height), 2)
        return out

    def summary(self) -> dict:
        com = self._usable_com()
        walking = bool(len(com) >= 20 and np.ptp(com[:, 0]) >= WALKING_TRAVEL_BH * np.percentile(com[:, 4], 90))
        metrics = {**self._gait_metrics(walking), **self._stability_metrics(com, walking),
                   "frames": self._frames}
        metrics.update(stratify(metrics))
        return metrics


def stratify(m: dict) -> dict:
    """Tier from measured metrics, with the reasons behind it.

    GREEN / AMBER / RED, or INDETERMINATE when the subject walked but the
    legs could not be observed independently and nothing else raised a flag:
    "could not see" must never be reported as "healthy".
    """
    tier, reasons = "GREEN", []

    def raise_to(level: str, why: str) -> None:
        nonlocal tier
        reasons.append(why)
        if _TIER_RANK[level] > _TIER_RANK[tier]:
            tier = level

    if m.get("gait_sufficient"):
        si = m["symmetry_index_pct"]
        if si >= TIER_SI_RED:
            raise_to("RED", f"knee-flexion symmetry index {si:.0f}% >= {TIER_SI_RED:.0f}%")
        elif si >= TIER_SI_AMBER:
            raise_to("AMBER", f"knee-flexion symmetry index {si:.0f}% >= {TIER_SI_AMBER:.0f}%")
    if m.get("stability_sufficient"):
        for key, label, amber, red in (
            ("com_drop_pct", "COM drop", TIER_DROP_AMBER, TIER_DROP_RED),
            ("sway_rms_pct", "postural sway", TIER_SWAY_AMBER, TIER_SWAY_RED),
        ):
            value = m.get(key)
            if value is None:
                continue
            if value >= red:
                raise_to("RED", f"{label} {value:.1f}% of height >= {red:.1f}%")
            elif value >= amber:
                raise_to("AMBER", f"{label} {value:.1f}% of height >= {amber:.1f}%")

    if not (m.get("gait_sufficient") or m.get("stability_sufficient")):
        return {"risk_tier": None, "tier_reasons": ["insufficient data to stratify"]}
    if m.get("walking") and not m.get("gait_sufficient"):
        why = ("legs not independently visible (fitted legs move together)"
               if m.get("legs_resolved") is False else "too few tracked strides")
        reasons.append(f"gait asymmetry not assessable: {why}")
        if tier == "GREEN":
            tier = "INDETERMINATE"
    return {"risk_tier": tier, "tier_reasons": reasons or ["all measured indicators nominal"]}