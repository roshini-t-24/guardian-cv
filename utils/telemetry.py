"""Module A — Mathematical frame-quality & SNR auditing.

Every number in this module is derived from the pixel matrix itself:

    Mean luminance          mu_L  = mean(I)                        on [0, 255]
    Laplacian sharpness     s2    = var(Laplacian(I))              CV_64F
    Shannon entropy         H(I)  = -SUM p_i * log2(p_i + eps)     256-bin calcHist
    Canny edge density      D     = |Canny(I, 35, 115)| / (W * H)
    Convex-hull solidity    S     = A(largest contour) / A(convexHull)

    Composite visual SNR confidence:

        C = 100 * clip(0.45*mu_hat + 0.30*H_hat + 0.15*D_hat + 0.10*S,
                       0.12, 0.994)          ->  C in [12.0 %, 99.4 %]

    where each "hat" term is the raw measurement divided by a full-scale
    constant (see below) and clipped to [0, 1].

Calibration
-----------
The full-scale constants are NOT theoretical maxima. Dividing luminance by
255, entropy by 8.0 bits and edge density by 20 % describes a frame no real
corridor ever produces, which pinned a healthy, well-lit frame at C ~ 60 %
— under the 72 % trigger — so the quality gate fired on every frame.

They are instead set to what a *well-exposed* ward frame measures, taken
from the three reference clips (480x360, MOG2+Canny mask):

                         mu_L    H      D        S      C (new)
    well lit (all clips) ~151   ~6.75  ~0.024   ~0.63   ~92 %
    night stress         ~54    ~5.6   ~0.014   ~0.54   ~55 %

The 72 % gate therefore sits between nominal and degraded. These numbers
come from synthetic clips; re-run the calibration on real ward footage
before trusting the thresholds clinically.

Glare is penalised too: above GLARE_ONSET the luminance term falls
linearly, so blown-out frames cannot score as "well exposed".

No filename inspection, no sine waves, no hardcoded confidence — the
camera is the only source of truth.
"""

from __future__ import annotations

import cv2
import numpy as np

EPSILON = 1e-9

# --- Active-perception trigger thresholds (imported by Module D) ----------
DEGRADATION_LUMINANCE = 85.0      # mean luminance below this => underexposed
DEGRADATION_CONFIDENCE = 72.0     # composite SNR below this => tracking risk

# --- Confidence model constants -------------------------------------------
CONFIDENCE_FLOOR_PCT = 12.0
CONFIDENCE_CEILING_PCT = 99.4

# Calibrated full-scale values (see module docstring).
LUMINANCE_FULL_SCALE = 128.0      # mid-grey or brighter == fully exposed
ENTROPY_FULL_SCALE = 7.0          # bits; textured ward scene, not the 8-bit ceiling
EDGE_DENSITY_FULL_SCALE = 0.03    # 3 % edge pixels == "fully textured" corridor
GLARE_ONSET = 190.0               # luminance above this starts to be penalised
GLARE_MAX_PENALTY = 0.40          # luminance term loses up to 40 % at mu_L = 255

WEIGHT_LUMINANCE = 0.45
WEIGHT_ENTROPY = 0.30
WEIGHT_EDGE_DENSITY = 0.15
WEIGHT_SOLIDITY = 0.10


def shannon_entropy(gray: np.ndarray) -> float:
    """Shannon image entropy H(I) = -SUM p_i log2(p_i + eps) over 256 bins.

    Uses cv2.calcHist on the actual grayscale histogram — never a model.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    probs = hist / max(float(gray.size), EPSILON)
    return float(-np.sum(probs * np.log2(probs + EPSILON)))


def _hull_solidity(mask: np.ndarray) -> float:
    """Area of the largest foreground contour divided by its convex-hull area.

    A clean human silhouette is convex-rich (S -> 1); fragmented noise
    blobs and half-lost trackers collapse toward 0.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    if area < 40.0 or hull_area < EPSILON:
        return 0.0
    return float(np.clip(area / hull_area, 0.0, 1.0))


def _luminance_term(luminance: float) -> float:
    """mu_hat in [0, 1]: rises to 1 at LUMINANCE_FULL_SCALE, falls with glare."""
    exposure = float(np.clip(luminance / LUMINANCE_FULL_SCALE, 0.0, 1.0))
    glare = float(np.clip(
        (luminance - GLARE_ONSET) / max(255.0 - GLARE_ONSET, EPSILON), 0.0, 1.0
    ))
    return exposure * (1.0 - GLARE_MAX_PENALTY * glare)


def compute_visual_telemetry(
    gray: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Compute the full deterministic telemetry vector for one frame.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel uint8 frame (BGR is auto-converted).
    mask : np.ndarray | None
        Optional uint8 foreground mask used for hull solidity.

    Returns
    -------
    dict with keys: luminance, sharpness, entropy, edge_density,
    hull_solidity, snr_confidence.
    """
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    luminance = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    entropy = shannon_entropy(gray)

    edges = cv2.Canny(gray, 35, 115)
    edge_density = float(np.count_nonzero(edges)) / max(float(gray.size), EPSILON)

    solidity = _hull_solidity(mask) if mask is not None else 0.0

    # Normalised contributions feeding the composite confidence model.
    mu_hat = _luminance_term(luminance)
    h_hat = float(np.clip(entropy / ENTROPY_FULL_SCALE, 0.0, 1.0))
    d_hat = float(np.clip(edge_density / EDGE_DENSITY_FULL_SCALE, 0.0, 1.0))

    composite = (
        WEIGHT_LUMINANCE * mu_hat
        + WEIGHT_ENTROPY * h_hat
        + WEIGHT_EDGE_DENSITY * d_hat
        + WEIGHT_SOLIDITY * solidity
    )
    confidence = 100.0 * float(np.clip(
        composite,
        CONFIDENCE_FLOOR_PCT / 100.0,
        CONFIDENCE_CEILING_PCT / 100.0,
    ))

    return {
        "luminance": round(luminance, 3),
        "sharpness": round(sharpness, 3),
        "entropy": round(entropy, 4),
        "edge_density": round(edge_density, 5),
        "hull_solidity": round(solidity, 4),
        "snr_confidence": round(confidence, 2),
    }


def quality_gate(telemetry: dict) -> tuple[bool, str]:
    """Return (degraded, human_reason) driving the active-perception trigger."""
    reasons: list[str] = []
    if telemetry["luminance"] < DEGRADATION_LUMINANCE:
        reasons.append(
            f"underexposed mu_L={telemetry['luminance']:.1f} < {DEGRADATION_LUMINANCE:.1f}"
        )
    if telemetry["snr_confidence"] < DEGRADATION_CONFIDENCE:
        reasons.append(
            f"SNR confidence {telemetry['snr_confidence']:.1f}% < {DEGRADATION_CONFIDENCE:.1f}%"
        )
    return (len(reasons) > 0, "; ".join(reasons) if reasons else "telemetry nominal")