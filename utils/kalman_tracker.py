"""Module B - 6-joint 4-D state-space Kalman filter bank.

One independent cv2.KalmanFilter(4, 2) per anatomical joint:
    state x_k = [x, y, vx, vy]^T  (pixels, pixels / frame)
Missing joints coast on the constant-velocity model (predict only); the
innovation residual of measured joints is the stumble / tracking-loss signal.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

JOINT_NAMES: Tuple[str, ...] = (
    "head",
    "hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
)

Measurement = Tuple[float, float]


def _build_filter(process_noise: float, measurement_noise: float) -> cv2.KalmanFilter:
    """Instantiate one constant-velocity cv2.KalmanFilter(4, 2)."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


class JointKalmanBank:
    """Bank of six constant-velocity Kalman filters with dropout coasting."""

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 4.0,
    ) -> None:
        self._filters = {
            name: _build_filter(process_noise, measurement_noise)
            for name in JOINT_NAMES
        }
        self._initialised = {name: False for name in JOINT_NAMES}

    def reset(self) -> None:
        """Forget all state - call when a new clip starts."""
        for name in JOINT_NAMES:
            self._initialised[name] = False

    def update(
        self,
        measurements: Mapping[str, Optional[Measurement]],
    ) -> dict:
        """Advance every filter by one frame.

        Returns dict with: smoothed_joints, joint_velocities,
        innovation_error, tracked_joints.
        """
        smoothed: dict[str, tuple[float, float]] = {}
        velocities: dict[str, float] = {}
        tracked: list[str] = []
        residuals: list[float] = []

        for name in JOINT_NAMES:
            kf = self._filters[name]
            measurement = measurements.get(name)

            if measurement is None:
                if not self._initialised[name]:
                    continue
                state = kf.predict()
            else:
                z = np.array(
                    [[np.float32(measurement[0])], [np.float32(measurement[1])]],
                    dtype=np.float32,
                )
                if not self._initialised[name]:
                    kf.statePost = np.array(
                        [[measurement[0]], [measurement[1]], [0.0], [0.0]],
                        dtype=np.float32,
                    )
                    self._initialised[name] = True
                    state = kf.statePost
                else:
                    predicted = kf.predict()
                    residual = z - kf.measurementMatrix @ predicted
                    residuals.append(float(np.linalg.norm(residual)))
                    state = kf.correct(z)

            smoothed[name] = (float(state[0, 0]), float(state[1, 0]))
            velocities[name] = float(np.hypot(state[2, 0], state[3, 0]))
            if measurement is not None:
                tracked.append(name)

        innovation = float(np.mean(residuals)) if residuals else 0.0
        return {
            "smoothed_joints": smoothed,
            "joint_velocities": velocities,
            "innovation_error": round(innovation, 3),
            "tracked_joints": tracked,
        }