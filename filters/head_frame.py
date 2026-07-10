"""Convert Madgwick sensor orientation into a neutral-relative head frame.

Quaternion notation used in this module is ``q_A_B``: the quaternion rotates
a vector expressed in frame B into frame A.  The Madgwick filter returns
``q_world_sensor``.

The fixed wearable mounting is ``q_head_sensor``.  Therefore::

    q_world_head = q_world_sensor * conjugate(q_head_sensor)

After recording a neutral, forward-looking head pose, its orientation is used
as the reference world frame::

    q_neutral_head = conjugate(q_world_head_neutral) * q_world_head

Euler angles should only be calculated after these quaternion operations.
Subtracting three Euler angles independently is not an equivalent operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from madgwick_filter import (
    Quaternion,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_euler,
)


def quaternion_conjugate(q: Sequence[float]) -> Quaternion:
    """Return the conjugate/inverse of a unit quaternion."""
    return [q[0], -q[1], -q[2], -q[3]]


def quaternion_dot(q1: Sequence[float], q2: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(q1, q2))


def average_quaternions(quaternions: Iterable[Sequence[float]]) -> Quaternion:
    """Average a compact cluster of quaternions, handling q/-q ambiguity.

    This normalized, sign-aligned mean is intended for a short static neutral
    calibration window.  It is not a general orientation trajectory average.
    """
    samples = [quaternion_normalize(list(q)) for q in quaternions]
    if not samples:
        raise ValueError("At least one quaternion is required for calibration")

    reference = samples[0]
    total = [0.0, 0.0, 0.0, 0.0]
    for sample in samples:
        if quaternion_dot(reference, sample) < 0.0:
            sample = [-value for value in sample]
        for index, value in enumerate(sample):
            total[index] += value
    return quaternion_normalize(total)


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Create a scalar-first quaternion from ZYX Euler angles in radians."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return quaternion_normalize([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


@dataclass
class HeadFrameTransform:
    """Apply fixed mounting correction and a neutral-pose reference.

    ``q_head_sensor`` maps sensor-frame vectors into the anatomical head frame.
    Keep it at identity only when the mounted sensor axes intentionally match
    the chosen head axes.
    """

    q_head_sensor: Quaternion = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0]
    )
    q_world_head_neutral: Quaternion | None = None

    def __post_init__(self) -> None:
        self.q_head_sensor = quaternion_normalize(list(self.q_head_sensor))
        if self.q_world_head_neutral is not None:
            self.q_world_head_neutral = quaternion_normalize(
                list(self.q_world_head_neutral)
            )

    def sensor_to_head_orientation(
        self, q_world_sensor: Sequence[float]
    ) -> Quaternion:
        """Return absolute head orientation ``q_world_head``."""
        return quaternion_normalize(
            quaternion_multiply(
                quaternion_normalize(list(q_world_sensor)),
                quaternion_conjugate(self.q_head_sensor),
            )
        )

    def set_neutral(self, q_world_sensor_samples: Iterable[Sequence[float]]) -> None:
        """Set neutral pose from a short, static group of filter outputs."""
        head_samples = [
            self.sensor_to_head_orientation(q)
            for q in q_world_sensor_samples
        ]
        self.q_world_head_neutral = average_quaternions(head_samples)

    @property
    def calibrated(self) -> bool:
        return self.q_world_head_neutral is not None

    def transform(self, q_world_sensor: Sequence[float]) -> Dict[str, float]:
        """Return neutral-relative head quaternion and ZYX Euler angles."""
        if self.q_world_head_neutral is None:
            raise RuntimeError("Head-frame neutral pose has not been calibrated")

        q_world_head = self.sensor_to_head_orientation(q_world_sensor)
        q_neutral_head = quaternion_normalize(
            quaternion_multiply(
                quaternion_conjugate(self.q_world_head_neutral),
                q_world_head,
            )
        )
        euler = quaternion_to_euler(q_neutral_head)
        return {
            "qw": q_neutral_head[0],
            "qx": q_neutral_head[1],
            "qy": q_neutral_head[2],
            "qz": q_neutral_head[3],
            "roll": euler["roll"],
            "pitch": euler["pitch"],
            "yaw": euler["yaw"],
        }
