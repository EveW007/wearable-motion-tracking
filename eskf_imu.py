# gyro-only orientation predictor for IMU ESKF

import numpy as np


def q_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)

    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    return q / n


def q_multiply(q1, q2):
    """
    Hamilton quaternion product.
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def q_from_gyro(omega_rad_s, dt):
    """
    Convert angular velocity into a small rotation quaternion.

    omega_rad_s: [gx, gy, gz] in rad/s
    dt: seconds
    """
    omega_rad_s = np.asarray(omega_rad_s, dtype=float)

    omega_norm = np.linalg.norm(omega_rad_s)

    if omega_norm < 1e-12 or dt <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    angle = omega_norm * dt
    axis = omega_rad_s / omega_norm

    half_angle = angle / 2.0

    dq = np.array([
        np.cos(half_angle),
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle),
    ])

    return q_normalize(dq)


def q_to_euler(q):
    """
    Convert quaternion [w, x, y, z] to roll, pitch, yaw in radians.
    """
    q = q_normalize(q)
    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


class GyroOnlyPredictor:
    """
    First step toward IMU ESKF.

    This only uses gyro integration:
        q_k = q_{k-1} ⊗ Exp(gyro * dt)

    No accelerometer correction yet.
    No covariance P yet.
    No gyro bias estimation yet.
    """

    def __init__(self, q0=None):
        if q0 is None:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self.q = q_normalize(q0)

    def step(self, gyro_rad_s, dt):
        dq = q_from_gyro(gyro_rad_s, dt)

        # Same convention as our Madgwick implementation:
        # current orientation updated by right multiplication
        self.q = q_multiply(self.q, dq)
        self.q = q_normalize(self.q)

        euler_rad = q_to_euler(self.q)

        return {
            "qw": self.q[0],
            "qx": self.q[1],
            "qy": self.q[2],
            "qz": self.q[3],
            "roll_rad": euler_rad[0],
            "pitch_rad": euler_rad[1],
            "yaw_rad": euler_rad[2],
        }