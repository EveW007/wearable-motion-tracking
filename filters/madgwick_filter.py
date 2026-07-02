"""
Python port of the provided TypeScript MadgwickFilter.

Inputs to MadgwickFilter.update():
    accel_x, accel_y, accel_z : accelerometer readings, normally m/s^2 or g
    gyro_x, gyro_y, gyro_z    : gyroscope readings in rad/s
    dt                        : sample interval in seconds

Output angles are in radians.
"""

# algorithm itself is based on the original Madgwick paper

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


# Expected gravity magnitude (m/s^2) and tolerance for init.
# Accepts 7.8-11.8 m/s^2 to handle noisy readings at startup.
GRAVITY_EXPECTED = 9.8
GRAVITY_TOLERANCE = 2.0


Quaternion = List[float]
Vector3 = List[float]


def quaternion_normalize(q: Quaternion) -> Quaternion:
    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return [1.0, 0.0, 0.0, 0.0]
    return [w / norm, x / norm, y / norm, z / norm]


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quaternion_from_gyro(gx: float, gy: float, gz: float, dt: float) -> Quaternion:
    """Return the incremental quaternion caused by angular velocity over dt.

    gx, gy, gz must be in rad/s, dt in seconds.
    """
    norm = math.sqrt(gx * gx + gy * gy + gz * gz)
    angle = norm * dt
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    half_angle = angle / 2.0
    s = math.sin(half_angle) / norm
    q = [math.cos(half_angle), gx * s, gy * s, gz * s]
    return quaternion_normalize(q)


def quaternion_to_euler(q: Quaternion) -> Dict[str, float]:
    """Convert quaternion [w, x, y, z] to roll, pitch, yaw in radians."""
    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return {"roll": roll, "pitch": pitch, "yaw": yaw}


def quaternion_slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]

    if dot < 0.0:
        q2 = [-q2[0], -q2[1], -q2[2], -q2[3]]
        dot = -dot

    # When quaternions are nearly aligned, SLERP's sin(theta) denominator
    # approaches zero. Fall back to normalised linear interpolation (NLERP).
    if dot > 0.9995:
        result = [
            q1[0] + t * (q2[0] - q1[0]),
            q1[1] + t * (q2[1] - q1[1]),
            q1[2] + t * (q2[2] - q1[2]),
            q1[3] + t * (q2[3] - q1[3]),
        ]
        return quaternion_normalize(result)

    theta0 = math.acos(abs(dot))
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0

    return [
        s0 * q1[0] + s1 * q2[0],
        s0 * q1[1] + s1 * q2[1],
        s0 * q1[2] + s1 * q2[2],
        s0 * q1[3] + s1 * q2[3],
    ]


def quaternion_rotate_vector(q: Quaternion, v: Vector3) -> Vector3:
    w, x, y, z = q
    vx, vy, vz = v

    # v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + w * v)
    cx1 = y * vz - z * vy
    cy1 = z * vx - x * vz
    cz1 = x * vy - y * vx

    tx = cx1 + w * vx
    ty = cy1 + w * vy
    tz = cz1 + w * vz

    cx2 = y * tz - z * ty
    cy2 = z * tx - x * tz
    cz2 = x * ty - y * tx

    return [vx + 2.0 * cx2, vy + 2.0 * cy2, vz + 2.0 * cz2]


def calculate_gravity_apple_convention(q: Quaternion, g: float = 9.807) -> Vector3:
    q_conj = [q[0], -q[1], -q[2], -q[3]]
    return quaternion_rotate_vector(q_conj, [0.0, 0.0, g])


def init_quaternion_from_gravity(accel_x: float, accel_y: float, accel_z: float) -> Quaternion:
    """Compute initial quaternion from gravity reading.

    Returns [w, x, y, z] such that calculate_gravity_apple_convention(q) ~= accel.
    """
    norm = math.sqrt(accel_x * accel_x + accel_y * accel_y + accel_z * accel_z)

    if abs(norm - GRAVITY_EXPECTED) < GRAVITY_TOLERANCE and norm > 0.0:
        gx = accel_x / norm
        gy = accel_y / norm
        gz = accel_z / norm

        # Rotation from [0,0,1] to gravity: axis = [0,0,1] x [gx,gy,gz]
        vx = -gy
        vy = gx
        v_norm = math.sqrt(vx * vx + vy * vy)
        c = gz

        if v_norm < 1e-6:
            return [1.0, 0.0, 0.0, 0.0] if c > 0.0 else [0.0, 1.0, 0.0, 0.0]

        angle = math.acos(max(-1.0, min(1.0, c)))
        half_angle = angle / 2.0
        s = math.sin(half_angle) / v_norm

        # Conjugate: calculate_gravity_apple_convention rotates by q^-1,
        # so we store the inverse of the [0,0,1] -> gravity rotation.
        return quaternion_normalize([
            math.cos(half_angle),
            -vx * s,
            -vy * s,
            0.0,
        ])

    return [1.0, 0.0, 0.0, 0.0]


def ahrs_nan_guard(q: Quaternion) -> Dict[str, float]:
    """Compute gravity + euler from q and return a standard AHRS result object."""
    gravity = calculate_gravity_apple_convention(q)
    euler = quaternion_to_euler(q)
    return {
        "gravityX": gravity[0],
        "gravityY": gravity[1],
        "gravityZ": gravity[2],
        "userAccelX": 0.0,
        "userAccelY": 0.0,
        "userAccelZ": 0.0,
        "qw": q[0],
        "qx": q[1],
        "qy": q[2],
        "qz": q[3],
        "roll": euler["roll"],
        "pitch": euler["pitch"],
        "yaw": euler["yaw"],
    }


@dataclass
class MadgwickFilter:
    beta: float = 0.1
    beta_mag: float = 0.0
    q: Quaternion = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    initialized: bool = False

    def init(self, accel_x: float, accel_y: float, accel_z: float) -> None:
        self.q = init_quaternion_from_gravity(accel_x, accel_y, accel_z)
        self.initialized = True

    def update(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        dt: float,
    ) -> Dict[str, float]:
        """Update IMU-only Madgwick filter.

        Gyroscope input must be in rad/s. dt must be in seconds.
        """
        values = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt]
        if any(math.isnan(v) for v in values):
            return ahrs_nan_guard(self.q)

        if not self.initialized:
            self.init(accel_x, accel_y, accel_z)

        q = self.q

        # Normalize accelerometer for gradient computation.
        accel_norm = math.sqrt(accel_x * accel_x + accel_y * accel_y + accel_z * accel_z)
        if accel_norm > 0.0:
            anx = accel_x / accel_norm
            any_ = accel_y / accel_norm
            anz = accel_z / accel_norm
        else:
            anx = any_ = anz = 0.0

        # Gradient descent - accelerometer objective function.
        f0 = 2.0 * (q[1] * q[3] - q[0] * q[2]) - anx
        f1 = 2.0 * (q[0] * q[1] + q[2] * q[3]) - any_
        f2 = 2.0 * (0.5 - q[1] * q[1] - q[2] * q[2]) - anz

        # Jacobian transposed * f (gradient).
        g0 = -2.0 * q[2] * f0 + 2.0 * q[1] * f1
        g1 = 2.0 * q[3] * f0 + 2.0 * q[0] * f1 - 4.0 * q[1] * f2
        g2 = -2.0 * q[0] * f0 + 2.0 * q[3] * f1 - 4.0 * q[2] * f2
        g3 = 2.0 * q[1] * f0 + 2.0 * q[2] * f1

        # Normalize gradient. This makes beta equivalent to estimated mean gyro error in rad/s.
        grad_norm = math.sqrt(g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3)
        if grad_norm > 1e-12:
            g0 /= grad_norm
            g1 /= grad_norm
            g2 /= grad_norm
            g3 /= grad_norm
        else:
            g0 = g1 = g2 = g3 = 0.0

        # Quaternion derivative from gyroscope.
        # This follows the provided TypeScript code directly; it does not negate gyro_z.
        q_dot_gyro = [
            0.5 * (-q[1] * gyro_x - q[2] * gyro_y - q[3] * gyro_z),
            0.5 * (q[0] * gyro_x + q[2] * gyro_z - q[3] * gyro_y),
            0.5 * (q[0] * gyro_y - q[1] * gyro_z + q[3] * gyro_x),
            0.5 * (q[0] * gyro_z + q[1] * gyro_y - q[2] * gyro_x),
        ]

        # Integrate: q = q + (qDotGyro - beta * gradient) * dt.
        self.q = quaternion_normalize([
            q[0] + (q_dot_gyro[0] - self.beta * g0) * dt,
            q[1] + (q_dot_gyro[1] - self.beta * g1) * dt,
            q[2] + (q_dot_gyro[2] - self.beta * g2) * dt,
            q[3] + (q_dot_gyro[3] - self.beta * g3) * dt,
        ])

        gravity = calculate_gravity_apple_convention(self.q)
        euler = quaternion_to_euler(self.q)

        return {
            "gravityX": gravity[0],
            "gravityY": gravity[1],
            "gravityZ": gravity[2],
            "userAccelX": accel_x - gravity[0],
            "userAccelY": accel_y - gravity[1],
            "userAccelZ": accel_z - gravity[2],
            "qw": self.q[0],
            "qx": self.q[1],
            "qy": self.q[2],
            "qz": self.q[3],
            "roll": euler["roll"],
            "pitch": euler["pitch"],
            "yaw": euler["yaw"],
        }


if __name__ == "__main__":
    # Tiny sanity test: flat, still IMU for a few samples.
    filt = MadgwickFilter(beta=0.1)
    for _ in range(5):
        out = filt.update(0.0, 0.0, 9.807, 0.0, 0.0, 0.0, 0.01)
    print(out)
