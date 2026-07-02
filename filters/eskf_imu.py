# gyro-only orientation predictor for IMU ESKF
# eskf_imu.py is a module that provides functions and classes for working with quaternions and implementing a gyro-only orientation predictor for an IMU (Inertial Measurement Unit) using an Error-State Kalman Filter (ESKF).


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

# v1: gyro → q prediction

def skew(v):
    """
    Return skew-symmetric matrix [v]x such that skew(v) @ w = v x w.
    """
    x, y, z = v
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])

#v2: gyro → predict q ; accel gravity → correct roll/pitch

def q_conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_rotate_vector(q, v):
    """
    Rotate vector v by quaternion q:
        v_rot = q ⊗ [0, v] ⊗ q*
    """
    q = q_normalize(q)
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    rotated = q_multiply(q_multiply(q, v_quat), q_conjugate(q))
    return rotated[1:4]

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

# v2.1 initial quaternion from accelerometer gravity direction (previously was predicted [1,0,0,0], while actually it should be predicted from accelerometer gravity direction, so that roll/pitch is correct at the beginning)

def init_quaternion_from_accel(accel_g):
    """
    Initialize quaternion from accelerometer gravity direction.
    Returns q such that predicted_gravity_body(q) ≈ accel_g / ||accel_g||.
    """
    accel_g = np.asarray(accel_g, dtype=float)
    norm = np.linalg.norm(accel_g)

    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])

    gx, gy, gz = accel_g / norm

    # Rotation from [0,0,1] to measured gravity direction
    vx = -gy
    vy = gx
    v_norm = np.sqrt(vx * vx + vy * vy)
    c = gz

    if v_norm < 1e-6:
        if c > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            return np.array([0.0, 1.0, 0.0, 0.0])

    angle = np.arccos(np.clip(c, -1.0, 1.0))
    half = angle / 2.0
    s = np.sin(half) / v_norm

    # Store inverse because predicted_gravity_body uses q_conj
    q = np.array([
        np.cos(half),
        -vx * s,
        -vy * s,
        0.0,
    ])

    return q_normalize(q)

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
    
class IMUESKF:
    """
    IMU-only Error-State Kalman Filter skeleton.

    Current version:
    - Uses gyroscope to propagate orientation.
    - Tracks gyro bias.
    - Propagates 6x6 covariance P.
    - No accelerometer correction yet.
    """

    def __init__(
        self,
        q0=None,
        bg0=None,
        P0=None,
        gyro_noise_std=0.02,
        gyro_bias_noise_std=0.001,
    ):
        if q0 is None:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self.q = q_normalize(q0)

        if bg0 is None:
            self.bg = np.zeros(3)
        else:
            self.bg = np.asarray(bg0, dtype=float)

        if P0 is None:
            # Error state: [delta_theta, delta_bg]
            self.P = np.eye(6) * 1e-3
        else:
            self.P = np.asarray(P0, dtype=float)

        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_noise_std = gyro_bias_noise_std

    def predict(self, gyro_rad_s, dt):
        gyro_rad_s = np.asarray(gyro_rad_s, dtype=float)

        if dt <= 0:
            return self.get_state()

        # 1. Remove current gyro bias estimate
        omega = gyro_rad_s - self.bg

        # 2. Nominal quaternion prediction
        dq = q_from_gyro(omega, dt)
        self.q = q_multiply(self.q, dq)
        self.q = q_normalize(self.q)

        # 3. Error-state covariance prediction
        # Error state: [delta_theta, delta_bg]
        F = np.eye(6)

        # attitude error dynamics
        F[0:3, 0:3] = np.eye(3) - skew(omega) * dt

        # gyro bias affects attitude integration
        F[0:3, 3:6] = -np.eye(3) * dt

        # Simple process noise
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = (self.gyro_noise_std ** 2) * dt * np.eye(3)
        Q[3:6, 3:6] = (self.gyro_bias_noise_std ** 2) * dt * np.eye(3)

        self.P = F @ self.P @ F.T + Q

        # Keep P symmetric to avoid numerical weirdness
        self.P = 0.5 * (self.P + self.P.T)

        return self.get_state()
    
    # v2: gyro → predict q ; accel gravity → correct roll/pitch --> added accel update function
    def update_accel(self, accel, accel_noise_std=0.08, gravity_ref=None):
        """
        Accelerometer update using gravity direction.

        accel: raw accelerometer measurement.
               If your sensor outputs g, this can be in g.
               We normalize it, so magnitude does not matter here.

        accel_noise_std: measurement noise for normalized gravity direction.
        """

        accel = np.asarray(accel, dtype=float)

        acc_norm = np.linalg.norm(accel)
        if acc_norm < 1e-12:
            return self.get_state()

        # Use direction only
        z = accel / acc_norm

        # Reference gravity direction in world frame.
        # Since your static accel is around [0.15, 0.10, 1.03],
        # your sensor convention seems to use +Z as gravity when flat.
        if gravity_ref is None:
            gravity_ref = np.array([0.0, 0.0, 1.0])

        # Predicted gravity direction in sensor/body frame.
        # This convention may need sign adjustment later, but start here.
        h = q_rotate_vector(q_conjugate(self.q), gravity_ref)
        h = h / np.linalg.norm(h)

        # Innovation
        r = z - h

        # Measurement Jacobian
        # For small attitude error, gravity direction changes by cross product.
        H = np.zeros((3, 6))
        H[:, 0:3] = skew(h)
        H[:, 3:6] = 0.0

        R = (accel_noise_std ** 2) * np.eye(3)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ r

        dtheta = dx[0:3]
        dbg = dx[3:6]

        # Inject correction into nominal state
        dq = q_from_gyro(dtheta, 1.0)   # Exp(dtheta)
        self.q = q_multiply(self.q, dq)
        self.q = q_normalize(self.q)

        self.bg = self.bg + dbg

        # Joseph form covariance update: more stable
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return self.get_state()


    def get_state(self):
        euler_rad = q_to_euler(self.q)

        return {
            "qw": self.q[0],
            "qx": self.q[1],
            "qy": self.q[2],
            "qz": self.q[3],
            "roll_rad": euler_rad[0],
            "pitch_rad": euler_rad[1],
            "yaw_rad": euler_rad[2],
            "bgx": self.bg[0],
            "bgy": self.bg[1],
            "bgz": self.bg[2],
            "P_diag": np.diag(self.P).copy(),
        }