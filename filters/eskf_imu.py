# gyro-only orientation predictor for IMU ESKF
# eskf_imu.py is a module that provides functions and classes for working with quaternions and implementing a gyro-only orientation predictor for an IMU (Inertial Measurement Unit) using an Error-State Kalman Filter (ESKF).


import numpy as np

# quaternion helper 

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
    IMU-only Error-State Kalman Filter.

    Nominal state:
        x = {q, bg}

    Error state:
        dx = [delta_theta, delta_bg]

    Current implementation:
    - Gyroscope prediction for quaternion orientation
    - Gyro bias state
    - 6x6 error covariance propagation
    - Accelerometer gravity-direction correction for roll/pitch
    - No magnetometer, so yaw is not absolutely observable
    """

    def __init__(
        self,
        q0=None,
        bg0=None,
        P0=None,
        gyro_noise_std=0.02,
        gyro_bias_noise_std=0.001,
        accel_noise_std=0.08,
        accel_gate=0.25,
    ):
        # Nominal orientation quaternion
        if q0 is None:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self.q = q_normalize(q0)

        # Nominal gyro bias
        if bg0 is None:
            self.bg = np.zeros(3)
        else:
            self.bg = np.asarray(bg0, dtype=float)

        # Error covariance over [delta_theta, delta_bg]
        if P0 is None:
            self.P = np.eye(6) * 1e-3
        else:
            self.P = np.asarray(P0, dtype=float)

        # Noise parameters
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_noise_std = gyro_bias_noise_std
        self.accel_noise_std = accel_noise_std

        # Accelerometer gate:
        # if ||a|| is too far from 1g, skip accel correction
        self.accel_gate = accel_gate

    # ------------------------------------------------------------
    # Prediction step
    # ------------------------------------------------------------
    def predict(self, gyro_rad_s, dt):
        """
        ESKF prediction step.

        Inputs:
            gyro_rad_s : measured gyro [rad/s]
            dt         : sample time [s]

        Nominal propagation:
            omega = gyro - bg
            q <- q ⊗ Exp(omega dt)

        Error covariance propagation:
            P <- Fd P Fd.T + Qd

        Current covariance propagation uses first-order discrete approximation.
        """
        gyro_rad_s = np.asarray(gyro_rad_s, dtype=float)

        if dt <= 0:
            return self.get_state()

        # 1. Bias-corrected angular rate
        omega = gyro_rad_s - self.bg

        # 2. Nominal quaternion prediction
        dq = q_from_gyro(omega, dt)
        self.q = q_multiply(self.q, dq)
        self.q = q_normalize(self.q)

        # 3. Discrete error-state transition matrix
        # Error state: dx = [delta_theta, delta_bg]
        Fd = np.eye(6)

        # delta_theta_{k+1} ≈ (I - [omega]x dt) delta_theta_k - I dt delta_bg
        Fd[0:3, 0:3] = np.eye(3) - skew(omega) * dt
        Fd[0:3, 3:6] = -np.eye(3) * dt

        # delta_bg_{k+1} ≈ delta_bg_k
        Fd[3:6, 3:6] = np.eye(3)

        # 4. Approximate discrete process noise
        Qd = np.zeros((6, 6))
        Qd[0:3, 0:3] = (self.gyro_noise_std ** 2) * dt * np.eye(3)
        Qd[3:6, 3:6] = (self.gyro_bias_noise_std ** 2) * dt * np.eye(3)

        # 5. Covariance propagation
        self.P = Fd @ self.P @ Fd.T + Qd

        # Keep P symmetric
        self.P = 0.5 * (self.P + self.P.T)

        return self.get_state()

    # ------------------------------------------------------------
    # Accelerometer measurement model
    # ------------------------------------------------------------
    def predicted_gravity_body(self, gravity_ref=None):
        """
        Predict gravity direction in sensor/body frame from current quaternion.

        gravity_ref is the reference gravity direction in the world frame.
        For this project we use +Z gravity convention:
            gravity_ref = [0, 0, 1]
        """
        if gravity_ref is None:
            gravity_ref = np.array([0.0, 0.0, 1.0])

        h = q_rotate_vector(q_conjugate(self.q), gravity_ref)
        h_norm = np.linalg.norm(h)

        if h_norm < 1e-12:
            return np.array([0.0, 0.0, 1.0])

        return h / h_norm

    def perturb_quaternion(self, q, delta_theta):
        """
        Right perturbation:
            q_perturbed = q ⊗ Exp(delta_theta)
        """
        dq = q_from_gyro(delta_theta, 1.0)
        return q_normalize(q_multiply(q, dq))

    def accel_measurement_model(self, accel_g):
        """
        Build accelerometer gravity-direction measurement.

        Measurement:
            z = accel / ||accel||

        Prediction:
            h(q) = predicted gravity direction in sensor/body frame

        Residual:
            r = z - h(q)

        Returns:
            valid, residual, H, R, innovation_norm, acc_norm
        """
        accel_g = np.asarray(accel_g, dtype=float)
        acc_norm = np.linalg.norm(accel_g)

        if acc_norm < 1e-12:
            return False, None, None, None, np.nan, acc_norm

        # Gate accelerometer update if acceleration is too far from 1g
        # This avoids using accelerometer when strong linear acceleration is present.
        if abs(acc_norm - 1.0) > self.accel_gate:
            return False, None, None, None, np.nan, acc_norm

        # Use direction only
        z = accel_g / acc_norm

        # Predicted gravity direction
        h0 = self.predicted_gravity_body()

        # Innovation / residual
        residual = z - h0
        innovation_norm = np.linalg.norm(residual)

        # Measurement noise
        R = (self.accel_noise_std ** 2) * np.eye(3)

        # Measurement Jacobian H
        # Use numerical Jacobian initially to avoid sign/convention mistakes.
        # Later this can be replaced by analytic H after frame convention validation.
        H = np.zeros((3, 6))
        eps = 1e-6

        for j in range(3):
            delta_theta = np.zeros(3)
            delta_theta[j] = eps

            q_perturbed = self.perturb_quaternion(self.q, delta_theta)

            # Temporarily compute h(q_perturbed)
            h_perturbed = q_rotate_vector(
                q_conjugate(q_perturbed),
                np.array([0.0, 0.0, 1.0])
            )
            h_perturbed = h_perturbed / np.linalg.norm(h_perturbed)

            H[:, j] = (h_perturbed - h0) / eps

        # Accelerometer does not directly measure gyro bias
        H[:, 3:6] = 0.0

        return True, residual, H, R, innovation_norm, acc_norm

    # ------------------------------------------------------------
    # Kalman correction
    # ------------------------------------------------------------
    def kalman_update(self, residual, H, R):
        """
        Standard Kalman correction over error state.

        S = H P H.T + R
        K = P H.T S^-1
        delta_x = K residual

        Covariance is updated using Joseph form for numerical stability.
        """
        S = H @ self.P @ H.T + R

        # More numerically stable than explicit inverse:
        # K = P H.T inv(S)
        K = np.linalg.solve(S, H @ self.P).T

        delta_x = K @ residual

        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        # Keep P symmetric
        self.P = 0.5 * (self.P + self.P.T)

        return delta_x

    # ------------------------------------------------------------
    # Error injection and reset
    # ------------------------------------------------------------
    def inject(self, delta_x):
        """
        Inject estimated error state into nominal state.

        delta_x = [delta_theta, delta_bg]

        q  <- q ⊗ Exp(delta_theta)
        bg <- bg + delta_bg
        """
        delta_theta = delta_x[0:3]
        delta_bg = delta_x[3:6]

        self.q = self.perturb_quaternion(self.q, delta_theta)
        self.bg = self.bg + delta_bg

    def reset_error_state(self, delta_x=None):
        """
        Reset error state after injection.

        In a full ESKF, the covariance should be transformed by a reset Jacobian G:
            P <- G P G.T

        Current prototype:
            use small-angle approximation G ≈ I

        So here we only enforce symmetry.
        """
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------
    # Accelerometer update wrapper
    # ------------------------------------------------------------
    def update_accel(self, accel_g):
        """
        Full accelerometer correction step.

        This function now only coordinates:
            1. measurement model
            2. Kalman update
            3. error injection
            4. error reset
        """
        valid, residual, H, R, innovation_norm, acc_norm = self.accel_measurement_model(accel_g)

        if not valid:
            state = self.get_state()
            state["used_accel"] = False
            state["innovation_norm"] = innovation_norm
            state["acc_norm"] = acc_norm
            return state

        delta_x = self.kalman_update(residual, H, R)

        self.inject(delta_x)
        self.reset_error_state(delta_x)

        state = self.get_state()
        state["used_accel"] = True
        state["innovation_norm"] = innovation_norm
        state["acc_norm"] = acc_norm
        return state

    # ------------------------------------------------------------
    # Full filter step
    # ------------------------------------------------------------
    def step(self, gyro_rad_s, accel_g, dt):
        """
        One full IMU ESKF step:
            predict using gyro
            correct using accelerometer
        """
        self.predict(gyro_rad_s, dt)
        return self.update_accel(accel_g)

    # ------------------------------------------------------------
    # State output
    # ------------------------------------------------------------
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