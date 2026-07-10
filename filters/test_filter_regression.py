import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eskf_imu import IMUESKF
from head_frame import (
    HeadFrameTransform,
    average_quaternions,
    quaternion_from_euler,
)
from madgwick_filter import (
    MadgwickFilter,
    init_quaternion_from_gravity,
    quaternion_multiply,
    quaternion_to_euler,
)


ROOT = Path(__file__).resolve().parents[1]


class MadgwickRegressionTests(unittest.TestCase):
    def test_gravity_initialisation_is_unit_independent(self):
        accel_g = np.array([0.09, 0.03, 1.04])
        q_from_g = np.asarray(init_quaternion_from_gravity(*accel_g))
        q_from_mps2 = np.asarray(init_quaternion_from_gravity(*(accel_g * 9.807)))

        # q and -q encode the same orientation, so compare absolute dot.
        self.assertAlmostEqual(abs(float(q_from_g @ q_from_mps2)), 1.0, places=12)

    def test_zero_acceleration_sample_uses_gyro_only(self):
        filt = MadgwickFilter(beta=0.1)
        out = filt.update(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1)
        self.assertAlmostEqual(out["roll"], 0.1, places=3)
        self.assertAlmostEqual(out["pitch"], 0.0, places=12)

    def test_project_beta_is_stable_on_recorded_static_data(self):
        df = pd.read_csv(ROOT / "data/raw/static_20s.csv")
        time_s = (df["timestamp"].to_numpy() - df["timestamp"].iloc[0]) / 1000.0
        dt = np.diff(time_s, prepend=time_s[0])
        dt[0] = np.median(dt[1:])

        accel = df[["ax", "ay", "az"]].to_numpy() * 9.807
        gyro = np.deg2rad(df[["gx", "gy", "gz"]].to_numpy())
        initial_static = time_s < 2.0
        gyro_bias = np.median(gyro[initial_static], axis=0)

        filt = MadgwickFilter(beta=0.005)
        tilt = []
        for accel_i, gyro_i, dt_i in zip(accel, gyro - gyro_bias, dt):
            out = filt.update(*accel_i, *gyro_i, dt_i)
            euler = quaternion_to_euler([out["qw"], out["qx"], out["qy"], out["qz"]])
            tilt.append([math.degrees(euler["roll"]), math.degrees(euler["pitch"])])

        tilt = np.asarray(tilt)[time_s >= 5.0]
        self.assertLess(float(np.max(np.ptp(tilt, axis=0))), 0.4)


class ESKFRegressionTests(unittest.TestCase):
    def test_zero_rate_update_observes_all_bias_axes(self):
        measured_bias = np.deg2rad(np.array([1.0, -1.5, 2.1]))
        eskf = IMUESKF(
            bg0=np.zeros(3),
            P0=np.eye(6) * 0.1,
            zero_rate_noise_std=1e-4,
        )

        state = eskf.update_zero_rate(measured_bias)
        estimated = np.array([state["bgx"], state["bgy"], state["bgz"]])
        np.testing.assert_allclose(estimated, measured_bias, atol=1e-6)


class HeadFrameRegressionTests(unittest.TestCase):
    def test_neutral_pose_is_zero(self):
        neutral = quaternion_from_euler(0.2, -0.1, 0.4)
        transform = HeadFrameTransform()
        transform.set_neutral([neutral, neutral])

        head = transform.transform(neutral)
        self.assertAlmostEqual(head["roll"], 0.0, places=12)
        self.assertAlmostEqual(head["pitch"], 0.0, places=12)
        self.assertAlmostEqual(head["yaw"], 0.0, places=12)

    def test_mounting_rotation_keeps_head_roll_on_head_roll_axis(self):
        q_head_sensor = quaternion_from_euler(0.0, 0.0, math.radians(90.0))
        q_neutral_sensor = q_head_sensor
        q_head_motion = quaternion_from_euler(math.radians(30.0), 0.0, 0.0)
        q_current_sensor = quaternion_multiply(q_head_motion, q_head_sensor)

        transform = HeadFrameTransform(q_head_sensor=q_head_sensor)
        transform.set_neutral([q_neutral_sensor])
        head = transform.transform(q_current_sensor)

        self.assertAlmostEqual(math.degrees(head["roll"]), 30.0, places=10)
        self.assertAlmostEqual(head["pitch"], 0.0, places=12)
        self.assertAlmostEqual(head["yaw"], 0.0, places=12)

    def test_neutral_average_handles_equivalent_quaternion_signs(self):
        q = quaternion_from_euler(0.1, 0.2, -0.3)
        mean = average_quaternions([q, [-value for value in q]])
        self.assertAlmostEqual(abs(float(np.dot(mean, q))), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
