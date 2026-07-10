# read IMU data from Arduino and run Madgwick filter in real-time

import csv
import math
import statistics
import time
import serial

from head_frame import HeadFrameTransform
from madgwick_filter import MadgwickFilter


PORT = "/dev/cu.usbmodem1101"  # 改成你的 port
BAUD = 115200

# 如果 Arduino gyro 输出是 deg/s，设 True
# 如果已经是 rad/s，设 False
GYRO_IN_DEG_PER_SEC = True

# 如果 accel 输出是 g，设 True
# 如果已经是 m/s^2，设 False
ACCEL_IN_G = True

G = 9.807
BETA = 0.005
GYRO_CALIBRATION_TIME_S = 5.0
MIN_CALIBRATION_SAMPLES = 20
HEAD_NEUTRAL_CALIBRATION_TIME_S = 2.0
MIN_NEUTRAL_CALIBRATION_SAMPLES = 10

# q_head_sensor maps sensor coordinates into anatomical head coordinates.
# Leave identity only while the mounted sensor axes are treated as head axes.
Q_HEAD_SENSOR = [1.0, 0.0, 0.0, 0.0]


def convert_accel(ax, ay, az):
    if ACCEL_IN_G:
        return ax * G, ay * G, az * G
    return ax, ay, az


def convert_gyro(gx, gy, gz):
    if GYRO_IN_DEG_PER_SEC:
        scale = math.pi / 180.0
        return gx * scale, gy * scale, gz * scale
    return gx, gy, gz


def parse_imu_line(line):
    parts = line.split(",")
    if len(parts) != 7:
        return None
    try:
        return tuple(map(float, parts))
    except ValueError:
        return None


def calibrate_gyro_bias(ser, duration_s=GYRO_CALIBRATION_TIME_S):
    """Estimate static gyro bias using a robust per-axis median."""
    print(f"Keep the sensor still: calibrating gyro for {duration_s:.1f} s...")
    samples = []
    deadline = time.monotonic() + duration_s

    while time.monotonic() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        parsed = parse_imu_line(line)
        if parsed is None:
            continue

        _, ax, ay, az, gx, gy, gz = parsed
        ax, ay, az = convert_accel(ax, ay, az)
        gx, gy, gz = convert_gyro(gx, gy, gz)

        # Reject obviously invalid acceleration samples.  Direction is not
        # used here; this check only catches bad records or strong motion.
        acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if 0.8 * G <= acc_norm <= 1.2 * G:
            samples.append((gx, gy, gz))

    if len(samples) < MIN_CALIBRATION_SAMPLES:
        raise RuntimeError(
            f"Only {len(samples)} valid calibration samples; keep the sensor "
            "still and check the serial sample rate."
        )

    bias = tuple(statistics.median(axis) for axis in zip(*samples))
    bias_dps = tuple(value * 180.0 / math.pi for value in bias)
    print(
        "Gyro bias [deg/s]: "
        f"x={bias_dps[0]:+.5f}, y={bias_dps[1]:+.5f}, z={bias_dps[2]:+.5f} "
        f"from {len(samples)} samples"
    )
    return bias


def calibrate_head_neutral(
    ser,
    filt,
    gyro_bias,
    duration_s=HEAD_NEUTRAL_CALIBRATION_TIME_S,
):
    """Record a static, forward-looking pose as zero head orientation."""
    print(
        f"Wear the sensor, look straight ahead, and keep still for "
        f"{duration_s:.1f} s..."
    )
    samples = []
    deadline = time.monotonic() + duration_s
    last_t_s = None

    while time.monotonic() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        parsed = parse_imu_line(line)
        if parsed is None:
            continue

        t_ms, ax, ay, az, gx, gy, gz = parsed
        t_s = t_ms / 1000.0
        if last_t_s is None:
            last_t_s = t_s
            continue

        dt = t_s - last_t_s
        last_t_s = t_s
        if dt <= 0.0 or dt > 1.0:
            continue

        ax, ay, az = convert_accel(ax, ay, az)
        gx, gy, gz = convert_gyro(gx, gy, gz)
        gyro_corrected = (
            gx - gyro_bias[0],
            gy - gyro_bias[1],
            gz - gyro_bias[2],
        )
        result = filt.update(
            ax, ay, az,
            gyro_corrected[0], gyro_corrected[1], gyro_corrected[2],
            dt,
        )

        acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
        gyro_norm = math.sqrt(sum(value * value for value in gyro_corrected))
        if 0.8 * G <= acc_norm <= 1.2 * G and gyro_norm <= math.radians(5.0):
            samples.append([
                result["qw"], result["qx"], result["qy"], result["qz"]
            ])

    # At the current ~9.6 Hz rate a 2 s window contains about 19 records, so
    # this threshold is intentionally separate from the 5 s gyro calibration.
    if len(samples) < MIN_NEUTRAL_CALIBRATION_SAMPLES:
        raise RuntimeError(
            f"Only {len(samples)} valid neutral samples; look straight ahead "
            "and keep the sensor still during calibration."
        )

    head_frame = HeadFrameTransform(q_head_sensor=Q_HEAD_SENSOR)
    head_frame.set_neutral(samples)
    print(
        f"Head neutral calibrated from {len(samples)} samples; "
        "angles are now zeroed."
    )
    return head_frame


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # 等 Arduino reset
    ser.reset_input_buffer()

    gyro_bias = calibrate_gyro_bias(ser)
    ser.reset_input_buffer()
    filt = MadgwickFilter(beta=BETA)
    head_frame = calibrate_head_neutral(ser, filt, gyro_bias)
    ser.reset_input_buffer()

    raw_file = open("raw_imu.csv", "w", newline="")
    out_file = open("orientation_output.csv", "w", newline="")

    raw_writer = csv.writer(raw_file)
    out_writer = csv.writer(out_file)

    raw_writer.writerow([
        "timestamp_s",
        "ax_mps2", "ay_mps2", "az_mps2",
        "gx_rads", "gy_rads", "gz_rads",
        "gx_corrected_rads", "gy_corrected_rads", "gz_corrected_rads",
    ])

    out_writer.writerow([
        "timestamp_s",
        "qw", "qx", "qy", "qz",
        "roll_rad", "pitch_rad", "yaw_rad",
        "roll_deg", "pitch_deg", "yaw_deg",
        "head_qw", "head_qx", "head_qy", "head_qz",
        "head_roll_rad", "head_pitch_rad", "head_yaw_rad",
        "head_roll_deg", "head_pitch_deg", "head_yaw_deg",
        "gravity_x", "gravity_y", "gravity_z",
        "user_accel_x", "user_accel_y", "user_accel_z",
    ])

    last_t = None

    print("Reading IMU... Press Ctrl+C to stop.")

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            parsed = parse_imu_line(line)
            if parsed is None:
                print("Skipping bad line:", line)
                continue

            t_ms, ax, ay, az, gx, gy, gz = parsed

            t_s = t_ms / 1000.0

            if last_t is None:
                last_t = t_s
                continue

            dt = t_s - last_t
            last_t = t_s

            if dt <= 0 or dt > 1:
                print("Skipping bad dt:", dt)
                continue

            ax, ay, az = convert_accel(ax, ay, az)
            gx, gy, gz = convert_gyro(gx, gy, gz)
            gx_corrected = gx - gyro_bias[0]
            gy_corrected = gy - gyro_bias[1]
            gz_corrected = gz - gyro_bias[2]

            result = filt.update(
                accel_x=ax,
                accel_y=ay,
                accel_z=az,
                gyro_x=gx_corrected,
                gyro_y=gy_corrected,
                gyro_z=gz_corrected,
                dt=dt,
            )

            raw_writer.writerow([
                t_s, ax, ay, az, gx, gy, gz,
                gx_corrected, gy_corrected, gz_corrected,
            ])

            roll_deg = result["roll"] * 180.0 / math.pi
            pitch_deg = result["pitch"] * 180.0 / math.pi
            yaw_deg = result["yaw"] * 180.0 / math.pi
            head = head_frame.transform([
                result["qw"], result["qx"], result["qy"], result["qz"]
            ])
            head_roll_deg = math.degrees(head["roll"])
            head_pitch_deg = math.degrees(head["pitch"])
            head_yaw_deg = math.degrees(head["yaw"])

            out_writer.writerow([
                t_s,
                result["qw"], result["qx"], result["qy"], result["qz"],
                result["roll"], result["pitch"], result["yaw"],
                roll_deg, pitch_deg, yaw_deg,
                head["qw"], head["qx"], head["qy"], head["qz"],
                head["roll"], head["pitch"], head["yaw"],
                head_roll_deg, head_pitch_deg, head_yaw_deg,
                result["gravityX"], result["gravityY"], result["gravityZ"],
                result["userAccelX"], result["userAccelY"], result["userAccelZ"],
            ])

            print(
                f"t={t_s:.2f}s | "
                f"head roll={head_roll_deg:7.2f}°, "
                f"pitch={head_pitch_deg:7.2f}°, "
                f"yaw={head_yaw_deg:7.2f}°"
            )

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        raw_file.close()
        out_file.close()
        ser.close()
        print("Saved raw_imu.csv and orientation_output.csv")


if __name__ == "__main__":
    main()
