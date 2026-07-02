# read csv, run gyro-only prediction, and plot results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from eskf_imu import GyroOnlyPredictor


# =========================
# Settings
# =========================

CSV_PATH = "data/raw/imu_test.csv"

# 先假设 Arduino gyro 输出是 deg/s
# 如果之后确认是 rad/s，再改成 False
GYRO_IN_DEG_PER_S = True

OUTPUT_PATH = "data/processed/gyro_only_orientation.csv"


# =========================
# Load CSV
# =========================

df = pd.read_csv(CSV_PATH)

print("CSV head:")
print(df.head())
print()
print("CSV shape:", df.shape)
print("CSV columns:", df.columns.tolist())


# =========================
# Time
# =========================

t = df["timestamp"].to_numpy(dtype=float)

# 如果 timestamp 是 Arduino millis()，相邻差值通常 > 1，比如 100 ms
# 如果 timestamp 已经是 seconds，相邻差值通常 < 1，比如 0.1 s
if np.median(np.diff(t)) > 1:
    time_s = (t - t[0]) / 1000.0
    print("Using timestamp as milliseconds.")
else:
    time_s = t - t[0]
    print("Using timestamp as seconds.")

dt = np.diff(time_s, prepend=time_s[0])

if len(dt) > 1:
    dt[0] = np.median(dt[1:])
else:
    dt[0] = 0.01

# 防止异常 dt 炸掉积分
median_dt = np.median(dt)
dt = np.clip(dt, 0.2 * median_dt, 5.0 * median_dt)

print("duration [s]:", time_s[-1] - time_s[0])
print("median dt [s]:", np.median(dt))
print("approx sampling rate [Hz]:", 1.0 / np.median(dt))


# =========================
# Read IMU raw data
# =========================

gx = df["gx"].to_numpy(dtype=float)
gy = df["gy"].to_numpy(dtype=float)
gz = df["gz"].to_numpy(dtype=float)

if GYRO_IN_DEG_PER_S:
    gx = np.deg2rad(gx)
    gy = np.deg2rad(gy)
    gz = np.deg2rad(gz)
    print("Converted gyro from deg/s to rad/s.")
else:
    print("Using gyro as rad/s.")

# =========================
# Remove static gyro bias
# =========================
# For this test, the board is static, so we estimate bias from the whole recording.
# Later, for real movements, use only the first 1-2 seconds while the board is still.

bgx = np.mean(gx)
bgy = np.mean(gy)
bgz = np.mean(gz)

print("Estimated gyro bias [deg/s]:", np.rad2deg([bgx, bgy, bgz]))

gx = gx - bgx
gy = gy - bgy
gz = gz - bgz

print("Gyro mean after bias removal [deg/s]:", np.rad2deg([np.mean(gx), np.mean(gy), np.mean(gz)]))

# =========================
# Run gyro-only prediction
# =========================

predictor = GyroOnlyPredictor()

results = []

for i in range(len(df)):
    gyro_rad_s = np.array([gx[i], gy[i], gz[i]])

    state = predictor.step(gyro_rad_s, dt[i])

    results.append({
        "time_s": time_s[i],
        "qw": state["qw"],
        "qx": state["qx"],
        "qy": state["qy"],
        "qz": state["qz"],
        "roll_deg": np.rad2deg(state["roll_rad"]),
        "pitch_deg": np.rad2deg(state["pitch_rad"]),
        "yaw_deg": np.rad2deg(state["yaw_rad"]),
    })

out = pd.DataFrame(results)


# =========================
# Save output
# =========================

Path("data/processed").mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"Saved gyro-only orientation to {OUTPUT_PATH}")


# =========================
# Plot Euler angles
# =========================

plt.figure(figsize=(10, 4))
plt.plot(out["time_s"], out["roll_deg"], label="roll")
plt.plot(out["time_s"], out["pitch_deg"], label="pitch")
plt.plot(out["time_s"], out["yaw_deg"], label="yaw")
plt.xlabel("time [s]")
plt.ylabel("angle [deg]")
plt.title("Gyro-only orientation prediction")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# Plot quaternion norm
# =========================

q_norm = np.sqrt(
    out["qw"] ** 2
    + out["qx"] ** 2
    + out["qy"] ** 2
    + out["qz"] ** 2
)

plt.figure(figsize=(10, 3))
plt.plot(out["time_s"], q_norm)
plt.xlabel("time [s]")
plt.ylabel("quaternion norm")
plt.title("Quaternion norm check")
plt.grid(True)
plt.show()

print("gyro mean [rad/s]:", np.mean(gx), np.mean(gy), np.mean(gz))
print("gyro mean [deg/s]:", np.rad2deg(np.mean(gx)), np.rad2deg(np.mean(gy)), np.rad2deg(np.mean(gz)))
print("gyro std [deg/s]:", np.rad2deg(np.std(gx)), np.rad2deg(np.std(gy)), np.rad2deg(np.std(gz)))