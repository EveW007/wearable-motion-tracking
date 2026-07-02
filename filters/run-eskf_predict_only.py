import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from eskf_imu import IMUESKF


CSV_PATH = "data/raw/imu_test.csv"
OUTPUT_PATH = "data/processed/eskf_predict_only.csv"

GYRO_IN_DEG_PER_S = True
USE_INITIAL_STATIC_BIAS = True
STATIC_TIME_S = 2.0


df = pd.read_csv(CSV_PATH)

print("CSV head:")
print(df.head())
print("CSV shape:", df.shape)
print("CSV columns:", df.columns.tolist())


# =========================
# Time
# =========================

t = df["timestamp"].to_numpy(dtype=float)

if np.median(np.diff(t)) > 1:
    time_s = (t - t[0]) / 1000.0
    print("Using timestamp as milliseconds.")
else:
    time_s = t - t[0]
    print("Using timestamp as seconds.")

dt = np.diff(time_s, prepend=time_s[0])
dt[0] = np.median(dt[1:])

median_dt = np.median(dt)
dt = np.clip(dt, 0.2 * median_dt, 5.0 * median_dt)

print("duration [s]:", time_s[-1] - time_s[0])
print("median dt [s]:", np.median(dt))
print("approx sampling rate [Hz]:", 1.0 / np.median(dt))


# =========================
# Gyro
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
# Initial gyro bias
# =========================

if USE_INITIAL_STATIC_BIAS:
    static_mask = time_s < STATIC_TIME_S

    bg0 = np.array([
        np.mean(gx[static_mask]),
        np.mean(gy[static_mask]),
        np.mean(gz[static_mask]),
    ])

    print("Initial gyro bias from first 2s [deg/s]:", np.rad2deg(bg0))
else:
    bg0 = np.zeros(3)
    print("Initial gyro bias set to zero.")


# =========================
# Run ESKF prediction only
# =========================

eskf = IMUESKF(bg0=bg0)

results = []

for i in range(len(df)):
    gyro_rad_s = np.array([gx[i], gy[i], gz[i]])

    state = eskf.predict(gyro_rad_s, dt[i])

    P_diag = state["P_diag"]

    results.append({
        "time_s": time_s[i],
        "qw": state["qw"],
        "qx": state["qx"],
        "qy": state["qy"],
        "qz": state["qz"],
        "roll_deg": np.rad2deg(state["roll_rad"]),
        "pitch_deg": np.rad2deg(state["pitch_rad"]),
        "yaw_deg": np.rad2deg(state["yaw_rad"]),
        "bgx_deg_s": np.rad2deg(state["bgx"]),
        "bgy_deg_s": np.rad2deg(state["bgy"]),
        "bgz_deg_s": np.rad2deg(state["bgz"]),
        "P_theta_x": P_diag[0],
        "P_theta_y": P_diag[1],
        "P_theta_z": P_diag[2],
        "P_bg_x": P_diag[3],
        "P_bg_y": P_diag[4],
        "P_bg_z": P_diag[5],
    })

out = pd.DataFrame(results)

Path("data/processed").mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"Saved ESKF predict-only output to {OUTPUT_PATH}")


# =========================
# Plot Euler angles
# =========================

plt.figure(figsize=(10, 4))
plt.plot(out["time_s"], out["roll_deg"], label="roll")
plt.plot(out["time_s"], out["pitch_deg"], label="pitch")
plt.plot(out["time_s"], out["yaw_deg"], label="yaw")
plt.xlabel("time [s]")
plt.ylabel("angle [deg]")
plt.title("ESKF predict-only orientation")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# Plot covariance diagonal
# =========================

plt.figure(figsize=(10, 4))
plt.plot(out["time_s"], out["P_theta_x"], label="P theta x")
plt.plot(out["time_s"], out["P_theta_y"], label="P theta y")
plt.plot(out["time_s"], out["P_theta_z"], label="P theta z")
plt.xlabel("time [s]")
plt.ylabel("covariance")
plt.title("ESKF attitude covariance prediction")
plt.legend()
plt.grid(True)
plt.show()