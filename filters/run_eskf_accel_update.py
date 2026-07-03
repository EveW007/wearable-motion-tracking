import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from eskf_imu import IMUESKF, init_quaternion_from_accel

from eskf_imu import IMUESKF


CSV_PATH = "data/raw/static_20s.csv"
OUTPUT_PATH = "data/processed/eskf_static_20s.csv"

GYRO_IN_DEG_PER_S = True
STATIC_TIME_S = 2.0

# Your accel appears to be in g, because static az ≈ 1.03
ACCEL_IN_G = True

# Acc update gating.
# Since accel is in g, static norm should be around 1.0.
ACC_NORM_EXPECTED = 1.0
ACC_NORM_TOL = 0.20


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

# include dt plot for debugging
plt.figure(figsize=(10, 3))
plt.plot(time_s, dt)
plt.xlabel("time [s]")
plt.ylabel("dt [s]")
plt.title("Sample interval dt")
plt.grid(True)
plt.show()

print("duration [s]:", time_s[-1] - time_s[0])
print("median dt [s]:", np.median(dt))
print("approx sampling rate [Hz]:", 1.0 / np.median(dt))


# =========================
# IMU data
# =========================

ax = df["ax"].to_numpy(dtype=float)
ay = df["ay"].to_numpy(dtype=float)
az = df["az"].to_numpy(dtype=float)

gx = df["gx"].to_numpy(dtype=float)
gy = df["gy"].to_numpy(dtype=float)
gz = df["gz"].to_numpy(dtype=float)

if GYRO_IN_DEG_PER_S:
    gx = np.deg2rad(gx)
    gy = np.deg2rad(gy)
    gz = np.deg2rad(gz)
    print("Converted gyro from deg/s to rad/s.")

acc_norm = np.sqrt(ax**2 + ay**2 + az**2)
print("acc norm median:", np.median(acc_norm))

# plot gyro for debugging
plt.figure(figsize=(10, 3))
plt.plot(time_s, np.rad2deg(gx), label="gx")
plt.plot(time_s, np.rad2deg(gy), label="gy")
plt.plot(time_s, np.rad2deg(gz), label="gz")
plt.xlabel("time [s]")
plt.ylabel("gyro [deg/s]")
plt.title("Raw gyroscope")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# Initial gyro bias from first static seconds
# =========================

static_mask = time_s < STATIC_TIME_S

bg0 = np.array([
    np.mean(gx[static_mask]),
    np.mean(gy[static_mask]),
    np.mean(gz[static_mask]),
])

print("Initial gyro bias [deg/s]:", np.rad2deg(bg0))


# =========================
# Run ESKF predict + accel update
# =========================

# v2.1 这样 roll/pitch 一开始不会从 0 被拉过去，而是直接从 gravity-consistent 的姿态开始

acc0 = np.array([
    np.mean(ax[static_mask]),
    np.mean(ay[static_mask]),
    np.mean(az[static_mask]),
])

q0 = init_quaternion_from_accel(acc0)

print("Initial accel mean [g]:", acc0)
print("Initial q0:", q0)

eskf = IMUESKF(q0=q0, bg0=bg0)

results = []

for i in range(len(df)):
    gyro_rad_s = np.array([gx[i], gy[i], gz[i]])
    accel = np.array([ax[i], ay[i], az[i]])

    state = eskf.predict(gyro_rad_s, dt[i])

    acc_norm_i = np.linalg.norm(accel)
    use_accel = abs(acc_norm_i - ACC_NORM_EXPECTED) < ACC_NORM_TOL

    innovation_norm = np.nan

    if use_accel:
        state = eskf.update_accel(accel)
        innovation_norm = state.get("innovation_norm", np.nan)

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

        "used_accel": int(use_accel),
        "innovation_norm": innovation_norm,
        "acc_norm": acc_norm_i,
    })

out = pd.DataFrame(results)

# innovation plot for debugging
plt.figure(figsize=(10, 3))
plt.plot(out["time_s"], out["innovation_norm"])
plt.xlabel("time [s]")
plt.ylabel("innovation norm")
plt.title("Accelerometer innovation")
plt.grid(True)
plt.show()

# =========================
# Relative Euler angles
# =========================

out["roll_rel_deg"] = out["roll_deg"] - out["roll_deg"].iloc[0]
out["pitch_rel_deg"] = out["pitch_deg"] - out["pitch_deg"].iloc[0]
out["yaw_rel_deg"] = out["yaw_deg"] - out["yaw_deg"].iloc[0]

plt.figure(figsize=(10, 4))
plt.plot(out["time_s"], out["roll_rel_deg"], label="roll relative")
plt.plot(out["time_s"], out["pitch_rel_deg"], label="pitch relative")
plt.plot(out["time_s"], out["yaw_rel_deg"], label="yaw relative")
plt.xlabel("time [s]")
plt.ylabel("relative angle [deg]")
plt.title("ESKF relative orientation from initial pose")
plt.legend()
plt.grid(True)
plt.show()

Path("data/processed").mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"Saved ESKF accel-update output to {OUTPUT_PATH}")


# =========================
# Plot Euler angles
# =========================

plt.figure(figsize=(10, 4))
plt.plot(out["time_s"], out["roll_deg"], label="roll")
plt.plot(out["time_s"], out["pitch_deg"], label="pitch")
plt.plot(out["time_s"], out["yaw_deg"], label="yaw")
plt.xlabel("time [s]")
plt.ylabel("angle [deg]")
plt.title("ESKF predict + accelerometer update")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# Plot acc norm and update gate
# =========================

plt.figure(figsize=(10, 3))
plt.plot(out["time_s"], out["acc_norm"], label="acc norm")
plt.axhline(ACC_NORM_EXPECTED, linestyle="--", label="expected")
plt.axhline(ACC_NORM_EXPECTED + ACC_NORM_TOL, linestyle="--", label="gate upper")
plt.axhline(ACC_NORM_EXPECTED - ACC_NORM_TOL, linestyle="--", label="gate lower")
plt.xlabel("time [s]")
plt.ylabel("acc norm [g]")
plt.title("Accelerometer norm gate")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# Plot covariance
# =========================

plt.figure(figsize=(10, 4))
plt.plot(out["time_s"], out["P_theta_x"], label="P theta x")
plt.plot(out["time_s"], out["P_theta_y"], label="P theta y")
plt.plot(out["time_s"], out["P_theta_z"], label="P theta z")
plt.xlabel("time [s]")
plt.ylabel("covariance")
plt.title("ESKF covariance after accel update")
plt.legend()
plt.grid(True)
plt.show()
