import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from madgwick_filter import MadgwickFilter


CSV_PATH = "data/raw/static_20s.csv"
OUTPUT_PATH = "data/processed/madgwick_static_20s.csv"

GYRO_IN_DEG_PER_S = True
ACCEL_IN_G = True

G = 9.807

# For first offline sanity check:
# True  = remove initial static gyro bias, useful for checking conventions/stability
# False = raw Madgwick baseline, useful for showing yaw drift from gyro bias
REMOVE_INITIAL_GYRO_BIAS = True
STATIC_TIME_S = 2.0

BETA = 0.1


def main():
    df = pd.read_csv(CSV_PATH)

    print("CSV head:")
    print(df.head())
    print("CSV shape:", df.shape)
    print("CSV columns:", df.columns.tolist())

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

    ax = df["ax"].to_numpy(dtype=float)
    ay = df["ay"].to_numpy(dtype=float)
    az = df["az"].to_numpy(dtype=float)

    gx = df["gx"].to_numpy(dtype=float)
    gy = df["gy"].to_numpy(dtype=float)
    gz = df["gz"].to_numpy(dtype=float)

    # Accel conversion:
    # Your CSV accel is in g, but this Madgwick implementation initializes using 9.8 m/s^2.
    if ACCEL_IN_G:
        ax_mps2 = ax * G
        ay_mps2 = ay * G
        az_mps2 = az * G
        print("Converted accel from g to m/s^2 for Madgwick.")
    else:
        ax_mps2 = ax
        ay_mps2 = ay
        az_mps2 = az

    # Gyro conversion:
    # Madgwick expects rad/s.
    if GYRO_IN_DEG_PER_S:
        gx_rad = np.deg2rad(gx)
        gy_rad = np.deg2rad(gy)
        gz_rad = np.deg2rad(gz)
        print("Converted gyro from deg/s to rad/s.")
    else:
        gx_rad = gx
        gy_rad = gy
        gz_rad = gz

    # Optional initial gyro bias removal.
    # This is not part of the basic Madgwick algorithm itself.
    # It is a preprocessing step for fair static/convention testing.
    static_mask = time_s < STATIC_TIME_S

    if REMOVE_INITIAL_GYRO_BIAS:
        bg0 = np.array([
            np.mean(gx_rad[static_mask]),
            np.mean(gy_rad[static_mask]),
            np.mean(gz_rad[static_mask]),
        ])

        print("Removed initial gyro bias [deg/s]:", np.rad2deg(bg0))

        gx_rad = gx_rad - bg0[0]
        gy_rad = gy_rad - bg0[1]
        gz_rad = gz_rad - bg0[2]
    else:
        print("Using raw gyro without initial bias removal.")

    filt = MadgwickFilter(beta=BETA)

    results = []

    for i in range(len(df)):
        out = filt.update(
            ax_mps2[i],
            ay_mps2[i],
            az_mps2[i],
            gx_rad[i],
            gy_rad[i],
            gz_rad[i],
            dt[i],
        )

        acc_norm_g = np.linalg.norm([ax[i], ay[i], az[i]])
        q_norm = np.linalg.norm([out["qw"], out["qx"], out["qy"], out["qz"]])

        results.append({
            "time_s": time_s[i],
            "qw": out["qw"],
            "qx": out["qx"],
            "qy": out["qy"],
            "qz": out["qz"],
            "q_norm": q_norm,
            "roll_deg": np.rad2deg(out["roll"]),
            "pitch_deg": np.rad2deg(out["pitch"]),
            "yaw_deg": np.rad2deg(out["yaw"]),
            "gravityX": out["gravityX"],
            "gravityY": out["gravityY"],
            "gravityZ": out["gravityZ"],
            "userAccelX": out["userAccelX"],
            "userAccelY": out["userAccelY"],
            "userAccelZ": out["userAccelZ"],
            "acc_norm_g": acc_norm_g,
        })

    out_df = pd.DataFrame(results)

    # Quick relative Euler angle check.
    # Later we will replace this with quaternion-relative orientation.
    out_df["roll_rel_deg"] = out_df["roll_deg"] - out_df["roll_deg"].iloc[0]
    out_df["pitch_rel_deg"] = out_df["pitch_deg"] - out_df["pitch_deg"].iloc[0]
    out_df["yaw_rel_deg"] = out_df["yaw_deg"] - out_df["yaw_deg"].iloc[0]

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved Madgwick offline output to {OUTPUT_PATH}")
    print("Final relative roll/pitch/yaw [deg]:")
    print(out_df[["roll_rel_deg", "pitch_rel_deg", "yaw_rel_deg"]].tail())

    # Plot absolute Euler angles
    plt.figure(figsize=(10, 4))
    plt.plot(out_df["time_s"], out_df["roll_deg"], label="roll")
    plt.plot(out_df["time_s"], out_df["pitch_deg"], label="pitch")
    plt.plot(out_df["time_s"], out_df["yaw_deg"], label="yaw")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.title("Madgwick offline orientation")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot relative Euler angles
    plt.figure(figsize=(10, 4))
    plt.plot(out_df["time_s"], out_df["roll_rel_deg"], label="roll relative")
    plt.plot(out_df["time_s"], out_df["pitch_rel_deg"], label="pitch relative")
    plt.plot(out_df["time_s"], out_df["yaw_rel_deg"], label="yaw relative")
    plt.xlabel("time [s]")
    plt.ylabel("relative angle [deg]")
    plt.title("Madgwick relative orientation from initial pose")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot quaternion norm
    plt.figure(figsize=(10, 3))
    plt.plot(out_df["time_s"], out_df["q_norm"])
    plt.xlabel("time [s]")
    plt.ylabel("quaternion norm")
    plt.title("Madgwick quaternion norm")
    plt.grid(True)
    plt.show()

    # Plot accel norm
    plt.figure(figsize=(10, 3))
    plt.plot(out_df["time_s"], out_df["acc_norm_g"])
    plt.axhline(1.0, linestyle="--", label="1 g")
    plt.xlabel("time [s]")
    plt.ylabel("acc norm [g]")
    plt.title("Raw accelerometer norm")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()