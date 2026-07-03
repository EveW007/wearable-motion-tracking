import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ESKF_PATH = "data/processed/eskf_roll_test_20s.csv"
MADGWICK_PATH = "data/processed/madgwick_roll_test_20s.csv"


def rms(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x * x))


def drift_rate_deg_per_s(time_s, angle_deg):
    time_s = np.asarray(time_s, dtype=float)
    angle_deg = np.asarray(angle_deg, dtype=float)

    # Linear fit: angle = slope * time + intercept
    slope, intercept = np.polyfit(time_s, angle_deg, 1)
    return slope


def main():
    eskf = pd.read_csv(ESKF_PATH)
    madgwick = pd.read_csv(MADGWICK_PATH)

    print("ESKF columns:", eskf.columns.tolist())
    print("Madgwick columns:", madgwick.columns.tolist())

    # Use each filter's own time base.
    # They should be from the same CSV, so lengths should usually match.
    print("ESKF shape:", eskf.shape)
    print("Madgwick shape:", madgwick.shape)

    # ------------------------------------------------------------
    # Plot relative roll
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(eskf["time_s"], eskf["roll_rel_deg"], label="ESKF roll rel")
    plt.plot(madgwick["time_s"], madgwick["roll_rel_deg"], label="Madgwick roll rel")
    plt.xlabel("time [s]")
    plt.ylabel("roll relative [deg]")
    plt.title("Static test: roll relative")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------------
    # Plot relative pitch
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(eskf["time_s"], eskf["pitch_rel_deg"], label="ESKF pitch rel")
    plt.plot(madgwick["time_s"], madgwick["pitch_rel_deg"], label="Madgwick pitch rel")
    plt.xlabel("time [s]")
    plt.ylabel("pitch relative [deg]")
    plt.title("Static test: pitch relative")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------------
    # Plot relative yaw
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(eskf["time_s"], eskf["yaw_rel_deg"], label="ESKF yaw rel")
    plt.plot(madgwick["time_s"], madgwick["yaw_rel_deg"], label="Madgwick yaw rel")
    plt.xlabel("time [s]")
    plt.ylabel("yaw relative [deg]")
    plt.title("Static test: yaw relative")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    metrics = []

    for name, df in [("ESKF", eskf), ("Madgwick", madgwick)]:
        roll_std = df["roll_rel_deg"].std()
        pitch_std = df["pitch_rel_deg"].std()
        yaw_std = df["yaw_rel_deg"].std()

        roll_rms = rms(df["roll_rel_deg"])
        pitch_rms = rms(df["pitch_rel_deg"])
        yaw_rms = rms(df["yaw_rel_deg"])

        yaw_drift = drift_rate_deg_per_s(df["time_s"], df["yaw_rel_deg"])

        roll_max_abs = np.max(np.abs(df["roll_rel_deg"]))
        pitch_max_abs = np.max(np.abs(df["pitch_rel_deg"]))
        yaw_max_abs = np.max(np.abs(df["yaw_rel_deg"]))

        metrics.append({
            "filter": name,
            "roll_std_deg": roll_std,
            "pitch_std_deg": pitch_std,
            "yaw_std_deg": yaw_std,
            "roll_rms_deg": roll_rms,
            "pitch_rms_deg": pitch_rms,
            "yaw_rms_deg": yaw_rms,
            "yaw_drift_deg_per_s": yaw_drift,
            "roll_max_abs_deg": roll_max_abs,
            "pitch_max_abs_deg": pitch_max_abs,
            "yaw_max_abs_deg": yaw_max_abs,
        })

    metrics_df = pd.DataFrame(metrics)
    print("\nStatic stability metrics:")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv("data/processed/static_comparison_metrics.csv", index=False)
    print("\nSaved metrics to data/processed/static_comparison_metrics.csv")


if __name__ == "__main__":
    main()