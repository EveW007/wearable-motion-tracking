"""Measure gyro offset change and calibrated angle drift from a static CSV.

The initial calibration window estimates a single startup offset. That same
constant is then held for the rest of the recording; it is deliberately not
re-estimated continuously, so subsequent bias drift remains visible.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


AXES = ("gx", "gy", "gz")


def load_csv(path: Path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < 2:
        raise ValueError("CSV must contain at least two data rows")

    # Accept both the dedicated drift recorder and existing project CSV names.
    timestamp_key = "timestamp_ms" if "timestamp_ms" in rows[0] else "timestamp"
    axis_keys = {
        axis: f"{axis}_dps" if f"{axis}_dps" in rows[0] else axis for axis in AXES
    }

    timestamp_ms = np.asarray([float(row[timestamp_key]) for row in rows])
    # Correct a possible Arduino millis() wrap (normally only after 49.7 days).
    timestamp_ms = np.unwrap(timestamp_ms, period=2**32)
    time_s = (timestamp_ms - timestamp_ms[0]) / 1000.0
    gyro = np.column_stack(
        [np.asarray([float(row[axis_keys[axis]]) for row in rows]) for axis in AXES]
    )

    temp = None
    if "temp_c" in rows[0]:
        temp = np.asarray([float(row["temp_c"]) for row in rows])
    return time_s, gyro, temp


def cumulative_trapezoid(values, time_s):
    result = np.zeros_like(values)
    dt = np.diff(time_s)
    result[1:] = np.cumsum(0.5 * (values[:-1] + values[1:]) * dt)
    return result


def write_window_means(path, time_s, gyro, bias0, window_s):
    window_index = np.floor(time_s / window_s).astype(int)
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["window_start_s"]
            + [f"{axis}_mean_dps" for axis in AXES]
            + [f"{axis}_bias_change_deg_per_hr" for axis in AXES]
        )
        for index in np.unique(window_index):
            mask = window_index == index
            mean = np.mean(gyro[mask], axis=0)
            writer.writerow([index * window_s, *mean, *((mean - bias0) * 3600)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a static gyro drift run")
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--calibration-min", type=float, default=10.0,
        help="Initial static interval used once to estimate offset (default: 10)",
    )
    parser.add_argument(
        "--window-min", type=float, default=5.0,
        help="Window used to track the changing bias (default: 5)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    time_s, gyro, temp = load_csv(args.csv)
    calibration_s = args.calibration_min * 60
    duration_s = time_s[-1]
    if duration_s <= calibration_s + 10:
        raise ValueError(
            f"Recording is only {duration_s / 60:.1f} min; it must extend at least "
            f"10 s beyond the {args.calibration_min:g} min calibration interval"
        )

    calibration_mask = time_s <= calibration_s
    evaluation_mask = time_s >= calibration_s
    bias0 = np.mean(gyro[calibration_mask], axis=0)

    eval_time = time_s[evaluation_mask] - time_s[evaluation_mask][0]
    residual = gyro[evaluation_mask] - bias0
    angles_deg = np.column_stack(
        [cumulative_trapezoid(residual[:, index], eval_time) for index in range(3)]
    )

    # Angle slope is the effective post-calibration orientation drift.
    angle_drift_deg_hr = np.asarray(
        [np.polyfit(eval_time, angles_deg[:, index], 1)[0] * 3600 for index in range(3)]
    )

    window_s = args.window_min * 60
    first_eval = (time_s >= calibration_s) & (time_s < calibration_s + window_s)
    last_eval = time_s >= max(calibration_s, duration_s - window_s)
    first_bias = np.mean(gyro[first_eval], axis=0)
    last_bias = np.mean(gyro[last_eval], axis=0)
    bias_change_deg_hr = (last_bias - first_bias) * 3600

    print(f"File: {args.csv}")
    print(f"Duration: {duration_s / 3600:.3f} h; samples: {len(time_s)}")
    print(f"Initial offset from first {args.calibration_min:g} min [deg/s]:")
    for axis, value in zip(AXES, bias0):
        print(f"  {axis}: {value:+.8f}")
    print("\nBias change: last window minus first post-calibration window [deg/hr]:")
    for axis, value in zip(AXES, bias_change_deg_hr):
        print(f"  {axis}: {value:+.3f}")
    print("\nEffective angle drift after one fixed calibration [deg/hr]:")
    for axis, value in zip(AXES, angle_drift_deg_hr):
        print(f"  {axis}: {value:+.3f}")
    print("\nFinal integrated angle error [deg]:")
    for axis, value in zip(AXES, angles_deg[-1]):
        print(f"  {axis}: {value:+.3f}")
    if temp is not None:
        print(f"\nTemperature range: {np.min(temp):.2f} to {np.max(temp):.2f} degC")

    out_path = args.out or args.csv.with_name(args.csv.stem + "_windows.csv")
    write_window_means(out_path, time_s, gyro, bias0, window_s)
    print(f"\nWindowed bias data saved to: {out_path}")


if __name__ == "__main__":
    main()
