"""Record a long static gyroscope run from gyro_drift_test.ino.

Example:
    python3 scripts/record_gyro_drift.py \
        --port /dev/cu.usbmodem1101 --hours 1 \
        --out data/raw/gyro_drift_1h.csv
"""

import argparse
import csv
import time
from pathlib import Path

import serial


def record(port: str, baud: int, duration_s: float, out_path: Path) -> None:
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening {port} at {baud} baud")
    print(f"Recording for {duration_s / 3600:.3f} hours -> {out_path}")
    print("Keep the board completely still. Do not touch the desk or USB cable.")

    with serial.Serial(port, baud, timeout=1) as ser, out_path.open(
        "w", newline=""
    ) as output:
        time.sleep(2)
        ser.reset_input_buffer()

        writer = csv.writer(output)
        writer.writerow(["timestamp_ms", "gx_dps", "gy_dps", "gz_dps", "temp_c"])

        start = time.monotonic()
        last_report = start
        samples = 0
        skipped = 0

        while time.monotonic() - start < duration_s:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line or line.startswith("timestamp") or line.startswith("["):
                continue

            fields = line.split(",")
            if len(fields) != 5:
                skipped += 1
                continue
            try:
                values = [float(value) for value in fields]
            except ValueError:
                skipped += 1
                continue

            writer.writerow(values)
            samples += 1

            now = time.monotonic()
            if now - last_report >= 60:
                elapsed = now - start
                output.flush()
                print(
                    f"{elapsed / 60:7.1f} min, {samples} samples, "
                    f"{samples / elapsed:5.1f} Hz, skipped={skipped}"
                )
                last_report = now

    elapsed = time.monotonic() - start
    print(f"Saved {samples} samples ({samples / elapsed:.2f} Hz) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a static gyro drift test")
    parser.add_argument("--port", default="/dev/cu.usbmodem1101")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record(args.port, args.baud, args.hours * 3600, args.out)


if __name__ == "__main__":
    main()
