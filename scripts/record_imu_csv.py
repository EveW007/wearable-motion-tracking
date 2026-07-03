# record IMU data from serial to CSV
# Usage example:
# /usr/local/bin/python3 record_imu_csv.py --out data/raw/roll_test_20s.csv --duration 20

import argparse
import serial
import csv
import time
from pathlib import Path


DEFAULT_PORT = "/dev/cu.usbmodem1101"
DEFAULT_BAUD = 115200


def record_imu_csv(port, baud, duration_s, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Opening serial...")
    print("Port:", port)
    print("Baud:", baud)
    print("Output:", out_path)
    print("Duration:", duration_s, "s")

    with serial.Serial(port, baud, timeout=1) as ser:
        print("Waiting for board to settle...")
        time.sleep(2)

        # Clear old serial buffer so we do not read stale data
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print(f"Recording for {duration_s} real seconds...")
        print("Important: keep the IMU still for the first 2 seconds.")

        start = time.monotonic()
        last_print = start
        count = 0
        skipped = 0

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])

            while True:
                now = time.monotonic()
                elapsed = now - start

                if elapsed >= duration_s:
                    break

                # Print progress once per second
                if now - last_print >= 1.0:
                    print(f"{elapsed:5.1f}s / {duration_s}s, samples = {count}, skipped = {skipped}")
                    last_print = now

                line = ser.readline().decode(errors="ignore").strip()

                if not line:
                    continue

                parts = line.replace("\t", ",").split(",")

                if len(parts) < 7:
                    skipped += 1
                    print("skip:", line)
                    continue

                try:
                    values = [float(x) for x in parts[:7]]
                except ValueError:
                    skipped += 1
                    print("skip:", line)
                    continue

                writer.writerow(values)
                count += 1

    elapsed = time.monotonic() - start
    print(f"Done. Wall-clock elapsed: {elapsed:.2f} s")
    print(f"Saved {count} samples to {out_path}")
    print(f"Approx sample rate: {count / elapsed:.2f} Hz")


def main():
    parser = argparse.ArgumentParser(description="Record raw IMU serial data to CSV.")

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path, e.g. data/raw/roll_test_20s.csv",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Recording duration in seconds. Default: 20",
    )

    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"Serial port. Default: {DEFAULT_PORT}",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Serial baud rate. Default: {DEFAULT_BAUD}",
    )

    args = parser.parse_args()

    record_imu_csv(
        port=args.port,
        baud=args.baud,
        duration_s=args.duration,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()