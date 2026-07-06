import argparse
import csv
import json
import time
from pathlib import Path

import serial


PORT = "/dev/cu.usbmodem1101"   # TODO: change this to your actual port
BAUD = 115200


def record_trial(trial_name, duration_s, notes=""):
    raw_dir = Path("data/raw")
    meta_dir = Path("data/metadata")
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    out_path = raw_dir / f"{trial_name}.csv"
    meta_path = meta_dir / f"{trial_name}.json"

    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. Use a new trial name or manually delete the old file."
        )

    print(f"Opening serial port: {PORT}")
    print(f"Trial: {trial_name}")
    print(f"Duration: {duration_s} s")
    print(f"Output: {out_path}")

    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        print("Waiting for board to settle...")
        time.sleep(2)

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print("Start recording now.")
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

                if now - last_print >= 1.0:
                    print(f"{elapsed:5.1f}s / {duration_s}s, samples={count}, skipped={skipped}")
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
    sample_rate = count / elapsed if elapsed > 0 else 0.0

    metadata = {
        "trial_name": trial_name,
        "duration_s_requested": duration_s,
        "duration_s_actual": elapsed,
        "samples": count,
        "skipped": skipped,
        "approx_sample_rate_hz": sample_rate,
        "port": PORT,
        "baud": BAUD,
        "notes": notes,
        "units": {
            "timestamp": "Arduino milliseconds",
            "accel": "g",
            "gyro": "deg/s"
        }
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print(f"Saved raw CSV to {out_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"Approx sample rate: {sample_rate:.2f} Hz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True, help="Trial name, e.g. roll_test_30s")
    parser.add_argument("--duration", type=float, required=True, help="Duration in seconds")
    parser.add_argument("--notes", default="", help="Optional notes about the trial")
    args = parser.parse_args()

    record_trial(args.trial, args.duration, args.notes)


if __name__ == "__main__":
    main()