# record 20s of IMU data to CSV

import serial
import csv
import time
from pathlib import Path

PORT = "/dev/cu.usbmodem1101"   # 改成你的真实 port
BAUD = 115200
DURATION_S = 20

out_path = Path("data/raw/static_20s.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

print("Opening serial...")
with serial.Serial(PORT, BAUD, timeout=1) as ser:
    print("Waiting for board to settle...")
    time.sleep(2)

    # 关键：清掉旧 buffer，不读旧数据
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print(f"Recording for {DURATION_S} real seconds...")

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

            if elapsed >= DURATION_S:
                break

            # 每秒打印一次，让你确认它没有瞬间结束
            if now - last_print >= 1.0:
                print(f"{elapsed:5.1f}s / {DURATION_S}s, samples = {count}, skipped = {skipped}")
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