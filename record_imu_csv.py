import serial
import csv
import time
from pathlib import Path

PORT = "/dev/cu.usbmodem1101"   # 改成你的 port
BAUD = 115200
DURATION_S = 20

out_path = Path("data/raw/imu_test.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

with serial.Serial(PORT, BAUD, timeout=2) as ser, open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])

    print("Waiting for serial...")
    time.sleep(2)
    ser.reset_input_buffer()

    start = time.time()
    count = 0

    while time.time() - start < DURATION_S:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        # 支持 comma 或 tab 分隔
        parts = line.replace("\t", ",").split(",")

        if len(parts) < 7:
            print("skip:", line)
            continue

        try:
            values = [float(x) for x in parts[:7]]
        except ValueError:
            print("skip:", line)
            continue

        writer.writerow(values)
        count += 1

print(f"Saved {count} samples to {out_path}")