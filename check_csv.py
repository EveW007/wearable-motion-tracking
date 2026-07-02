import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/imu_test.csv")

print(df.head())
print(df.shape)
print(df.columns)

# 1. 读取数据
t = df["timestamp"].to_numpy(dtype=float)

ax = df["ax"].to_numpy(dtype=float)
ay = df["ay"].to_numpy(dtype=float)
az = df["az"].to_numpy(dtype=float)

gx = df["gx"].to_numpy(dtype=float)
gy = df["gy"].to_numpy(dtype=float)
gz = df["gz"].to_numpy(dtype=float)

# 2. timestamp 转成相对秒
# 如果 timestamp 像 Arduino millis()，差值大概是 100，就除以 1000
# 如果 timestamp 已经是秒，差值大概是 0.1，就不用除
if np.median(np.diff(t)) > 1:
    time_s = (t - t[0]) / 1000.0
else:
    time_s = t - t[0]

dt = np.diff(time_s)

print("duration [s]:", time_s[-1] - time_s[0])
print("median dt [s]:", np.median(dt))
print("approx sampling rate [Hz]:", 1 / np.median(dt))

# 3. 检查 accel / gyro 大小
acc_norm = np.sqrt(ax**2 + ay**2 + az**2)
gyro_norm = np.sqrt(gx**2 + gy**2 + gz**2)

print("acc norm median:", np.median(acc_norm))
print("gyro norm median:", np.median(gyro_norm))

# 4. 画 accel norm
plt.figure(figsize=(10, 3))
plt.plot(time_s, acc_norm)
plt.axhline(9.807, linestyle="--", label="9.807 m/s²")
plt.axhline(1.0, linestyle="--", label="1 g")
plt.xlabel("time [s]")
plt.ylabel("acc norm")
plt.title("Accelerometer norm")
plt.legend()
plt.grid(True)
plt.show()

# 5. 画 gyro
plt.figure(figsize=(10, 3))
plt.plot(time_s, gx, label="gx")
plt.plot(time_s, gy, label="gy")
plt.plot(time_s, gz, label="gz")
plt.xlabel("time [s]")
plt.ylabel("gyro raw")
plt.title("Gyroscope raw data")
plt.legend()
plt.grid(True)
plt.show()