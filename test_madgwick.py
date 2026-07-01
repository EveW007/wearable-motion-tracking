from madgwick_filter import MadgwickFilter
import math

# 建立 filter
filt = MadgwickFilter(beta=0.1)

# 模拟静止状态：accel = gravity, gyro = 0
out = filt.update(
    accel_x=0.0,
    accel_y=0.0,
    accel_z=9.807,
    gyro_x=0.0,
    gyro_y=0.0,
    gyro_z=0.0,
    dt=0.01,
)

print("Quaternion:")
print(out["qw"], out["qx"], out["qy"], out["qz"])

print("Euler angles in radians:")
print("roll =", out["roll"])
print("pitch =", out["pitch"])
print("yaw =", out["yaw"])

print("Euler angles in degrees:")
print("roll =", out["roll"] * 180 / math.pi)
print("pitch =", out["pitch"] * 180 / math.pi)
print("yaw =", out["yaw"] * 180 / math.pi)

print("Gravity:")
print(out["gravityX"], out["gravityY"], out["gravityZ"])