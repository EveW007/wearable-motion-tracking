# Coordinate and head frame

## 为什么需要这一层

Madgwick 输出的是**传感器的姿态**，不是天然的“头部姿态”。如果板子在头上有安装偏角，直接把 Madgwick 的 roll/pitch/yaw 当成点头、侧倾、转头，会出现：

- 正视时角度不为零；
- 点头同时混入 roll 或 yaw；
- 每次重新佩戴后的基线不同；
- 直接对 Euler 角做首帧相减时，在 ±180° 附近跳变，并在大角度下产生轴耦合。

因此 Madgwick 后面需要一个独立的 head-frame 输出层。它不改变滤波器，也不改变 `beta`。

## 三个坐标系

- **Sensor frame, S**：IMU 芯片的 x/y/z，由板子和驱动定义。
- **World frame, W**：Madgwick 的参考系；本项目约定 `+Z` 沿 accelerometer 静止时的 `+g` 方向。IMU-only 条件下绝对 yaw 不可观测。
- **Head frame, H**：解剖/动作语义坐标。建议最终约定 `+X` 向前、`+Y` 向左、`+Z` 向上，并使用右手定则；实际正负号必须用佩戴后的动作试验确认。

代码中的四元数采用 `[w, x, y, z]` Hamilton 格式，`q_A_B` 表示把 B 坐标中的向量旋转到 A 坐标。

Madgwick 给出：

```text
q_world_sensor
```

固定安装标定给出：

```text
q_head_sensor
```

所以传感器姿态转成头部姿态为：

```text
q_world_head = q_world_sensor * conjugate(q_head_sensor)
```

记录正视中立位 `q_world_head_neutral` 后，最终输出为：

```text
q_neutral_head = conjugate(q_world_head_neutral) * q_world_head
```

最后才把 `q_neutral_head` 转为 ZYX yaw-pitch-roll。

## 当前代码做了什么

- `filters/head_frame.py`：安装变换、中立位四元数平均、head quaternion 和 Euler 输出。
- `filters/run_madgwick_offline.py`：使用开头 2 秒静止窗口标定中立位，写出 `head_q*` 和 `head_*_deg`。
- `filters/live_madgwick.py`：gyro bias 标定后，再要求佩戴者正视静止 2 秒；屏幕和 CSV 同时输出 head-frame 角度。

旧的 `roll_rel_deg/pitch_rel_deg/yaw_rel_deg` 仍保留，但现在指向 quaternion-relative head angles，不再是 Euler 首帧相减。

## 现在怎样使用

1. 暂时保持脚本里的 `Q_HEAD_SENSOR = [1, 0, 0, 0]`。这表示暂时假设 sensor axes 与 head axes 对齐。
2. 佩戴设备，保持正视和静止，完成 gyro bias 与 2 秒 head-neutral 标定。
3. 分别做纯点头、纯侧倾、纯左右转头，检查是否分别主要落在 pitch、roll、yaw，并检查正负号。
4. 完成六面 accelerometer 测试和三轴 gyro 正转测试，填写 sensor x/y/z 在实际板子上的物理方向。
5. 根据最终佩戴方向确定 `q_head_sensor`，替换 offline/live 脚本中的 `Q_HEAD_SENSOR`，然后重复单轴验证。

## 不能靠中立位解决的事

静止 accelerometer 只能提供重力方向，不能识别绕重力轴的安装旋转，也不能提供绝对 heading。因此：

- sensor 绕竖直方向装偏 90° 时，单纯“正视静止 2 秒”不足以可靠区分 head X/Y；还需要明确的安装几何或已知动作标定；
- yaw 仍会随 gyro bias 漂移；head frame 只定义零点和动作坐标，不会让 yaw 变成绝对可观测；
- 线性加速度误当重力的问题仍属于 Madgwick measurement correction 风险，head-frame 变换不会消除它。
