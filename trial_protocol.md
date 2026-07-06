# IMU Trial Protocol

## General recording rules

- All trials start with at least 2 seconds static.
- The initial static period is used for gyro bias estimation and initial accelerometer tilt alignment.
- Keep USB cable still during recording.
- Avoid touching the table or moving the laptop during static trials.
- Accelerometer is recorded in g.
- Gyroscope is recorded in deg/s.
- Timestamp is Arduino milliseconds.

## Trials

### static_20s.csv

Purpose:
Static stability test.

Protocol:
- 0–20s: sensor remains still.

Expected:
- roll/pitch relative close to zero
- yaw may drift slightly
- accelerometer norm near 1g

---

### roll_test_20s.csv

Purpose:
Single-axis roll response test.

Protocol:
- 0–2s: static
- 2–8s: roll sensor slowly
- 8–14s: return to initial pose
- 14–20s: static

Expected:
- roll changes clearly
- pitch/yaw should remain smaller, except for coupling/mounting effects

---

### pitch_test_20s.csv

Purpose:
Single-axis pitch response test.

Protocol:
- 0–2s: static
- 2–8s: pitch sensor slowly
- 8–14s: return to initial pose
- 14–20s: static

Expected:
- pitch changes clearly
- roll/yaw should remain smaller, except for coupling/mounting effects

---

### yaw_test_20s.csv

Purpose:
Short-term yaw tracking test.

Protocol:
- 0–2s: static
- 2–8s: rotate sensor around vertical axis
- 8–14s: return to initial pose
- 14–20s: static

Expected:
- yaw changes clearly
- yaw drift may remain due to IMU-only setup

---

### maneuver_like_45_90.csv

Purpose:
Preliminary maneuver-level orientation tracking.

Protocol:
- 0–3s: static
- 3–8s: rotate approximately 45 degrees
- 8–13s: hold
- 13–18s: rotate further approximately 90 degrees
- 18–23s: hold
- 23–30s: return to initial pose

Expected:
- filter should detect phase changes and hold periods
- exact angle accuracy is not yet guaranteed without ground truth