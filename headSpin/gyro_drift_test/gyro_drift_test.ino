#include <Adafruit_TinyUSB.h>
#include "LSM6DS3.h"
#include "Wire.h"

// XIAO nRF52840 Sense onboard LSM6DS3TR-C/LSM6DS3 at I2C address 0x6A.
LSM6DS3 imu(I2C_MODE, 0x6A);

// The sensor runs at 104 Hz. We stream at 100 Hz so the serial link can keep up.
constexpr uint32_t OUTPUT_INTERVAL_US = 10000;
uint32_t nextOutputUs = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  // These settings MUST be assigned before begin(). The Seeed library defaults
  // to +/-2000 dps (0.070 dps/LSB); +/-245 dps gives 0.00875 dps/LSB.
  imu.settings.gyroRange = 245;
  imu.settings.gyroSampleRate = 104;
  imu.settings.gyroBandWidth = 50;
  imu.settings.accelEnabled = 0;

  if (imu.begin() != 0) {
    Serial.println("[ERROR] IMU initialization failed");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("timestamp_ms,gx_dps,gy_dps,gz_dps,temp_c");
  nextOutputUs = micros();
}

void loop() {
  const uint32_t nowUs = micros();
  if ((int32_t)(nowUs - nextOutputUs) < 0) {
    return;
  }

  // Advance by a fixed interval instead of delay(), reducing timestamp jitter.
  nextOutputUs += OUTPUT_INTERVAL_US;

  const float gx = imu.readFloatGyroX();
  const float gy = imu.readFloatGyroY();
  const float gz = imu.readFloatGyroZ();
  const float tempC = imu.readTempC();

  Serial.print(millis());
  Serial.print(',');
  Serial.print(gx, 6);
  Serial.print(',');
  Serial.print(gy, 6);
  Serial.print(',');
  Serial.print(gz, 6);
  Serial.print(',');
  Serial.println(tempC, 3);
}
