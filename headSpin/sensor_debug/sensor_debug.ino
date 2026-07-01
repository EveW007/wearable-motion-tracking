#include <Adafruit_TinyUSB.h>
#include "LSM6DS3.h"
#include "Wire.h"

LSM6DS3 myIMU(I2C_MODE, 0x6A);

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  if (myIMU.begin() != 0) {
    Serial.println("[ERROR] IMU init failed! Check wiring.");
    while (1) {
      delay(100);
    }
  }

  Serial.println("Time(ms),AccX(g),AccY(g),AccZ(g),GyroX(dps),GyroY(dps),GyroZ(dps)");
}

void loop() {
  float ax = myIMU.readFloatAccelX();
  float ay = myIMU.readFloatAccelY();
  float az = myIMU.readFloatAccelZ();

  float gx = myIMU.readFloatGyroX();
  float gy = myIMU.readFloatGyroY();
  float gz = myIMU.readFloatGyroZ();

  Serial.print(millis());
  Serial.print(",");
  Serial.print(ax, 6);
  Serial.print(",");
  Serial.print(ay, 6);
  Serial.print(",");
  Serial.print(az, 6);
  Serial.print(",");
  Serial.print(gx, 6);
  Serial.print(",");
  Serial.print(gy, 6);
  Serial.print(",");
  Serial.println(gz, 6);

  delay(100);
}