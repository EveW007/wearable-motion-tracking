#include "LSM6DS3.h"

LSM6DS3 IMU(I2C_MODE, 0x6A);

void setup() {
  Serial.begin(115200);

  while (!Serial) {
    delay(10);
  }

  if (IMU.begin() != 0) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.println("IMU initialized!");
}

void loop() {
  Serial.println("still running");
  delay(1000);

float x = IMU.readFloatAccelX();
  float y = IMU.readFloatAccelY();
  float z = IMU.readFloatAccelZ();

  Serial.print(x);
  Serial.print("\t");
  Serial.print(y);
  Serial.print("\t");
  Serial.println(z);

  delay(500);

  float gx = IMU.readFloatGyroX();
  float gy = IMU.readFloatGyroY();
  float gz = IMU.readFloatGyroZ();

  Serial.print(gx);
  Serial.print("\t");
  Serial.print(gy);
  Serial.print("\t");
  Serial.println(gz);

  delay(500);

}