#include <Adafruit_TinyUSB.h>

void setup() {
  Serial.begin(115200);
  while (!Serial);          // Wait for USB serial to connect
  Serial.println("USB Serial OK - XIAO nRF52840");
}

void loop() {
  Serial.print("Tick: ");
  Serial.println(millis());
  delay(1000);
}