/*
 * ======================================================================================
 *  AI FLASH FLOOD & LANDSLIDE EARLY WARNING SYSTEM - ESP32 / ESP32-S3 SENSOR
 * NODE FIRMWARE
 * ======================================================================================
 *  Target Hardware:
 *    1. ESP32-S3 (ESP32-S3-DevKitC-1 / ESP32-S3 WROOM / NodeMCU ESP32-S3)
 *    2. Classic ESP32 (ESP32 DevKit V1 / ESP-WROOM-32 / NodeMCU ESP32 30/38
 * pin)
 *
 *  PIN CONNECTIONS (Choose your board type below in Section 2):
 *  --------------------------------------------------------------------------------------
 *  FOR ESP32-S3:
 *    - DHT11 Temp & Humidity      : VCC -> 3.3V | GND -> GND | DATA/OUT -> GPIO 4 (IO4)
 *    - MPU-6050 (GY-521) 6-Axis   : VCC -> 5V/VIN or 3.3V | GND -> GND | SDA -> GPIO 8 | SCL -> GPIO 9 | AD0 -> GND
 *    - Soil Moisture Sensor       : VCC -> 3.3V | GND -> GND | AOUT -> GPIO 1 (IO1 - ADC1)
 *    - HW-038 Raindrop Sensor     : VCC -> 3.3V | GND -> GND | A0/AOUT -> GPIO 2 (IO2 - ADC1) | D0 -> GPIO 3 (IO3)
 *    - Piezo Buzzer               : (+) -> GPIO 5 (IO5) | (-) -> GND
 *    - Status LEDs                : Green -> GPIO 6 (IO6) | Yellow -> GPIO 7 (IO7) | Red -> GPIO 10 (IO10)
 *
 *  FOR CLASSIC ESP32 (DevKit V1):
 *    - DHT11 Temp & Humidity      : VCC -> 3.3V | GND -> GND | DATA/OUT -> GPIO 4
 *    - MPU-6050 (GY-521) 6-Axis   : VCC -> 5V/VIN or 3.3V | GND -> GND | SDA -> GPIO 21 | SCL -> GPIO 22 | AD0 -> GND
 *    - Soil Moisture Sensor       : VCC -> 3.3V | GND -> GND | AOUT -> GPIO 34 (ADC1)
 *    - HW-038 Raindrop Sensor     : VCC -> 3.3V | GND -> GND | A0/AOUT -> GPIO 35 (ADC1) | D0 -> GPIO 18
 *    - Piezo Buzzer               : (+) -> GPIO 5 | (-) -> GND
 *    - Status LEDs                : Green -> GPIO 19 | Yellow -> GPIO 23 | Red -> GPIO 25
 * ======================================================================================
 */

#include <ArduinoJson.h>
#include <DHT.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <Wire.h>

// ======================================================================================
// 1. CONFIGURATION: WIFI & BACKEND SERVER SETTINGS
// ======================================================================================
const char *WIFI_SSID = "Hunters"; // Your 2.4 GHz WiFi SSID (Must be 2.4 GHz!)
const char *WIFI_PASSWORD = "12345678"; // Your WiFi Password

// Backend Telemetry Endpoint (Your computer's active local IP address on 'Hunters' Wi-Fi or Cloud HTTPS URL)
const char *SERVER_URL = "http://10.45.63.139:5000/api/telemetry";

// Node ID & Fast Telemetry Streaming Interval (2000 ms = 2 seconds)
const char *NODE_ID = "ESP32_FLOOD_NODE_01";
const unsigned long TELEMETRY_INTERVAL_MS = 2000;

// ======================================================================================
// 2. BOARD SELECTION & PIN DEFINITIONS
// ======================================================================================
// ---> UNCOMMENT THE LINE THAT MATCHES YOUR PHYSICAL BOARD:
#define BOARD_ESP32_S3 // Enable if using ESP32-S3 (DevKitC-1 / WROOM)
// #define BOARD_ESP32_CLASSIC  // Enable if using Standard ESP32 (ESP-WROOM-32 / DevKit V1)

#if defined(BOARD_ESP32_CLASSIC)
// Standard ESP32 Pinout (I2C default on GPIO 21 & 22)
#define PIN_DHT 4
#define PIN_MPU_SDA 21     // Standard ESP32 SDA
#define PIN_MPU_SCL 22     // Standard ESP32 SCL
#define PIN_SOIL_ANALOG 34 // ADC1_CH6 (WiFi-safe)
#define PIN_RAIN_ANALOG 35 // ADC1_CH7 (WiFi-safe)
#define PIN_RAIN_DIGITAL 18
#define PIN_BUZZER 5
#define PIN_LED_GREEN 19
#define PIN_LED_YELLOW 23
#define PIN_LED_RED 25
#else
// ESP32-S3 Pinout
#define PIN_DHT 4          // GPIO 4
#define PIN_MPU_SDA 8      // GPIO 8 - I2C SDA
#define PIN_MPU_SCL 9      // GPIO 9 - I2C SCL
#define PIN_SOIL_ANALOG 1  // GPIO 1 - ADC1_CH0
#define PIN_RAIN_ANALOG 2  // GPIO 2 - ADC1_CH1
#define PIN_RAIN_DIGITAL 3 // GPIO 3
#define PIN_BUZZER 5       // GPIO 5
#define PIN_LED_GREEN 6    // GPIO 6
#define PIN_LED_YELLOW 7   // GPIO 7
#define PIN_LED_RED 10     // GPIO 10
#endif

#define DHT_TYPE DHT11

// ADC Calibration (12-bit ADC: 0 to 4095)
// Dry air: ~3800-4095, Submerged in water: ~1000-1400
const int SOIL_DRY_ADC = 3800;
const int SOIL_WET_ADC = 1200;

// HW-038 Rain / Water Level Sensor: Dry plate: ~4095, Wet plate with water drops: ~1000-1600
const int RAIN_DRY_ADC = 4095;
const int RAIN_WET_ADC = 1000;

// Maximum Measurement Scale in mm or mm/h (Default: 100.0 mm - Change to 150, 200, 300 mm as desired):
const float WATER_LEVEL_MAX_SCALE_MM = 100.0f;

// ======================================================================================
// 3. CALIBRATED MULTI-TIER RISK THRESHOLDS (METEOROLOGICAL & SEISMIC MATRIX)
// ======================================================================================
// Water Level / Rainfall Rate Thresholds (mm or mm/h)
const float THRESH_RAIN_LIGHT_MMH     = 15.0f; // Level 1: Advisory (Rising Water / Light-Moderate Rain)
const float THRESH_RAIN_HEAVY_MMH     = 45.0f; // Level 2: High Alert (High Water Level / Heavy Downpour)
const float THRESH_RAIN_EXTREME_MMH   = 75.0f; // Level 3: Critical Flood Warning (Extreme Water Surge / Cloudburst)

// Cumulative Rainfall Depth Thresholds (mm)
const float THRESH_ACCUM_MODERATE_MM  = 25.0f; // Level 1
const float THRESH_ACCUM_HEAVY_MM     = 50.0f; // Level 2
const float THRESH_ACCUM_EXTREME_MM   = 100.0f; // Level 3

// Soil Moisture Saturation Thresholds (%)
const float THRESH_SOIL_ELEVATED_PCT  = 60.0f; // Level 1
const float THRESH_SOIL_HEAVY_PCT     = 75.0f; // Level 2
const float THRESH_SOIL_SATURATED_PCT = 85.0f; // Level 3 (Infiltration Exhausted)
const float THRESH_SOIL_WATERLOG_PCT  = 88.0f; // Level 2 Immediate

// Terrain Slope Tilt Thresholds (Degrees)
const float THRESH_SLOPE_HILLSIDE_DEG = 20.0f; // Moderate Hill Slope
const float THRESH_SLOPE_STEEP_DEG    = 30.0f; // Steep Mountain Slope
const float THRESH_SLOPE_CLIFF_DEG    = 35.0f; // Extreme Cliff Inclination

// Seismic Vibration Thresholds (g)
const float THRESH_VIBE_NOTABLE_G     = 0.15f; // Level 1
const float THRESH_VIBE_SIGNIFICANT_G = 0.20f; // Level 2
const float THRESH_VIBE_SEVERE_G      = 0.35f; // Level 3 (Landslide Slippage)

// ======================================================================================
// 4. GLOBAL SENSORS & VARIABLES
// ======================================================================================
DHT dht(PIN_DHT, DHT_TYPE);

uint8_t mpuActiveAddress = 0; // 0x68 (AD0->GND) or 0x69 (AD0->VCC/3.3V)
bool mpuOnline = false;

unsigned long lastTelemetryTime = 0;
unsigned long lastWiFiCheckTime = 0;
unsigned long lastMPUScanTime = 0;
float cumulativeRainfallProxy = 0.0;

// Persistent last valid values for DHT11
float lastTemp = 24.5;
float lastHumidity = 65.0;

/**
 * Autonomous Local Edge Fallback Risk Evaluation (Active when offline / disconnected)
 */
int evaluateEdgeRiskLocal(float rainRate, float rainAccum, float soilPct, float slopeDeg, float vibeG) {
  // LEVEL 3 (SEVERE RISK / CRITICAL EMERGENCY)
  if (rainRate >= THRESH_RAIN_EXTREME_MMH ||
      (soilPct >= THRESH_SOIL_SATURATED_PCT && rainRate >= THRESH_RAIN_HEAVY_MMH) ||
      rainAccum >= THRESH_ACCUM_EXTREME_MM ||
      (slopeDeg >= THRESH_SLOPE_STEEP_DEG && soilPct >= 80.0f && (rainRate >= 20.0f || vibeG >= 0.25f)) ||
      (slopeDeg >= THRESH_SLOPE_CLIFF_DEG && vibeG >= 0.40f)) {
    return 3;
  }
  // LEVEL 2 (HIGH RISK / WATCH & WARNING)
  if (rainRate >= THRESH_RAIN_HEAVY_MMH ||
      rainAccum >= THRESH_ACCUM_HEAVY_MM ||
      (soilPct >= THRESH_SOIL_HEAVY_PCT && (rainRate >= 15.0f || slopeDeg >= 25.0f)) ||
      (slopeDeg >= THRESH_SLOPE_STEEP_DEG && (vibeG >= THRESH_VIBE_NOTABLE_G || rainRate >= THRESH_RAIN_LIGHT_MMH)) ||
      soilPct >= THRESH_SOIL_WATERLOG_PCT) {
    return 2;
  }
  // LEVEL 1 (MODERATE RISK / ADVISORY)
  if (rainRate >= THRESH_RAIN_LIGHT_MMH ||
      rainAccum >= THRESH_ACCUM_MODERATE_MM ||
      soilPct >= THRESH_SOIL_ELEVATED_PCT ||
      (slopeDeg >= THRESH_SLOPE_HILLSIDE_DEG && (soilPct >= 50.0f || rainRate >= 5.0f)) ||
      vibeG >= THRESH_VIBE_SIGNIFICANT_G) {
    return 1;
  }
  return 0; // Low Risk / Baseline All Clear
}

// ======================================================================================
// 4. I2C BUS RECOVERY & MPU-6050 DIRECT DRIVER
// ======================================================================================

/**
 * Recovers stuck I2C bus if a slave device held SDA low during sudden reset.
 */
void recoverI2CBus(int sdaPin, int sclPin) {
  pinMode(sdaPin, INPUT_PULLUP);
  pinMode(sclPin, OUTPUT);
  digitalWrite(sclPin, HIGH);

  // Send 9 clock pulses on SCL to force stuck slave to release SDA
  for (int i = 0; i < 9; i++) {
    if (digitalRead(sdaPin) == HIGH)
      break;
    digitalWrite(sclPin, LOW);
    delayMicroseconds(10);
    digitalWrite(sclPin, HIGH);
    delayMicroseconds(10);
  }

  // Generate STOP condition
  pinMode(sdaPin, OUTPUT);
  digitalWrite(sdaPin, LOW);
  delayMicroseconds(10);
  digitalWrite(sclPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(sdaPin, HIGH);
  delayMicroseconds(10);
}

/**
 * Initializes and configures MPU-6050 (wakes up power mgmt, sets clock to gyro
 * PLL).
 */
bool wakeAndConfigureMPU(uint8_t addr) {
  // Test connection
  Wire.beginTransmission(addr);
  if (Wire.endTransmission() != 0)
    return false;

  // 1. Reset device signal paths and power state
  Wire.beginTransmission(addr);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0x80); // Reset bit
  Wire.endTransmission();
  delay(50);

  // 2. Wake up MPU-6050 and set Clock Source to PLL with X axis gyroscope
  Wire.beginTransmission(addr);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0x01); // PLL with X-Gyro reference
  if (Wire.endTransmission() != 0)
    return false;
  delay(20);

  // 3. Enable all accelerometer and gyroscope axes (PWR_MGMT_2 = 0x00)
  Wire.beginTransmission(addr);
  Wire.write(0x6C); // PWR_MGMT_2
  Wire.write(0x00); // 0 = all axes on
  Wire.endTransmission();
  delay(10);

  // 4. Set Accelerometer Range to ±4G (Register 0x1C = 0x08)
  Wire.beginTransmission(addr);
  Wire.write(0x1C); // ACCEL_CONFIG
  Wire.write(0x08); // ±4g
  Wire.endTransmission();

  // 5. Set Gyro Range to ±500°/s (Register 0x1B = 0x08)
  Wire.beginTransmission(addr);
  Wire.write(0x1B); // GYRO_CONFIG
  Wire.write(0x08); // ±500°/s
  Wire.endTransmission();

  // 6. Set Digital Low Pass Filter (Register 0x1A = 0x03 -> ~42Hz bandwidth)
  Wire.beginTransmission(addr);
  Wire.write(0x1A); // CONFIG
  Wire.write(0x03);
  Wire.endTransmission();

  return true;
}

/**
 * Scans I2C bus and reports all connected devices.
 */
bool scanAndInitMPU() {
  Serial.println("[I2C SCAN] Scanning I2C bus...");
  int nDevices = 0;
  uint8_t foundAddress = 0;

  for (uint8_t address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    uint8_t error = Wire.endTransmission();

    if (error == 0) {
      Serial.printf("  -> [I2C FOUND] Device responding at address 0x%02X",
                    address);
      if (address == 0x68) {
        Serial.println(" (MPU-6050 Default: AD0->GND)");
        foundAddress = address;
      } else if (address == 0x69) {
        Serial.println(" (MPU-6050 Alternate: AD0->VCC)");
        foundAddress = address;
      } else {
        Serial.println();
      }
      nDevices++;
    }
  }

  if (nDevices == 0) {
    Serial.println("  -> [I2C SCAN] No I2C devices found!");
    Serial.printf("     Wiring Check: SDA->GPIO%d, SCL->GPIO%d, VCC->5V/3.3V, "
                  "GND->GND, AD0->GND\n",
                  PIN_MPU_SDA, PIN_MPU_SCL);
    mpuOnline = false;
    mpuActiveAddress = 0;
    return false;
  }

  uint8_t candidateAddresses[2] = {0x68, 0x69};
  if (foundAddress == 0x68 || foundAddress == 0x69) {
    candidateAddresses[0] = foundAddress;
  }

  for (int i = 0; i < 2; i++) {
    uint8_t addr = candidateAddresses[i];
    if (wakeAndConfigureMPU(addr)) {
      mpuActiveAddress = addr;
      mpuOnline = true;
      Serial.printf("[I2C OK] MPU-6050 Motion Sensor successfully initialized at 0x%02X!\n", addr);
      return true;
    }
  }

  mpuOnline = false;
  mpuActiveAddress = 0;
  Serial.println("[WARN] MPU-6050 found on bus but failed configuration. Retrying...");
  return false;
}

/**
 * Reads raw 14 bytes from MPU-6050 and converts to 3D Pitch Angle and Seismic
 * Vibration.
 */
bool readMPURaw(float &pitchDeg, float &vibrationMag) {
  if (!mpuOnline || mpuActiveAddress == 0) {
    pitchDeg = 0.0;
    vibrationMag = 0.0;
    return false;
  }

  // Request 14 bytes starting from ACCEL_XOUT_H (Register 0x3B)
  Wire.beginTransmission(mpuActiveAddress);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) {
    Wire.beginTransmission(mpuActiveAddress);
    Wire.write(0x3B);
    if (Wire.endTransmission(true) != 0) {
      mpuOnline = false;
      return false;
    }
  }

  int count = Wire.requestFrom((int)mpuActiveAddress, 14, true);
  if (count < 14 || Wire.available() < 14) {
    mpuOnline = false;
    return false;
  }

  int16_t rawAx = (Wire.read() << 8) | Wire.read();
  int16_t rawAy = (Wire.read() << 8) | Wire.read();
  int16_t rawAz = (Wire.read() << 8) | Wire.read();
  int16_t rawTemp = (Wire.read() << 8) | Wire.read();
  int16_t rawGx = (Wire.read() << 8) | Wire.read();
  int16_t rawGy = (Wire.read() << 8) | Wire.read();
  int16_t rawGz = (Wire.read() << 8) | Wire.read();

  // If reading returned all 0s or all -1s, sensor is unseated
  if ((rawAx == 0 && rawAy == 0 && rawAz == 0) ||
      (rawAx == -1 && rawAy == -1 && rawAz == -1)) {
    return false;
  }

  // Convert raw values for ±4g range: sensitivity is 8192 LSB/g (1g = 9.80665 m/s^2)
  float ax = (float)rawAx / 8192.0f * 9.80665f;
  float ay = (float)rawAy / 8192.0f * 9.80665f;
  float az = (float)rawAz / 8192.0f * 9.80665f;

  // Convert raw gyro for ±500°/s: sensitivity is 65.5 LSB/(°/s)
  float gx = (float)rawGx / 65.5f;
  float gy = (float)rawGy / 65.5f;
  float gz = (float)rawGz / 65.5f;

  // Calculate 3D Tilt Angle (0° is flat table, 90° is vertical cliff)
  float totalAcc = sqrt(ax * ax + ay * ay + az * az);
  if (totalAcc > 0.5f) {
    float cosTilt = constrain(abs(az) / totalAcc, 0.0f, 1.0f);
    pitchDeg = acos(cosTilt) * (180.0f / PI);
  } else {
    pitchDeg = 0.0f;
  }

  // Calculate vibration magnitude (deviation from 1G gravity + rotational speed)
  vibrationMag = abs(totalAcc - 9.80665f);
  float gyroSpeed = sqrt(gx * gx + gy * gy + gz * gz);
  if (gyroSpeed > 2.0f) {
    vibrationMag += (gyroSpeed / 100.0f);
  }

  return true;
}

// ======================================================================================
// 5. HARDWARE INITIALIZATION & WIFI
// ======================================================================================

void setupHardware() {
  Serial.begin(115200);

  // Wait up to 2.5 seconds for USB CDC Serial Monitor to connect
  unsigned long startWait = millis();
  while (!Serial && (millis() - startWait < 2500)) {
    delay(10);
  }
  delay(500);

  Serial.println("\n=======================================================");
  Serial.println("  AI FLASH FLOOD EARLY WARNING - ESP32 SENSOR NODE");
  Serial.println("=======================================================");
#if defined(BOARD_ESP32_CLASSIC)
  Serial.println("  Board Profile: CLASSIC ESP32 (SDA=21, SCL=22)");
#else
  Serial.println("  Board Profile: ESP32-S3 (SDA=8, SCL=9)");
#endif
  Serial.printf("  Backend Server: %s\n", SERVER_URL);
  Serial.println("=======================================================");

  // Set ADC 12-bit resolution and attenuation
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_SOIL_ANALOG, ADC_11db);
  analogSetPinAttenuation(PIN_RAIN_ANALOG, ADC_11db);

  // Initialize GPIOs
  pinMode(PIN_RAIN_DIGITAL, INPUT_PULLUP);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);

  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_LED_GREEN, HIGH);
  digitalWrite(PIN_LED_YELLOW, LOW);
  digitalWrite(PIN_LED_RED, LOW);

  // Initialize DHT11 on GPIO 4
  Serial.printf("[INIT] Initializing DHT11 on GPIO %d...\n", PIN_DHT);
  dht.begin();

  // Recover I2C bus if held low, then start Wire
  Serial.printf("[INIT] Initializing I2C Bus on SDA=GPIO%d, SCL=GPIO%d (100kHz)...\n",
                PIN_MPU_SDA, PIN_MPU_SCL);
  recoverI2CBus(PIN_MPU_SDA, PIN_MPU_SCL);
  Wire.begin(PIN_MPU_SDA, PIN_MPU_SCL);
  Wire.setClock(100000);
  Wire.setTimeOut(250);
  delay(100);

  scanAndInitMPU();
}

void printWiFiDiagnostics() {
  wl_status_t st = WiFi.status();
  switch (st) {
  case WL_NO_SSID_AVAIL:
    Serial.println("[WIFI ERROR] SSID not found! Ensure WiFi network is 2.4 GHz (ESP32 cannot connect to 5 GHz).");
    break;
  case WL_CONNECT_FAILED:
    Serial.println("[WIFI ERROR] Connection failed! Check password.");
    break;
  case WL_CONNECTION_LOST:
    Serial.println("[WIFI ERROR] Connection lost.");
    break;
  case WL_DISCONNECTED:
    Serial.println("[WIFI STATUS] Disconnected / Connecting...");
    break;
  case WL_CONNECTED:
    Serial.printf("[WIFI OK] Connected! IP: %s | Signal (RSSI): %d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
    break;
  default:
    Serial.printf("[WIFI STATUS] Status code: %d\n", st);
    break;
  }
}

void initWiFi() {
  Serial.println("\n-------------------------------------------------------");
  Serial.print("[WIFI] Connecting to SSID: ");
  Serial.println(WIFI_SSID);
  Serial.println("[NOTE] ESP32 requires a 2.4 GHz Wi-Fi band.");

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 25) {
    delay(500);
    Serial.print(".");
    timeout++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[WIFI OK] Successfully connected to '%s'!\n", WIFI_SSID);
    Serial.print("[WIFI OK] ESP32 Local IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[WIFI WARN] Initial Wi-Fi connection timed out.");
    printWiFiDiagnostics();
  }
  Serial.println("-------------------------------------------------------\n");
}

void checkWiFiConnection() {
  if (WiFi.status() == WL_CONNECTED)
    return;

  if (millis() - lastWiFiCheckTime < 10000)
    return;

  lastWiFiCheckTime = millis();
  Serial.print("[WIFI RECONNECT] Attempting reconnection to: ");
  Serial.println(WIFI_SSID);
  printWiFiDiagnostics();
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void triggerBuzzer(bool active) {
  digitalWrite(PIN_BUZZER, active ? HIGH : LOW);
}

void updateAlertActuators(int riskLevel) {
  digitalWrite(PIN_LED_GREEN, LOW);
  digitalWrite(PIN_LED_YELLOW, LOW);
  digitalWrite(PIN_LED_RED, LOW);
  triggerBuzzer(false);

  if (riskLevel == 0) {
    digitalWrite(PIN_LED_GREEN, HIGH);
  } else if (riskLevel == 1) {
    digitalWrite(PIN_LED_YELLOW, HIGH);
  } else if (riskLevel == 2) {
    digitalWrite(PIN_LED_YELLOW, HIGH);
    digitalWrite(PIN_LED_RED, HIGH);
    triggerBuzzer(true);
    delay(80);
    triggerBuzzer(false);
  } else if (riskLevel >= 3) {
    digitalWrite(PIN_LED_RED, HIGH);
    triggerBuzzer(true);
  }
}

// ======================================================================================
// 6. MAIN SETUP & LOOP
// ======================================================================================

void setup() {
  setupHardware();
  initWiFi();
}

void loop() {
  checkWiFiConnection();

  // Background auto-retry for MPU-6050 if unplugged/replugged
  if (!mpuOnline && (millis() - lastMPUScanTime > 5000)) {
    lastMPUScanTime = millis();
    scanAndInitMPU();
  }

  unsigned long currentMillis = millis();
  if (currentMillis - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = currentMillis;

    // -------------------------------------------------------------
    // 1. Read DHT11 Temperature & Humidity (GPIO 4)
    // -------------------------------------------------------------
    float tempC = dht.readTemperature();
    float humidity = dht.readHumidity();

    if (!isnan(tempC) && tempC >= -10.0 && tempC <= 75.0) {
      lastTemp = tempC;
    }
    if (!isnan(humidity) && humidity >= 0.0 && humidity <= 100.0) {
      lastHumidity = humidity;
    }

    // -------------------------------------------------------------
    // 2. Read Capacitive/Resistive Soil Moisture Sensor
    // -------------------------------------------------------------
    int soilRaw = analogRead(PIN_SOIL_ANALOG);
    float soilPct = 0.0;
    if (SOIL_DRY_ADC > SOIL_WET_ADC) {
      soilPct = (float)(SOIL_DRY_ADC - soilRaw) /
                (float)(SOIL_DRY_ADC - SOIL_WET_ADC) * 100.0f;
    }
    soilPct = constrain(soilPct, 0.0f, 100.0f);

    // -------------------------------------------------------------
    // 3. Read HW-038 Raindrop Sensor
    // -------------------------------------------------------------
    int rainRaw = analogRead(PIN_RAIN_ANALOG);
    int rainDigitalTrigger = digitalRead(PIN_RAIN_DIGITAL); // LOW when wet, HIGH when dry

    float rainPct = 0.0;

    // Polarity detection:
    if (rainDigitalTrigger == HIGH && rainRaw < 500) {
      rainPct = 0.0f;
    } else if (rainRaw < 2500 && rainRaw > 50) {
      rainPct = (float)(RAIN_DRY_ADC - rainRaw) /
                (float)(RAIN_DRY_ADC - RAIN_WET_ADC) * 100.0f;
    } else if (rainRaw >= 500 && rainDigitalTrigger == LOW) {
      rainPct = (float)rainRaw / 3500.0f * 100.0f;
    }

    if (rainDigitalTrigger == LOW) {
      rainPct = max(rainPct, 40.0f);
    }
    rainPct = constrain(rainPct, 0.0f, 100.0f);

    // Convert percentage to mm water level / rain intensity based on configured scale
    float rainRateMmH = (rainPct / 100.0f) * WATER_LEVEL_MAX_SCALE_MM; 
    float deltaAccumMm = (rainRateMmH / 3600.0f) * (TELEMETRY_INTERVAL_MS / 1000.0f);
    cumulativeRainfallProxy += deltaAccumMm;

    // -------------------------------------------------------------
    // 4. Read MPU-6050 Motion, Tilt & Vibration
    // -------------------------------------------------------------
    float pitchDeg = 0.0;
    float vibrationMag = 0.0;
    bool mpuReadSuccess = readMPURaw(pitchDeg, vibrationMag);

    // -------------------------------------------------------------
    // 5. Print Live Sensor Values to Serial Monitor
    // -------------------------------------------------------------
    Serial.println("\n-------------------------------------------------------");
    Serial.printf("[ESP32 SENSORS LIVE REPORT]\n");
    Serial.printf("  🌡️ Temp / Humidity : %.1f°C | %.0f%%\n", lastTemp, lastHumidity);
    Serial.printf("  🌱 Soil Moisture   : %.1f%%  [Raw ADC: %d]\n", soilPct, soilRaw);
    Serial.printf("  🌧️ Rain (HW-038)   : %.1f mm/h (%.1f%%)  [Raw ADC: %d | Digital: %s]\n",
                  rainRateMmH, rainPct, rainRaw,
                  rainDigitalTrigger == LOW ? "TRIGGERED (WET)" : "DRY");
    Serial.printf("  📐 Terrain Tilt    : %.1f°  [MPU 0x%02X: %s]\n", pitchDeg,
                  mpuActiveAddress,
                  mpuReadSuccess ? "ONLINE ✅" : "OFFLINE / RETRYING ❌");
    Serial.printf("  ⚡ Seismic Vibe    : %.2fg\n", vibrationMag);
    Serial.println("-------------------------------------------------------");

    // -------------------------------------------------------------
    // 6. Serialize to JSON Payload
    // -------------------------------------------------------------
#if ARDUINOJSON_VERSION_MAJOR >= 7
    JsonDocument doc;
#else
    StaticJsonDocument<512> doc;
#endif

    doc["node_id"] = NODE_ID;
    doc["temperature_c"] = round(lastTemp * 10.0) / 10.0;
    doc["humidity_pct"] = round(lastHumidity * 10.0) / 10.0;
    doc["soil_moisture_pct"] = round(soilPct * 10.0) / 10.0;
    doc["soil_moisture_raw"] = soilRaw;
    doc["rain_intensity_pct"] = round(rainPct * 10.0) / 10.0;
    doc["rain_rate_mm_h"] = round(rainRateMmH * 10.0) / 10.0;
    doc["rain_accum_mm"] = round(cumulativeRainfallProxy * 10.0) / 10.0;
    doc["slope_degrees"] = round(pitchDeg * 10.0) / 10.0;
    doc["vibration_g"] = round(vibrationMag * 100.0) / 100.0;

    String jsonPayload;
    serializeJson(doc, jsonPayload);

    // -------------------------------------------------------------
    // 7. Send Telemetry to Python AI Server via HTTP/HTTPS POST
    // -------------------------------------------------------------
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      String sUrl = String(SERVER_URL);
      WiFiClient client;
      WiFiClientSecure secClient;

      if (sUrl.startsWith("https://")) {
        secClient.setInsecure(); // Support cloud / ngrok / tunnel HTTPS certificates without root cert bundle
        http.begin(secClient, SERVER_URL);
      } else {
        http.begin(client, SERVER_URL);
      }

      http.addHeader("Content-Type", "application/json");
      http.setTimeout(4000);

      int httpResponseCode = http.POST(jsonPayload);

      if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.printf("[SERVER RESPONSE %d] %s\n", httpResponseCode, response.c_str());

#if ARDUINOJSON_VERSION_MAJOR >= 7
        JsonDocument respDoc;
#else
        StaticJsonDocument<512> respDoc;
#endif
        DeserializationError err = deserializeJson(respDoc, response);
        if (!err) {
          int predictedRisk = respDoc["predicted_risk_level"] | 0;
          const char *riskCat = respDoc["risk_category"] | "Low Risk";
          float conf = respDoc["confidence_score"] | 95.0f;

          Serial.printf(">> AI PREDICTION: %s (Level %d) | Confidence: %.1f%%\n", riskCat, predictedRisk, conf);
          updateAlertActuators(predictedRisk);
        }
      } else {
        Serial.printf("[HTTP ERROR %d] Telemetry POST to '%s' failed!\n", httpResponseCode, SERVER_URL);
        int localRisk = evaluateEdgeRiskLocal(rainRateMmH, cumulativeRainfallProxy, soilPct, pitchDeg, vibrationMag);
        Serial.printf(">> [EDGE FALLBACK] Local Risk Assessment: Level %d\n", localRisk);
        updateAlertActuators(localRisk);
      }
      http.end();
    } else {
      Serial.println("[OFFLINE] WiFi Disconnected. Using Local Edge Fallback...");
      int localRisk = evaluateEdgeRiskLocal(rainRateMmH, cumulativeRainfallProxy, soilPct, pitchDeg, vibrationMag);
      updateAlertActuators(localRisk);
    }
  }

  delay(20);
}
