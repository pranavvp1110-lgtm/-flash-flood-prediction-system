# ESP32 / ESP32-S3 Flash Flood & Landslide IoT Node: Wiring & Setup Guide

This guide details how to wire every physical sensor, configure network and cloud connectivity (local Wi-Fi IP and public tunnel), and access persistent sensor telemetry storage.

---

## 📌 1. Pin Mapping & Connections

### Option A: ESP32-S3 (ESP32-S3-DevKitC-1, ESP32-S3 WROOM)
*(In Arduino sketch: `#define BOARD_ESP32_S3`)*

| Sensor / Module | Sensor Pin | ESP32-S3 Pin Label | Notes |
| :--- | :--- | :--- | :--- |
| **MPU-6050 (GY-521)** | **VCC**<br>**GND**<br>**SDA**<br>**SCL**<br>**AD0** | **5V / VIN** (or 3.3V)<br>**GND**<br>**IO8** (GPIO 8)<br>**IO9** (GPIO 9)<br>**GND** | ⚡ **Critical:** Connect **AD0 to GND** to fix I2C address at `0x68`. Power with **5V/VIN** if module has onboard 662K 3.3V regulator. |
| **DHT11** (Temp & Humidity) | **VCC**<br>**GND**<br>**DATA / OUT** | **3.3V**<br>**GND**<br>**IO4** (GPIO 4) | Single-bus data pin |
| **Capacitive Soil Moisture** | **VCC**<br>**GND**<br>**AOUT** | **3.3V**<br>**GND**<br>**IO1** (GPIO 1 - ADC1) | ADC1 is WiFi-safe |
| **HW-038 Raindrop Sensor** | **VCC**<br>**GND**<br>**A0 / AOUT**<br>**D0 / DOUT** | **3.3V**<br>**GND**<br>**IO2** (GPIO 2 - ADC1)<br>**IO3** (GPIO 3) | A0: Analog Rain Level<br>D0: Digital Rain Threshold |
| **Piezo Buzzer** | **(+) Positive**<br>**(-) Negative** | **IO5** (GPIO 5)<br>**GND** | Early warning siren |
| **Status LEDs** | **Green (+)**<br>**Yellow (+)**<br>**Red (+)**<br>**GND (-)** | **IO6** (GPIO 6)<br>**IO7** (GPIO 7)<br>**IO10** (GPIO 10)<br>**GND** | In series with 220Ω - 330Ω resistors |

---

### Option B: Classic ESP32 (ESP32 DevKit V1 / ESP-WROOM-32 / 30 & 38 Pin)
*(In Arduino sketch: `#define BOARD_ESP32_CLASSIC`)*

| Sensor / Module | Sensor Pin | ESP32 Pin Label | Notes |
| :--- | :--- | :--- | :--- |
| **MPU-6050 (GY-521)** | **VCC**<br>**GND**<br>**SDA**<br>**SCL**<br>**AD0** | **VIN / 5V** (or 3.3V)<br>**GND**<br>**GPIO 21** (SDA)<br>**GPIO 22** (SCL)<br>**GND** | Standard ESP32 hardware I2C pins. **AD0 -> GND**. |
| **DHT11** (Temp & Humidity) | **VCC**<br>**GND**<br>**DATA / OUT** | **3.3V**<br>**GND**<br>**GPIO 4** | Single-bus data pin |
| **Capacitive Soil Moisture** | **VCC**<br>**GND**<br>**AOUT** | **3.3V**<br>**GND**<br>**GPIO 34** (ADC1_CH6) | ADC1 is WiFi-safe |
| **HW-038 Raindrop Sensor** | **VCC**<br>**GND**<br>**A0 / AOUT**<br>**D0 / DOUT** | **3.3V**<br>**GND**<br>**GPIO 35** (ADC1_CH7)<br>**GPIO 18** | A0: Analog Rain<br>D0: Digital Rain |
| **Piezo Buzzer** | **(+) Positive**<br>**(-) Negative** | **GPIO 5**<br>**GND** | Early warning siren |
| **Status LEDs** | **Green (+)**<br>**Yellow (+)**<br>**Red (+)** | **GPIO 19**<br>**GPIO 23**<br>**GPIO 25** | In series with 220Ω resistors |

---

## 🌐 2. Connecting ESP32 to Backend (Not Localhost)

ESP32 microcontrollers cannot connect to `localhost` because `localhost` refers to the ESP32 itself. Use either your **Local Wi-Fi IP** or a **Public Cloud Tunnel**.

### Method 1: Local Wi-Fi / Hotspot IP (Recommended for Same Network)
1. Start the server on your computer:
   ```bash
   python src/iot_server.py
   ```
2. The server automatically detects and displays your active local network IP address:
   ```
   📡 ESP32 WI-FI INGESTION ENDPOINTS:
       • http://192.168.1.15:5000/api/telemetry
   ```
3. In [`esp32_flood_monitor.ino`](file:///c:/Users/PRANAV/OneDrive/Desktop/flood_risk_level/firmware/esp32_flood_monitor/esp32_flood_monitor.ino#L54), update:
   ```cpp
   const char *WIFI_SSID = "Your_WiFi_SSID";
   const char *WIFI_PASSWORD = "Your_WiFi_Password";
   const char *SERVER_URL = "http://192.168.1.15:5000/api/telemetry";
   ```

### Method 2: Public Cloud / Global Internet Access (Ngrok Tunnel)
To send telemetry from ANY Wi-Fi network or cellular hotspot anywhere in the world:
1. Start the server with the `--tunnel` flag:
   ```bash
   python src/iot_server.py --tunnel
   ```
2. Copy the generated public HTTPS URL (e.g., `https://abcdef.ngrok-free.app/api/telemetry`).
3. Set `SERVER_URL` in [`esp32_flood_monitor.ino`](file:///c:/Users/PRANAV/OneDrive/Desktop/flood_risk_level/firmware/esp32_flood_monitor/esp32_flood_monitor.ino#L54):
   ```cpp
   const char *SERVER_URL = "https://abcdef.ngrok-free.app/api/telemetry";
   ```

---

## 📦 3. Real-Time Sensor Telemetry Persistent Storage

All incoming real-time sensor packets from the ESP32 and AI predictions are permanently recorded in two formats:

1. **SQLite Database (`data/sensor_readings.db`)**:
   - Indexed time-series table storing all sensor values (temperature, humidity, soil moisture %, raw ADC, rain rate mm/h, cumulative rain depth mm, slope pitch angle, vibration g, predicted risk level 0-3, risk category, confidence %, and action advisories).
2. **Time-Series CSV Log (`data/sensor_readings.csv`)**:
   - Continuous append-only CSV file viewable directly in Microsoft Excel, Google Sheets, or Pandas.

### REST APIs for Querying and Exporting Data:
- `GET http://<SERVER_IP>:5000/api/latest` : Current real-time state and sensor readings.
- `GET http://<SERVER_IP>:5000/api/records?limit=100` : Query stored historical records.
- `GET http://<SERVER_IP>:5000/api/stats` : Aggregate statistics (total stored records, sensor averages, risk counts).
- `GET http://<SERVER_IP>:5000/api/export/csv` : One-click download of the complete sensor history CSV file.

---

## 🔍 4. Troubleshooting MPU-6050 Sensor

1. **AD0 Pin is Floating (Most Common Cause)**:
   - Connect **AD0 directly to GND** (sets fixed address `0x68`).
2. **Voltage Drop on GY-521 Board**:
   - Connect MPU-6050 **VCC to the 5V / VIN pin** of the ESP32.
3. **Wrong I2C Pins**:
   - On **ESP32-S3**: SDA -> `IO8`, SCL -> `IO9`.
   - On **Classic ESP32 (WROOM-32)**: SDA -> `GPIO 21`, SCL -> `GPIO 22`.
4. **I2C Bus Lockup**:
   - The firmware includes automatic I2C clock recovery (`recoverI2CBus()`) on startup.
