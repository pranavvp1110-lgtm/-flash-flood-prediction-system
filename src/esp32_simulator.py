"""
ESP32 Multi-Sensor Hardware Telemetry Stream Simulator.
Simulates realistic sensor readings from DHT11, MPU6050, Soil Moisture, and HW-038 Raindrop sensor,
streaming them to the IoT prediction server to demonstrate real-time AI early warning triggers.
"""

import sys
import time
import requests
import argparse
import random
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_SERVER_URL = "http://127.0.0.1:5000/api/telemetry"


def simulate_esp32_stream(
    server_url=DEFAULT_SERVER_URL,
    node_id="ESP32_FLOOD_NODE_01",
    scenario="monsoon_surge",
    duration_seconds=60,
    interval_seconds=2,
):
    print(f"\n=======================================================")
    print(f"  STARTING ESP32 SENSOR TELEMETRY STREAM")
    print(f"=======================================================")
    print(f"  Target Server : {server_url}")
    print(f"  Node ID       : {node_id}")
    print(f"  Scenario      : {scenario}")
    print(f"  Interval      : {interval_seconds}s (Duration: {duration_seconds}s)\n")

    start_time = time.time()
    packet_count = 0

    accumulated_rain = 5.0
    soil_saturation = 0.55

    while time.time() - start_time < duration_seconds:
        elapsed = time.time() - start_time
        progress = elapsed / duration_seconds

        if scenario == "monsoon_surge":
            # Ramp up rain and soil saturation gradually to simulate sudden storm surge
            rain_rate = 5.0 + progress * 75.0 + random.uniform(-3, 3)
            accumulated_rain += (rain_rate / 3600.0) * interval_seconds * 15.0 # Accelerated time
            soil_saturation = min(0.98, 0.50 + progress * 0.45 + random.uniform(-0.01, 0.01))
            temp_c = 25.0 - progress * 4.0 + random.uniform(-0.3, 0.3)
            humidity = min(99.0, 78.0 + progress * 20.0 + random.uniform(-1, 1))
            slope_deg = 15.0 + random.uniform(-0.5, 0.5)

        elif scenario == "cloudburst_disaster":
            # Violent sudden downpour with severe ground vibration and soil saturation spike
            rain_rate = 85.0 + random.uniform(-5, 15)
            accumulated_rain += 8.0 + random.uniform(1, 4)
            soil_saturation = min(0.99, 0.88 + random.uniform(0.01, 0.08))
            temp_c = 19.5 + random.uniform(-0.5, 0.5)
            humidity = 98.0 + random.uniform(-1, 1)
            slope_deg = 32.0 + random.uniform(-1.0, 1.0) # Mountain slope

        else: # "normal_dry"
            rain_rate = random.uniform(0.0, 2.0)
            accumulated_rain = 5.0 + random.uniform(0, 2)
            soil_saturation = random.uniform(0.30, 0.45)
            temp_c = random.uniform(22.0, 27.0)
            humidity = random.uniform(55.0, 70.0)
            slope_deg = 12.0 + random.uniform(-0.2, 0.2)

        payload = {
            "node_id": node_id,
            "temperature_c": round(temp_c, 1),
            "humidity_pct": round(humidity, 1),
            "soil_moisture_pct": round(soil_saturation * 100.0, 1),
            "rain_rate_mm_h": round(max(0.0, rain_rate), 1),
            "rain_accum_mm": round(accumulated_rain, 1),
            "slope_degrees": round(slope_deg, 1),
            "vibration_g": round(random.uniform(0.01, 0.08), 2),
        }

        try:
            r = requests.post(server_url, json=payload, timeout=3)
            packet_count += 1
            if r.status_code == 200:
                resp = r.json()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Packet #{packet_count} | Rain: {payload['rain_accum_mm']}mm | Soil: {payload['soil_moisture_pct']}% | Slope: {payload['slope_degrees']}° -> AI Risk: {resp.get('risk_category')} (Level {resp.get('predicted_risk_level')}) [Confidence: {resp.get('confidence_score')}%]")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Server returned HTTP {r.status_code}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection failed: {e}")

        time.sleep(interval_seconds)

    print(f"\nSimulation complete. Sent {packet_count} telemetry packets.\n")


def main():
    parser = argparse.ArgumentParser(description="ESP32 Sensor Telemetry Stream Simulator")
    parser.add_argument("--url", type=str, default=DEFAULT_SERVER_URL, help="IoT Server endpoint")
    parser.add_argument("--node_id", type=str, default="ESP32_FLOOD_NODE_01", help="Target node ID")
    parser.add_argument("--scenario", type=str, default="monsoon_surge", choices=["monsoon_surge", "cloudburst_disaster", "normal_dry"], help="Simulation scenario")
    parser.add_argument("--duration", type=int, default=30, help="Simulation duration in seconds")
    parser.add_argument("--interval", type=int, default=2, help="Packet interval in seconds")

    args = parser.parse_args()
    simulate_esp32_stream(
        server_url=args.url,
        node_id=args.node_id,
        scenario=args.scenario,
        duration_seconds=args.duration,
        interval_seconds=args.interval,
    )


if __name__ == "__main__":
    main()
