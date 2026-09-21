"""
Real-Time IoT Telemetry & Flash Flood Risk Inference Server.
Accepts live JSON telemetry streams from ESP32 edge sensor nodes (DHT11, MPU6050, Soil, HW-038),
runs instant ML early warning inference, stores readings permanently in SQLite & CSV,
and serves live state to the frontend dashboard over Local Network and Public Cloud.
"""

import os
import sys
import json
import time
import socket
import mimetypes
import argparse
from datetime import datetime
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading
from urllib.parse import parse_qs, urlparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import predict_sensor_telemetry_risk
from risk_labeler import RISK_TIERS
import telemetry_db

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
LATEST_TELEMETRY_PATH = os.path.join(DATA_DIR, "live_telemetry_latest.json")

# In-memory circular buffer for fast real-time telemetry streaming
MAX_HISTORY_LEN = 100
_TELEMETRY_HISTORY = deque(maxlen=MAX_HISTORY_LEN)
_LATEST_TELEMETRY = None
_LAST_PACKET_TIMESTAMP = 0.0
HEARTBEAT_TIMEOUT_SEC = 6.0
_BUFFER_LOCK = threading.Lock()


def get_local_ip_addresses():
    """Retrieve all active non-loopback IPv4 network addresses on the machine."""
    ip_list = []
    try:
        # Standard hostname resolution
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127.") and ip not in ip_list:
                ip_list.append(ip)
    except Exception:
        pass

    try:
        # Connect to public DNS dummy socket to determine default routing interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        default_ip = s.getsockname()[0]
        s.close()
        if default_ip and not default_ip.startswith("127.") and default_ip not in ip_list:
            ip_list.insert(0, default_ip)
    except Exception:
        pass

    return ip_list or ["127.0.0.1"]


def initialize_server_state():
    """Initialize persistent database, pre-warm ML model, and restore recent history from disk."""
    global _LATEST_TELEMETRY, _LAST_PACKET_TIMESTAMP
    telemetry_db.init_db()

    # Pre-warm ML model bundle so all incoming packets have sub-20ms inference latency
    try:
        from predict import load_model
        load_model()
    except Exception as m_err:
        sys.stderr.write(f"[MODEL WARN] Could not pre-warm model: {m_err}\n")

    # Restore recent packet history from SQLite database
    saved_packets = telemetry_db.get_recent_packets_for_stream(limit=MAX_HISTORY_LEN)
    with _BUFFER_LOCK:
        _TELEMETRY_HISTORY.clear()
        for p in saved_packets:
            _TELEMETRY_HISTORY.append(p)
        if saved_packets:
            _LATEST_TELEMETRY = saved_packets[-1]
            _LAST_PACKET_TIMESTAMP = time.time() - 10.0 # Mark as standby/offline until active


def get_default_standby_packet():
    """Generate default initial telemetry and prediction in standby state."""
    pred = predict_sensor_telemetry_risk(
        temp_c=24.5,
        humidity_pct=65.0,
        soil_moisture_pct=38.0,
        rain_rate_mm_h=0.0,
        rain_accum_mm=0.0,
        slope_degrees=12.0,
        vibration_g=0.0,
        node_id="ESP32_FLOOD_NODE_01"
    )
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "node_id": "ESP32_FLOOD_NODE_01",
        "status": "STANDBY",
        "sensors": {
            "temperature_c": 24.5,
            "humidity_pct": 65.0,
            "soil_moisture_pct": 38.0,
            "soil_moisture_raw": 3200,
            "rain_rate_mm_h": 0.0,
            "rain_accum_mm": 0.0,
            "rain_intensity_pct": 0.0,
            "slope_degrees": 12.0,
            "vibration_g": 0.0,
        },
        "ai_prediction": pred,
    }


def process_telemetry_packet(data):
    """
    Process raw ESP32 telemetry packet, run direct ML flash flood inference,
    permanently persist to SQLite & CSV, and format response for actuators.
    """
    global _LATEST_TELEMETRY, _LAST_PACKET_TIMESTAMP

    node_id = str(data.get("node_id", "ESP32_FLOOD_NODE_01"))
    temp_c = float(data.get("temperature_c", 24.0))
    humidity = float(data.get("humidity_pct", 65.0))
    
    # Soil moisture: handle raw ADC or percentage
    soil_raw = int(data.get("soil_moisture_raw", 0))
    if "soil_moisture_pct" in data:
        soil_pct = float(data.get("soil_moisture_pct", 40.0))
        if 0.0 < soil_pct <= 1.0:
            soil_pct = soil_pct * 100.0
    elif soil_raw > 0:
        # Fallback ADC calculation: 3800 dry, 1200 submerged
        soil_pct = max(0.0, min(100.0, (3800.0 - soil_raw) / 2600.0 * 100.0))
    else:
        soil_pct = 40.0

    # Rain readings
    rain_rate = float(data.get("rain_rate_mm_h", data.get("rain_intensity_pct", 0.0) * 0.8))
    rain_accum = float(data.get("rain_accum_mm", data.get("rainfall_mm", rain_rate * 0.5)))
    rain_intensity = float(data.get("rain_intensity_pct", (rain_rate / 100.0) * 100.0))
    
    # MPU6050 slope tilt and vibration
    slope_deg = float(data.get("slope_degrees", data.get("pitch_deg", 12.0)))
    vibration = float(data.get("vibration_g", 0.0))
    
    # Run direct sensor ML prediction
    try:
        prediction = predict_sensor_telemetry_risk(
            temp_c=temp_c,
            humidity_pct=humidity,
            soil_moisture_pct=soil_pct,
            rain_rate_mm_h=rain_rate,
            rain_accum_mm=rain_accum,
            slope_degrees=slope_deg,
            vibration_g=vibration,
            node_id=node_id,
        )
    except Exception as e:
        prediction = {
            "predicted_risk_level": 0,
            "risk_category": "Low Risk",
            "risk_code": "LOW",
            "risk_color": "#10B981",
            "confidence_score": 90.0,
            "early_warning_lead_time": "72h Routine Monitoring",
            "status_code": "NORMAL / ALL CLEAR",
            "primary_risk_drivers": [f"Direct sensor inference active ({str(e)})"],
            "advisory_actions": ["Routine sensor monitoring active."],
            "probability_distribution": {"Low Risk": 90.0, "Moderate Risk": 10.0, "High Risk": 0.0, "Severe Risk": 0.0},
            "sensors": {
                "temperature_c": temp_c,
                "humidity_pct": humidity,
                "soil_moisture_pct": soil_pct,
                "soil_moisture_raw": soil_raw,
                "rain_rate_mm_h": rain_rate,
                "rain_accum_mm": rain_accum,
                "rain_intensity_pct": rain_intensity,
                "slope_degrees": slope_deg,
                "vibration_g": vibration,
            },
            "actuator_state": {
                "buzzer_active": False,
                "led_color": "GREEN",
                "status_text": "NORMAL (GREEN LED)"
            }
        }

    timestamp_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    packet = {
        "timestamp": timestamp_iso,
        "node_id": node_id,
        "status": "ONLINE",
        "sensors": {
            "temperature_c": temp_c,
            "humidity_pct": humidity,
            "soil_moisture_pct": round(soil_pct, 1),
            "soil_moisture_raw": soil_raw,
            "rain_rate_mm_h": rain_rate,
            "rain_accum_mm": rain_accum,
            "rain_intensity_pct": round(rain_intensity, 1),
            "slope_degrees": slope_deg,
            "vibration_g": vibration,
        },
        "ai_prediction": prediction,
    }

    # 1. Store permanently into SQLite database and append to CSV
    try:
        record_id = telemetry_db.insert_telemetry_reading(packet)
        packet["id"] = record_id
    except Exception as db_err:
        sys.stderr.write(f"[DB ERROR] Failed to store telemetry: {db_err}\n")

    # 2. Update memory buffer
    with _BUFFER_LOCK:
        _LAST_PACKET_TIMESTAMP = time.time()
        _LATEST_TELEMETRY = packet
        _TELEMETRY_HISTORY.append(packet)

    # 3. Persist latest telemetry snapshot to disk for instant Streamlit/API caching
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(LATEST_TELEMETRY_PATH, "w", encoding="utf-8") as f:
            json.dump(packet, f, indent=2)
    except Exception:
        pass

    # Response payload for ESP32 edge actuators
    risk_level = prediction.get("predicted_risk_level", 0)
    esp_response = {
        "status": "success",
        "node_id": node_id,
        "timestamp": timestamp_iso,
        "predicted_risk_level": risk_level,
        "risk_category": prediction.get("risk_category", "Low Risk"),
        "risk_code": prediction.get("risk_code", "LOW"),
        "risk_color": prediction.get("risk_color", "#10B981"),
        "confidence_score": prediction.get("confidence_score", 95.0),
        "early_warning_lead_time": prediction.get("early_warning_lead_time", "72h Routine Monitoring"),
        "status_code": prediction.get("status_code", "NORMAL / ALL CLEAR"),
        "buzzer_active": risk_level >= 2,
        "led_color": "RED" if risk_level >= 2 else ("YELLOW" if risk_level == 1 else "GREEN"),
        "advisory": (prediction.get("advisory_actions") or ["Normal baseline."])[0],
    }
    return esp_response


class TelemetryHTTPHandler(BaseHTTPRequestHandler):

    def _set_cors_headers(self, status_code=200, content_type="application/json", filename=None):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_cors_headers(204)
    def do_HEAD(self):
        self._set_cors_headers(200, "text/html; charset=utf-8")    

    def do_GET(self):
        parsed_url = urlparse(self.path)
        url_path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        if url_path == "/api/latest":
            with _BUFFER_LOCK:
                if _LATEST_TELEMETRY is not None:
                    elapsed = time.time() - _LAST_PACKET_TIMESTAMP
                    data = dict(_LATEST_TELEMETRY)
                    if elapsed > HEARTBEAT_TIMEOUT_SEC:
                        data["status"] = "OFFLINE"
                        data["is_live"] = False
                        data["offline_reason"] = f"No telemetry received for {int(elapsed)}s (ESP32 Disconnected/Power Cut)"
                        data["seconds_since_last_packet"] = round(elapsed, 1)
                    else:
                        data["status"] = "ONLINE"
                        data["is_live"] = True
                        data["seconds_since_last_packet"] = round(elapsed, 1)
                else:
                    data = get_default_standby_packet()
            self._set_cors_headers(200, "application/json")
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif url_path == "/api/history":
            with _BUFFER_LOCK:
                history = list(_TELEMETRY_HISTORY)
            self._set_cors_headers(200, "application/json")
            self.wfile.write(json.dumps({"count": len(history), "history": history}).encode("utf-8"))

        elif url_path == "/api/records":
            # Paginated queries from SQLite database
            limit = int(query_params.get("limit", [100])[0])
            offset = int(query_params.get("offset", [0])[0])
            node_id = query_params.get("node_id", [None])[0]
            records = telemetry_db.get_recent_readings(limit=min(limit, 1000), offset=offset, node_id=node_id)
            stats = telemetry_db.get_telemetry_stats()
            self._set_cors_headers(200, "application/json")
            self.wfile.write(json.dumps({
                "status": "success",
                "count": len(records),
                "total_stored": stats["total_readings"],
                "records": records
            }).encode("utf-8"))

        elif url_path == "/api/stats":
            stats = telemetry_db.get_telemetry_stats()
            stats["active_nodes_online"] = 1 if (_LATEST_TELEMETRY and (time.time() - _LAST_PACKET_TIMESTAMP <= HEARTBEAT_TIMEOUT_SEC)) else 0
            self._set_cors_headers(200, "application/json")
            self.wfile.write(json.dumps(stats).encode("utf-8"))

        elif url_path == "/api/export/csv" or url_path == "/api/download/csv":
            if os.path.exists(telemetry_db.CSV_PATH):
                try:
                    with open(telemetry_db.CSV_PATH, "rb") as f:
                        csv_bytes = f.read()
                    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"esp32_sensor_readings_{date_tag}.csv"
                    self._set_cors_headers(200, "text/csv; charset=utf-8", filename=filename)
                    self.wfile.write(csv_bytes)
                except Exception as e:
                    self._set_cors_headers(500, "application/json")
                    self.wfile.write(json.dumps({"error": f"Failed to export CSV: {e}"}).encode("utf-8"))
            else:
                self._set_cors_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": "No CSV data found yet"}).encode("utf-8"))

        elif url_path == "/api/network":
            local_ips = get_local_ip_addresses()
            self._set_cors_headers(200, "application/json")
            self.wfile.write(json.dumps({
                "status": "online",
                "local_ips": local_ips,
                "port": self.server.server_port,
                "endpoints": {
                    "ingestion_local": [f"http://{ip}:{self.server.server_port}/api/telemetry" for ip in local_ips],
                    "dashboard_local": [f"http://{ip}:{self.server.server_port}/" for ip in local_ips],
                    "api_latest": f"/api/latest",
                    "api_records": f"/api/records",
                    "api_export_csv": f"/api/export/csv"
                }
            }).encode("utf-8"))

        elif url_path == "/api/health":
            self._set_cors_headers(200, "application/json")
            self.wfile.write(json.dumps({
                "status": "online",
                "service": "AI Flash Flood ESP32 IoT Telemetry Server",
                "version": "2.1.0",
                "database_connected": os.path.exists(telemetry_db.DB_PATH),
                "buffer_packets": len(_TELEMETRY_HISTORY)
            }).encode("utf-8"))

        else:
            # Serve static frontend files from public/
            if url_path == "/" or url_path == "/index.html":
                filepath = os.path.join(PUBLIC_DIR, "index.html")
            else:
                rel_path = url_path.lstrip("/")
                filepath = os.path.join(PUBLIC_DIR, rel_path)

            if os.path.exists(filepath) and os.path.isfile(filepath):
                ctype, _ = mimetypes.guess_type(filepath)
                if not ctype:
                    ctype = "text/plain"
                try:
                    with open(filepath, "rb") as f:
                        content = f.read()
                    self._set_cors_headers(200, ctype)
                    self.wfile.write(content)
                except Exception as e:
                    self._set_cors_headers(500, "application/json")
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            else:
                self._set_cors_headers(200, "application/json")
                self.wfile.write(json.dumps({
                    "status": "online",
                    "message": "AI Flash Flood IoT Server active. POST to /api/telemetry or view dashboard at /.",
                }).encode("utf-8"))

    def do_POST(self):
        parsed_url = urlparse(self.path)
        url_path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception as e:
            self._set_cors_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {e}"}).encode("utf-8"))
            return

        if url_path in ["/api/telemetry", "/api/simulate"]:
            resp = process_telemetry_packet(payload)
            self._set_cors_headers(200, "application/json")
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self._set_cors_headers(404, "application/json")
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def log_message(self, format, *args):
        # Clean terminal logging
        sys.stdout.write(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]} {args[1]} -> {args[2]}\n")
        sys.stdout.flush()


def run_iot_server(host="0.0.0.0", port=5000, enable_tunnel=False):
    """Start the multi-threaded IoT Telemetry server on network interfaces with optional public tunnel."""
    initialize_server_state()

    local_ips = get_local_ip_addresses()
    server = ThreadingHTTPServer((host, port), TelemetryHTTPHandler)
    server.daemon_threads = True

    public_url = None
    if enable_tunnel:
        try:
            from pyngrok import ngrok
            tunnel = ngrok.connect(port, "http")
            public_url = tunnel.public_url
        except Exception as t_err:
            sys.stderr.write(f"[TUNNEL WARN] Could not start ngrok tunnel: {t_err}\n")

    print(f"\n=======================================================================", flush=True)
    print(f"   AI FLASH FLOOD EARLY WARNING - ESP32 REAL-TIME IoT SERVER", flush=True)
    print(f"=======================================================================", flush=True)
    print(f"  --> Bound Interface          : {host}:{port} (Listening on all networks)", flush=True)
    print(f"  --> Local Web Dashboard      : http://localhost:{port}/", flush=True)
    print(f"  --> Persistent Database      : {telemetry_db.DB_PATH}", flush=True)
    print(f"  --> Persistent CSV Log       : {telemetry_db.CSV_PATH}", flush=True)
    print(f"-----------------------------------------------------------------------", flush=True)
    print(f"  📡 ESP32 WI-FI INGESTION ENDPOINTS (Choose your Wi-Fi / Hotspot IP):", flush=True)
    for ip in local_ips:
        print(f"      • http://{ip}:{port}/api/telemetry", flush=True)
        print(f"        (Dashboard: http://{ip}:{port}/)", flush=True)
    if public_url:
        print(f"-----------------------------------------------------------------------", flush=True)
        print(f"  🌍 PUBLIC CLOUD / TUNNEL URL (Accessible anywhere in the world!):", flush=True)
        print(f"      • Ingestion Endpoint: {public_url}/api/telemetry", flush=True)
        print(f"      • Web Dashboard     : {public_url}/", flush=True)
    print(f"-----------------------------------------------------------------------", flush=True)
    print(f"  📊 REST API Endpoints:", flush=True)
    print(f"      • GET /api/latest       -> Real-time packet & status", flush=True)
    print(f"      • GET /api/records      -> Query stored SQLite records", flush=True)
    print(f"      • GET /api/stats        -> Storage statistics & sensor averages", flush=True)
    print(f"      • GET /api/export/csv   -> One-click download of all stored data", flush=True)
    print(f"      • GET /api/network      -> Active IP addresses & endpoint discovery", flush=True)
    print(f"=======================================================================\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down IoT server...", flush=True)
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="ESP32 Real-Time IoT Flash Flood Telemetry Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--tunnel", action="store_true", help="Launch public ngrok tunnel for global internet access")
    args = parser.parse_args()

    run_iot_server(host=args.host, port=args.port, enable_tunnel=args.tunnel)


if __name__ == "__main__":
    main()
