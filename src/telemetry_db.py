"""
Persistent Real-Time Sensor Telemetry Storage Engine.
Stores real-time ESP32 edge telemetry and AI inference results in SQLite
and an append-only time-series CSV log with query, export, and statistics APIs.
"""

import os
import csv
import json
import sqlite3
import threading
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "sensor_readings.db")
CSV_PATH = os.path.join(DATA_DIR, "sensor_readings.csv")

_DB_LOCK = threading.Lock()

CSV_HEADERS = [
    "id",
    "timestamp",
    "node_id",
    "status",
    "temperature_c",
    "humidity_pct",
    "soil_moisture_pct",
    "soil_moisture_raw",
    "rain_rate_mm_h",
    "rain_accum_mm",
    "rain_intensity_pct",
    "slope_degrees",
    "vibration_g",
    "predicted_risk_level",
    "risk_category",
    "risk_code",
    "risk_color",
    "confidence_score",
    "status_code",
    "early_warning_lead_time",
    "buzzer_active",
    "led_color",
    "primary_risk_drivers",
    "advisory_actions"
]


def get_db_connection(db_path=DB_PATH):
    """Create and return a SQLite database connection with row factory."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH, csv_path=CSV_PATH):
    """Initialize the SQLite database schema and CSV log file if they do not exist."""
    with _DB_LOCK:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = get_db_connection(db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sensor_readings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        status TEXT DEFAULT 'ONLINE',
                        temperature_c REAL,
                        humidity_pct REAL,
                        soil_moisture_pct REAL,
                        soil_moisture_raw INTEGER,
                        rain_rate_mm_h REAL,
                        rain_accum_mm REAL,
                        rain_intensity_pct REAL,
                        slope_degrees REAL,
                        vibration_g REAL,
                        predicted_risk_level INTEGER,
                        risk_category TEXT,
                        risk_code TEXT,
                        risk_color TEXT,
                        confidence_score REAL,
                        status_code TEXT,
                        early_warning_lead_time TEXT,
                        buzzer_active INTEGER,
                        led_color TEXT,
                        primary_risk_drivers TEXT,
                        advisory_actions TEXT,
                        raw_payload TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON sensor_readings(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_node_id ON sensor_readings(node_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_level ON sensor_readings(predicted_risk_level)")
        finally:
            conn.close()

        # Initialize CSV with headers if not present or empty
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)


def insert_telemetry_reading(packet, db_path=DB_PATH, csv_path=CSV_PATH):
    """
    Store a validated telemetry packet and AI risk prediction into SQLite and append to CSV.
    Returns the newly generated record ID.
    """
    init_db(db_path, csv_path)

    timestamp = packet.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    node_id = packet.get("node_id", "ESP32_FLOOD_NODE_01")
    status = packet.get("status", "ONLINE")

    sensors = packet.get("sensors", {})
    temp_c = float(sensors.get("temperature_c", 24.0)) if sensors.get("temperature_c") is not None else None
    humidity_pct = float(sensors.get("humidity_pct", 65.0)) if sensors.get("humidity_pct") is not None else None
    soil_moisture_pct = float(sensors.get("soil_moisture_pct", 40.0)) if sensors.get("soil_moisture_pct") is not None else None
    soil_moisture_raw = int(sensors.get("soil_moisture_raw", 0)) if sensors.get("soil_moisture_raw") is not None else None
    rain_rate_mm_h = float(sensors.get("rain_rate_mm_h", 0.0)) if sensors.get("rain_rate_mm_h") is not None else None
    rain_accum_mm = float(sensors.get("rain_accum_mm", 0.0)) if sensors.get("rain_accum_mm") is not None else None
    rain_intensity_pct = float(sensors.get("rain_intensity_pct", (rain_rate_mm_h or 0.0) / 1.0)) if sensors.get("rain_intensity_pct") is not None else None
    slope_degrees = float(sensors.get("slope_degrees", 12.0)) if sensors.get("slope_degrees") is not None else None
    vibration_g = float(sensors.get("vibration_g", 0.0)) if sensors.get("vibration_g") is not None else None

    pred = packet.get("ai_prediction", {})
    pred_level = int(pred.get("predicted_risk_level", 0)) if pred.get("predicted_risk_level") is not None else 0
    risk_category = pred.get("risk_category", "Low Risk")
    risk_code = pred.get("risk_code", "LOW")
    risk_color = pred.get("risk_color", "#10B981")
    confidence_score = float(pred.get("confidence_score", 95.0)) if pred.get("confidence_score") is not None else 95.0
    status_code = pred.get("status_code", "NORMAL / ALL CLEAR")
    lead_time = pred.get("early_warning_lead_time", "72h Routine Monitoring")
    
    actuator = pred.get("actuator_state", {})
    buzzer_active = 1 if (actuator.get("buzzer_active") or pred_level >= 2) else 0
    led_color = actuator.get("led_color", "RED" if pred_level >= 2 else ("YELLOW" if pred_level == 1 else "GREEN"))

    drivers = pred.get("primary_risk_drivers", [])
    advisories = pred.get("advisory_actions", [])
    drivers_json = json.dumps(drivers) if isinstance(drivers, (list, dict)) else str(drivers)
    advisories_json = json.dumps(advisories) if isinstance(advisories, (list, dict)) else str(advisories)
    raw_payload_json = json.dumps(packet)

    record_id = None
    with _DB_LOCK:
        conn = get_db_connection(db_path)
        try:
            with conn:
                cursor = conn.execute("""
                    INSERT INTO sensor_readings (
                        timestamp, node_id, status, temperature_c, humidity_pct,
                        soil_moisture_pct, soil_moisture_raw, rain_rate_mm_h, rain_accum_mm,
                        rain_intensity_pct, slope_degrees, vibration_g, predicted_risk_level,
                        risk_category, risk_code, risk_color, confidence_score, status_code,
                        early_warning_lead_time, buzzer_active, led_color,
                        primary_risk_drivers, advisory_actions, raw_payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp, node_id, status, temp_c, humidity_pct,
                    soil_moisture_pct, soil_moisture_raw, rain_rate_mm_h, rain_accum_mm,
                    rain_intensity_pct, slope_degrees, vibration_g, pred_level,
                    risk_category, risk_code, risk_color, confidence_score, status_code,
                    lead_time, buzzer_active, led_color,
                    drivers_json, advisories_json, raw_payload_json
                ))
                record_id = cursor.lastrowid
        finally:
            conn.close()

        # Append to CSV log
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record_id, timestamp, node_id, status, temp_c, humidity_pct,
                    soil_moisture_pct, soil_moisture_raw, rain_rate_mm_h, rain_accum_mm,
                    rain_intensity_pct, slope_degrees, vibration_g, pred_level,
                    risk_category, risk_code, risk_color, confidence_score, status_code,
                    lead_time, buzzer_active, led_color,
                    drivers_json, advisories_json
                ])
        except Exception:
            pass

    return record_id


def get_recent_readings(limit=100, offset=0, node_id=None, db_path=DB_PATH):
    """Fetch the most recent telemetry readings ordered by ID descending."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        if node_id:
            query = "SELECT * FROM sensor_readings WHERE node_id = ? ORDER BY id DESC LIMIT ? OFFSET ?"
            rows = conn.execute(query, (node_id, limit, offset)).fetchall()
        else:
            query = "SELECT * FROM sensor_readings ORDER BY id DESC LIMIT ? OFFSET ?"
            rows = conn.execute(query, (limit, offset)).fetchall()

        results = []
        for r in rows:
            row_dict = dict(r)
            # Parse json fields safely
            for json_col in ["primary_risk_drivers", "advisory_actions", "raw_payload"]:
                if row_dict.get(json_col):
                    try:
                        row_dict[json_col] = json.loads(row_dict[json_col])
                    except Exception:
                        pass
            results.append(row_dict)
        return results
    finally:
        conn.close()


def get_recent_packets_for_stream(limit=100, db_path=DB_PATH):
    """
    Fetch recent telemetry packets formatted specifically for the live frontend waveform stream
    ordered chronologically (ascending).
    """
    rows = get_recent_readings(limit=limit, offset=0, db_path=db_path)
    # Reverse to ascending chronological order
    rows.reverse()

    packets = []
    for r in rows:
        drivers = r.get("primary_risk_drivers", [])
        if isinstance(drivers, str):
            try:
                drivers = json.loads(drivers)
            except Exception:
                drivers = [drivers]
        advisories = r.get("advisory_actions", [])
        if isinstance(advisories, str):
            try:
                advisories = json.loads(advisories)
            except Exception:
                advisories = [advisories]

        packet = {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "node_id": r["node_id"],
            "status": r.get("status", "ONLINE"),
            "sensors": {
                "temperature_c": r["temperature_c"],
                "humidity_pct": r["humidity_pct"],
                "soil_moisture_pct": r["soil_moisture_pct"],
                "soil_moisture_raw": r.get("soil_moisture_raw", 0),
                "rain_rate_mm_h": r["rain_rate_mm_h"],
                "rain_accum_mm": r["rain_accum_mm"],
                "rain_intensity_pct": r.get("rain_intensity_pct", 0.0),
                "slope_degrees": r["slope_degrees"],
                "vibration_g": r["vibration_g"],
            },
            "ai_prediction": {
                "predicted_risk_level": r["predicted_risk_level"],
                "risk_category": r["risk_category"],
                "risk_code": r.get("risk_code", "LOW"),
                "risk_color": r.get("risk_color", "#10B981"),
                "confidence_score": r.get("confidence_score", 95.0),
                "status_code": r.get("status_code", "NORMAL / ALL CLEAR"),
                "early_warning_lead_time": r.get("early_warning_lead_time", "72h Monitoring"),
                "primary_risk_drivers": drivers,
                "advisory_actions": advisories,
                "actuator_state": {
                    "buzzer_active": bool(r.get("buzzer_active", 0)),
                    "led_color": r.get("led_color", "GREEN"),
                }
            }
        }
        packets.append(packet)
    return packets


def get_telemetry_stats(db_path=DB_PATH):
    """Compute aggregate storage and telemetry statistics."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        total_count = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
        if total_count == 0:
            return {
                "total_readings": 0,
                "earliest_timestamp": None,
                "latest_timestamp": None,
                "unique_nodes": 0,
                "avg_temperature_c": 0.0,
                "avg_humidity_pct": 0.0,
                "avg_soil_moisture_pct": 0.0,
                "avg_rain_rate_mm_h": 0.0,
                "max_rain_rate_mm_h": 0.0,
                "max_rain_accum_mm": 0.0,
                "risk_tier_counts": {"Level 0": 0, "Level 1": 0, "Level 2": 0, "Level 3": 0},
                "storage_db_size_bytes": os.path.getsize(db_path) if os.path.exists(db_path) else 0,
                "storage_csv_size_bytes": os.path.getsize(CSV_PATH) if os.path.exists(CSV_PATH) else 0,
            }

        first_ts = conn.execute("SELECT timestamp FROM sensor_readings ORDER BY id ASC LIMIT 1").fetchone()[0]
        last_ts = conn.execute("SELECT timestamp FROM sensor_readings ORDER BY id DESC LIMIT 1").fetchone()[0]
        unique_nodes = conn.execute("SELECT COUNT(DISTINCT node_id) FROM sensor_readings").fetchone()[0]

        aggs = conn.execute("""
            SELECT 
                AVG(temperature_c), AVG(humidity_pct), AVG(soil_moisture_pct),
                AVG(rain_rate_mm_h), MAX(rain_rate_mm_h), MAX(rain_accum_mm)
            FROM sensor_readings
        """).fetchone()

        risk_rows = conn.execute("""
            SELECT predicted_risk_level, COUNT(*) 
            FROM sensor_readings 
            GROUP BY predicted_risk_level
        """).fetchall()

        risk_tier_counts = {"Level 0 (Low)": 0, "Level 1 (Moderate)": 0, "Level 2 (High)": 0, "Level 3 (Severe)": 0}
        for r_lvl, count in risk_rows:
            key_map = {0: "Level 0 (Low)", 1: "Level 1 (Moderate)", 2: "Level 2 (High)", 3: "Level 3 (Severe)"}
            if r_lvl in key_map:
                risk_tier_counts[key_map[r_lvl]] = count

        return {
            "total_readings": total_count,
            "earliest_timestamp": first_ts,
            "latest_timestamp": last_ts,
            "unique_nodes": unique_nodes,
            "avg_temperature_c": round(aggs[0] or 0.0, 1),
            "avg_humidity_pct": round(aggs[1] or 0.0, 1),
            "avg_soil_moisture_pct": round(aggs[2] or 0.0, 1),
            "avg_rain_rate_mm_h": round(aggs[3] or 0.0, 1),
            "max_rain_rate_mm_h": round(aggs[4] or 0.0, 1),
            "max_rain_accum_mm": round(aggs[5] or 0.0, 1),
            "risk_tier_counts": risk_tier_counts,
            "storage_db_size_bytes": os.path.getsize(db_path) if os.path.exists(db_path) else 0,
            "storage_csv_size_bytes": os.path.getsize(CSV_PATH) if os.path.exists(CSV_PATH) else 0,
        }
    finally:
        conn.close()


def clear_all_readings(db_path=DB_PATH, csv_path=CSV_PATH):
    """Clear all stored readings from SQLite and reset CSV header."""
    with _DB_LOCK:
        if os.path.exists(db_path):
            conn = get_db_connection(db_path)
            try:
                with conn:
                    conn.execute("DELETE FROM sensor_readings")
                    conn.execute("VACUUM")
            finally:
                conn.close()

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
