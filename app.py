"""
Flash Flood Early Risk Prediction & Real-Time IoT Early Warning Dashboard.
Live Telemetry Monitor for ESP32 Edge Sensor Node (DHT11, MPU6050, Soil Moisture, HW-038).
Connects to persistent SQLite database (data/sensor_readings.db) and CSV time-series log.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# Path configurations
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LATEST_TELEMETRY_PATH = os.path.join(DATA_DIR, "live_telemetry_latest.json")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import predict_sensor_telemetry_risk
from risk_labeler import RISK_TIERS
import telemetry_db

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Flash Flood Early Warning - ESP32 Real-Time IoT Hub",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #06B6D4 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    
    .header-bar {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
    }
    
    .metric-card {
        background: rgba(18, 24, 38, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.3rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }
    
    .sensor-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .sensor-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    
    .sensor-val {
        font-size: 2.1rem;
        font-weight: 900;
        color: #38BDF8;
        line-height: 1.1;
        margin: 0.2rem 0;
    }
    
    .sensor-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        margin-top: 0.3rem;
    }
    
    .risk-badge-low {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid #10B981;
        color: #10B981;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.85rem;
        display: inline-block;
    }
    .risk-badge-moderate {
        background: rgba(245, 158, 11, 0.18);
        border: 1px solid #F59E0B;
        color: #F59E0B;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.85rem;
        display: inline-block;
    }
    .risk-badge-high {
        background: rgba(249, 115, 22, 0.18);
        border: 1px solid #F97316;
        color: #F97316;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.85rem;
        display: inline-block;
    }
    .risk-badge-severe {
        background: rgba(239, 68, 68, 0.25);
        border: 1px solid #EF4444;
        color: #EF4444;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 900;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .driver-item {
        background: rgba(15, 23, 42, 0.65);
        border-left: 3px solid #3B82F6;
        padding: 0.65rem 0.9rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #E2E8F0;
    }
    
    .advisory-item {
        background: rgba(15, 23, 42, 0.65);
        border-left: 3px solid #F59E0B;
        padding: 0.65rem 0.9rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #FEF08A;
    }
</style>
""", unsafe_allow_html=True)


def parse_timestamp_flexible(ts_str):
    """Parse timestamp string supporting multiple standard date formats."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts_str, fmt)
        except Exception:
            pass
    return None


def load_live_telemetry():
    """Load latest telemetry packet from disk or return default standby state."""
    if os.path.exists(LATEST_TELEMETRY_PATH):
        try:
            with open(LATEST_TELEMETRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                ts_str = data.get("timestamp", "")
                ts_dt = parse_timestamp_flexible(ts_str)
                if ts_dt:
                    age = (datetime.now() - ts_dt).total_seconds()
                    if age > 6.0:
                        data["status"] = "OFFLINE"
                        data["is_live"] = False
                        data["age_seconds"] = int(age)
                    else:
                        data["status"] = "ONLINE"
                        data["is_live"] = True
                        data["age_seconds"] = int(age)
                return data
        except Exception:
            pass

    # Default fallback state
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
        "is_live": False,
        "age_seconds": 0,
        "sensors": {
            "temperature_c": 24.5,
            "humidity_pct": 65.0,
            "soil_moisture_pct": 38.0,
            "soil_moisture_raw": 3200,
            "rain_rate_mm_h": 0.0,
            "rain_accum_mm": 0.0,
            "slope_degrees": 12.0,
            "vibration_g": 0.0,
        },
        "ai_prediction": pred,
    }


def main():
    packet = load_live_telemetry()
    sensors = packet.get("sensors", {})
    pred = packet.get("ai_prediction", {})
    status = packet.get("status", "STANDBY")
    is_live = packet.get("is_live", False)
    age_sec = packet.get("age_seconds", 0)
    node_id = packet.get("node_id", "ESP32_FLOOD_NODE_01")
    timestamp = packet.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Header section
    st.markdown('<div class="main-title">🌊 AI Flash Flood Early Warning System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Live Edge Telemetry Stream from ESP32 Multi-Sensor Microcontroller | Persistent SQLite & CSV Storage</div>', unsafe_allow_html=True)

    # Top Status & Control Bar
    top_c1, top_c2, top_c3, top_c4 = st.columns([1.5, 1.2, 1, 0.8])
    with top_c1:
        if status == "ONLINE" and is_live:
            st.success(f"🟢 **ESP32 STREAM ACTIVE** (`{node_id}`)")
        elif status == "OFFLINE" or not is_live:
            st.error(f"🔴 **ESP32 DISCONNECTED / STANDBY** (`{node_id}` - {age_sec}s ago)")
        else:
            st.warning(f"🟡 **STANDBY - WAITING FOR ESP32** (`{node_id}`)")
    with top_c2:
        st.info(f"⏱️ **Last Telemetry**: `{timestamp}`")
    with top_c3:
        auto_refresh = st.checkbox("⚡ Live Auto-Update", value=False, help="Automatically refreshes every 2 seconds when new telemetry arrives.")
    with top_c4:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # ==================================================================================
    # 1. LIVE SENSOR TELEMETRY ROW (5 PHYSICAL SENSORS)
    # ==================================================================================
    st.subheader("📡 Live ESP32 Sensor Readings")

    temp = float(sensors.get("temperature_c", 24.0))
    humidity = float(sensors.get("humidity_pct", 65.0))
    soil = float(sensors.get("soil_moisture_pct", 40.0))
    rain_rate = float(sensors.get("rain_rate_mm_h", 0.0))
    rain_accum = float(sensors.get("rain_accum_mm", 0.0))
    slope = float(sensors.get("slope_degrees", 12.0))
    vibration = float(sensors.get("vibration_g", 0.0))

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)

    with sc1:
        st.markdown(f"""
        <div class="sensor-card">
            <div class="sensor-chip">DHT11 DIGITAL</div>
            <div class="sensor-val">{temp:.1f}°C</div>
            <div class="sensor-label">🌡️ Ambient Temp</div>
            <div style="font-size: 0.78rem; color: {'#F59E0B' if temp > 35 else '#10B981'}; margin-top: 0.3rem;">
                {'High Heat' if temp > 35 else 'Normal Temp'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        st.markdown(f"""
        <div class="sensor-card">
            <div class="sensor-chip">DHT11 DIGITAL</div>
            <div class="sensor-val">{humidity:.0f}%</div>
            <div class="sensor-label">💧 Relative Humidity</div>
            <div style="font-size: 0.78rem; color: {'#EF4444' if humidity >= 90 else ('#F59E0B' if humidity >= 75 else '#10B981')}; margin-top: 0.3rem;">
                {'Vapor Saturated' if humidity >= 90 else ('High Moisture' if humidity >= 75 else 'Optimal RH')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sc3:
        soil_color = '#EF4444' if soil >= 85 else ('#F59E0B' if soil >= 70 else '#10B981')
        st.markdown(f"""
        <div class="sensor-card">
            <div class="sensor-chip">CAPACITIVE ADC</div>
            <div class="sensor-val" style="color: {soil_color};">{soil:.1f}%</div>
            <div class="sensor-label">🌱 Soil Saturation</div>
            <div style="font-size: 0.78rem; color: {soil_color}; margin-top: 0.3rem;">
                {'Waterlogged (Critical)' if soil >= 85 else ('High Saturation' if soil >= 70 else 'Stable Subsurface')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sc4:
        rain_color = '#EF4444' if rain_rate >= 50 else ('#F59E0B' if rain_rate >= 20 else '#38BDF8')
        st.markdown(f"""
        <div class="sensor-card">
            <div class="sensor-chip">HW-038 RAINDROP</div>
            <div class="sensor-val" style="color: {rain_color};">{rain_rate:.1f}</div>
            <div class="sensor-label">🌧️ Rain (mm/h)</div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.3rem;">
                Depth: <b style="color: #F8FAFC;">{rain_accum:.1f} mm</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sc5:
        slope_color = '#EF4444' if slope >= 25 or vibration >= 0.15 else ('#F59E0B' if slope >= 15 else '#10B981')
        st.markdown(f"""
        <div class="sensor-card">
            <div class="sensor-chip">MPU-6050 6-AXIS</div>
            <div class="sensor-val" style="color: {slope_color};">{slope:.1f}°</div>
            <div class="sensor-label">📐 Slope Pitch Angle</div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.3rem;">
                Vibration: <b style="color: #F8FAFC;">{vibration:.2f}g</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ==================================================================================
    # 2. HERO REAL-TIME AI FLASH FLOOD RISK ASSESSMENT & ADVISORIES
    # ==================================================================================
    col_risk_main, col_risk_side = st.columns([1.4, 1])

    risk_level = pred.get("predicted_risk_level", 0)
    risk_cat = pred.get("risk_category", "Low Risk")
    risk_code = pred.get("risk_code", "LOW")
    risk_color = pred.get("risk_color", "#10B981")
    risk_desc = pred.get("risk_description", "Baseline hydrological conditions normal.")
    confidence = pred.get("confidence_score", 95.0)
    lead_time = pred.get("early_warning_lead_time", "72h Routine Monitoring")
    status_code = pred.get("status_code", "NORMAL / ALL CLEAR")
    drivers = pred.get("primary_risk_drivers", ["All sensors operating within baseline safe parameters."])
    advisories = pred.get("advisory_actions", ["Routine environmental monitoring active."])
    probabilities = pred.get("probability_distribution", {"Low Risk": 90.0, "Moderate Risk": 10.0, "High Risk": 0.0, "Severe Risk": 0.0})

    with col_risk_main:
        st.subheader("⚡ Real-Time AI Flash Flood Inference")

        badge_class = f"risk-badge-{risk_code.lower()}"
        buzzer_active = risk_level >= 2
        led_color = "RED" if risk_level >= 2 else ("YELLOW" if risk_level == 1 else "GREEN")

        st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid {risk_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                <span class="{badge_class}">{risk_cat.upper()} (LEVEL {risk_level})</span>
                <span style="font-weight: 700; color: #94A3B8;">Confidence: <span style="color: #F8FAFC;">{confidence:.1f}%</span></span>
            </div>
            <div style="font-size: 1.45rem; font-weight: 800; color: {risk_color if risk_level >= 2 else '#F8FAFC'}; margin-bottom: 0.4rem;">
                {status_code}
            </div>
            <div style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.4;">
                {risk_desc}
            </div>
            <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.08); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; font-size: 0.88rem;">
                <div>⏳ <b>Early Warning Lead Time:</b> <span style="color: #38BDF8;">{lead_time}</span></div>
                <div>🔔 <b>Hardware Siren:</b> <span style="color: {'#EF4444' if buzzer_active else '#10B981'}; font-weight: 700;">{'ACTIVE 🚨' if buzzer_active else 'OFF 🔇'}</span> | <b>LED:</b> <span style="color: {risk_color}; font-weight: 700;">{led_color}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 📊 Neural Risk Probability Distribution")
        prob_df = pd.DataFrame(list(probabilities.items()), columns=["Risk Tier", "Probability (%)"])
        st.bar_chart(prob_df.set_index("Risk Tier"), color="#38BDF8", height=160)

    with col_risk_side:
        st.subheader("🔍 Contributing Sensor Drivers")
        for d in drivers:
            st.markdown(f'<div class="driver-item">⚡ {d}</div>', unsafe_allow_html=True)

        st.subheader("🛡️ Civil Defense Action Protocol")
        for a in advisories:
            st.markdown(f'<div class="advisory-item">⚠️ {a}</div>', unsafe_allow_html=True)

    # ==================================================================================
    # 3. LIVE HISTORICAL TELEMETRY WAVEFORM FROM PERSISTENT DATABASE
    # ==================================================================================
    st.markdown("---")
    st.subheader("📈 Real-Time Persistent Telemetry Waveform")

    # Load from persistent SQLite database
    db_records = telemetry_db.get_recent_readings(limit=35)
    if db_records:
        chart_rows = []
        for r in reversed(db_records):
            t_str = str(r["timestamp"])
            time_label = t_str.split(" ")[1] if " " in t_str else t_str
            chart_rows.append({
                "Time": time_label,
                "Rain Rate (mm/h)": float(r.get("rain_rate_mm_h") or 0.0),
                "Soil Moisture (%)": float(r.get("soil_moisture_pct") or 0.0),
                "Humidity (%)": float(r.get("humidity_pct") or 0.0),
                "Temperature (°C)": float(r.get("temperature_c") or 0.0),
                "Slope Pitch (°)": float(r.get("slope_degrees") or 0.0),
            })
        chart_df = pd.DataFrame(chart_rows).set_index("Time")
        st.line_chart(chart_df, height=260)
    else:
        st.info("No historical readings recorded in database yet. Start streaming from ESP32 or simulator.")

    # ==================================================================================
    # 4. PERSISTENT DATABASE RECORDS VIEWER & CSV EXPORT
    # ==================================================================================
    st.markdown("---")
    st.subheader("📦 Persistent Sensor Telemetry Database (`data/sensor_readings.db`)")

    stats = telemetry_db.get_telemetry_stats()
    st1, st2, st3, st4 = st.columns(4)
    st1.metric("Total Readings Saved", f"{stats['total_readings']:,}")
    st1_avg_t = stats.get("avg_temperature_c", 0.0)
    st2.metric("Avg Temperature", f"{st1_avg_t:.1f}°C")
    st3.metric("Avg Soil Saturation", f"{stats['avg_soil_moisture_pct']:.1f}%")
    st4.metric("Max Rain Intensity", f"{stats['max_rain_rate_mm_h']:.1f} mm/h")

    if db_records:
        df_display = pd.DataFrame(db_records)[[
            "id", "timestamp", "node_id", "temperature_c", "humidity_pct",
            "soil_moisture_pct", "rain_rate_mm_h", "rain_accum_mm", "slope_degrees",
            "vibration_g", "predicted_risk_level", "risk_category", "status"
        ]].rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "node_id": "Node ID",
            "temperature_c": "Temp (°C)",
            "humidity_pct": "Humidity (%)",
            "soil_moisture_pct": "Soil (%)",
            "rain_rate_mm_h": "Rain Rate (mm/h)",
            "rain_accum_mm": "Accum (mm)",
            "slope_degrees": "Slope (°)",
            "vibration_g": "Vibe (g)",
            "predicted_risk_level": "Risk Level",
            "risk_category": "Risk Tier",
            "status": "Status"
        })
        st.dataframe(df_display, use_container_width=True, height=240)

        # Download CSV button
        if os.path.exists(telemetry_db.CSV_PATH):
            with open(telemetry_db.CSV_PATH, "r", encoding="utf-8") as f:
                csv_data = f.read()
            st.download_button(
                label="📥 Download All Readings as CSV",
                data=csv_data,
                file_name=f"esp32_sensor_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Live auto-refresh cycle
    if auto_refresh:
        time.sleep(2.0)
        st.rerun()


if __name__ == "__main__":
    main()
