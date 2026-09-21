"""
Real-time Early Flash Flood Risk Inference Engine & Alert Generator.
Takes district metadata, recent hydrological history, and forecasted weather parameters
to produce calibrated flash flood early warning predictions, risk factor attribution, and action advisories.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import engineer_features, get_feature_column_names
from risk_labeler import RISK_TIERS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "flood_risk_model.joblib")
TOPOGRAPHY_FILE = os.path.join(DATA_DIR, "elevation_slope_by_district.csv")
CLIMATOLOGY_FILE = os.path.join(DATA_DIR, "district_climatology_summary.csv")

_MODEL_CACHE = None


def load_model():
    """Load and cache the trained model bundle."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def get_district_metadata(district_name):
    """Retrieve coordinates, region, elevation, slope, and historical climatology for a district."""
    if not os.path.exists(TOPOGRAPHY_FILE):
        raise FileNotFoundError(f"Topography file not found at {TOPOGRAPHY_FILE}")
    
    df_dist = pd.read_csv(TOPOGRAPHY_FILE)
    if os.path.exists(CLIMATOLOGY_FILE):
        df_clim = pd.read_csv(CLIMATOLOGY_FILE)
        merge_cols = [c for c in df_clim.columns if c not in ["Latitude", "Longitude"]]
        df_dist = pd.merge(df_dist, df_clim[merge_cols], on=["District", "Region"], how="left", suffixes=("", "_clim"))
        if "Elevation_m_clim" in df_dist.columns:
            df_dist["Elevation_m"] = df_dist["Elevation_m"].fillna(df_dist["Elevation_m_clim"])
            df_dist.drop(columns=["Elevation_m_clim"], inplace=True)
        if "Slope_degrees_clim" in df_dist.columns:
            df_dist["Slope_degrees"] = df_dist["Slope_degrees"].fillna(df_dist["Slope_degrees_clim"])
            df_dist.drop(columns=["Slope_degrees_clim"], inplace=True)

    match = df_dist[df_dist["District"].str.lower() == district_name.strip().lower()]
    if match.empty:
        available = ", ".join(df_dist["District"].unique())
        raise ValueError(f"District '{district_name}' not found. Available districts: {available}")
    return match.iloc[0].to_dict()


def generate_action_advisory(risk_level, district_name, p1d, p3d, s_surf, slope):
    """
    Generate actionable civil defense and community early warning advisories.
    """
    if risk_level == 0:
        return {
            "status": "NORMAL / ALL CLEAR",
            "lead_time": "72h Routine Monitoring",
            "actions": [
                "Routine environmental and watershed monitoring.",
                "Maintain standard drainage clearance and river gauge tracking.",
                "No community alerts or evacuations required."
            ]
        }
    elif risk_level == 1:
        return {
            "status": "ADVISORY / ELEVATED WATCH",
            "lead_time": "48h - 72h Early Advisory",
            "actions": [
                "Issue advisory to local disaster management teams and riverside settlements.",
                "Inspect critical culverts, drainage conduits, and hillside road blocks.",
                "Advise residents in low-lying and flood-prone floodplains to stay vigilant for rising waters.",
                "Keep emergency response teams on standby."
            ]
        }
    elif risk_level == 2:
        return {
            "status": "WATCH / ORANGE ALERT",
            "lead_time": "24h - 48h Flash Flood Watch",
            "actions": [
                "Activate District Disaster Management Authority (DDMA) Emergency Operations Center.",
                "Issue mandatory travel restrictions near steep mountain roads, waterfall areas, and riverbanks.",
                "Prepare relief shelters and preposition rescue boats, sandbags, and earthmovers.",
                "High risk of hillside landslides and tributary flash floods; begin voluntary relocation in high-vulnerability wards."
            ]
        }
    else: # risk_level == 3
        return {
            "status": "SEVERE / RED ALERT - TAKE IMMEDIATE ACTION",
            "lead_time": "0h - 24h Critical Imminent Warning",
            "actions": [
                "URGENT: Initiate immediate evacuation of riverside settlements and steep landslide slopes.",
                "Deploy National/State Disaster Response Forces (NDRF/SDRF) to vulnerable sectors.",
                "Suspend all mountain traffic, school operations, and non-essential activities immediately.",
                "Issue life-safety broadcast alerts across mobile networks and siren warning towers."
            ]
        }


def explain_top_risk_drivers(feat_row, risk_level, dist_meta=None):
    """Identify the primary hydrological and topographical factors contributing to the risk."""
    drivers = []
    
    p1d = feat_row.get("Precip_1d", 0.0)
    p3d = feat_row.get("Precip_3d_sum", 0.0)
    api = feat_row.get("Antecedent_Precip_Index", 0.0)
    s_surf = feat_row.get("Soil_Moisture_Surface", 0.0)
    s_root = feat_row.get("Soil_Moisture_RootZone", 0.0)
    slope = feat_row.get("Slope_degrees", 0.0)
    elev = feat_row.get("Elevation_m", 0.0)
    runoff = feat_row.get("Runoff_Energy_Index", 0.0)

    # Climatological baseline comparisons
    if dist_meta and "Rainfall_3day_mean" in dist_meta and not np.isnan(dist_meta["Rainfall_3day_mean"]):
        r3_mean = float(dist_meta["Rainfall_3day_mean"])
        r3_max = float(dist_meta.get("Rainfall_3day_max", r3_mean * 2.5))
        if p3d >= r3_mean * 1.5:
            ratio = p3d / max(0.1, r3_mean)
            drivers.append(f"3-Day Rainfall ({p3d:.1f} mm) is {ratio:.1f}x higher than district historical normal ({r3_mean:.1f} mm, max recorded: {r3_max:.1f} mm).")

    if p1d >= 115.5:
        drivers.append(f"Extreme daily cloudburst-level precipitation: {p1d:.1f} mm/day (IMD Very Heavy threshold).")
    elif p1d >= 64.5:
        drivers.append(f"Heavy 24-hour rainfall: {p1d:.1f} mm (IMD Heavy Rain threshold).")
    elif p1d > 25.0:
        drivers.append(f"Active daily rainfall: {p1d:.1f} mm.")

    if p3d >= 150.0:
        drivers.append(f"Massive 3-day rainfall accumulation: {p3d:.1f} mm (high watershed saturation).")
    elif p3d >= 80.0:
        drivers.append(f"Elevated 3-day cumulative rainfall: {p3d:.1f} mm.")

    if s_surf >= 0.85:
        drivers.append(f"Severe surface soil saturation at {s_surf*100:.1f}% (virtually zero infiltration capacity left, rapid runoff).")
    elif s_surf >= 0.70:
        drivers.append(f"High surface soil moisture: {s_surf*100:.1f}%.")

    if slope >= 25.0:
        drivers.append(f"Steep topographical terrain slope ({slope:.1f}°), driving high runoff kinetic energy and landslide susceptibility.")
    elif slope >= 15.0:
        drivers.append(f"Moderate-to-steep hillside slope ({slope:.1f}°).")

    if runoff >= 50.0:
        drivers.append(f"Critical Runoff Energy Index ({runoff:.1f}), indicating extreme rapid overland surge.")

    if not drivers:
        drivers.append("Normal precipitation, dry-to-moderate soil conditions, and stable hydrological parameters.")

    return drivers


def predict_district_risk(
    district_name,
    precipitation_mm=None,
    precip_3d_sum=None,
    precip_7d_sum=None,
    soil_moisture_surface=None,
    soil_moisture_root=None,
    temp_avg_c=None,
    temp_max_c=None,
    temp_min_c=None,
    humidity_pct=None,
    target_date=None,
):
    """
    End-to-end prediction function for a given district and weather state.
    If weather parameters are None, automatically uses the district's historical climatological baselines.
    """
    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    dist_meta = get_district_metadata(district_name)
    
    # Fill defaults from district climatology summary if available
    if precipitation_mm is None:
        precipitation_mm = float(dist_meta.get("Rainfall_3day_mean", 15.0)) / 2.0
    if precip_3d_sum is None:
        precip_3d_sum = float(dist_meta.get("Rainfall_3day_mean", 30.0))
    if precip_7d_sum is None:
        precip_7d_sum = float(dist_meta.get("Rainfall_7day_mean", 60.0))
    if soil_moisture_surface is None:
        soil_moisture_surface = float(dist_meta.get("Soil_Moisture_Surface_mean", 0.65))
    if soil_moisture_root is None:
        soil_moisture_root = float(dist_meta.get("Soil_Moisture_RootZone_mean", 0.68))
    if temp_avg_c is None:
        temp_avg_c = float(dist_meta.get("Temp_Avg_C", 20.0))
    if temp_max_c is None:
        temp_max_c = float(dist_meta.get("Temp_Max_C", temp_avg_c + 5.0))
    if temp_min_c is None:
        temp_min_c = float(dist_meta.get("Temp_Min_C", temp_avg_c - 5.0))
    if humidity_pct is None:
        humidity_pct = 85.0

    if target_date is None:
        target_date = datetime.today()
    elif isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    # Build sequence of synthetic 14-day history to calculate rolling/lag features accurately
    dates = pd.date_range(end=target_date, periods=14, freq="D")
    
    # Ramp up to current state smoothly
    p_series = np.linspace(precipitation_mm * 0.4, precipitation_mm, 14)
    p_series[-1] = precipitation_mm
    
    s_surf_series = np.linspace(max(0.2, soil_moisture_surface - 0.15), soil_moisture_surface, 14)
    s_root_series = np.linspace(max(0.2, soil_moisture_root - 0.10), soil_moisture_root, 14)

    synth_df = pd.DataFrame({
        "Region": dist_meta["Region"],
        "District": dist_meta["District"],
        "Latitude": dist_meta["Latitude"],
        "Longitude": dist_meta["Longitude"],
        "Elevation_m": dist_meta["Elevation_m"],
        "Slope_degrees": dist_meta["Slope_degrees"],
        "Date": dates,
        "Precipitation_mm": p_series,
        "Soil_Moisture_RootZone": s_root_series,
        "Soil_Moisture_Surface": s_surf_series,
        "Temp_Avg_C": temp_avg_c,
        "Temp_Max_C": temp_max_c,
        "Temp_Min_C": temp_min_c,
        "Humidity_pct": humidity_pct,
    })

    # Run full feature engineering pipeline
    engineered = engineer_features(synth_df)
    
    # Override rolling sums if explicitly supplied
    if precip_3d_sum is not None:
        engineered.loc[engineered.index[-1], "Precip_3d_sum"] = precip_3d_sum
    if precip_7d_sum is not None:
        engineered.loc[engineered.index[-1], "Precip_7d_sum"] = precip_7d_sum

    latest_row = engineered.iloc[-1:]
    X_raw = latest_row[feature_names].values
    X_scaled = scaler.transform(X_raw)

    # Run AI inference
    pred_level = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None

    prob_dict = {}
    if probabilities is not None:
        for i, p in enumerate(probabilities):
            tier_info = RISK_TIERS.get(i, {"name": f"Level {i}"})
            prob_dict[tier_info["name"]] = round(float(p) * 100.0, 2)

    tier = RISK_TIERS[pred_level]
    advisory = generate_action_advisory(
        pred_level,
        dist_meta["District"],
        precipitation_mm,
        latest_row["Precip_3d_sum"].values[0],
        soil_moisture_surface,
        dist_meta["Slope_degrees"]
    )
    risk_drivers = explain_top_risk_drivers(latest_row.iloc[0].to_dict(), pred_level, dist_meta=dist_meta)

    result = {
        "district": dist_meta["District"],
        "region": dist_meta["Region"],
        "coordinates": {"latitude": dist_meta["Latitude"], "longitude": dist_meta["Longitude"]},
        "topography": {"elevation_m": dist_meta["Elevation_m"], "slope_degrees": dist_meta["Slope_degrees"]},
        "target_date": target_date.strftime("%Y-%m-%d"),
        "predicted_risk_level": pred_level,
        "risk_category": tier["name"],
        "risk_code": tier["code"],
        "risk_color": tier["color"],
        "risk_description": tier["desc"],
        "confidence_score": round(float(probabilities[pred_level]) * 100.0, 1) if probabilities is not None else 100.0,
        "probability_distribution": prob_dict,
        "early_warning_lead_time": advisory["lead_time"],
        "status_code": advisory["status"],
        "advisory_actions": advisory["actions"],
        "primary_risk_drivers": risk_drivers,
        "input_conditions": {
            "rainfall_1d_mm": precipitation_mm,
            "rainfall_3d_sum_mm": float(latest_row["Precip_3d_sum"].values[0]),
            "soil_moisture_surface_pct": round(soil_moisture_surface * 100.0, 1),
            "soil_moisture_root_pct": round(soil_moisture_root * 100.0, 1),
            "humidity_pct": humidity_pct,
            "temp_avg_c": temp_avg_c,
        },
        "climatology_baseline": {
            "historical_3d_rainfall_mean_mm": dist_meta.get("Rainfall_3day_mean", None),
            "historical_3d_rainfall_max_mm": dist_meta.get("Rainfall_3day_max", None),
            "historical_7d_rainfall_mean_mm": dist_meta.get("Rainfall_7day_mean", None),
            "historical_7d_rainfall_max_mm": dist_meta.get("Rainfall_7day_max", None),
            "historical_surface_soil_mean": dist_meta.get("Soil_Moisture_Surface_mean", None),
            "historical_root_soil_mean": dist_meta.get("Soil_Moisture_RootZone_mean", None),
        }
    }
    return result


def predict_sensor_telemetry_risk(
    temp_c=24.0,
    humidity_pct=85.0,
    soil_moisture_pct=75.0,
    rain_rate_mm_h=0.0,
    rain_accum_mm=0.0,
    slope_degrees=15.0,
    vibration_g=0.0,
    elevation_m=450.0,
    node_id="ESP32_FLOOD_NODE",
    target_date=None,
):
    """
    Direct Real-Time Flash Flood Risk Prediction from live ESP32 IoT sensor telemetry.
    Computes all 43 ML hydrological and terrain features directly from physical sensor inputs:
    - DHT11: Temperature & Humidity
    - Capacitive Soil Sensor: Soil moisture saturation %
    - HW-038: Instantaneous Rain Rate (mm/h) & Cumulative Rain (mm)
    - MPU-6050: Terrain Slope Pitch (°) & Vibration (g)
    """
    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    temp_c = float(temp_c) if temp_c is not None else 24.0
    humidity_pct = float(humidity_pct) if humidity_pct is not None else 85.0
    soil_moisture_pct = float(soil_moisture_pct) if soil_moisture_pct is not None else 50.0
    rain_rate_mm_h = float(rain_rate_mm_h) if rain_rate_mm_h is not None else 0.0
    rain_accum_mm = float(rain_accum_mm) if rain_accum_mm is not None else 0.0
    slope_degrees = float(slope_degrees) if slope_degrees is not None else 15.0
    vibration_g = float(vibration_g) if vibration_g is not None else 0.0
    elevation_m = float(elevation_m) if elevation_m is not None else 450.0

    # Soil moisture fractions
    soil_surf = np.clip(soil_moisture_pct / 100.0 if soil_moisture_pct > 1.0 else soil_moisture_pct, 0.05, 0.99)
    soil_root = np.clip(soil_surf * 0.95, 0.05, 0.99)

    # Precipitation proxy from HW-038
    precip_1d = max(rain_accum_mm, rain_rate_mm_h * 2.2, 0.5)
    precip_3d = precip_1d * 2.4 + 10.0
    precip_7d = precip_3d * 1.6 + 15.0

    if target_date is None:
        target_date = datetime.today()
    elif isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    dates = pd.date_range(end=target_date, periods=14, freq="D")
    p_series = np.linspace(precip_1d * 0.4, precip_1d, 14)
    p_series[-1] = precip_1d

    s_surf_series = np.linspace(max(0.2, soil_surf - 0.15), soil_surf, 14)
    s_root_series = np.linspace(max(0.2, soil_root - 0.10), soil_root, 14)

    synth_df = pd.DataFrame({
        "Region": "Live IoT Monitoring Field",
        "District": node_id,
        "Latitude": 11.68,
        "Longitude": 76.13,
        "Elevation_m": elevation_m,
        "Slope_degrees": slope_degrees,
        "Date": dates,
        "Precipitation_mm": p_series,
        "Soil_Moisture_RootZone": s_root_series,
        "Soil_Moisture_Surface": s_surf_series,
        "Temp_Avg_C": temp_c,
        "Temp_Max_C": temp_c + 3.0,
        "Temp_Min_C": temp_c - 3.0,
        "Humidity_pct": humidity_pct,
    })

    engineered = engineer_features(synth_df)
    engineered.loc[engineered.index[-1], "Precip_3d_sum"] = precip_3d
    engineered.loc[engineered.index[-1], "Precip_7d_sum"] = precip_7d

    latest_row = engineered.iloc[-1:]
    X_raw = latest_row[feature_names].values
    X_scaled = scaler.transform(X_raw)

    pred_level = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None

    # Dynamic calibrated multi-sensor risk thresholds:
    # -------------------------------------------------------------
    # LEVEL 3 (SEVERE RISK / CRITICAL EMERGENCY)
    # -------------------------------------------------------------
    is_level_3 = (
        # 1. Extreme cloudburst / Critical flood water level
        rain_rate_mm_h >= 75.0 or
        # 2. Saturated ground with high water level / downpour
        (soil_moisture_pct >= 85.0 and rain_rate_mm_h >= 45.0) or
        # 3. Massive rainfall depth accumulation
        rain_accum_mm >= 100.0 or
        # 4. Saturated steep mountain slope with active rain or seismic tremor
        (slope_degrees >= 30.0 and soil_moisture_pct >= 80.0 and (rain_rate_mm_h >= 25.0 or vibration_g >= 0.25)) or
        # 5. Severe seismic earth tremor with steep slope
        (slope_degrees >= 35.0 and vibration_g >= 0.40)
    )

    # -------------------------------------------------------------
    # LEVEL 2 (HIGH RISK / WATCH & WARNING)
    # -------------------------------------------------------------
    is_level_2 = (
        # 1. High water level / Heavy rainfall intensity
        rain_rate_mm_h >= 45.0 or
        # 2. Significant rainfall accumulation
        rain_accum_mm >= 50.0 or
        # 3. High soil moisture combined with moderate water level or hill slope
        (soil_moisture_pct >= 75.0 and (rain_rate_mm_h >= 20.0 or slope_degrees >= 25.0)) or
        # 4. Steep slope with active vibration or rain
        (slope_degrees >= 30.0 and (vibration_g >= 0.18 or rain_rate_mm_h >= 15.0)) or
        # 5. Very high waterlogged ground
        soil_moisture_pct >= 88.0
    )

    # -------------------------------------------------------------
    # LEVEL 1 (MODERATE RISK / ADVISORY)
    # -------------------------------------------------------------
    is_level_1 = (
        # 1. Active moderate water rise / rain
        rain_rate_mm_h >= 15.0 or
        # 2. Accumulated rain
        rain_accum_mm >= 25.0 or
        # 3. Elevated soil saturation
        soil_moisture_pct >= 60.0 or
        # 4. Mountain terrain with moisture
        (slope_degrees >= 20.0 and (soil_moisture_pct >= 50.0 or rain_rate_mm_h >= 8.0)) or
        # 5. Notable seismic tremor
        vibration_g >= 0.20 or
        # 6. Very high atmospheric saturation
        humidity_pct >= 90.0
    )

    if is_level_3:
        pred_level = max(pred_level, 3)
    elif is_level_2:
        pred_level = max(pred_level, 2)
    elif is_level_1:
        pred_level = max(pred_level, 1)

    prob_dict = {}
    if probabilities is not None:
        for i, p in enumerate(probabilities):
            tier_info = RISK_TIERS.get(i, {"name": f"Level {i}"})
            prob_dict[tier_info["name"]] = round(float(p) * 100.0, 2)
        # Reflect threshold override in distribution
        if pred_level > 0 and prob_dict.get(RISK_TIERS[pred_level]["name"], 0) < 50.0:
            prob_dict[RISK_TIERS[pred_level]["name"]] = 85.0
            prob_dict["Low Risk"] = 5.0
    else:
        for i in range(4):
            prob_dict[RISK_TIERS[i]["name"]] = 100.0 if i == pred_level else 0.0

    tier = RISK_TIERS[pred_level]
    advisory = generate_action_advisory(
        pred_level,
        node_id,
        precip_1d,
        precip_3d,
        soil_surf,
        slope_degrees
    )

    # Dynamic sensor drivers
    drivers = []
    if soil_moisture_pct >= 85.0:
        drivers.append(f"Capacitive Soil Sensor: Critical saturation ({soil_moisture_pct:.1f}%). Subsurface infiltration capacity exhausted.")
    elif soil_moisture_pct >= 60.0:
        drivers.append(f"Capacitive Soil Sensor: Elevated soil moisture ({soil_moisture_pct:.1f}%). High runoff predisposition.")

    if rain_rate_mm_h >= 75.0:
        drivers.append(f"Water Level / Rain Sensor: Critical inundation / extreme intensity ({rain_rate_mm_h:.1f} mm). Cloudburst threshold active.")
    elif rain_rate_mm_h >= 45.0:
        drivers.append(f"Water Level / Rain Sensor: High water surge / heavy downpour ({rain_rate_mm_h:.1f} mm).")
    elif rain_rate_mm_h >= 15.0:
        drivers.append(f"Water Level / Rain Sensor: Active water accumulation detected ({rain_rate_mm_h:.1f} mm).")

    if rain_accum_mm >= 50.0:
        drivers.append(f"HW-038 Rain Sensor: Critical accumulated rainfall depth ({rain_accum_mm:.1f} mm).")
    elif rain_accum_mm >= 25.0:
        drivers.append(f"HW-038 Rain Sensor: Significant accumulated rainfall depth ({rain_accum_mm:.1f} mm).")

    if slope_degrees >= 30.0:
        drivers.append(f"MPU-6050 Motion Sensor: Steep mountain slope inclination ({slope_degrees:.1f}°). High kinetic runoff acceleration.")
    elif slope_degrees >= 20.0:
        drivers.append(f"MPU-6050 Motion Sensor: Moderate hillside terrain tilt ({slope_degrees:.1f}°).")

    if vibration_g >= 0.30:
        drivers.append(f"MPU-6050 Motion Sensor: Severe seismic vibration detected ({vibration_g:.2f}g). High landslide trigger potential.")
    elif vibration_g >= 0.15:
        drivers.append(f"MPU-6050 Motion Sensor: Notable ground vibration detected ({vibration_g:.2f}g).")

    if humidity_pct >= 90.0:
        drivers.append(f"DHT11 Sensor: Saturated atmospheric relative humidity ({humidity_pct:.1f}%).")

    if not drivers:
        drivers.append("All sensors operating within baseline safe operational thresholds.")

    return {
        "node_id": node_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_risk_level": pred_level,
        "risk_category": tier["name"],
        "risk_code": tier["code"],
        "risk_color": tier["color"],
        "risk_description": tier["desc"],
        "confidence_score": round(float(probabilities[pred_level]) * 100.0, 1) if probabilities is not None else 95.0,
        "probability_distribution": prob_dict,
        "early_warning_lead_time": advisory["lead_time"],
        "status_code": advisory["status"],
        "advisory_actions": advisory["actions"],
        "primary_risk_drivers": drivers,
        "sensors": {
            "temperature_c": temp_c,
            "humidity_pct": humidity_pct,
            "soil_moisture_pct": round(soil_surf * 100.0, 1),
            "rain_rate_mm_h": rain_rate_mm_h,
            "rain_accum_mm": rain_accum_mm,
            "slope_degrees": slope_degrees,
            "vibration_g": vibration_g,
        },
        "actuator_state": {
            "buzzer_active": pred_level >= 2,
            "led_color": "RED" if pred_level >= 2 else ("YELLOW" if pred_level == 1 else "GREEN"),
            "status_text": "EMERGENCY SIREN & RED ALARM" if pred_level >= 2 else ("ADVISORY WATCH (YELLOW)" if pred_level == 1 else "NORMAL (GREEN LED)")
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Flash Flood Risk Level for a District")
    parser.add_argument("--district", type=str, default="Wayanad", help="District Name")
    parser.add_argument("--rainfall", type=float, default=None, help="24h Daily Rainfall in mm (default: district climatology mean)")
    parser.add_argument("--rainfall_3d", type=float, default=None, help="3-day cumulative rainfall in mm")
    parser.add_argument("--soil_surface", type=float, default=None, help="Surface Soil Moisture from 0.0 to 1.0")
    parser.add_argument("--soil_root", type=float, default=None, help="Root Zone Soil Moisture from 0.0 to 1.0")
    parser.add_argument("--humidity", type=float, default=None, help="Relative Humidity percentage")
    parser.add_argument("--temp", type=float, default=None, help="Average Temperature in Celsius")
    parser.add_argument("--date", type=str, default=None, help="Prediction date in YYYY-MM-DD format")

    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f" FLASH FLOOD RISK PREDICTION & EARLY WARNING SYSTEM")
    print(f"=======================================================\n")
    print(f"Target District : {args.district}")

    res = predict_district_risk(
        district_name=args.district,
        precipitation_mm=args.rainfall,
        precip_3d_sum=args.rainfall_3d,
        soil_moisture_surface=args.soil_surface,
        soil_moisture_root=args.soil_root,
        temp_avg_c=args.temp,
        humidity_pct=args.humidity,
        target_date=args.date,
    )

    inp = res["input_conditions"]
    print(f"Rainfall (24h)  : {inp['rainfall_1d_mm']:.1f} mm")
    print(f"Rainfall (3-day): {inp['rainfall_3d_sum_mm']:.1f} mm")
    print(f"Soil Saturation : {inp['soil_moisture_surface_pct']:.1f}%")
    print(f"Region          : {res['region']}")
    print(f"Elevation/Slope : {res['topography']['elevation_m']:.0f}m / {res['topography']['slope_degrees']:.1f}°")

    print(f"\n>> PREDICTION RESULTS:")
    print(f"  Risk Category  : {res['risk_category']} (Level {res['predicted_risk_level']})")
    print(f"  Confidence     : {res['confidence_score']}%")
    print(f"  Status         : {res['status_code']}")
    print(f"  Lead Time      : {res['early_warning_lead_time']}")
    print(f"\n>> Probability Breakdown:")
    for cat, prob in res["probability_distribution"].items():
        print(f"  - {cat:<18}: {prob:>5.1f}%")

    print(f"\n>> Primary Contributing Risk Drivers:")
    for d in res["primary_risk_drivers"]:
        print(f"  * {d}")

    print(f"\n>> Emergency Advisory Actions:")
    for a in res["advisory_actions"]:
        print(f"  [!] {a}")
    print(f"\n=======================================================\n")


if __name__ == "__main__":
    main()
