"""
Hydrological Flash Flood Risk Labeling & Ground Truth Engine.
Calculates continuous Flash Flood Risk Index (FFRI: 0-1) and multi-class risk tiers (0 to 3)
grounded in Flash Flood Guidance (FFG) principles and physical runoff physics.
"""

import numpy as np
import pandas as pd

RISK_TIERS = {
    0: {"name": "Low Risk", "code": "LOW", "color": "#10B981", "desc": "Normal baseline conditions. No flood threat."},
    1: {"name": "Moderate Risk", "code": "MODERATE", "color": "#F59E0B", "desc": "Elevated moisture/rainfall. Advisory: Monitor local streams and weather updates."},
    2: {"name": "High Risk", "code": "HIGH", "color": "#F97316", "desc": "High saturation and heavy rainfall on steep slopes. Watch: Flash flood likely within 24-48h."},
    3: {"name": "Severe Risk", "code": "SEVERE", "color": "#EF4444", "desc": "Extreme cloudburst/monsoon deluge on saturated soil. Warning: Flash flood/debris flow imminent."},
}


def calculate_flash_flood_risk_index(df):
    """
    Computes a continuous Flash Flood Risk Index (FFRI, 0.0 to 1.0)
    combining 4 core hydrological pillars:
    1. Short-term Precipitation Forcing (Weight: 35%)
    2. Multi-day Cumulative & Antecedent Moisture (Weight: 25%)
    3. Soil Saturation & Runoff Infiltration Deficit (Weight: 25%)
    4. Topographic Acceleration (Slope & Elevation headwater gravity) (Weight: 15%)
    """
    p1d = df["Precip_1d"].clip(0, 300)
    p3d = df["Precip_3d_sum"].clip(0, 500)
    api = df["Antecedent_Precip_Index"].clip(0, 400)
    s_surf = df["Soil_Moisture_Surface"].clip(0, 1)
    s_root = df["Soil_Moisture_RootZone"].clip(0, 1)
    slope = df["Slope_degrees"].clip(0, 50)
    elevation = df["Elevation_m"].clip(0, 5500)
    
    # 1. Normalized short-term precipitation score (0-1)
    # 150mm/day reaches saturation threshold
    score_p1d = 1.0 - np.exp(-p1d / 65.0)
    
    # 2. Normalized cumulative & antecedent precipitation score (0-1)
    score_p3d = 1.0 - np.exp(-p3d / 130.0)
    score_api = 1.0 - np.exp(-api / 110.0)
    score_precip_cum = 0.6 * score_p3d + 0.4 * score_api
    
    # 3. Soil Saturation Score (nonlinear response when saturation exceeds 75%)
    combined_soil = 0.65 * s_surf + 0.35 * s_root
    # Sigmoidal activation around 0.75 saturation threshold
    score_soil = 1.0 / (1.0 + np.exp(-12.0 * (combined_soil - 0.72)))
    
    # 4. Topographical kinetic multiplier
    sin_slope = np.sin(np.radians(slope))
    score_topography = sin_slope * 0.75 + np.clip(elevation / 4500.0, 0, 0.25)
    
    # Core linear-nonlinear blend
    base_index = (
        0.35 * score_p1d +
        0.25 * score_precip_cum +
        0.25 * score_soil +
        0.15 * score_topography
    )
    
    # High-risk compounding coupling:
    # When soil is >85% saturated AND 1-day rain is heavy (>65mm), runoff coefficient spikes to near 100%
    extreme_coupling = (s_surf > 0.82) & (p1d > 60.0)
    surge_boost = np.where(extreme_coupling, 0.18 * score_p1d * (1.0 + sin_slope), 0.0)
    
    # Cloudburst instant multiplier (>120mm on steep slope >20 deg)
    cloudburst_boost = np.where((p1d > 115.0) & (slope > 18.0), 0.15, 0.0)
    
    raw_ffri = base_index + surge_boost + cloudburst_boost
    ffri = np.clip(raw_ffri, 0.0, 1.0)
    return np.round(ffri, 4)


def assign_risk_classes(ffri_series):
    """
    Map continuous FFRI to standard 4-tier disaster management risk levels:
    0: Low Risk (< 0.28)
    1: Moderate Risk (0.28 to 0.58)
    2: High Risk (0.58 to 0.78)
    3: Severe Risk (>= 0.78)
    """
    conditions = [
        ffri_series < 0.28,
        (ffri_series >= 0.28) & (ffri_series < 0.58),
        (ffri_series >= 0.58) & (ffri_series < 0.78),
        ffri_series >= 0.78,
    ]
    choices = [0, 1, 2, 3]
    return np.select(conditions, choices, default=0)


def label_dataset(df):
    """
    Add FFRI and Risk_Level to the dataset.
    """
    df_labeled = df.copy()
    df_labeled["Flash_Flood_Risk_Index"] = calculate_flash_flood_risk_index(df_labeled)
    df_labeled["Risk_Level"] = assign_risk_classes(df_labeled["Flash_Flood_Risk_Index"])
    df_labeled["Risk_Label"] = df_labeled["Risk_Level"].map(lambda x: RISK_TIERS[x]["name"])
    return df_labeled
