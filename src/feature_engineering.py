"""
Feature Engineering Engine for Flash Flood Risk Prediction.
Calculates hydrological, meteorological, topographical, and seasonal features.
"""

import numpy as np
import pandas as pd


def compute_antecedent_precipitation_index(precip_series, decay=0.85):
    """
    Compute Antecedent Precipitation Index (API):
    API_t = P_t + decay * API_{t-1}
    Standard hydrological metric for soil moisture memory and flood susceptibility.
    """
    n = len(precip_series)
    api = np.zeros(n)
    p_values = precip_series.values
    
    if n > 0:
        api[0] = p_values[0]
        for t in range(1, n):
            api[t] = p_values[t] + decay * api[t - 1]
    return api


def engineer_features(df_input):
    """
    Transform raw daily district time series into high-dimensional hydrological feature set.
    df_input must contain:
    [Region, District, Latitude, Longitude, Elevation_m, Slope_degrees, Date,
     Precipitation_mm, Soil_Moisture_RootZone, Soil_Moisture_Surface,
     Temp_Avg_C, Temp_Max_C, Temp_Min_C, Humidity_pct]
    """
    df = df_input.copy()
    df.sort_values(["District", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1. Temporal & Harmonic Monsoon Seasonality Features
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day_of_Year"] = df["Date"].dt.dayofyear
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    
    # Cyclic Fourier representations of the calendar year
    df["Sin_Day_of_Year"] = np.sin(2 * np.pi * df["Day_of_Year"] / 365.25)
    df["Cos_Day_of_Year"] = np.cos(2 * np.pi * df["Day_of_Year"] / 365.25)
    
    # Monsoon active window indicator (SW Monsoon Jun-Sep, NE Monsoon Oct-Nov)
    df["Is_Monsoon_Season"] = df["Month"].isin([6, 7, 8, 9]).astype(int)
    df["Is_Post_Monsoon"] = df["Month"].isin([10, 11]).astype(int)

    # 2. Group-wise Rolling and Hydrological Lag Features
    engineered_districts = []

    for district, group in df.groupby("District", sort=False):
        g = group.copy()
        p = g["Precipitation_mm"]
        s_surf = g["Soil_Moisture_Surface"]
        s_root = g["Soil_Moisture_RootZone"]
        t_avg = g["Temp_Avg_C"]
        rh = g["Humidity_pct"]
        slope = g["Slope_degrees"].iloc[0]
        elevation = g["Elevation_m"].iloc[0]

        # Antecedent Precipitation Index (API)
        g["Antecedent_Precip_Index"] = compute_antecedent_precipitation_index(p, decay=0.85)

        # Multi-window rolling precipitation sums and statistics
        g["Precip_1d"] = p
        g["Precip_3d_sum"] = p.rolling(window=3, min_periods=1).sum()
        g["Precip_7d_sum"] = p.rolling(window=7, min_periods=1).sum()
        g["Precip_14d_sum"] = p.rolling(window=14, min_periods=1).sum()
        g["Precip_3d_max"] = p.rolling(window=3, min_periods=1).max()
        g["Precip_7d_mean"] = p.rolling(window=7, min_periods=1).mean()
        
        # Lags for early warning lead-time
        g["Precip_lag1"] = p.shift(1).fillna(0.0)
        g["Precip_lag2"] = p.shift(2).fillna(0.0)
        g["Precip_3d_lag1"] = g["Precip_3d_sum"].shift(1).fillna(0.0)

        # Soil Moisture Saturation & Dynamics
        g["Combined_Soil_Saturation"] = 0.65 * s_surf + 0.35 * s_root
        g["Soil_Moisture_Surface_3d_mean"] = s_surf.rolling(window=3, min_periods=1).mean()
        g["Soil_Moisture_Root_7d_mean"] = s_root.rolling(window=7, min_periods=1).mean()
        
        # Soil moisture surges (delta)
        g["Soil_Surface_Delta_24h"] = (s_surf - s_surf.shift(1)).fillna(0.0)
        g["Soil_Surface_Delta_48h"] = (s_surf - s_surf.shift(2)).fillna(0.0)
        
        # Meteorological & Instability Features
        g["Diurnal_Temp_Range"] = g["Temp_Max_C"] - g["Temp_Min_C"]
        g["Temp_Drop_24h"] = (t_avg.shift(1) - t_avg).fillna(0.0)
        g["Humidity_3d_mean"] = rh.rolling(window=3, min_periods=1).mean()
        g["Atmospheric_Moisture_Index"] = (rh / 100.0) * (1.0 + np.clip(t_avg / 30.0, 0, 1.5))

        # Topographical-Hydrological Interactions
        slope_rad = np.radians(slope)
        sin_slope = np.sin(slope_rad)
        g["Sin_Slope"] = sin_slope
        
        # Runoff Energy Index: Precipitation * Soil Saturation * sin(Slope)
        g["Runoff_Energy_Index"] = g["Precip_3d_sum"] * s_surf * sin_slope
        g["Surface_Runoff_Proxy"] = p * np.maximum(0.0, s_surf - 0.5) * 2.0 * (1.0 + sin_slope)
        g["Elevation_Precip_Load"] = (elevation / 1000.0) * p

        # IMD Rainfall threshold indicators
        g["Is_Heavy_Rain"] = (p >= 64.5).astype(int)        # IMD Heavy Rain (>64.5 mm)
        g["Is_Very_Heavy_Rain"] = (p >= 115.5).astype(int)  # IMD Very Heavy Rain (>115.5 mm)
        g["Is_Extremely_Heavy_Rain"] = (p >= 204.4).astype(int) # IMD Extremely Heavy (>204.4 mm)

        engineered_districts.append(g)

    final_df = pd.concat(engineered_districts, ignore_index=True)
    return final_df


def get_feature_column_names():
    """Returns the list of feature column names used for ML model training."""
    return [
        "Latitude",
        "Longitude",
        "Elevation_m",
        "Slope_degrees",
        "Sin_Slope",
        "Day_of_Year",
        "Month",
        "Sin_Day_of_Year",
        "Cos_Day_of_Year",
        "Is_Monsoon_Season",
        "Is_Post_Monsoon",
        "Precipitation_mm",
        "Precip_1d",
        "Precip_3d_sum",
        "Precip_7d_sum",
        "Precip_14d_sum",
        "Precip_3d_max",
        "Precip_7d_mean",
        "Precip_lag1",
        "Precip_lag2",
        "Precip_3d_lag1",
        "Antecedent_Precip_Index",
        "Soil_Moisture_Surface",
        "Soil_Moisture_RootZone",
        "Combined_Soil_Saturation",
        "Soil_Moisture_Surface_3d_mean",
        "Soil_Moisture_Root_7d_mean",
        "Soil_Surface_Delta_24h",
        "Soil_Surface_Delta_48h",
        "Temp_Avg_C",
        "Temp_Max_C",
        "Temp_Min_C",
        "Diurnal_Temp_Range",
        "Temp_Drop_24h",
        "Humidity_pct",
        "Humidity_3d_mean",
        "Atmospheric_Moisture_Index",
        "Runoff_Energy_Index",
        "Surface_Runoff_Proxy",
        "Elevation_Precip_Load",
        "Is_Heavy_Rain",
        "Is_Very_Heavy_Rain",
        "Is_Extremely_Heavy_Rain",
    ]
