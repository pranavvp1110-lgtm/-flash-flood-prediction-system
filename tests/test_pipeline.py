"""
Unit and Integration Tests for Flash Flood Risk Prediction Pipeline.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add src to python path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_collector import load_districts, load_climatology_summary
from feature_engineering import engineer_features, get_feature_column_names, compute_antecedent_precipitation_index
from risk_labeler import calculate_flash_flood_risk_index, assign_risk_classes, label_dataset
from predict import predict_district_risk, get_district_metadata, load_model


class TestFloodRiskPipeline(unittest.TestCase):

    def test_district_metadata(self):
        """Test that all 34 districts are properly loaded with valid coordinates, topography, and climatology."""
        df = load_districts()
        self.assertEqual(len(df), 34, "Expected exactly 34 monitored districts.")
        
        required_cols = ["Region", "District", "Latitude", "Longitude", "Elevation_m", "Slope_degrees"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column {col} in district metadata.")
        
        # Verify valid geographical and physical ranges
        self.assertTrue((df["Latitude"] >= 8.0).all() and (df["Latitude"] <= 38.0).all())
        self.assertTrue((df["Longitude"] >= 70.0).all() and (df["Longitude"] <= 98.0).all())
        self.assertTrue((df["Elevation_m"] >= 0.0).all() and (df["Elevation_m"] <= 8848.0).all())
        self.assertTrue((df["Slope_degrees"] >= 0.0).all() and (df["Slope_degrees"] <= 90.0).all())

    def test_climatology_summary_data(self):
        """Test that district_climatology_summary.csv loads properly with all 34 districts."""
        clim_df = load_climatology_summary()
        self.assertIsNotNone(clim_df, "district_climatology_summary.csv should be loaded successfully.")
        self.assertEqual(len(clim_df), 34, "Climatology table should contain all 34 districts.")
        
        expected_cols = [
            "District", "Region", "Soil_Moisture_RootZone_mean", "Soil_Moisture_Surface_mean",
            "Rainfall_3day_mean", "Rainfall_3day_max", "Rainfall_7day_mean", "Rainfall_7day_max",
            "Temp_Avg_C", "Temp_Max_C", "Temp_Min_C", "Elevation_m", "Slope_degrees"
        ]
        for col in expected_cols:
            self.assertIn(col, clim_df.columns, f"Missing column {col} in climatology summary.")

    def test_antecedent_precipitation_index(self):
        """Test API calculation behaves with decay factor."""
        precip = pd.Series([10.0, 0.0, 0.0, 20.0])
        api = compute_antecedent_precipitation_index(precip, decay=0.8)
        self.assertAlmostEqual(api[0], 10.0)
        self.assertAlmostEqual(api[1], 8.0)
        self.assertAlmostEqual(api[2], 6.4)
        self.assertAlmostEqual(api[3], 20.0 + 0.8 * 6.4)

    def test_feature_engineering_pipeline(self):
        """Test that feature engineering creates all required features without remaining NaNs."""
        dates = pd.date_range("2023-07-01", periods=10, freq="D")
        sample_df = pd.DataFrame({
            "Region": "Kerala Western Ghats",
            "District": "Wayanad",
            "Latitude": 11.75,
            "Longitude": 76.125,
            "Elevation_m": 788.0,
            "Slope_degrees": 9.35,
            "Date": dates,
            "Precipitation_mm": [5.0, 10.0, 45.0, 120.0, 80.0, 20.0, 5.0, 0.0, 15.0, 60.0],
            "Soil_Moisture_RootZone": [0.4, 0.45, 0.6, 0.85, 0.9, 0.82, 0.75, 0.7, 0.72, 0.8],
            "Soil_Moisture_Surface": [0.35, 0.48, 0.7, 0.95, 0.98, 0.88, 0.78, 0.65, 0.72, 0.85],
            "Temp_Avg_C": [24.0] * 10,
            "Temp_Max_C": [28.0] * 10,
            "Temp_Min_C": [20.0] * 10,
            "Humidity_pct": [85.0] * 10,
        })
        
        feat_df = engineer_features(sample_df)
        feature_cols = get_feature_column_names()
        
        for col in feature_cols:
            self.assertIn(col, feat_df.columns, f"Expected feature column '{col}' missing.")
            self.assertFalse(feat_df[col].isna().any(), f"Feature '{col}' contains NaN values.")

    def test_risk_labeling(self):
        """Test risk labeling maps index to 0-3 properly."""
        extreme_df = pd.DataFrame({
            "Precip_1d": [180.0, 220.0, 250.0],
            "Precip_3d_sum": [300.0, 400.0, 470.0],
            "Antecedent_Precip_Index": [150.0, 220.0, 300.0],
            "Soil_Moisture_Surface": [0.95, 0.96, 0.98],
            "Soil_Moisture_RootZone": [0.90, 0.92, 0.94],
            "Slope_degrees": [35.0, 35.0, 35.0],
            "Elevation_m": [2500.0, 2500.0, 2500.0],
        })
        ffri_extreme = calculate_flash_flood_risk_index(extreme_df)
        self.assertTrue((ffri_extreme >= 0.80).all(), "Severe scenario should yield FFRI >= 0.80")
        
        classes_extreme = assign_risk_classes(ffri_extreme)
        self.assertTrue((classes_extreme == 3).all(), "Severe scenario should be classified as Level 3 (Severe).")

        safe_df = pd.DataFrame({
            "Precip_1d": [0.0, 0.0, 1.0],
            "Precip_3d_sum": [0.0, 0.0, 1.0],
            "Antecedent_Precip_Index": [2.0, 1.6, 2.28],
            "Soil_Moisture_Surface": [0.20, 0.18, 0.22],
            "Soil_Moisture_RootZone": [0.25, 0.24, 0.25],
            "Slope_degrees": [5.0, 5.0, 5.0],
            "Elevation_m": [200.0, 200.0, 200.0],
        })
        ffri_safe = calculate_flash_flood_risk_index(safe_df)
        self.assertTrue((ffri_safe < 0.28).all(), "Dry safe scenario should yield FFRI < 0.28")
        
        classes_safe = assign_risk_classes(ffri_safe)
        self.assertTrue((classes_safe == 0).all(), "Dry safe scenario should be classified as Level 0 (Low).")

    def test_end_to_end_prediction(self):
        """Test real-time prediction and risk attribution for a district."""
        res = predict_district_risk(
            district_name="Wayanad",
            precipitation_mm=130.0,
            precip_3d_sum=250.0,
            soil_moisture_surface=0.92,
            soil_moisture_root=0.88,
        )
        self.assertIn("predicted_risk_level", res)
        self.assertIn(res["predicted_risk_level"], [0, 1, 2, 3])
        self.assertGreaterEqual(res["predicted_risk_level"], 2, "High rainfall/saturation on Wayanad should trigger High or Severe risk.")
        self.assertIn("confidence_score", res)
        self.assertIn("primary_risk_drivers", res)
        self.assertTrue(len(res["primary_risk_drivers"]) > 0)
        self.assertIn("climatology_baseline", res)

    def test_sensor_telemetry_prediction(self):
        """Test direct real-time ESP32 IoT sensor prediction without district dependency."""
        from predict import predict_sensor_telemetry_risk
        
        # Test extreme sensor reading
        severe_res = predict_sensor_telemetry_risk(
            temp_c=22.0,
            humidity_pct=95.0,
            soil_moisture_pct=92.0,
            rain_rate_mm_h=70.0,
            rain_accum_mm=130.0,
            slope_degrees=28.0,
            vibration_g=0.18
        )
        self.assertEqual(severe_res["predicted_risk_level"], 3, "Extreme sensor values should yield Level 3 Severe Risk.")
        self.assertTrue(severe_res["actuator_state"]["buzzer_active"], "Siren buzzer should be active during severe flood alert.")
        self.assertEqual(severe_res["actuator_state"]["led_color"], "RED")
        self.assertTrue(len(severe_res["primary_risk_drivers"]) > 0)

        # Test safe sensor reading
        safe_res = predict_sensor_telemetry_risk(
            temp_c=26.0,
            humidity_pct=55.0,
            soil_moisture_pct=30.0,
            rain_rate_mm_h=0.0,
            rain_accum_mm=0.0,
            slope_degrees=8.0,
            vibration_g=0.0
        )
        self.assertEqual(safe_res["predicted_risk_level"], 0, "Normal dry sensor values should yield Level 0 Low Risk.")
        self.assertFalse(safe_res["actuator_state"]["buzzer_active"])
        self.assertEqual(safe_res["actuator_state"]["led_color"], "GREEN")

    def test_telemetry_database_storage(self):
        """Test SQLite database initialization, insertion, querying, and CSV export."""
        import tempfile
        import telemetry_db

        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = os.path.join(tmpdir, "test_sensors.db")
            test_csv = os.path.join(tmpdir, "test_sensors.csv")

            telemetry_db.init_db(db_path=test_db, csv_path=test_csv)
            self.assertTrue(os.path.exists(test_db))
            self.assertTrue(os.path.exists(test_csv))

            sample_packet = {
                "timestamp": "2026-09-21 14:00:00",
                "node_id": "TEST_ESP32_NODE",
                "status": "ONLINE",
                "sensors": {
                    "temperature_c": 28.5,
                    "humidity_pct": 72.0,
                    "soil_moisture_pct": 65.0,
                    "soil_moisture_raw": 2200,
                    "rain_rate_mm_h": 18.0,
                    "rain_accum_mm": 12.5,
                    "rain_intensity_pct": 18.0,
                    "slope_degrees": 14.2,
                    "vibration_g": 0.04
                },
                "ai_prediction": {
                    "predicted_risk_level": 1,
                    "risk_category": "Moderate Risk",
                    "risk_code": "MODERATE",
                    "risk_color": "#F59E0B",
                    "confidence_score": 88.5,
                    "status_code": "ADVISORY / ELEVATED WATCH",
                    "early_warning_lead_time": "48h Early Advisory",
                    "primary_risk_drivers": ["Active rainfall detected (18.0 mm/h)."],
                    "advisory_actions": ["Inspect critical culverts."],
                    "actuator_state": {
                        "buzzer_active": False,
                        "led_color": "YELLOW"
                    }
                }
            }

            rec_id = telemetry_db.insert_telemetry_reading(sample_packet, db_path=test_db, csv_path=test_csv)
            self.assertIsNotNone(rec_id)
            self.assertGreater(rec_id, 0)

            # Query recent readings
            readings = telemetry_db.get_recent_readings(limit=10, db_path=test_db)
            self.assertEqual(len(readings), 1)
            self.assertEqual(readings[0]["node_id"], "TEST_ESP32_NODE")
            self.assertEqual(readings[0]["temperature_c"], 28.5)
            self.assertEqual(readings[0]["predicted_risk_level"], 1)

            # Stream packets
            stream_packets = telemetry_db.get_recent_packets_for_stream(limit=10, db_path=test_db)
            self.assertEqual(len(stream_packets), 1)
            self.assertEqual(stream_packets[0]["sensors"]["temperature_c"], 28.5)

            # Stats
            stats = telemetry_db.get_telemetry_stats(db_path=test_db)
            self.assertEqual(stats["total_readings"], 1)
            self.assertEqual(stats["unique_nodes"], 1)
            self.assertEqual(stats["avg_temperature_c"], 28.5)

            # CSV file check
            with open(test_csv, "r", encoding="utf-8") as f:
                csv_lines = f.readlines()
            self.assertEqual(len(csv_lines), 2, "CSV should contain header line + 1 record line.")
            self.assertIn("TEST_ESP32_NODE", csv_lines[1])

    def test_iot_server_telemetry_pipeline(self):
        """Test iot_server process_telemetry_packet end-to-end integration."""
        from iot_server import process_telemetry_packet

        raw_telemetry = {
            "node_id": "ESP32_UNIT_TEST",
            "temperature_c": 23.4,
            "humidity_pct": 82.0,
            "soil_moisture_pct": 45.0,
            "rain_rate_mm_h": 0.0,
            "rain_accum_mm": 0.0,
            "slope_degrees": 10.0,
            "vibration_g": 0.01
        }

        resp = process_telemetry_packet(raw_telemetry)
        self.assertEqual(resp["status"], "success")
        self.assertEqual(resp["node_id"], "ESP32_UNIT_TEST")
        self.assertIn("predicted_risk_level", resp)
        self.assertIn("actuator_state", resp) if "actuator_state" in resp else self.assertIn("buzzer_active", resp)


if __name__ == "__main__":
    unittest.main()
