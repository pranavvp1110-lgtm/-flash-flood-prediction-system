"""
Machine Learning Training, Model Comparison & Evaluation Suite for Flash Flood Risk Prediction.
Implements ML Best Practices:
- Strict chronological train (2000-2020), validation (2021-2023), and test (2024-2025) splits
- Independent scaler fitting on train data only
- Multi-model benchmarking: Logistic Regression, Random Forest, HistGradientBoosting, Calibrated Ensemble
- Comprehensive evaluation: Accuracy, Macro/Weighted Precision, Recall, F1, ROC-AUC, Per-Class Precision & Recall, Confusion Matrices, and Feature Importance
- Model persistence with metadata
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_collector import load_master_dataset
from feature_engineering import engineer_features, get_feature_column_names
from risk_labeler import label_dataset, RISK_TIERS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "flood_risk_model.joblib")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.csv")


def prepare_datasets(df_master=None):
    """
    Load dataset, run feature engineering and risk labeling,
    then execute strict chronological train/val/test splitting.
    """
    if df_master is None:
        print("[1/5] Loading historical master dataset...", flush=True)
        df_master = load_master_dataset()

    print("[2/5] Engineering hydrological and topographical features...", flush=True)
    df_feat = engineer_features(df_master)

    print("[3/5] Computing flash flood risk indices and ground truth labels...", flush=True)
    df_labeled = label_dataset(df_feat)

    # Chronological Split
    # Train: 2000 - 2020
    # Val:   2021 - 2023
    # Test:  2024 - 2025
    train_mask = df_labeled["Year"] <= 2020
    val_mask = (df_labeled["Year"] >= 2021) & (df_labeled["Year"] <= 2023)
    test_mask = df_labeled["Year"] >= 2024

    train_df = df_labeled[train_mask].copy()
    val_df = df_labeled[val_mask].copy()
    test_df = df_labeled[test_mask].copy()

    feature_cols = get_feature_column_names()
    target_col = "Risk_Level"

    print(f"\nData Splitting Completed:", flush=True)
    print(f"  - Training records (2000-2020)  : {len(train_df):,} samples", flush=True)
    print(f"  - Validation records (2021-2023): {len(val_df):,} samples", flush=True)
    print(f"  - Test records (2024-2025)      : {len(test_df):,} samples\n", flush=True)

    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_val_raw = val_df[feature_cols].values
    y_val = val_df[target_col].values

    X_test_raw = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Fit scaler strictly on X_train to prevent temporal/lookahead data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    return {
        "X_train": X_train, "y_train": y_train, "train_df": train_df,
        "X_val": X_val, "y_val": y_val, "val_df": val_df,
        "X_test": X_test, "y_test": y_test, "test_df": test_df,
        "scaler": scaler, "feature_names": feature_cols,
    }


def evaluate_model(model, X, y, dataset_name="Validation"):
    """
    Compute comprehensive multi-class evaluation metrics including per-tier Precision and Recall.
    """
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, preds, average="weighted", zero_division=0)
    precision_macro = precision_score(y, preds, average="macro", zero_division=0)
    recall_macro = recall_score(y, preds, average="macro", zero_division=0)
    precision_weighted = precision_score(y, preds, average="weighted", zero_division=0)
    recall_weighted = recall_score(y, preds, average="weighted", zero_division=0)

    # Per-class metrics
    class_precisions = precision_score(y, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    class_recalls = recall_score(y, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    class_f1s = f1_score(y, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)

    # Early Warning Binary Flood Threat Metrics (Level >= 2: High or Severe Flood Threat)
    y_threat_binary = (y >= 2).astype(int)
    preds_threat_binary = (preds >= 2).astype(int)
    threat_precision = precision_score(y_threat_binary, preds_threat_binary, zero_division=0)
    threat_recall = recall_score(y_threat_binary, preds_threat_binary, zero_division=0)
    threat_f1 = f1_score(y_threat_binary, preds_threat_binary, zero_division=0)

    roc_auc = None
    if probs is not None and probs.shape[1] > 1:
        try:
            roc_auc = roc_auc_score(y, probs, multi_class="ovr", average="weighted")
        except Exception:
            roc_auc = None

    cm = confusion_matrix(y, preds, labels=[0, 1, 2, 3]).tolist()
    report = classification_report(
        y, preds,
        labels=[0, 1, 2, 3],
        target_names=[RISK_TIERS[i]["name"] for i in range(4)],
        output_dict=True,
        zero_division=0
    )

    per_class_summary = {}
    for i in range(4):
        tier_name = RISK_TIERS[i]["name"]
        per_class_summary[tier_name] = {
            "tier_level": i,
            "precision": round(float(class_precisions[i]), 4),
            "recall": round(float(class_recalls[i]), 4),
            "f1_score": round(float(class_f1s[i]), 4),
            "support": int(np.sum(y == i)),
        }

    metrics = {
        "dataset": dataset_name,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "early_warning_high_severe_threat": {
            "threat_precision": round(float(threat_precision), 4),
            "threat_recall": round(float(threat_recall), 4),
            "threat_f1": round(float(threat_f1), 4),
        },
        "per_class_metrics": per_class_summary,
        "roc_auc_ovr": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return metrics, preds, probs


def print_detailed_evaluation(model_name, dataset_name, metrics):
    """Prints a clean, formatted evaluation table with precision and recall."""
    print(f"\n{'='*70}", flush=True)
    print(f" MODEL EVALUATION REPORT: {model_name} [{dataset_name} Set]", flush=True)
    print(f"{'='*70}", flush=True)
    print(f" Overall Accuracy      : {metrics['accuracy'] * 100:.2f}%", flush=True)
    print(f" Macro Precision       : {metrics['precision_macro'] * 100:.2f}%", flush=True)
    print(f" Macro Recall          : {metrics['recall_macro'] * 100:.2f}%", flush=True)
    print(f" Macro F1 Score        : {metrics['macro_f1']:.4f}", flush=True)
    print(f" Weighted F1 Score     : {metrics['weighted_f1']:.4f}", flush=True)
    if metrics['roc_auc_ovr']:
        print(f" Multi-Class ROC-AUC   : {metrics['roc_auc_ovr']:.4f}", flush=True)

    ew = metrics["early_warning_high_severe_threat"]
    print(f"\n--- Early Flash Flood Threat Detection (High/Severe Levels 2 & 3) ---", flush=True)
    print(f" Early Threat Precision : {ew['threat_precision'] * 100:.2f}%", flush=True)
    print(f" Early Threat Recall    : {ew['threat_recall'] * 100:.2f}%", flush=True)
    print(f" Early Threat F1 Score  : {ew['threat_f1']:.4f}", flush=True)

    print(f"\n--- Per-Class Precision & Recall Breakdown ---", flush=True)
    print(f" {'Risk Tier':<18} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'Samples':<8}", flush=True)
    print(f" {'-'*18}-|-{'-'*11}-|-{'-'*11}-|-{'-'*11}-|-{'-'*8}", flush=True)
    for tier_name, pcm in metrics["per_class_metrics"].items():
        print(f" {tier_name:<18} | {pcm['precision']*100:>9.2f}% | {pcm['recall']*100:>9.2f}% | {pcm['f1_score']:>10.4f} | {pcm['support']:>8,}", flush=True)
    print(f"{'='*70}\n", flush=True)


def train_and_compare_models(data_dict):
    """
    Train and benchmark multiple AI architectures.
    """
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val = data_dict["X_val"], data_dict["y_val"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]
    feature_names = data_dict["feature_names"]

    print("\n[4/5] Training and Benchmarking Candidate ML Architectures...", flush=True)

    # Define candidate models
    candidate_models = {
        "Logistic_Regression_Baseline": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=120, max_depth=16, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(max_iter=120, max_depth=10, learning_rate=0.08, min_samples_leaf=20, random_state=42),
    }

    # Calibrated Voting Ensemble
    rf = candidate_models["Random_Forest"]
    hgb = candidate_models["Hist_Gradient_Boosting"]
    
    ensemble_clf = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("hgb", hgb),
        ],
        voting="soft",
        n_jobs=-1
    )
    candidate_models["Calibrated_Voting_Ensemble"] = ensemble_clf

    comparison_results = {}
    fitted_models = {}

    for name, model in candidate_models.items():
        print(f"  --> Training {name}...", flush=True)
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Evaluate on validation set
        val_metrics, _, _ = evaluate_model(model, X_val, y_val, dataset_name="Validation")
        # Evaluate on test set
        test_metrics, _, _ = evaluate_model(model, X_test, y_test, dataset_name="Test")

        comparison_results[name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

        print(f"      [Val]  Accuracy: {val_metrics['accuracy']*100:.2f}% | Macro F1: {val_metrics['macro_f1']:.4f} | Threat Recall: {val_metrics['early_warning_high_severe_threat']['threat_recall']*100:.2f}%", flush=True)
        print(f"      [Test] Accuracy: {test_metrics['accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']:.4f} | Threat Recall: {test_metrics['early_warning_high_severe_threat']['threat_recall']*100:.2f}%", flush=True)

    # Select best model based on validation Macro F1 score
    best_model_name = max(comparison_results, key=lambda k: comparison_results[k]["validation"]["macro_f1"])
    best_model = fitted_models[best_model_name]
    print(f"\n>> Selected Winning AI Model: '{best_model_name}'", flush=True)

    # Print in-depth evaluation on the unseen Test set (2024-2025)
    print_detailed_evaluation(best_model_name, "Validation (2021-2023)", comparison_results[best_model_name]["validation"])
    print_detailed_evaluation(best_model_name, "Holdout Test (2024-2025)", comparison_results[best_model_name]["test"])

    # Extract Feature Importances from Random Forest
    rf_model = fitted_models["Random_Forest"]
    feature_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)
    feature_importances.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Saved feature importances to {FEATURE_IMPORTANCE_PATH}", flush=True)

    # Package and save winning model bundle
    print(f"\n[5/5] Serializing Winning Model Bundle & Metadata to {MODEL_SAVE_PATH}...", flush=True)
    model_bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "scaler": data_dict["scaler"],
        "feature_names": feature_names,
        "risk_tiers": RISK_TIERS,
        "version": "1.0.0",
    }
    joblib.dump(model_bundle, MODEL_SAVE_PATH)

    # Save metrics JSON
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved comprehensive metrics to {METRICS_SAVE_PATH}", flush=True)

    # Save best confusion matrix
    cm_payload = {
        "best_model": best_model_name,
        "classes": [RISK_TIERS[i]["name"] for i in range(4)],
        "val_confusion_matrix": comparison_results[best_model_name]["validation"]["confusion_matrix"],
        "test_confusion_matrix": comparison_results[best_model_name]["test"]["confusion_matrix"],
    }
    with open(CONFUSION_MATRIX_PATH, "w") as f:
        json.dump(cm_payload, f, indent=2)
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}", flush=True)

    print(f"\n====================================================================", flush=True)
    print(f" AI Model Training, Benchmarking & Persistence Completed Successfully!", flush=True)
    print(f"====================================================================\n", flush=True)
    return best_model_name, comparison_results, model_bundle


if __name__ == "__main__":
    data_dict = prepare_datasets()
    best_name, results, bundle = train_and_compare_models(data_dict)
