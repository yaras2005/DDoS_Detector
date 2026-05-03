import json
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.detection_rules import rule_based_screening
from src.preprocess import preprocess_uploaded_data, load_label_encoder

MODEL_DIR = "models"
RESULTS_DIR = "results"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
BEST_MODEL_META_PATH = os.path.join(RESULTS_DIR, "best_model.json")


def load_best_model():
    return joblib.load(BEST_MODEL_PATH)


def load_best_model_name() -> str:
    if not os.path.exists(BEST_MODEL_META_PATH):
        return "best_model"
    with open(BEST_MODEL_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("best_model", "best_model")


def analyze_uploaded_file(df: pd.DataFrame, label_column: str = None, attack_threshold=0.30) -> Dict[str, object]:
    rule_result = rule_based_screening(df)
    X = preprocess_uploaded_data(df, label_column=label_column)
    model = load_best_model()
    model_name = load_best_model_name()

    le = load_label_encoder()

    predictions_encoded = model.predict(X)
    predictions_labels = le.inverse_transform(predictions_encoded)

    total_rows = len(predictions_labels)
    class_counts = {
        str(cls): int((predictions_labels == cls).sum())
        for cls in le.classes_
    }

    normal_class = "Normal Traffic" if "Normal Traffic" in le.classes_ else le.classes_[0]
    normal_count = int((predictions_labels == normal_class).sum())
    attack_count = total_rows - normal_count
    attack_ratio = attack_count / max(total_rows, 1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        normal_idx = list(le.classes_).index(normal_class)
        attack_proba = 1 - proba[:, normal_idx]
        confidence = float(attack_proba.mean())
    else:
        confidence = float(attack_ratio)

    ml_flag = attack_ratio >= attack_threshold
    final_flag = bool(rule_result["rule_flag"] or ml_flag)

    if final_flag:
        non_normal = [p for p in predictions_labels if p != normal_class]
        if non_normal:
            dominant_attack = max(set(non_normal), key=non_normal.count)
            final_label = f"Potential {dominant_attack} activity detected"
        else:
            final_label = "Potential DDoS activity detected"
    else:
        final_label = "Traffic appears normal"

    return {
        "final_label": final_label,
        "final_flag": final_flag,
        "best_model": model_name,
        "ml_flag": ml_flag,
        "rule_flag": rule_result["rule_flag"],
        "confidence": round(confidence * 100, 2),
        "attack_count": attack_count,
        "normal_count": normal_count,
        "total_rows": total_rows,
        "attack_ratio": round(attack_ratio * 100, 2),
        "rule_score": rule_result["rule_score"],
        "rule_reason": rule_result["reason"],
        "suspicious_columns": rule_result.get("suspicious_columns", []),
        "class_counts": class_counts,
        "predictions_labels": predictions_labels.tolist(),
    }