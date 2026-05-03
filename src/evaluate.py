import json
import os
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from preprocess import load_label_encoder  

RESULTS_DIR = "results"
MODEL_DIR = "models"
BEST_MODEL_META_PATH = os.path.join(RESULTS_DIR, "best_model.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")


def evaluate_models(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.DataFrame, str]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    le = load_label_encoder()
    class_names = le.classes_

    rows = []
    best_model_name = None
    best_f1 = -1.0

    for name, model in models.items():
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
        rec = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

        print(f"\n===== {name.upper()} =====")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("\nClassification Report:")
        # CHANGED: use actual class names instead of 0/1
        print(classification_report(y_test, predictions, target_names=class_names, zero_division=0))

        report_dict = classification_report(
            y_test,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        with open(os.path.join(RESULTS_DIR, f"{name}_per_class_metrics.json"), "w") as f:
            json.dump(report_dict, f, indent=2)

        cm = confusion_matrix(y_test, predictions)        

        rows.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "Confusion Matrix": cm.tolist(),
        })

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(xticks_rotation=45)  
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()  # CHANGED: prevent label cutoff
        plt.savefig(os.path.join(RESULTS_DIR, f"{name}_confusion_matrix.png"))
        plt.close()

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    results_df = pd.DataFrame(rows).sort_values(by="F1-score", ascending=False)
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["Accuracy"])
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0.9, 1.0)
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"))
    plt.close()

    best_model = models[best_model_name]
    joblib.dump(best_model, BEST_MODEL_PATH)

    with open(BEST_MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "selection_metric": "weighted_f1"
            },
            f,
            indent=2
        )

    print("\nBest model based on F1-score:")
    print(results_df.iloc[0][["Model", "Accuracy", "Precision", "Recall", "F1-score"]])

    return results_df, best_model_name