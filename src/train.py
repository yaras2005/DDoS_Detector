import os
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")


def train_models(X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, object], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            solver='lbfgs'
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=8,
            random_state=42,
            n_jobs=-1,
        ),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    return trained_models, X_train, y_train, X_test, y_test