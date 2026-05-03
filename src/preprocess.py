import json
import os
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

ARTIFACT_DIR = "artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor_columns.json")
LABEL_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "label_encoder.pkl")


def load_data(file_path: str, sample_size: Optional[int] = 100000, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().dropna()
    return df


def encode_labels(y: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y.astype(str).str.strip()))
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(le, LABEL_ENCODER_PATH)
    return y_encoded, le


def split_features_labels(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[label_column])
    y = df[label_column]  
    return X, y


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X, drop_first=True)


def save_feature_columns(columns: List[str]) -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(PREPROCESSOR_PATH, "w", encoding="utf-8") as f:
        json.dump(columns, f, indent=2)


def load_feature_columns() -> List[str]:
    with open(PREPROCESSOR_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_encoder() -> LabelEncoder:
    return joblib.load(LABEL_ENCODER_PATH)


def align_features_to_training_schema(X: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=expected_columns, fill_value=0)
    return X


def preprocess_pipeline(file_path: str, label_column: str, sample_size: Optional[int] = 100000) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    df = load_data(file_path, sample_size=sample_size)
    df = basic_cleaning(df)
    X, y = split_features_labels(df, label_column)
    X = encode_categorical_features(X)
    save_feature_columns(list(X.columns))
    y_encoded, le = encode_labels(y)  
    return X, y_encoded, le


def preprocess_uploaded_data(df: pd.DataFrame, label_column: Optional[str] = None) -> pd.DataFrame:
    df = basic_cleaning(df)
    if label_column and label_column in df.columns:
        df = df.drop(columns=[label_column])
    expected_columns = load_feature_columns()
    return align_features_to_training_schema(df, expected_columns)