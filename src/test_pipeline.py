from preprocess import preprocess_pipeline
from train import train_models
from evaluate import evaluate_models

FILE_PATH = "data/ddos_dataset.csv"
LABEL_COLUMN = "Attack Type"


def main():
    X, y, le = preprocess_pipeline(FILE_PATH, LABEL_COLUMN)

    print(f"Classes detected: {list(le.classes_)}")  
    print(f"Dataset shape: {X.shape}")

    models, X_train, y_train, X_test, y_test = train_models(X, y)

    results_df, best_model_name = evaluate_models(models, X_test, y_test)

    print("\nPipeline completed successfully.")
    print(f"Best model selected for deployment: {best_model_name}")
    print(results_df[["Model", "Accuracy", "Precision", "Recall", "F1-score"]])


if __name__ == "__main__":
    main()