import pandas as pd

FILE_PATH = "data/ddos_dataset.csv"


def inspect_dataset(file_path: str):
    df = pd.read_csv(file_path)

    print("First 5 rows:")
    print(df.head(), "\n")

    print("Shape:")
    print(df.shape, "\n")

    print("Columns:")
    print(df.columns.tolist(), "\n")

    print("Data types:")
    print(df.dtypes, "\n")

    print("Missing values:")
    print(df.isnull().sum(), "\n")

    print("Duplicate rows:")
    print(df.duplicated().sum(), "\n")

    print("Possible label columns:")
    for col in df.columns:
        if "label" in col.lower() or "class" in col.lower() or "attack" in col.lower():
            print(f"\nColumn: {col}")
            print(df[col].value_counts())


if __name__ == "__main__":
    inspect_dataset(FILE_PATH)