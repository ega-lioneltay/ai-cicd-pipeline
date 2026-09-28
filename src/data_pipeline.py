import pandas as pd
from sklearn.datasets import load_wine

def ingest() -> pd.DataFrame:
    """Automate data collection from source."""
    raw = load_wine(as_frame=True)
    df = raw.frame
    df.to_csv("data/raw.csv", index=False)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers and normalize inputs."""
    df = df.dropna()
    for col in df.columns.drop("target"):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    return df

def validate(df: pd.DataFrame) -> bool:
    """Verify data meets quality thresholds."""
    if df.isnull().values.any():
        return False
    if len(df) < 50:
        return False
    if not set(df["target"].unique()).issubset({0, 1, 2}):
        return False
    return True

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for model consumption."""
    features = df.drop(columns=["target"])
    normalized = (features - features.mean()) / features.std()
    normalized["target"] = df["target"].values
    return normalized

if __name__ == "__main__":
    df = ingest()
    df = clean(df)
    assert validate(df), "Data failed validation checks"
    df = transform(df)
    df.to_csv("data/processed.csv", index=False)
    print(f"Processed {len(df)} rows.")