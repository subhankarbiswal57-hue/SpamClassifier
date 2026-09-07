import pandas as pd
from sklearn.model_selection import train_test_split

def get_splits(csv_path="data/processed/spam_clean.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path).dropna(subset=["clean_text"])
    return train_test_split(
        df["clean_text"], df["label"],
        test_size=test_size, stratify=df["label"], random_state=random_state
    )
