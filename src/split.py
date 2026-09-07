import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_splits(csv_path=None, test_size=0.2, random_state=42):
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "processed", "spam_clean.csv")
        
    df = pd.read_csv(csv_path).dropna(subset=["clean_text"])
    return train_test_split(
        df["clean_text"], df["label"],
        test_size=test_size, stratify=df["label"], random_state=random_state
    )

