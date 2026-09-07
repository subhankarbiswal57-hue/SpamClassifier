import os, sys, re, string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK data is downloaded
try:
    STOP = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOP = set(stopwords.words('english'))

STEMMER = PorterStemmer()

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Remove digits where appropriate
    text = re.sub(r"\d+", " ", text)
    tokens = [STEMMER.stem(w) for w in text.split() if w not in STOP and len(w) > 1]
    return " ".join(tokens)

def load_and_clean(path=None, text_col=None, label_col=None):
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "data", "raw", "spam.csv")
    
    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}")
    
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    
    # Find columns if not provided
    if text_col is None or label_col is None:
        cols_lower = [c.lower() for c in df.columns]
        if "label" in cols_lower and "message" in cols_lower:
            label_col = df.columns[cols_lower.index("label")]
            text_col = df.columns[cols_lower.index("message")]
        elif "v1" in cols_lower and "v2" in cols_lower:
            label_col = df.columns[cols_lower.index("v1")]
            text_col = df.columns[cols_lower.index("v2")]
        elif "label" in cols_lower and "text" in cols_lower:
            label_col = df.columns[cols_lower.index("label")]
            text_col = df.columns[cols_lower.index("text")]
        else:
            label_col, text_col = df.columns[0], df.columns[1]

    df = df[[label_col, text_col]].dropna().drop_duplicates()
    df.columns = ["label", "text"]
    
    # Map labels to 0 (ham/not spam) and 1 (spam)
    label_map = {"ham": 0, "spam": 1, 0: 0, 1: 1, "0": 0, "1": 1, "legitimate": 0}
    df["label"] = df["label"].astype(str).str.lower().map(label_map).fillna(df["label"]).astype(int)
    
    df["clean_text"] = df["text"].apply(clean_text)
    # Remove any empty clean_text entries
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    return df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base_dir, "data", "processed"), exist_ok=True)
    
    df = load_and_clean()
    out_path = os.path.join(base_dir, "data", "processed", "spam_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Cleaned {len(df)} rows -> {out_path}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    main()

