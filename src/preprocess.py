import re, string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = [STEMMER.stem(w) for w in text.split() if w not in STOP and len(w) > 1]
    return " ".join(tokens)

def load_and_clean(path="data/raw/spam.csv", text_col="message", label_col="label"):
    df = pd.read_csv(path, encoding="latin-1")[[label_col, text_col]].dropna().drop_duplicates()
    df.columns = ["label", "text"]
    df["clean_text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).fillna(df["label"])
    return df

def main():
    df = load_and_clean()
    df.to_csv("data/processed/spam_clean.csv", index=False)
    print(f"Cleaned {len(df)} rows -> data/processed/spam_clean.csv")

if __name__ == "__main__":
    main()
