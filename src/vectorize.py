import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def fit_vectorizer(train_texts, max_features=3000, save_path=None):
    if save_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vec.fit_transform(train_texts)
    joblib.dump(vec, save_path)
    return vec, X

def load_vectorizer(path=None):
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
    return joblib.load(path)

