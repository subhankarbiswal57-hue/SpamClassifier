import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def fit_vectorizer(train_texts, max_features=3000, save_path="models/tfidf_vectorizer.pkl"):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vec.fit_transform(train_texts)
    joblib.dump(vec, save_path)
    return vec, X

def load_vectorizer(path="models/tfidf_vectorizer.pkl"):
    return joblib.load(path)
