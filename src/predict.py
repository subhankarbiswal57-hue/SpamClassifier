import joblib
from preprocess import clean_text

def predict_message(text, model_path="models/logistic_regression_model.pkl",
                     vec_path="models/tfidf_vectorizer.pkl"):
    model = joblib.load(model_path)
    vec = joblib.load(vec_path)
    X = vec.transform([clean_text(text)])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    return ("Spam" if pred == 1 else "Not Spam"), round(prob, 3)
