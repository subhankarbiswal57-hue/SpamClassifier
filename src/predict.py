import os, sys
import joblib

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text

def predict_message(text, model_path=None, vec_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path = os.path.join(base_dir, "models", "naive_bayes_model.pkl")
    if vec_path is None:
        vec_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
        
    model = joblib.load(model_path)
    vec = joblib.load(vec_path)
    
    cleaned = clean_text(text)
    X = vec.transform([cleaned])
    pred = model.predict(X)[0]
    
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0].max())
    else:
        prob = 1.0
        
    label = "Spam" if pred == 1 else "Not Spam"
    return label, round(prob, 4)

if __name__ == "__main__":
    sample_text = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"
    label, conf = predict_message(sample_text)
    print(f"Sample Text: {sample_text}")
    print(f"Prediction: {label} (Confidence: {conf})")

