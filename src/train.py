import os, sys
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from split import get_splits
from vectorize import fit_vectorizer

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Splitting dataset...")
    X_train_txt, X_test_txt, y_train, y_test = get_splits()
    
    print("Vectorizing training text using TF-IDF (max_features=3000, ngram_range=(1,2))...")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    vec, X_train = fit_vectorizer(X_train_txt, save_path=vec_path)
    X_test = vec.transform(X_test_txt)

    print("Training Multinomial Naive Bayes model...")
    nb = MultinomialNB().fit(X_train, y_train)
    
    print("Training Logistic Regression model...")
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)

    joblib.dump(nb, os.path.join(models_dir, "naive_bayes_model.pkl"))
    joblib.dump(lr, os.path.join(models_dir, "logistic_regression_model.pkl"))
    joblib.dump((X_test, y_test), os.path.join(models_dir, "test_split.pkl"))
    print(f"Training complete. Models saved to {models_dir}")

if __name__ == "__main__":
    main()

