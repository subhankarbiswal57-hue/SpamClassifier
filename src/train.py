import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from split import get_splits
from vectorize import fit_vectorizer

def main():
    X_train_txt, X_test_txt, y_train, y_test = get_splits()
    vec, X_train = fit_vectorizer(X_train_txt)
    X_test = vec.transform(X_test_txt)

    nb = MultinomialNB().fit(X_train, y_train)
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    joblib.dump(nb, "models/naive_bayes_model.pkl")
    joblib.dump(lr, "models/logistic_regression_model.pkl")
    joblib.dump((X_test, y_test), "models/test_split.pkl")
    print("Training complete. Models saved to models/")

if __name__ == "__main__":
    main()
