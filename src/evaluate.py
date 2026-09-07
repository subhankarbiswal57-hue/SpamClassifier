import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{name}.png")
    plt.close()
    return metrics, classification_report(y_test, preds)

def main():
    X_test, y_test = joblib.load("models/test_split.pkl")
    nb = joblib.load("models/naive_bayes_model.pkl")
    lr = joblib.load("models/logistic_regression_model.pkl")

    with open("outputs/metrics_report.txt", "w") as f:
        for name, model in [("nb", nb), ("lr", lr)]:
            metrics, report = evaluate_model(model, X_test, y_test, name)
            f.write(f"=== {name.upper()} ===\n{metrics}\n{report}\n\n")
            print(name, metrics)

if __name__ == "__main__":
    main()
