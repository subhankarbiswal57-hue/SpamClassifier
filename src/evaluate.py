import os, sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

def evaluate_model(model, X_test, y_test, name, out_dir):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }
    cm = confusion_matrix(y_test, preds)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Not Spam (0)", "Spam (1)"], 
                yticklabels=["Not Spam (0)", "Spam (1)"])
    plt.title(f"Confusion Matrix - {name.upper()}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix_{name.lower()}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    report = classification_report(y_test, preds, target_names=["Not Spam", "Spam"])
    return metrics, report, cm

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    X_test, y_test = joblib.load(os.path.join(models_dir, "test_split.pkl"))
    nb = joblib.load(os.path.join(models_dir, "naive_bayes_model.pkl"))
    lr = joblib.load(os.path.join(models_dir, "logistic_regression_model.pkl"))

    results = {}
    reports = {}
    cms = {}

    for name, model in [("nb", nb), ("lr", lr)]:
        metrics, report, cm = evaluate_model(model, X_test, y_test, name, out_dir)
        results[name] = metrics
        reports[name] = report
        cms[name] = cm

    # Determine best model based on F1-score
    best_model = max(results.keys(), key=lambda k: (results[k]["f1"], results[k]["accuracy"]))
    best_name = "Naive Bayes (MultinomialNB)" if best_model == "nb" else "Logistic Regression"

    report_content = []
    report_content.append("=" * 65)
    report_content.append("TASK 1: SPAM CLASSIFIER - MODEL EVALUATION REPORT")
    report_content.append("=" * 65)
    report_content.append("")
    report_content.append("SIDE-BY-SIDE METRICS COMPARISON:")
    report_content.append(f"{'Metric':<15} | {'Multinomial NB (nb)':<22} | {'Logistic Regression (lr)':<24}")
    report_content.append("-" * 65)
    for m in ["accuracy", "precision", "recall", "f1"]:
        report_content.append(f"{m.capitalize():<15} | {results['nb'][m]:<22.4f} | {results['lr'][m]:<24.4f}")
    report_content.append("-" * 65)
    report_content.append(f"BEST MODEL IDENTIFIED: {best_name} (Higher F1-Score: {results[best_model]['f1']:.4f})\n")

    for name, full_name in [("nb", "Multinomial Naive Bayes"), ("lr", "Logistic Regression")]:
        report_content.append("=" * 65)
        report_content.append(f"DETAILED REPORT: {full_name.upper()} ({name.upper()})")
        report_content.append("=" * 65)
        report_content.append("Metrics:")
        for k, v in results[name].items():
            report_content.append(f"  {k.capitalize()}: {v:.4f}")
        report_content.append("\nConfusion Matrix:")
        report_content.append(f"  TN: {cms[name][0,0]}, FP: {cms[name][0,1]}")
        report_content.append(f"  FN: {cms[name][1,0]}, TP: {cms[name][1,1]}")
        report_content.append("\nClassification Report:")
        report_content.append(reports[name])
        report_content.append("")

    report_text = "\n".join(report_content)
    
    metrics_path = os.path.join(out_dir, "metrics_report.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nEvaluation complete. Artifacts saved in {out_dir}")

if __name__ == "__main__":
    main()

