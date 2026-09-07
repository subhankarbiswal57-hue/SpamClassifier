# Task 1: SMS / Text Spam Classifier

A production-ready Machine Learning pipeline to classify text messages as **Spam** or **Not Spam (Ham)**. Built according to the specifications in `rnd task.pdf` (Task 1: Spam Classifier).

---

## 📌 1. Objective & Overview
The goal of this project is to build, evaluate, and deploy text classification models capable of distinguishing spam SMS messages from legitimate communications.

The pipeline comprises:
1. **Text Preprocessing**: Lowercasing, punctuation removal, digit removal, tokenization, English stopword filtering, and Porter Stemming.
2. **Feature Extraction**: TF-IDF Vectorization (`max_features=3000`, `ngram_range=(1, 2)`).
3. **Dataset Partitioning**: Stratified 80/20 train-test split (`random_state=42`).
4. **Model Training**: Multi-model training using **Multinomial Naive Bayes** and **Logistic Regression** (`max_iter=1000`).
5. **Evaluation**: Comprehensive metrics including Accuracy, Precision, Recall, F1-Score, Classification Report, and Confusion Matrix plots.
6. **Inference & CLI Application**: Real-time prediction function and interactive terminal application.

---

## 📊 2. Dataset Information
- **Dataset Chosen**: **SMS Spam Collection (UCI Machine Learning Repository)**
  - Source URL: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
  - Raw Location: `data/raw/spam.csv` (5,574 messages)
  - Processed Clean Location: `data/processed/spam_clean.csv` (5,159 deduplicated, cleaned messages)
  - Class Distribution:
    - `0` (Not Spam / Ham): 4,506 samples (87.3%)
    - `1` (Spam): 653 samples (12.7%)

---

## 📂 3. Repository Structure
```
task1_spam_classifier/
├── data/
│   ├── raw/
│   │   └── spam.csv                       # Untouched raw dataset from UCI
│   └── processed/
│       └── spam_clean.csv                 # Cleaned, stemmed, and encoded dataset
├── src/
│   ├── __init__.py
│   ├── preprocess.py                      # Text normalization and NLTK pipeline
│   ├── split.py                           # 80/20 stratified split
│   ├── vectorize.py                       # TF-IDF vectorizer fitting and transform
│   ├── train.py                           # Model training (MultinomialNB & LogisticRegression)
│   ├── evaluate.py                        # Metrics calculation, reporting, & confusion matrix generation
│   └── predict.py                         # Single text prediction function
├── models/
│   ├── tfidf_vectorizer.pkl               # Fitted TF-IDF model
│   ├── naive_bayes_model.pkl              # Trained MultinomialNB model
│   ├── logistic_regression_model.pkl      # Trained LogisticRegression model
│   └── test_split.pkl                     # Serialized test split (X_test, y_test)
├── outputs/
│   ├── confusion_matrix_nb.png            # Confusion matrix plot for Naive Bayes
│   ├── confusion_matrix_lr.png            # Confusion matrix plot for Logistic Regression
│   └── metrics_report.txt                 # Detailed side-by-side evaluation report
├── app.py                                 # Interactive CLI application
├── run_pipeline.py                        # End-to-end automated pipeline script
├── requirements.txt                       # Project dependencies
└── README.md                              # Documentation and benchmarks
```

---

## ⚙️ 4. Installation & Setup

1. **Clone and enter the repository**:
   ```bash
   cd SpamClassifier
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 5. Execution & Usage

### A. Run End-to-End Pipeline (One Command)
To run preprocessing, training, and evaluation in one automated step:
```bash
python run_pipeline.py
```

### B. Run Individual Stages
```bash
# 1. Preprocess raw data
python src/preprocess.py

# 2. Train models
python src/train.py

# 3. Evaluate models and generate artifacts
python src/evaluate.py
```

### C. Run Interactive CLI Application
```bash
python app.py
```
*Tip: In the CLI, type `examples` to run built-in test cases, or input custom text strings to classify.*

---

## 📈 6. Evaluation Results & Model Comparison

Evaluated on the held-out test split (1,032 samples: 901 Not Spam, 131 Spam):

| Metric | Multinomial Naive Bayes (`nb`) | Logistic Regression (`lr`) |
| :--- | :---: | :---: |
| **Accuracy** | **97.58%** (0.9758) | 96.80% (0.9680) |
| **Precision (Spam)** | **100.00%** (1.0000) | 98.04% (0.9804) |
| **Recall (Spam)** | **80.92%** (0.8092) | 76.34% (0.7634) |
| **F1-Score (Spam)** | **89.45%** (0.8945) | 85.84% (0.8584) |

### 🏆 Best Model
**Multinomial Naive Bayes (`MultinomialNB`)** is identified as the best performing model, achieving **100% precision with 0 False Positives** (crucial for spam detection to prevent legitimate emails from being incorrectly marked as spam) and a higher overall **F1-score of 0.8945**.

Confusion Matrix Artifacts:
- `outputs/confusion_matrix_nb.png`
- `outputs/confusion_matrix_lr.png`
- `outputs/metrics_report.txt`

---

## 🧪 7. Sample Predictions
```python
from src.predict import predict_message

label, confidence = predict_message("WINNER!! You have won a 1,000 cash prize! Call now to claim.")
print(f"Prediction: {label} (Confidence: {confidence:.2%})")
# Output: Prediction: Spam (Confidence: 99.67%)

label, confidence = predict_message("Hey, are we still meeting for lunch at 1pm tomorrow?")
print(f"Prediction: {label} (Confidence: {confidence:.2%})")
# Output: Prediction: Not Spam (Confidence: 99.23%)
```
