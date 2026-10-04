# 💳 Fraud Detection using Logistic Regression

A machine learning project that detects potentially fraudulent credit card transactions using **Logistic Regression**. The project includes data preprocessing, feature scaling, model training, and evaluation using multiple classification metrics.

## 📌 Project Overview

Credit card fraud detection is a **binary classification** problem where transactions need to be classified as either legitimate or fraudulent.

In this project, **Logistic Regression** is trained on transaction data to predict whether a transaction belongs to the fraud class.

The model is evaluated using metrics such as Precision, Recall, F1-Score, and ROC-AUC, which are especially important for fraud detection because fraudulent transactions are typically much less frequent than legitimate transactions.

## 🎯 Objectives

* Load and explore the credit card transaction dataset
* Separate features and target variable
* Split the data into training and testing sets
* Apply feature scaling using StandardScaler
* Train a Logistic Regression classifier
* Generate fraud predictions and probabilities
* Evaluate the model using multiple classification metrics
* Visualize the Confusion Matrix
* Visualize the ROC Curve

## 📊 Dataset

The project uses a **credit card transaction dataset** containing transaction-related features and a target column named `Class`.

The target variable represents:

* `0` — Legitimate transaction
* `1` — Fraudulent transaction

The dataset is highly imbalanced, making metrics such as **Precision, Recall, F1-Score, and ROC-AUC** important for evaluating the model.

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification.

In this project, it predicts whether a transaction is:

```text
0 → Legitimate
1 → Fraudulent
```

## 🔄 Machine Learning Workflow

```text
Credit Card Dataset
        ↓
Data Loading
        ↓
Feature / Target Separation
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Logistic Regression
        ↓
Predictions
        ↓
Classification Metrics
        ↓
Confusion Matrix + ROC Curve
```

## 📏 Evaluation Metrics

The model is evaluated using:

* **Accuracy** — Overall percentage of correct predictions
* **Precision** — How many transactions predicted as fraud were actually fraudulent
* **Recall** — How many actual fraudulent transactions were detected
* **F1-Score** — Balance between Precision and Recall
* **ROC-AUC** — Measures the model's ability to distinguish between legitimate and fraudulent transactions
* **Confusion Matrix** — Shows correct and incorrect predictions for each class

## 📈 Results

The model achieved the following performance on the test set:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 99.91% |
| Precision | 82.67% |
| Recall    | 63.27% |
| F1-Score  | 71.68% |
| ROC-AUC   | 96.05% |

### Key Observation

Although the model achieves very high accuracy, accuracy alone is not sufficient for fraud detection because the dataset is highly imbalanced.

The **Recall of 63.27%** indicates that the model successfully identified a significant portion of fraudulent transactions, while the **Precision of 82.67%** indicates that most transactions predicted as fraudulent were actually fraud.

The **ROC-AUC of 96.05%** indicates strong overall discrimination between fraudulent and legitimate transactions.

## 📊 Visualizations

The project generates:

### Confusion Matrix

The confusion matrix provides a detailed view of:

* True Negatives
* False Positives
* False Negatives
* True Positives

### ROC Curve

The ROC curve visualizes the model's ability to distinguish fraudulent transactions from legitimate transactions across different classification thresholds.

## 📁 Project Structure

```text
fraud-detection/
│
├── fraud_detection.py
├── requirements.txt
├── README.md

```

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fraud-detection.git
```

Move into the project directory:

```bash
cd fraud-detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📦 Requirements

The project requires:

```text
pandas
matplotlib
scikit-learn
```

## ▶️ Running the Project

Place the dataset in the appropriate location and update the dataset path in the Python script if required.

Then run:

```bash
python fraud_detection.py
```

The program will display:

* Dataset shape
* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Classification Report
* Confusion Matrix
* ROC Curve

## 📚 Key Learning Outcomes

Through this project, I practiced:

* Binary classification
* Logistic Regression
* Data preprocessing
* Feature scaling
* Train-test splitting
* Classification metrics
* Handling imbalanced classification problems
* Confusion Matrix analysis
* ROC-AUC evaluation
* Model performance interpretation
* Data visualization using Matplotlib

## 🔮 Future Improvements

Possible improvements include:

* Applying techniques for handling class imbalance such as SMOTE
* Testing tree-based models such as Random Forest and XGBoost
* Performing hyperparameter tuning
* Comparing multiple classification algorithms
* Optimizing the classification threshold
* Adding Precision-Recall curves
* Deploying the fraud detection model as a web application

## 👩‍💻 Author

**Laiba Aamer**

BS Artificial Intelligence Student

## 📄 License

This project is created for educational and learning purposes.
