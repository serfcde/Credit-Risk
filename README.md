# Loan Default Risk Prediction Model

A comprehensive machine learning project that predicts loan default risk using statistical analysis and predictive modeling techniques. This project analyzes historical lending data to identify key risk factors and build accurate predictive models.

## 📋 Project Overview

This project builds and evaluates multiple machine learning models to predict whether a loan will be charged off (defaulted) or fully paid. Using data from LendingClub (2007-2018), we apply statistical methods and machine learning algorithms to assess credit risk effectively.

**Target Variable:**
- **0 (Fully Paid):** Loan repaid successfully
- **1 (Charged Off):** Loan defaulted

---

## 🔍 Key Steps & Analysis

### 1. **Data Loading & Exploration**
- Load LendingClub dataset (107,882 loans)
- Initial exploratory data analysis (EDA)
- Identify data quality issues

### 2. **Data Cleaning & Preprocessing**
- Removed ID columns (`id`, `member_id`)
- Dropped columns with >90% missing values
- Filtered to include only "Fully Paid" and "Charged Off" loans
- Binary encoding of target variable
- Handled missing values using median imputation

### 3. **Feature Selection (Mutual Information Analysis)**
- Applied Mutual Information (MI) classification scoring
- Identified 6 most informative features:
  - `recoveries` (amount recovered post-charge-off)
  - `collection_recovery_fee` (fees from collections)
  - `int_rate` (loan interest rate)
  - `fico_range_low` (minimum FICO score)
  - `fico_range_high` (maximum FICO score)
  - `installment` (monthly payment amount)

### 4. **Exploratory Data Analysis (EDA)**
- **Distribution Analysis:** Histograms with KDE plots for feature distributions
- **Bivariate Analysis:** 
  - Box plots: Features vs. Loan Status
  - Violin plots: Distribution comparisons by default status
- **Correlation Analysis:**
  - Interest rate vs. FICO scores
  - Installment vs. Interest rate
  - FICO score ranges
  - Recovery metrics scatter plots

### 5. **Model Development & Training**
Three classifiers were trained and evaluated:

#### Models Implemented:
1. **Logistic Regression** - Statistical baseline model
2. **Decision Tree Classifier** - Rule-based interpretation
3. **Random Forest Classifier** - Ensemble method

#### Key Techniques:
- **Train-Test Split:** 80-20 split with stratification
- **Feature Scaling:** StandardScaler for Logistic Regression
- **Cross-Validation:** 5-fold CV for robust performance estimation

### 6. **Hyperparameter Tuning**
Used GridSearchCV to optimize model parameters:

**Logistic Regression:**
- Penalty: L1/L2 regularization
- C: Regularization strength [0.01, 0.1, 1, 10, 100]

**Random Forest:**
- n_estimators: [100, 200, 300]
- max_depth: [5, 10, 15, None]
- min_samples_split: [2, 5, 10]
- max_features: ['sqrt', 'log2']

### 7. **Model Evaluation**
Comprehensive evaluation metrics:
- **Classification Report:** Precision, Recall, F1-Score
- **Confusion Matrix:** Visual representation of predictions
- **Cross-Validation Scores:** Robustness validation
- **ROC-AUC Curves:** Trade-off between true and false positive rates
- **Feature Importance:** Identification of most influential features

### 8. **Results & Insights**
- Model performance comparison across all three classifiers
- ROC curves showing model discrimination ability
- Feature importance ranking from Random Forest
- Identification of key risk factors

### 9. **Model Persistence**
Trained models saved for production deployment:
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`

---

## 📊 Dataset Features

| Feature | Description | Type |
|---------|-------------|------|
| `recoveries` | Amount recovered post-charge-off | Numerical |
| `collection_recovery_fee` | Fees collected during recovery process | Numerical |
| `int_rate` | Loan interest rate (%) | Numerical |
| `fico_range_low` | Minimum FICO credit score | Numerical |
| `fico_range_high` | Maximum FICO credit score | Numerical |
| `installment` | Monthly loan payment amount | Numerical |

---

## 🛠️ Technologies & Libraries

```python
# Data Processing
pandas
numpy
scikit-learn

# Modeling
scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)
GridSearchCV, cross_val_score

# Evaluation
classification_report, confusion_matrix, roc_curve, roc_auc_score

# Visualization
matplotlib
seaborn

# Model Persistence
joblib
```

---

## 📈 Key Findings

1. **Most Important Features:**
   - Interest rate, FICO scores, and installment amount are strong default indicators
   - Recovery metrics provide valuable post-charge-off insights

2. **Model Performance:**
   - Random Forest generally outperforms other models
   - Cross-validation reveals consistent performance across data folds
   - ROC-AUC provides clear model discrimination ability

3. **Risk Patterns:**
   - Higher interest rates correlate with increased default risk
   - FICO scores show inverse relationship with default probability
   - Monthly installment amounts influence repayment capacity

---

## 📂 Project Structure

```
Credit Risk/
├── Credit Risk Assessment Model.ipynb  # Main analysis notebook
└── README.md                           # This file
```

---

## 🚀 How to Use

### Prerequisites:
- Python 3.7+
- Required libraries: pandas, scikit-learn, matplotlib, seaborn

### Running the Analysis:
1. Open the Jupyter notebook: `Credit Risk Assessment Model.ipynb`
2. Execute cells sequentially to:
   - Load and explore the dataset
   - Perform feature selection
   - Train and evaluate models
   - Generate visualizations and insights

### Making Predictions:
```python
import joblib

# Load trained model
model = joblib.load('random_forest_model.pkl')

# Predict on new data
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]
```

---

## 💡 Insights for Stakeholders

- **Risk Assessment:** Models can effectively predict loan default risk
- **Decision Support:** Feature importance guides lending decisions
- **Resource Allocation:** Prioritize monitoring for high-risk loans
- **Policy Making:** Adjust interest rates based on risk profiles

---

## 📝 Notes

- Models are trained on historical data (2007-2018) from LendingClub
- Performance metrics are calculated on held-out test set
- Hyperparameter tuning optimizes for accuracy; other metrics (precision, recall) can be tuned based on business needs
- Feature selection based on Mutual Information captures non-linear relationships

---

## 📧 Contact & Attribution

**Project:** Loan Default Risk Prediction Model  
**Date:** January 2026  
**Purpose:** Resume portfolio project demonstrating ML & statistical analysis skills

---

## 📜 License

This project uses the LendingClub dataset for educational purposes.

