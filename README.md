# Personality Trait Classification: Introvert vs Extrovert

This project implements multiple machine learning algorithms to classify individuals as Introverts or Extroverts based on their social behavior and personality traits.

Kaggle hackathon: https://www.kaggle.com/competitions/playground-series-s5e7/overview/abstract

## Project Structure

```
├── adaboost.py              # AdaBoost classifier implementation
├── catboost.py              # CatBoost classifier implementation
├── decision_tree.py         # Decision Tree classifier
├── grad_boosting.py         # Gradient Boosting classifier
├── k_folds_neural_net.py    # Neural Network with K-Folds cross-validation
├── knn.py                   # K-Nearest Neighbors classifier
├── lightgmb.py              # LightGBM classifier implementation
├── log_regression.py        # Logistic Regression classifier
├── naive_bayes.py           # Naive Bayes classifier
├── neural_net.py            # Standard Neural Network implementation
├── preprocessing.py         # Data preprocessing and feature engineering
├── random_forest.py         # Random Forest classifier
├── xgboosting.py            # XGBoost classifier implementation
└── README.md                # This file
```

## Data Preprocessing

- Handling Missing Values
- Encoding Categorical Variables
- Feature Scaling
- Data Splitting

## Available Models

### 1. **Traditional Machine Learning**
- `log_regression.py` - Logistic Regression
- `knn.py` - K-Nearest Neighbors
- `decision_tree.py` - Decision Tree
- `random_forest.py` - Random Forest
- `naive_bayes.py` - Naive Bayes

### 2. **Ensemble Methods & Boosting**
- `adaboost.py` - AdaBoost Classifier
- `grad_boosting.py` - Gradient Boosting
- `xgboosting.py` - XGBoost
- `lightgmb.py` - LightGBM
- `catboost.py` - CatBoost

### 3. **Neural Networks**
- `neural_net.py` - Standard Neural Network
- `k_folds_neural_net.py` - Neural Network with K-Folds Cross-Validation
