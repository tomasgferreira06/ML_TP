# Wine Quality Classification Using Physicochemical Properties

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

## Overview

This project was developed as part of the **Machine Learning** course in the **Master's in Artificial Intelligence** program at the **University of Coimbra**. It applies supervised learning methods to the task of predicting wine quality from measurable physicochemical attributes, framed as both a binary and a multiclass classification problem.

The dataset consists of 1,599 red wine samples described by 11 continuous features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol) and a discrete quality score ranging from 3 to 8. The project evaluates three classifiers: **Support Vector Machines (SVM)**, **Neural Networks (MLP)**, and **Random Forest** ; under both classification scenarios, including hyperparameter optimization, feature engineering, feature selection, and class imbalance handling via SMOTE.

## Theoretical Background

The methodology draws on standard supervised classification principles and addresses several practical challenges common in real-world ML pipelines:

- **Class Imbalance**: Over 80% of samples concentrate in quality scores 5 and 6. In the binary scenario, the binarization threshold (quality ≥ 6) produces a near-balanced split (~53%/47%). For the multiclass case, SMOTE (Synthetic Minority Over-sampling Technique) is applied to generate synthetic samples for underrepresented classes via k-nearest neighbor interpolation in feature space.
- **Feature Scaling**: Given the heterogeneous magnitude of features (e.g., total sulfur dioxide ~ 10² vs. chlorides ~ 10⁻²), standardization (z-score normalization) is applied to prevent scale-sensitive algorithms (SVM, MLP) from being dominated by high-magnitude variables.
- **Regularization and Generalization**: Hyperparameter tuning prioritizes the train-test generalization gap over raw training accuracy. Configurations exhibiting high training performance with large gaps are rejected in favor of models with lower variance and more stable test-set behavior.
- **Feature Engineering**: Derived features (SO₂ ratio, acidity index, alcohol-density difference, total acidity) are constructed to capture nonlinear physicochemical interactions not explicitly present in the original feature set.
- **Ensemble Ranking for Feature Selection**: A consensus-based ranking strategy combines Random Forest importance (Gini impurity), Mutual Information, and Permutation Importance (evaluated on an MLP) to produce a robust feature hierarchy less biased toward any single method.

## Problem Statement

The project addresses two classification scenarios:

**Scenario A — Binary Classification**: Classify each wine as "Under-Average" (quality ≤ 5) or "Above-Average" (quality ≥ 6). This reduces the problem to a single decision boundary and serves as a baseline for evaluating feature discriminability.

**Scenario B — Multiclass Classification**: Predict the exact quality score (3–8) for each wine sample. This scenario introduces significant additional complexity due to class overlap in adjacent categories and severe class imbalance in the extremes.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Environment | Jupyter Notebook |
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| ML Models | `scikit-learn` (MLPClassifier, RandomForestClassifier, SVC) |
| Resampling | `imbalanced-learn` (SMOTE) |
| Statistics | `scipy` (IQR-based outlier detection) |
| Preprocessing | `StandardScaler`, `train_test_split`, `ParameterGrid` |
| Evaluation | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`, `roc_auc_score` |
| Feature Analysis | `mutual_info_classif`, `permutation_importance` |


## How to Run

```bash
# Clone the repository
git https://github.com/tomasgferreira06/ML_TP.git
cd ML_TP

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy jupyter

# Launch the notebook
jupyter notebook projetoML.ipynb
```

> **Note**: The notebook expects `wine_quality.csv` in the same directory. All experiments use `random_state=42` for reproducibility.
