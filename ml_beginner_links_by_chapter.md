# Beginner-Friendly Machine Learning Links by Chapter


## Introduction

- [scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html) - A gentle overview of the basic ML workflow in Python.
- [Python Tutorial](https://docs.python.org/3/tutorial/index.html) - Good background if the reader is new to Python syntax.

## Chapter 1 - Learning the Python Ecosystem

- [Python Releases for Windows](https://www.python.org/downloads/windows/) - Official Windows download page for Python.
- [Using Python on Windows](https://docs.python.org/3/using/windows.html) - Official installation and setup guide.
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) - First stop for arrays and numeric computing.
- [Installing scikit-learn](https://scikit-learn.org/stable/install.html) - Clear install instructions, including virtual environments.
- [Matplotlib Quick Start](https://matplotlib.org/stable/users/explain/quick_start.html) - Beginner-friendly plotting basics.

## Chapter 2 - Dataset Load and Exploration

- [load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) - The exact dataset used in the book, with sample counts and feature info.
- [scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html) - Shows the basic `X`, `y`, fit, and predict pattern.
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) - Helpful for understanding shapes, arrays, and slicing while exploring data.

## Chapter 3 - Train/Test Split

- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) - Official reference with simple examples.
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) - Explains why splitting matters before evaluation.
- [Common Pitfalls and Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html) - Beginner-friendly explanation of leakage and incorrect preprocessing.

## Chapter 4 - Feature Scaling

- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) - The exact scaler discussed in many beginner pipelines.
- [Common Pitfalls and Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html) - Especially useful for understanding why scaling must be fit only on training data.
- [scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html) - Shows scaling inside a pipeline with `LogisticRegression`.

## Chapter 5 - Model Training

- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - Official classifier reference with examples.
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - Official tree-ensemble reference.
- [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) - Simple metric for early experiments.
- [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) - Great for learning precision, recall, and F1 in one place.

## Chapter 6 - Fitting

- [Common Pitfalls and Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html) - Practical examples of leakage, bad evaluation, and over-optimistic results.
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) - Helps beginners see the difference between one split and more reliable evaluation.
- [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) - Small API surface, good for understanding fit quality quickly.

## Chapter 7 - Cross-Validation

- [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) - Good next step once `cross_val_score` makes sense.
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) - Broader conceptual explanation of folds and validation strategy.
- [scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html) - Includes a short, readable cross-validation example.

## Chapter 8 - Feature Importance

- [Permutation Feature Importance](https://scikit-learn.org/stable/modules/permutation_importance.html) - Best beginner explanation of feature importance that stays conceptually honest.
- [permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html) - Direct API reference once the concept clicks.
- [Permutation Importance vs Random Forest Feature Importance](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html) - Helpful example comparing two common interpretations.

## Chapter 9 - Save / Load Model

- [Model Persistence](https://scikit-learn.org/stable/model_persistence.html) - Official overview of saving and loading trained models.
- [Common Pitfalls and Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html) - Useful here because persistence only works safely when preprocessing stays consistent.
- [Installing scikit-learn](https://scikit-learn.org/stable/install.html) - Handy if a beginner is recreating the environment before reloading a saved model.

## Chapter 10 - Visualization

- [Matplotlib Quick Start](https://matplotlib.org/stable/users/explain/quick_start.html) - Best first stop for making plots in Python.
- [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) - Official explanation of the matrix values.
- [ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) - Easiest way to plot confusion matrices.
- [Evaluate the performance of a classifier with Confusion Matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) - Example-driven walkthrough.

## Chapter 11 - Summing Up Your First Real Classifier

- [scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html) - Best single recap of the end-to-end workflow.
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) - Good review of evaluation habits.
- [Model Persistence](https://scikit-learn.org/stable/model_persistence.html) - Helps tie the training workflow to actual reuse.
- [Common Pitfalls and Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html) - Strong closing reference for avoiding beginner mistakes.

- © 2026 David Grunwald. All rights reserved.
