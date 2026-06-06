"""
=============================================================
  Simple Machine Learning Demo  |  Key Concepts Explained
=============================================================
  Concepts covered:
    1. Dataset loading & exploration
    2. Train/test split
    3. Feature scaling (normalization)
    4. Model training (Logistic Regression + Random Forest)
    5. Evaluation (accuracy, confusion matrix, classification report)
    6. Overfitting vs underfitting
    7. Cross-validation
    8. Feature importance
    9. Saving and loading a model
   10. Visualization of results

  Verified runnable with:
    Python 3.13   scikit-learn 1.8   matplotlib 3.10   numpy 2.4
  Install:  pip install scikit-learn matplotlib numpy
  Run:      python ml_demo.py
  Outputs (written to the current directory):
    rf_model.pkl, scaler.pkl, ml_demo_results.png
=============================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import pickle

# ─────────────────────────────────────────────────────────────
# 1.  LOAD & EXPLORE THE DATASET
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1: Load & Explore the Dataset")
print("="*60)

# Breast cancer dataset: 569 samples, 30 numeric features
# Target: 0 = malignant, 1 = benign
data = load_breast_cancer()
X = data.data          # feature matrix  (569 x 30)
y = data.target        # label vector    (569,)

print(f"  Dataset      : Breast Cancer Wisconsin")
print(f"  Samples      : {X.shape[0]}")
print(f"  Features     : {X.shape[1]}")
print(f"  Classes      : {data.target_names.tolist()}  (0=malignant, 1=benign)")
print(f"  Class balance: malignant={np.sum(y==0)}, benign={np.sum(y==1)}")

# ─────────────────────────────────────────────────────────────
# 2.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2: Train / Test Split")
print("="*60)

# KEY CONCEPT: Never train and evaluate on the same data.
# We hold out 20% for testing so the model never "sees" it during training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,     # 20% held out for testing
    random_state=42,    # reproducibility seed
    stratify=y          # preserve class ratios in both splits
)

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")
print(f"  (stratify=y keeps class ratios identical in both splits)")

# ─────────────────────────────────────────────────────────────
# 3.  FEATURE SCALING
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3: Feature Scaling (StandardScaler)")
print("="*60)

# KEY CONCEPT: Many algorithms (Logistic Regression, SVM, KNN) are sensitive
# to feature magnitude. StandardScaler transforms each feature to mean=0, std=1.
# IMPORTANT: fit ONLY on training data, then transform both train and test.
# Fitting on test data would be "data leakage."
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform train
X_test_scaled  = scaler.transform(X_test)        # transform test only

print(f"  Before scaling - feature 0: mean={X_train[:,0].mean():.2f}, std={X_train[:,0].std():.2f}")
print(f"  After  scaling - feature 0: mean={X_train_scaled[:,0].mean():.2f}, std={X_train_scaled[:,0].std():.2f}")
print(f"  (Fit scaler on TRAIN only -- fitting on test = data leakage!)")

# ─────────────────────────────────────────────────────────────
# 4.  TRAIN MODELS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4: Train Two Models")
print("="*60)

# --- Model A: Logistic Regression ---
# KEY CONCEPT: Linear model that outputs a probability (0-1) via the sigmoid
# function. Fast, interpretable, works well when classes are linearly separable.
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
print("  [A] Logistic Regression trained")

# --- Model B: Random Forest ---
# KEY CONCEPT: Ensemble of decision trees. Each tree sees a random subset of
# data and features (bagging). Final prediction = majority vote across trees.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
print("  [B] Random Forest (100 trees) trained")

# ─────────────────────────────────────────────────────────────
# 5.  EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5: Evaluate on Test Set")
print("="*60)

# Predictions
lr_pred = lr_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test_scaled)

lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"\n  Logistic Regression Accuracy : {lr_acc:.4f} ({lr_acc*100:.1f}%)")
print(f"  Random Forest Accuracy       : {rf_acc:.4f} ({rf_acc*100:.1f}%)")

print("\n  --- Logistic Regression Classification Report ---")
print(classification_report(y_test, lr_pred, target_names=data.target_names))

print("  --- Random Forest Classification Report ---")
print(classification_report(y_test, rf_pred, target_names=data.target_names))

# KEY CONCEPT: Precision vs Recall
# Precision = of all predicted positives, how many were actually positive?
# Recall    = of all actual positives, how many did we catch?
# In medical contexts, high Recall (sensitivity) is critical -- missing
# a malignant case is more dangerous than a false alarm.

# ─────────────────────────────────────────────────────────────
# 6.  OVERFITTING vs UNDERFITTING
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6: Overfitting vs Underfitting")
print("="*60)

# KEY CONCEPT:
# Underfitting = model too simple, bad on BOTH train and test
# Overfitting  = model too complex, great on train but bad on test
# Good fit     = good on BOTH

lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train_scaled))
rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train_scaled))

print(f"\n  Logistic Regression  | Train: {lr_train_acc:.4f}  Test: {lr_acc:.4f}  Gap: {lr_train_acc-lr_acc:.4f}")
print(f"  Random Forest        | Train: {rf_train_acc:.4f}  Test: {rf_acc:.4f}  Gap: {rf_train_acc-rf_acc:.4f}")
print(f"\n  NOTE: RF has perfect train accuracy -- slight overfitting.")
print(f"  The gap between train and test accuracy indicates overfitting severity.")

# ─────────────────────────────────────────────────────────────
# 7.  CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 7: Cross-Validation (k-fold)")
print("="*60)

# KEY CONCEPT: Instead of a single train/test split, k-fold CV splits data
# into k folds. The model trains on k-1 folds and tests on the remaining fold,
# rotating through all k combinations. Gives a more reliable accuracy estimate.
cv_scores_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"\n  Logistic Regression 5-fold CV: {cv_scores_lr.round(4)}  Mean={cv_scores_lr.mean():.4f}")
print(f"  Random Forest       5-fold CV: {cv_scores_rf.round(4)}  Mean={cv_scores_rf.mean():.4f}")
print(f"\n  Low variance across folds = model is stable and generalizes well.")

# ─────────────────────────────────────────────────────────────
# 8.  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 8: Feature Importance (Random Forest)")
print("="*60)

# KEY CONCEPT: Tree-based models can rank which features contributed most
# to predictions. Useful for understanding the model and reducing dimensionality.
importances = rf_model.feature_importances_
top5_idx = np.argsort(importances)[::-1][:5]

print("\n  Top 5 most important features:")
for rank, idx in enumerate(top5_idx, 1):
    print(f"    {rank}. {data.feature_names[idx]:<35} importance={importances[idx]:.4f}")

# ─────────────────────────────────────────────────────────────
# 9.  SAVE & RELOAD THE MODEL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 9: Save & Reload the Model (pickle)")
print("="*60)

# KEY CONCEPT: Trained models can be serialized (pickled) so they can be
# loaded later for inference without retraining.
#
# CRITICAL -- always save the SCALER alongside the model. The model learned
# patterns from SCALED data (mean=0, std=1). At inference time, any new input
# must be transformed the exact same way before being passed to the model.
#
# Rule: Scaler + Model are a matched pair. Deploy them together. Always.
model_path  = "rf_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path,  "wb") as f:
    pickle.dump(rf_model, f)
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"  Model saved to  : {os.path.abspath(model_path)}")
print(f"  Scaler saved to : {os.path.abspath(scaler_path)}")

# Reload and verify -- correct inference pattern: scale first, then predict
with open(model_path,  "rb") as f:
    loaded_model = pickle.load(f)
with open(scaler_path, "rb") as f:
    loaded_scaler = pickle.load(f)

# Step 1: scale the RAW input using the SAME fitted scaler
# Step 2: pass the scaled input to the model
verify_pred = loaded_model.predict(loaded_scaler.transform(X_test))
verify_acc  = accuracy_score(y_test, verify_pred)

# Use np.isclose, not '==', when comparing floating-point results.
matches = np.isclose(verify_acc, rf_acc)
print(f"  Reloaded model accuracy: {verify_acc:.4f}  (matches original: {matches})")
print(f"  Inference pattern: scaler.transform(raw_input) -> model.predict(scaled_input)")

# ─────────────────────────────────────────────────────────────
# 10. VISUALIZE RESULTS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 10: Generate Visualization")
print("="*60)

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#f8f8f8')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

DARK    = '#1a1a1a'
RED     = '#c0392b'
BLUE    = '#2e6da4'
GREEN   = '#27ae60'
ORANGE  = '#e67e22'

# --- Plot 1: Confusion Matrix - Logistic Regression ---
ax1 = fig.add_subplot(gs[0, 0])
cm_lr = confusion_matrix(y_test, lr_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=data.target_names)
disp.plot(ax=ax1, colorbar=False, cmap='Blues')
ax1.set_title('Confusion Matrix\nLogistic Regression', fontsize=11, fontweight='bold', color=DARK)
ax1.set_xlabel('Predicted', fontsize=9)
ax1.set_ylabel('Actual', fontsize=9)

# --- Plot 2: Confusion Matrix - Random Forest ---
ax2 = fig.add_subplot(gs[0, 1])
cm_rf = confusion_matrix(y_test, rf_pred)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=data.target_names)
disp2.plot(ax=ax2, colorbar=False, cmap='Greens')
ax2.set_title('Confusion Matrix\nRandom Forest', fontsize=11, fontweight='bold', color=DARK)
ax2.set_xlabel('Predicted', fontsize=9)
ax2.set_ylabel('Actual', fontsize=9)

# --- Plot 3: Train vs Test Accuracy (Overfitting) ---
ax3 = fig.add_subplot(gs[0, 2])
models     = ['Logistic\nRegression', 'Random\nForest']
train_accs = [lr_train_acc, rf_train_acc]
test_accs  = [lr_acc, rf_acc]
x = np.arange(len(models))
w = 0.32
bars1 = ax3.bar(x - w/2, train_accs, w, label='Train', color=BLUE,   alpha=0.85, edgecolor='white')
bars2 = ax3.bar(x + w/2, test_accs,  w, label='Test',  color=ORANGE, alpha=0.85, edgecolor='white')
ax3.set_ylim(0.93, 1.005)
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=9)
ax3.set_ylabel('Accuracy', fontsize=9)
ax3.set_title('Train vs Test Accuracy\n(Overfitting Check)', fontsize=11, fontweight='bold', color=DARK)
ax3.legend(fontsize=9)
ax3.set_facecolor('#ffffff')
for bar in bars1:
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

# --- Plot 4: Cross-Validation Scores ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(range(1,6), cv_scores_lr, 'o-', color=BLUE,  label=f'LR  (mean={cv_scores_lr.mean():.3f})', lw=2, ms=7)
ax4.plot(range(1,6), cv_scores_rf, 's-', color=GREEN, label=f'RF  (mean={cv_scores_rf.mean():.3f})', lw=2, ms=7)
ax4.axhline(cv_scores_lr.mean(), color=BLUE,  ls='--', alpha=0.4, lw=1)
ax4.axhline(cv_scores_rf.mean(), color=GREEN, ls='--', alpha=0.4, lw=1)
ax4.set_xlabel('Fold', fontsize=9)
ax4.set_ylabel('Accuracy', fontsize=9)
ax4.set_title('5-Fold Cross-Validation\nStability Check', fontsize=11, fontweight='bold', color=DARK)
ax4.legend(fontsize=8)
ax4.set_ylim(0.93, 1.01)
ax4.set_facecolor('#ffffff')
ax4.set_xticks(range(1,6))

# --- Plot 5: Top 10 Feature Importances ---
ax5 = fig.add_subplot(gs[1, 1:])
top10_idx   = np.argsort(importances)[::-1][:10]
top10_vals  = importances[top10_idx]
top10_names = [data.feature_names[i] for i in top10_idx]
colors_bar  = [RED if i == 0 else BLUE for i in range(10)]
bars = ax5.barh(range(10), top10_vals[::-1], color=colors_bar[::-1], alpha=0.85, edgecolor='white')
ax5.set_yticks(range(10))
ax5.set_yticklabels(top10_names[::-1], fontsize=8.5)
ax5.set_xlabel('Importance Score', fontsize=9)
ax5.set_title('Top 10 Feature Importances\n(Random Forest)', fontsize=11, fontweight='bold', color=DARK)
ax5.set_facecolor('#ffffff')
for bar, val in zip(bars, top10_vals[::-1]):
    ax5.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8)

fig.suptitle('Machine Learning Demo  |  Breast Cancer Dataset  |  scikit-learn',
             fontsize=14, fontweight='bold', color=DARK, y=0.98)

out_path = "ml_demo_results.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  Visualization saved to: {os.path.abspath(out_path)}")

print("\n" + "="*60)
print("  ALL STEPS COMPLETE")
print("="*60)
