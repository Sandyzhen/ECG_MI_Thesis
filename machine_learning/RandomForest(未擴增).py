# machine_learning/random_forest_comparison.py

"""
本檔案包含三種版本的 Random Forest 模型訓練與評估：
1. 無採樣版本（原始資料）
2. 有採樣版本（如 SMOTE 平衡後資料）
3. 1:1 切資料版本（手動平衡資料集）

每種版本皆針對不同評估指標（F1、Accuracy、Recall、AUC）進行調參與分析。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    roc_curve, confusion_matrix, auc
)

#======================= Part 1: 無採樣版本 ==========================
# ✅ 用原始資料，針對 F1 與 AUC 各自調參
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

best_f1, best_auc = 0, 0
best_f1_params, best_auc_params = {}, {}

for n in param_grid['n_estimators']:
    for d in param_grid['max_depth']:
        for split in param_grid['min_samples_split']:
            for leaf in param_grid['min_samples_leaf']:
                model = RandomForestClassifier(
                    n_estimators=n, max_depth=d,
                    min_samples_split=split, min_samples_leaf=leaf,
                    bootstrap=True, random_state=42
                )
                model.fit(X_train, y_train)

                # F1 調參
                f1 = f1_score(y_val, model.predict(X_val))
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_params = model.get_params()

                # AUC 調參
                prob = model.predict_proba(X_val)[:, 1]
                auc_val = roc_auc_score(y_val, prob)
                if auc_val > best_auc:
                    best_auc = auc_val
                    best_auc_params = model.get_params()

print("[無採樣] Best F1:", best_f1)
print("F1 Params:", best_f1_params)
print("Best AUC:", best_auc)
print("AUC Params:", best_auc_params)

#======================= Part 2: 有採樣版本 ==========================
# ✅ 使用 SMOTE 後資料進行訓練，分別針對四個指標訓練四個模型
rf_f1 = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=2, min_samples_leaf=2, bootstrap=True, random_state=42)
rf_acc = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, min_samples_leaf=1, bootstrap=True, random_state=42)
rf_rec = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=2, min_samples_leaf=2, bootstrap=True, random_state=42)
rf_auc = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=2, min_samples_leaf=2, bootstrap=True, random_state=42)

models_resampled = [rf_f1, rf_acc, rf_rec, rf_auc]

for i, model in enumerate(models_resampled, 1):
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    print(f"[有採樣] Model {i}:")
    print(classification_report(y_test, y_pred))
    auc_val = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("AUC:", auc_val)

# 混淆矩陣與特徵重要性（以 rf_acc 模型為例）
cm = confusion_matrix(y_test, rf_acc.predict(X_test))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Resampled)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

importances = rf_acc.feature_importances_
top_features = pd.DataFrame({
    'Feature': X_train_res.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(6, 5))
colors = plt.cm.Blues(np.linspace(0.5, 1, 10))
plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color=colors)
plt.xlabel('Importance')
plt.title('Top Features (Resampled)')
plt.tight_layout()
plt.show()

#======================= Part 3: 1:1 切資料版本 ==========================
# ✅ 模擬人工切分後平衡資料
rf_f1 = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)
rf_acc = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)
rf_rec = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)
rf_auc = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)

models_bal = [rf_f1, rf_acc, rf_rec, rf_auc]

for i, model in enumerate(models_bal, 1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"[1:1切資料] Model {i}:")
    print(classification_report(y_test, y_pred))
    auc_val = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("AUC:", auc_val)

# 特徵重要性（以 recall 模型為例）
importances = rf_rec.feature_importances_
top_features = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(6, 5))
colors = plt.cm.Blues(np.linspace(0.5, 1, 10))
plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color=colors)
plt.xlabel('Importance')
plt.title('Top Features (1:1 Split)')
plt.tight_layout()
plt.show()
