"""
本檔案包含三個版本的 XGBoost 模型比較：
1. 資料無擴增（未採樣）
2. 資料無擴增（有採樣，SMOTE 或其他）
3. 資料無擴增（1:1 切資料）

每個版本皆針對不同評估指標（Accuracy, Recall, F1, AUC）做調參與測試。
並包含模型訓練、預測、評估、混淆矩陣與特徵重要性分析。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc
)

#========================= Part 1: 無採樣版本 ==========================
# ✅ Grid search 調參，以 Recall 為目標
param_grid = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.1, 0.2, 0.3],
    'reg_alpha': [0.05, 0.1],
    'gamma': [0, 0.1, 0.5],
    'max_depth': [2, 4, 6],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

best_recall_score = 0
best_params = {}

for n in param_grid['n_estimators']:
    for lr in param_grid['learning_rate']:
        for ra in param_grid['reg_alpha']:
            for g in param_grid['gamma']:
                for md in param_grid['max_depth']:
                    for ss in param_grid['subsample']:
                        for cb in param_grid['colsample_bytree']:
                            model = XGBClassifier(
                                n_estimators=n, learning_rate=lr, reg_alpha=ra, gamma=g,
                                max_depth=md, subsample=ss, colsample_bytree=cb,
                                eval_metric='logloss', random_state=42
                            )
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
                            recall = recall_score(y_val, preds)
                            if recall > best_recall_score:
                                best_recall_score = recall
                                best_params = model.get_params()

print("[無採樣] Best recall score:", best_recall_score)
print("Best Parameters:", best_params)

#=================================== Part 2: 有採樣版本 ===================================
# ✅ 使用最佳參數訓練於 SMOTE/ADASYN 處理後資料
xgb_model_resampled = XGBClassifier(**best_params)
xgb_model_resampled.fit(X_train_res, y_train_res)

pred_test_res = xgb_model_resampled.predict(X_test)
print("[有採樣] classification_report:")
print(classification_report(y_test, pred_test_res))

# 混淆矩陣
cm_res = confusion_matrix(y_test, pred_test_res)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_res, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Resampled Data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# 特徵重要性
importances = xgb_model_resampled.feature_importances_
feature_df = pd.DataFrame({
    'Feature': X_train_res.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(6, 5))
colors = plt.cm.Blues(np.linspace(0.5, 1, 10))
plt.barh(feature_df['Feature'][::-1], feature_df['Importance'][::-1], color=colors)
plt.xlabel('Importance')
plt.title('Top 10 Features (Resampled)')
plt.tight_layout()
plt.show()

# ROC 曲線
proba = xgb_model_resampled.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)
print("AUC (Resampled):", roc_auc)

#========================= Part 3: 1:1 切資料版本 ==========================
# ✅ 使用 1:1 切資料版本（例如平衡過類別後切資料）
xgb_model_balanced = XGBClassifier(colsample_bytree=1, reg_alpha=0.05, gamma=0.5,
                                    learning_rate=0.1, max_depth=8, n_estimators=300,
                                    subsample=0.8, random_state=42)
xgb_model_balanced.fit(X_train, y_train)

pred_test_bal = xgb_model_balanced.predict(X_test)
print("[1:1切資料] classification_report:")
print(classification_report(y_test, pred_test_bal))

# 混淆矩陣
cm_bal = confusion_matrix(y_test, pred_test_bal)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bal, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (1:1 Split)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# 特徵重要性
importances = xgb_model_balanced.feature_importances_
feature_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(6, 5))
colors = plt.cm.Blues(np.linspace(0.5, 1, 10))
plt.barh(feature_df['Feature'][::-1], feature_df['Importance'][::-1], color=colors)
plt.xlabel('Importance')
plt.title('Top 10 Features (1:1 Split)')
plt.tight_layout()
plt.show()

# ROC 曲線
proba_bal = xgb_model_balanced.predict_proba(X_test)[:, 1]
fpr_bal, tpr_bal, _ = roc_curve(y_test, proba_bal)
roc_auc_bal = auc(fpr_bal, tpr_bal)
print("AUC (1:1 Split):", roc_auc_bal)
