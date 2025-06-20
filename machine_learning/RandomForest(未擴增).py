# Random Forest 模型實驗
# 資料分為三種處理方式：
# (1) 原始資料（未採樣）
# (2) 採樣資料（如欠抽樣、過抽樣後的 X_train_res）
# (3) 1:1 配平資料（X_train_bal）

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#================= 1. 採樣後資料（X_train_res） ===================
param_grid = {
    'n_estimators': [100, 200, 300,400],  # 減少樹的數量範圍
    'max_depth': [2,4, 6, 8],  # 限制最大深度，避免過於複雜的模型
    'min_samples_split': [2, 5],  # 增加分裂所需的最小樣本數
    'min_samples_leaf': [1, 2],  # 增加葉節點的最小樣本數
    'bootstrap': [True],  # 使用bootstrap樣本來增加訓練的隨機性
}

best_auc_res = 0
best_params_res = {}

for n in param_grid['n_estimators']:
    for d in param_grid['max_depth']:
        for s in param_grid['min_samples_split']:
            for l in param_grid['min_samples_leaf']:
                model = RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_split=s,
                                               min_samples_leaf=l, bootstrap=True, random_state=42)
                model.fit(X_train_res, y_train_res)
                prob = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, prob)
                if auc > best_auc_res:
                    best_auc_res = auc
                    best_params_res = model.get_params()

print("[採樣] Best ROC AUC:", best_auc_res)
print("[採樣] Best Params:", best_params_res)

model_res = RandomForestClassifier(**best_params_res)
model_res.fit(X_train_res, y_train_res)
y_pred_res = model_res.predict(X_test)
y_prob_res = model_res.predict_proba(X_test)[:, 1]

print("[採樣] Accuracy:", accuracy_score(y_test, y_pred_res))
print("[採樣] Precision:", precision_score(y_test, y_pred_res))
print("[採樣] Recall:", recall_score(y_test, y_pred_res))
print("[採樣] F1-score:", f1_score(y_test, y_pred_res))
print("[採樣] ROC AUC:", roc_auc_score(y_test, y_prob_res))

cm_res = confusion_matrix(y_test, y_pred_res)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_res, annot=True, fmt='d', cmap='Blues')
plt.title("[採樣] Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#================= 2. 原始資料（未採樣） ===================
best_auc_raw = 0
best_params_raw = {}

for n in param_grid['n_estimators']:
    for d in param_grid['max_depth']:
        for s in param_grid['min_samples_split']:
            for l in param_grid['min_samples_leaf']:
                model = RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_split=s,
                                               min_samples_leaf=l, bootstrap=True, random_state=42)
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, prob)
                if auc > best_auc_raw:
                    best_auc_raw = auc
                    best_params_raw = model.get_params()

print("[未採樣] Best ROC AUC:", best_auc_raw)
print("[未採樣] Best Params:", best_params_raw)

model_raw = RandomForestClassifier(**best_params_raw)
model_raw.fit(X_train, y_train)
y_pred_raw = model_raw.predict(X_test)
y_prob_raw = model_raw.predict_proba(X_test)[:, 1]

print("[未採樣] Accuracy:", accuracy_score(y_test, y_pred_raw))
print("[未採樣] Precision:", precision_score(y_test, y_pred_raw))
print("[未採樣] Recall:", recall_score(y_test, y_pred_raw))
print("[未採樣] F1-score:", f1_score(y_test, y_pred_raw))
print("[未採樣] ROC AUC:", roc_auc_score(y_test, y_prob_raw))

#================= 3. 1:1 配平資料 ===================
best_auc_bal = 0
best_params_bal = {}

for n in param_grid['n_estimators']:
    for d in param_grid['max_depth']:
        for s in param_grid['min_samples_split']:
            for l in param_grid['min_samples_leaf']:
                model = RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_split=s,
                                               min_samples_leaf=l, bootstrap=True, random_state=42)
                model.fit(X_train_bal, y_train_bal)
                prob = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, prob)
                if auc > best_auc_bal:
                    best_auc_bal = auc
                    best_params_bal = model.get_params()

print("[1:1] Best ROC AUC:", best_auc_bal)
print("[1:1] Best Params:", best_params_bal)

model_bal = RandomForestClassifier(**best_params_bal)
model_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = model_bal.predict(X_test)
y_prob_bal = model_bal.predict_proba(X_test)[:, 1]

print("[1:1] Accuracy:", accuracy_score(y_test, y_pred_bal))
print("[1:1] Precision:", precision_score(y_test, y_pred_bal))
print("[1:1] Recall:", recall_score(y_test, y_pred_bal))
print("[1:1] F1-score:", f1_score(y_test, y_pred_bal))
print("[1:1] ROC AUC:", roc_auc_score(y_test, y_prob_bal))

#===============================================
# 總結：依據五個指標（Accuracy, Precision, Recall, F1-score, AUC）綜合比較三種資料處理方式的結果，
# 決定最終採用哪一組模型進行訓練與部署。可搭配 DataFrame 或圖表視覺化整理輸出指標以利比較。

# 提醒：請確認變數 X_train_bal, y_train_bal 為 1:1 配平後資料
