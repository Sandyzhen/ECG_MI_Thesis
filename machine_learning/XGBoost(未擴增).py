# XGBoost 模型實驗
# 資料分為三種處理方式：
# (1) 原始資料（未採樣）
# (2) 採樣資料（如 SMOTE 後的 X_train_res）
# (3) 1:1 配平資料（X_train_bal）

from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, classification_report, confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#================= 1. 採樣後資料（X_train_res） ===================
param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'learning_rate': [0.1,0.2,0.3],
    'reg_alpha': [0.05, 0.1],
    'gamma': [0, 0.1, 0.5],
    'max_depth': [2,4,6,8],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

best_auc_res = 0
best_params_res = {}

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
                                eval_metric='logloss', random_state=42, use_label_encoder=False
                            )
                            model.fit(X_train_res, y_train_res)
                            proba = model.predict_proba(X_val)[:, 1]
                            auc_val = roc_auc_score(y_val, proba)
                            if auc_val > best_auc_res:
                                best_auc_res = auc_val
                                best_params_res = model.get_params()

print("[採樣] Best ROC AUC:", best_auc_res)
print("[採樣] Best Params:", best_params_res)

xgb_model_res = XGBClassifier(**best_params_res)
xgb_model_res.fit(X_train_res, y_train_res)
y_pred_res = xgb_model_res.predict(X_test)
y_prob_res = xgb_model_res.predict_proba(X_test)[:, 1]

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
    for lr in param_grid['learning_rate']:
        for ra in param_grid['reg_alpha']:
            for g in param_grid['gamma']:
                for md in param_grid['max_depth']:
                    for ss in param_grid['subsample']:
                        for cb in param_grid['colsample_bytree']:
                            model = XGBClassifier(
                                n_estimators=n, learning_rate=lr, reg_alpha=ra, gamma=g,
                                max_depth=md, subsample=ss, colsample_bytree=cb,
                                eval_metric='logloss', random_state=42, use_label_encoder=False
                            )
                            model.fit(X_train, y_train)
                            proba = model.predict_proba(X_val)[:, 1]
                            auc_val = roc_auc_score(y_val, proba)
                            if auc_val > best_auc_raw:
                                best_auc_raw = auc_val
                                best_params_raw = model.get_params()

print("[未採樣] Best ROC AUC:", best_auc_raw)
print("[未採樣] Best Params:", best_params_raw)

xgb_model_raw = XGBClassifier(**best_params_raw)
xgb_model_raw.fit(X_train, y_train)
y_pred_raw = xgb_model_raw.predict(X_test)
y_prob_raw = xgb_model_raw.predict_proba(X_test)[:, 1]

print("[未採樣] Accuracy:", accuracy_score(y_test, y_pred_raw))
print("[未採樣] Precision:", precision_score(y_test, y_pred_raw))
print("[未採樣] Recall:", recall_score(y_test, y_pred_raw))
print("[未採樣] F1-score:", f1_score(y_test, y_pred_raw))
print("[未採樣] ROC AUC:", roc_auc_score(y_test, y_prob_raw))

#================= 3. 1:1 配平資料 ===================
best_auc_bal = 0
best_params_bal = {}

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
                                eval_metric='logloss', random_state=42, use_label_encoder=False
                            )
                            model.fit(X_train_bal, y_train_bal)
                            proba = model.predict_proba(X_val)[:, 1]
                            auc_val = roc_auc_score(y_val, proba)
                            if auc_val > best_auc_bal:
                                best_auc_bal = auc_val
                                best_params_bal = model.get_params()

print("[1:1] Best ROC AUC:", best_auc_bal)
print("[1:1] Best Params:", best_params_bal)

xgb_model_bal = XGBClassifier(**best_params_bal)
xgb_model_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = xgb_model_bal.predict(X_test)
y_prob_bal = xgb_model_bal.predict_proba(X_test)[:, 1]

print("[1:1] Accuracy:", accuracy_score(y_test, y_pred_bal))
print("[1:1] Precision:", precision_score(y_test, y_pred_bal))
print("[1:1] Recall:", recall_score(y_test, y_pred_bal))
print("[1:1] F1-score:", f1_score(y_test, y_pred_bal))
print("[1:1] ROC AUC:", roc_auc_score(y_test, y_prob_bal))

#===============================================
# 總結：依據五個指標（Accuracy, Precision, Recall, F1-score, AUC）綜合比較三種資料處理方式的結果，
# 決定最終採用哪一組模型進行訓練與部署。可搭配 DataFrame 或圖表視覺化整理輸出指標以利比較。

# 提醒：請確認變數 X_train_bal, y_train_bal 為 1:1 配平後資料
