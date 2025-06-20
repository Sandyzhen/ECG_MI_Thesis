import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score, precision_score,
                             classification_report, confusion_matrix, roc_curve, auc)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 📂 資料載入
os.chdir("C:\\Users\\sandy\\Desktop\\心電圖資料集")
ECG_data = pd.read_csv("ECG特徵值持續時間+人口學+資料擴增20(MI).csv")
ECG_data["age"] = np.where(ECG_data["age"].isnull(), np.nanmean(ECG_data["age"]), ECG_data["age"])
ECG_data = ECG_data.rename(columns={'filtered_diagnostic_subclass': 'filtered_diagnostic_superclass'})

# 🎯 篩選資料平衡樣本
mi_sample = ECG_data[ECG_data['filtered_diagnostic_superclass'] == 1].sample(n=5400, random_state=42)
norm_sample = ECG_data[ECG_data['filtered_diagnostic_superclass'] == 0].sample(n=5400, random_state=42)
data = pd.concat([mi_sample, norm_sample])

# 🎯 分離 X, y
X = data.drop(columns=['filtered_diagnostic_superclass'])
y = data['filtered_diagnostic_superclass']

# 📊 分割訓練集、驗證集、測試集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

# 定義參數網格
param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'learning_rate': [0.1,0.2,0.3],
    'reg_alpha': [0.05, 0.1],
    'gamma': [0, 0.1, 0.5],
    'max_depth': [2,4,6,8],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}


best_recall_score = 0
best_params = {}

for n in param_grid['n_estimators']:
    for lr in param_grid['learning_rate']:
        for alpha in param_grid['reg_alpha']:
            for gamma in param_grid['gamma']:
                for depth in param_grid['max_depth']:
                    model = XGBClassifier(
                        n_estimators=n,
                        learning_rate=lr,
                        reg_alpha=alpha,
                        gamma=gamma,
                        max_depth=depth,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric='logloss',
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    recall = recall_score(y_val, preds)
                    if recall > best_recall_score:
                        best_recall_score = recall
                        best_params = model.get_params()

print("\nBest Recall:", best_recall_score)
print("Best Params:", best_params)

# 🚀 訓練最佳模型（無擴增資料、無採樣）
final_model = XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

# 📈 預測與評估
preds = final_model.predict(X_test)
probs = final_model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1-score:", f1_score(y_test, preds))
print(classification_report(y_test, preds))

# 📊 混淆矩陣
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# 🔍 特徵重要性
feature_importances = final_model.feature_importances_
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
top_10 = features_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(6, 5))
colors = plt.cm.Blues(np.linspace(0.5, 1, 10))
plt.barh(top_10['Feature'][::-1], top_10['Importance'][::-1], color=colors, edgecolor='k')
plt.xlabel('Importance')
plt.title('Top 10 Important Features - XGBoost')
plt.tight_layout()
plt.show()

# 📉 AUC 曲線
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)
