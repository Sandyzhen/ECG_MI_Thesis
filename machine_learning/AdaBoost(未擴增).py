# AdaBoost 模型實驗
# 資料分為三種處理方式：
# (1) 原始資料（未採樣）
# (2) 採樣資料（如欠抽樣、過抽樣後的 X_train_res）
# (3) 1:1 配平資料
#================= 1. 採樣後資料（X_train_res） ===================
# 搜尋最佳參數，評估指標以 ROC AUC 為準
param_grid = {
    'n_estimators':[50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'estimator__max_depth': [1, 2, 3, 4]
}

best_roc_auc_score = 0
best_params = {}

for n in param_grid['n_estimators']:
    for lr in param_grid['learning_rate']:
        for depth in param_grid['estimator__max_depth']:
            estimator = DecisionTreeClassifier(max_depth=depth, random_state=42, min_samples_split=6, min_samples_leaf=4)
            model = AdaBoostClassifier(estimator=estimator, n_estimators=n, learning_rate=lr, random_state=42)
            model.fit(X_train_res, y_train_res)
            pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, pred_proba)
            if auc > best_roc_auc_score:
                best_roc_auc_score = auc
                best_params = {'n_estimators': n, 'learning_rate': lr, 'estimator__max_depth': depth}

print("[採樣] Best ROC AUC:", best_roc_auc_score)
print("[採樣] Best Params:", best_params)

# 根據最佳參數訓練模型
best_estimator = DecisionTreeClassifier(max_depth=best_params['estimator__max_depth'], 
                                        min_samples_split=6, min_samples_leaf=4, random_state=42)
best_model = AdaBoostClassifier(estimator=best_estimator,
                                 n_estimators=best_params['n_estimators'],
                                 learning_rate=best_params['learning_rate'], random_state=42)
best_model.fit(X_train_res, y_train_res)
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

# 指標計算
print("[採樣] Accuracy:", accuracy_score(y_test, y_pred))
print("[採樣] Precision:", precision_score(y_test, y_pred))
print("[採樣] Recall:", recall_score(y_test, y_pred))
print("[採樣] F1-score:", f1_score(y_test, y_pred))
print("[採樣] ROC AUC:", roc_auc_score(y_test, y_prob))

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("[採樣] Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#================= 2. 原始資料（未採樣） ===================
best_auc = 0
best_params_raw = {}

for n in param_grid['n_estimators']:
    for lr in param_grid['learning_rate']:
        for depth in param_grid['estimator__max_depth']:
            estimator = DecisionTreeClassifier(max_depth=depth, random_state=42, min_samples_split=6, min_samples_leaf=4)
            model = AdaBoostClassifier(estimator=estimator, n_estimators=n, learning_rate=lr, random_state=42)
            model.fit(X_train, y_train)
            pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, pred_proba)
            if auc > best_auc:
                best_auc = auc
                best_params_raw = {'n_estimators': n, 'learning_rate': lr, 'estimator__max_depth': depth}

print("[未採樣] Best ROC AUC:", best_auc)
print("[未採樣] Best Params:", best_params_raw)

# 訓練與測試
estimator_raw = DecisionTreeClassifier(max_depth=best_params_raw['estimator__max_depth'], 
                                       min_samples_split=6, min_samples_leaf=4, random_state=42)
best_model_raw = AdaBoostClassifier(estimator=estimator_raw,
                                    n_estimators=best_params_raw['n_estimators'],
                                    learning_rate=best_params_raw['learning_rate'], random_state=42)
best_model_raw.fit(X_train, y_train)
y_pred_raw = best_model_raw.predict(X_test)
y_prob_raw = best_model_raw.predict_proba(X_test)[:,1]

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
        for depth in param_grid['estimator__max_depth']:
            estimator = DecisionTreeClassifier(max_depth=depth, random_state=42, min_samples_split=6, min_samples_leaf=4)
            model = AdaBoostClassifier(estimator=estimator, n_estimators=n, learning_rate=lr, random_state=42)
            model.fit(X_train_bal, y_train_bal)
            pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, pred_proba)
            if auc > best_auc_bal:
                best_auc_bal = auc
                best_params_bal = {'n_estimators': n, 'learning_rate': lr, 'estimator__max_depth': depth}

print("[1:1] Best ROC AUC:", best_auc_bal)
print("[1:1] Best Params:", best_params_bal)

# 訓練與測試
estimator_bal = DecisionTreeClassifier(max_depth=best_params_bal['estimator__max_depth'], 
                                       min_samples_split=6, min_samples_leaf=4, random_state=42)
best_model_bal = AdaBoostClassifier(estimator=estimator_bal,
                                    n_estimators=best_params_bal['n_estimators'],
                                    learning_rate=best_params_bal['learning_rate'], random_state=42)
best_model_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = best_model_bal.predict(X_test)
y_prob_bal = best_model_bal.predict_proba(X_test)[:,1]

print("[1:1] Accuracy:", accuracy_score(y_test, y_pred_bal))
print("[1:1] Precision:", precision_score(y_test, y_pred_bal))
print("[1:1] Recall:", recall_score(y_test, y_pred_bal))
print("[1:1] F1-score:", f1_score(y_test, y_pred_bal))
print("[1:1] ROC AUC:", roc_auc_score(y_test, y_prob_bal))

#===============================================
# 總結：依據五個指標（Accuracy, Precision, Recall, F1-score, AUC）綜合比較三種資料處理方式的結果，
# 決定最終採用哪一組模型進行訓練與部署。可搭配 DataFrame 或圖表視覺化整理輸出指標以利比較。

# 提醒：請確認變數 X_train_bal, y_train_bal 為 1:1 配平後資料
