"""
此腳本針對 ECG 特徵值資料集進行資料清洗、1:1 平衡處理、Train/Val/Test 分割，供後續模型訓練使用。
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

# === 資料讀取與處理 ===
os.chdir("C:\\Users\\sandy\\Desktop\\心電圖資料集")
ECG_data = pd.read_csv("ECG特徵值持續時間+人口學+資料擴增20(MI).csv")
ECG_data["age"] = np.where(ECG_data["age"].isnull(), np.nanmean(ECG_data["age"]), ECG_data["age"])
ECG_data = ECG_data.rename(columns={'filtered_diagnostic_subclass': 'filtered_diagnostic_superclass'})

# === 1:1 隨機抽樣平衡 MI 與 NORM ===
mi_sample = ECG_data[ECG_data['filtered_diagnostic_superclass'] == 1].sample(n=5400, random_state=42)
norm_sample = ECG_data[ECG_data['filtered_diagnostic_superclass'] == 0].sample(n=5400, random_state=42)
data_balanced = pd.concat([mi_sample, norm_sample])

X = data_balanced.drop(columns=['filtered_diagnostic_superclass'])
y = data_balanced['filtered_diagnostic_superclass']

# === 資料切分（Train:Val:Test = 70:20:10）===
X_train, X_combine, y_train, y_combine = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_combine, y_combine, test_size=1/3, stratify=y_combine, random_state=42)

# === 欠抽樣處理（只作用在訓練集）===
rus = RandomUnderSampler(random_state=25)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

# === 顯示各類別樣本數 ===
print(\"訓練集樣本數（原始）:\", y_train.value_counts().to_dict())
print(\"訓練集樣本數（欠抽樣後）:\", y_train_res.value_counts().to_dict())
print(\"驗證集樣本數:\", y_val.value_counts().to_dict())
print(\"測試集樣本數:\", y_test.value_counts().to_dict())
