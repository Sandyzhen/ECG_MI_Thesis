import time
import numpy as np
import wfdb
import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy.fftpack import fft, ifft 
from scipy import signal




def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = "C:\\Users\\sandy\\Desktop\\新增資料夾\\"
sampling_rate = 500
# 读取文件并转换标签
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# 获取原始信号数据
X = load_raw_data(Y, sampling_rate, path)
# 获取scp_statements.csv中的诊断信息
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))
# 添加诊断信息

Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)



Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
# 只保留包含單一標籤的資料，多標籤視為空值
Y['diagnostic_superclass_2'] = Y['diagnostic_superclass'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else [])

mi_count = Y['diagnostic_superclass'].apply(lambda x: "MI" in x).sum()

mi_count = Y['diagnostic_superclass'].apply(lambda x: "NORM" in x).sum()

df_filter = Y[Y['diagnostic_superclass'].apply(lambda x: "MI" in x or "NORM" in x)]


def filter_diagnostic_superclass(x):
    if "MI" in x:
        return ["MI"]
    elif "NORM" in x:
        return ["NORM"]
    else:
        return []

df_filter['filtered_diagnostic_superclass'] = df_filter['diagnostic_superclass'].apply(filter_diagnostic_superclass)
# 先計算 "MI" 和 "NORM" 的數量
num_mi = df_filter['diagnostic_superclass'].apply(lambda x: "MI" in x).sum()
num_norm = df_filter['diagnostic_superclass'].apply(lambda x: "NORM" in x).sum()
# 印出結果


# 假設你的 DataFrame 為 df，欲處理的欄位為 'diagnostic_superclass'
df_filter['filtered_diagnostic_superclass'] = df_filter['filtered_diagnostic_superclass'].apply(lambda x: x[0]).str.replace('\["', '').str.replace('\"]', '')
df_filter['filtered_diagnostic_superclass'] = df_filter['filtered_diagnostic_superclass'].replace({'NORM': 0, 'MI': 1})


X = load_raw_data(df_filter, sampling_rate, path)




import numpy as np
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

window_size = 10  # 這是一個範例值，您可以根據您的數據和需求進行調整

# 假設 X 是您的3D數據
smoothed_data_ma = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[2]):
        smoothed_data_ma[i,:,j] = moving_average(X[i,:,j], window_size)


import numpy as np
import neurokit2 as nk

def filter_patient_data(patient_data):
    # patient_data 的形狀應為 (5000, 12)
    filtered_data = np.zeros_like(patient_data)
    for i in range(patient_data.shape[1]):  # 對於每一個通道
        filtered_data[:, i] = nk.signal_filter(patient_data[:, i], lowcut=0.5, highcut=50, method="butterworth", order=2)
    return filtered_data

# 初始化一個與 X 形狀相同的數組來保存濾波後的數據
X_filtered = np.zeros_like(smoothed_data_ma)

for j in range(smoothed_data_ma.shape[0]):  # 對於每一位病人
    X_filtered[j] = filter_patient_data(smoothed_data_ma[j])

# 現在 X_filtered 包含所有病人濾波後的數據


def detrend_patient_data(patient_data, order=10):
    # patient_data 的形狀應為 (5000, 12)
    detrended_data = np.zeros_like(patient_data)
    for i in range(patient_data.shape[1]):  # 對於每一個通道
        detrended_data[:, i] = nk.signal_detrend(patient_data[:, i], order=order)
    return detrended_data

# 初始化一個與 X_filtered 形狀相同的數組來保存校正後的數據
X_detrended = np.zeros_like(X_filtered)

for j in range(X_filtered.shape[0]):  # 對於每一位病人
    X_detrended[j] = detrend_patient_data(X_filtered[j])
