import numpy as np
import matplotlib.pyplot as plt

# ========== 計算 RR 間期 ==========

def calculate_rr_intervals(r_peaks_all_patients, sampling_rate=500):
    num_patients, num_leads = r_peaks_all_patients.shape
    rr_intervals_all_patients = np.empty((num_patients, num_leads), dtype=object)

    for i in range(num_patients):
        for j in range(num_leads):
            r_peaks = r_peaks_all_patients[i, j]
            if isinstance(r_peaks, str) or r_peaks is None or len(r_peaks) < 2:
                rr_intervals_all_patients[i, j] = np.array([])
                continue
            rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # 轉為毫秒
            rr_intervals_all_patients[i, j] = rr_intervals

    return rr_intervals_all_patients

# ========== 計算平均 RR 間期 ==========

def calculate_mean_rr_interval(rr_intervals_all_patients):
    num_patients, num_leads = rr_intervals_all_patients.shape
    mean_rr_interval_all_patients = np.empty((num_patients, num_leads))

    for i in range(num_patients):
        for j in range(num_leads):
            rr_intervals = rr_intervals_all_patients[i, j]
            if rr_intervals is not None and len(rr_intervals) > 0:
                mean_rr_interval_all_patients[i, j] = np.mean(rr_intervals)
            else:
                mean_rr_interval_all_patients[i, j] = np.nan

    return mean_rr_interval_all_patients

# ========== 計算心率（HR） ==========

def calculate_heart_rate(rr_intervals_all_patients):
    num_patients, num_leads = rr_intervals_all_patients.shape
    heart_rate_all_patients = np.empty((num_patients, num_leads))

    for i in range(num_patients):
        for j in range(num_leads):
            rr_intervals = rr_intervals_all_patients[i, j]
            if rr_intervals is not None and len(rr_intervals) > 0:
                mean_rr_interval_sec = np.mean(rr_intervals) / 1000.0
                heart_rate_all_patients[i, j] = 60.0 / mean_rr_interval_sec
            else:
                heart_rate_all_patients[i, j] = np.nan

    return heart_rate_all_patients

# ========== RR 間期標註圖（單一病人） ==========

def plot_rr_intervals(X_cleaned, r_peaks_all_patients, rr_intervals_all_patients, patient_index, lead_index):
    ecg_signal = X_cleaned[patient_index, :, lead_index]
    r_peaks = r_peaks_all_patients[patient_index, lead_index]
    rr_intervals = rr_intervals_all_patients[patient_index, lead_index]

    if isinstance(r_peaks, str) or len(r_peaks) < 2:
        print("No valid R peaks to display.")
        return

    plt.figure(figsize=(20, 5))
    plt.plot(ecg_signal)
    plt.title(f'ECG signal with RR intervals (Patient {patient_index+1}, Lead {lead_index+1})')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')

    for r1, r2, rr in zip(r_peaks[:-1], r_peaks[1:], rr_intervals):
        plt.axvspan(r1, r2, color='yellow', alpha=0.3)
        plt.text((r1 + r2) / 2, ecg_signal[r1], f'{rr:.0f} ms', color='black', ha='center', fontsize=8)

    plt.tight_layout()
    # plt.savefig(f"figures/rr_intervals_p{patient_index+1}_lead{lead_index+1}.png")  # 可開啟這行存圖
    plt.show()

# ========== 範例呼叫方式（請在主程式執行） ==========

# rr_intervals_all_patients = calculate_rr_intervals(r_peaks_all_patients)
# mean_rr_interval_all_patients = calculate_mean_rr_interval(rr_intervals_all_patients)
# heart_rate_all_patients = calculate_heart_rate(rr_intervals_all_patients)
# average_heart_rate_per_patient = np.nanmean(heart_rate_all_patients, axis=1)
# plot_rr_intervals(X_cleaned, r_peaks_all_patients, rr_intervals_all_patients, 23, 0)
