#找出 S 波的起始點（從 S 峰往左找斜率下降處）
def find_s_wave_onsets(cleaned_base, s_peak_indices, r_peaks, sampling_rate=500):
    s_wave_onsets = []
    backward_window = int(0.05 * sampling_rate)  # 往前搜尋 50ms 的範圍
    derivative = np.diff(cleaned_base)  # 計算一階導數（斜率）

    for s_peak in s_peak_indices:
        start = s_peak
        while start > s_peak - backward_window and derivative[start] > -0.01:
            start -= 1  # 直到斜率變負才停止，代表 S 波起始
        s_wave_onsets.append(start)

    return np.array(s_wave_onsets)

# 計算所有病人所有導聯的 S 波持續時間（從 S 波起點到 QRS offset）
def compute_s_duration_for_all_patients(X_detrended, r_peaks_all_patients, sampling_rate=500):
    num_patients, _, num_leads = X_detrended.shape
    s_durations_all = np.empty((num_patients, num_leads), dtype=object)

    for patient_idx in range(num_patients):
        for lead_idx in range(num_leads):
            if np.all(r_peaks_all_patients[patient_idx, lead_idx] == 'INCOMPLETE'):
                continue

            ecg_signal = X_detrended[patient_idx, :, lead_idx]
            r_peaks = r_peaks_all_patients[patient_idx, lead_idx]

            # 定義 QRS 區域（前後各 70ms）
            interval = int(0.070 * sampling_rate)
            qrs_regions = [(r - interval, r + interval) for r in r_peaks]

            # 找出 Q、S 點
            q_peaks = find_q_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            s_peaks = find_s_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)

            # 找出 QRS offset（結束點）與 S 波起點
            qrs_offsets = find_qrs_offset(ecg_signal, s_peaks, sampling_rate)
            s_wave_onsets = find_s_wave_onsets(ecg_signal, s_peaks, r_peaks)

            # 計算每段 S 波的持續時間（單位毫秒）
            ms_per_sample = 1000 / sampling_rate
            s_durations_ms = [(end - start) * ms_per_sample for start, end in zip(s_wave_onsets, qrs_offsets)]

            s_durations_all[patient_idx, lead_idx] = s_durations_ms

    return s_durations_all

#計算所有 S 波時間
s_durations_all_patients = compute_s_duration_for_all_patients(X_detrended, r_peaks_all_patients)

#計算每位病人每個導聯的 S 波平均持續時間
def calculate_average_s_durations(s_durations_all_patients):
    num_patients, num_leads = s_durations_all_patients.shape
    average_s_durations = np.empty((num_patients, num_leads))

    for i in range(num_patients):
        for j in range(num_leads):
            intervals = s_durations_all_patients[i, j]
            if intervals is not None and len(intervals) > 0:
                average_s_durations[i, j] = np.mean(intervals)
            else:
                average_s_durations[i, j] = np.nan

    return average_s_durations

average_s_durations = calculate_average_s_durations(s_durations_all_patients)

#計算每位病人 S 波持續時間的全導聯平均值
def calculate_global_average_s_durations(average_s_durations):
    global_averages = np.nanmean(average_s_durations, axis=1)  # 忽略 NaN
    return global_averages

global_average_s_durations = calculate_global_average_s_durations(average_s_durations)

#轉成 DataFrame 顯示
df = pd.DataFrame(global_average_s_durations)


# 找出 Q 波的結束點（從 Q 峰往右找，直到斜率變正）
def find_q_wave_offsets(cleaned_base, q_peak_indices, r_peaks, sampling_rate=500):
    q_wave_offsets = []
    forward_window = int(0.05 * sampling_rate)  # 往右搜尋 50ms
    derivative = np.diff(cleaned_base)

    for q_peak in q_peak_indices:
        end = q_peak
        while end < q_peak + forward_window and derivative[end] < 0.01:
            end += 1
        q_wave_offsets.append(end)

    return np.array(q_wave_offsets)

# 計算所有病人 Q 波持續時間（QRS onset 到 Q 波結束）
def compute_q_duration_for_all_patients(X_detrended, r_peaks_all_patients, sampling_rate=500):
    num_patients, _, num_leads = X_detrended.shape
    q_durations_all = np.empty((num_patients, num_leads), dtype=object)

    for patient_idx in range(num_patients):
        for lead_idx in range(num_leads):
            if np.all(r_peaks_all_patients[patient_idx, lead_idx] == 'INCOMPLETE'):
                continue

            ecg_signal = X_detrended[patient_idx, :, lead_idx]
            r_peaks = r_peaks_all_patients[patient_idx, lead_idx]

            interval = int(0.070 * sampling_rate)
            qrs_regions = [(r - interval, r + interval) for r in r_peaks]

            # 找出 Q 點與相關區域
            q_peaks = find_q_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            s_peaks = find_s_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)

            qrs_onsets = find_qrs_onset(ecg_signal, q_peaks, sampling_rate)
            q_wave_offsets = find_q_wave_offsets(ecg_signal, q_peaks, r_peaks)

            ms_per_sample = 1000 / sampling_rate
            q_durations_ms = [(end - start) * ms_per_sample for start, end in zip(qrs_onsets, q_wave_offsets)]

            q_durations_all[patient_idx, lead_idx] = q_durations_ms

    return q_durations_all

#計算所有 Q 波時間
q_durations_all_patients = compute_q_duration_for_all_patients(X_detrended, r_peaks_all_patients)

#計算每位病人每個導聯的 Q 波平均時間
def calculate_average_q_durations(q_durations_all_patients):
    num_patients, num_leads = q_durations_all_patients.shape
    average_q_durations = np.empty((num_patients, num_leads))

    for i in range(num_patients):
        for j in range(num_leads):
            intervals = q_durations_all_patients[i, j]
            if intervals is not None and len(intervals) > 0:
                average_q_durations[i, j] = np.mean(intervals)
            else:
                average_q_durations[i, j] = np.nan

    return average_q_durations

average_q_durations = calculate_average_q_durations(q_durations_all_patients)

#計算每位病人 Q 波全導聯的平均值
def calculate_global_average_q_durations(average_q_durations):
    global_averages = np.nanmean(average_q_durations, axis=1)
    return global_averages

global_average_q_durations = calculate_global_average_q_durations(average_q_durations)

#轉為 DataFrame
df = pd.DataFrame(global_average_q_durations)
