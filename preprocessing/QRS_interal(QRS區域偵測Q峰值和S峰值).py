#=======================S峰值和Q峰值偵測===========================
# 在 R 峰與 QRS 區域右半部之間尋找 S 峰（最小值）
def find_s_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate=500):
    s_peaks = []
    for r_pos, (start, end) in zip(r_peaks, qrs_regions):
        rs_region = ecg_signal[r_pos:end]
        s_pos = r_pos + np.argmin(rs_region)  # 找最小值位置，代表 S 點
        s_peaks.append(s_pos)
    return np.array(s_peaks)

# 在 QRS 區域左半部與 R 峰之間尋找 Q 峰（最小值）
def find_q_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate=500):
    q_peaks = []
    for r_pos, (start, end) in zip(r_peaks, qrs_regions):
        qr_region = ecg_signal[start:r_pos]
        q_pos = start + np.argmin(qr_region)  # 找最小值位置，代表 Q 點
        q_peaks.append(q_pos)
    return np.array(q_peaks)
  
#=======================QRS起始點峰值和QRS結束點偵測===========================
# 計算從某一點開始的斜率（作為波形變化速率的估計）
def compute_slope(signal, idx, window_size):
    if idx + window_size < len(signal):
        return (signal[idx + window_size] - signal[idx]) / window_size
    else:
        return 0  # 避免超出邊界

# 根據 Q 峰點往前推，尋找 QRS 起始點（slope 最小）
def find_qrs_onset(ecg_signal, q_peaks, sampling_rate=500):
    window_size = int(0.02 * sampling_rate)  # 40ms 的視窗
    qrs_onsets = []
    for q_peak in q_peaks:
        min_slope = float('inf')
        onset = q_peak - 1  # 預設初始點
        for i in range(q_peak - 1, q_peak - window_size, -1):
            if i < 0:
                break
            slope = compute_slope(ecg_signal, i, window_size)
            if slope < min_slope:
                min_slope = slope
                onset = i
        qrs_onsets.append(onset)
    return np.array(qrs_onsets)

# 根據 S 峰點往後推，尋找 QRS 結束點（slope 最小）
def find_qrs_offset(ecg_signal, s_peaks, sampling_rate=500):
    window_size = int(0.02 * sampling_rate)  # 40ms 的視窗
    qrs_offsets = []
    for s_peak in s_peaks:
        min_slope = float('inf')
        offset = s_peak + 1  # 預設起始點
        for i in range(s_peak + 1, s_peak + window_size):
            if i >= len(ecg_signal):
                break
            slope = compute_slope(ecg_signal, i, window_size)
            if slope < min_slope:
                min_slope = slope
                offset = i
        qrs_offsets.append(offset)
    return np.array(qrs_offsets)

# 計算所有病人、所有導聯的 QRS 時間長度
def compute_qrs_duration_for_all_patients(X_detrended, r_peaks_all_patients, sampling_rate=500):
    num_patients, _, num_leads = X_detrended.shape
    qrs_durations_all = np.empty((num_patients, num_leads), dtype=object)  # 儲存每位病人的所有 QRS durations

    for patient_idx in range(num_patients):
        for lead_idx in range(num_leads):
            if np.all(r_peaks_all_patients[patient_idx, lead_idx] == 'INCOMPLETE'):
                continue  # 忽略無效資料

            ecg_signal = X_detrended[patient_idx, :, lead_idx]
            r_peaks = r_peaks_all_patients[patient_idx, lead_idx]

            # 定義每個 R 峰附近的 QRS 區域（左右各取 60ms）
            interval = int(0.060 * sampling_rate)
            qrs_regions = [(r - interval, r + interval) for r in r_peaks]

            # 找出 Q、S 點與 QRS 起止點
            s_peaks = find_s_peaks_within_region(ecg_signal, r_peaks, qrs_regions)
            q_peaks = find_q_peaks_within_region(ecg_signal, r_peaks, qrs_regions)
            qrs_onsets = find_qrs_onset(ecg_signal, q_peaks)
            qrs_offsets = find_qrs_offset(ecg_signal, s_peaks)

            # 將 sample index 差距轉換為毫秒
            ms_per_sample = 1000 / sampling_rate
            qrs_durations_ms = [(offset - onset) * ms_per_sample for onset, offset in zip(qrs_onsets, qrs_offsets)]

            qrs_durations_all[patient_idx, lead_idx] = qrs_durations_ms

    return qrs_durations_all

# 呼叫計算所有病人的 QRS durations
qrs_durations_all_patients = compute_qrs_duration_for_all_patients(X_detrended, r_peaks_all_patients)

# 計算每位病人每個導聯的平均 QRS duration
def calculate_average_qrs_durations(qrs_durations_all_patients):
    num_patients, num_leads = qrs_durations_all_patients.shape
    average_qrs_durations = np.empty((num_patients, num_leads))

    for i in range(num_patients):
        for j in range(num_leads):
            intervals = qrs_durations_all_patients[i, j]
            if intervals is not None and len(intervals) > 0:
                average_qrs_durations[i, j] = np.mean(intervals)
            else:
                average_qrs_durations[i, j] = np.nan  # 若無資料則填 NaN

    return average_qrs_durations

average_qrs_durations = calculate_average_qrs_durations(qrs_durations_all_patients)

# 計算每位病人 12 導聯的 QRS duration 全域平均值
def calculate_global_average_qrs_wave_durations(average_qrs_durations):
    global_averages = np.nanmean(average_qrs_durations, axis=1)  # 忽略 NaN 計算平均
    return global_averages

global_average_qrs_durations = calculate_global_average_qrs_wave_durations(average_qrs_durations)
