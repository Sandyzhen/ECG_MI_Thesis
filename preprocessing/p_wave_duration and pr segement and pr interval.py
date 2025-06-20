import numpy as np
import pandas as pd

# =============================
# Function: 找出 QRS 的起點 (onset)
# =============================
def find_qrs_onset(ecg_signal, q_peaks, sampling_rate=500):
    window_size = int(0.02 * sampling_rate)  # 使用 40ms 視窗檢查斜率變化
    qrs_onsets = []
    for q_peak in q_peaks:
        min_slope = float('inf')
        onset = q_peak - 1  # 預設為 Q 點前一個點
        for i in range(q_peak - 1, q_peak - window_size, -1):
            if i < 0:
                break
            slope = compute_slope(ecg_signal, i, window_size)
            if slope < min_slope:
                min_slope = slope
                onset = i
        qrs_onsets.append(onset)
    return np.array(qrs_onsets)
  

# =============================
# Function: 偵測 P 峰與等電基線
# =============================
def detect_p_peaks_and_isoelectric_line(ecg_signal, qrs_onsets, sampling_rate=500, search_window=0.15):
    p_peaks = []
    isoelectric_lines = []
    window_samples = int(search_window * sampling_rate)

    for onset in qrs_onsets:
        start = max(0, onset - window_samples)
        end = onset
        segment = ecg_signal[start:end]

        local_max_idx = np.argmax(segment) + start
        local_min_idx = np.argmin(segment) + start

        isoelectric_value = ecg_signal[onset]
        isoelectric_lines.append(isoelectric_value)

        # 選擇離基線最遠的極值作為 P 波
        if abs(ecg_signal[local_max_idx] - isoelectric_value) > abs(ecg_signal[local_min_idx] - isoelectric_value):
            p_peaks.append(local_max_idx)
        else:
            p_peaks.append(local_min_idx)

    return p_peaks, isoelectric_lines

# =============================
# Function: 根據等電基線偵測 P 波起始與結束
# =============================
def detect_p_wave_boundaries(ecg_signal, p_peaks, isoelectric_values, qrs_onsets, sampling_rate=500, search_window=0.07):
    window_samples = int(search_window * sampling_rate)
    p_wave_starts = []
    p_wave_ends = []

    for p_peak, isoelectric_value, qrs_onset in zip(p_peaks, isoelectric_values, qrs_onsets):
        # P 波起始：由 P 峰向左回推，找訊號重新接近等電基線
        start_search_start = max(0, p_peak - window_samples)
        start_segment = ecg_signal[start_search_start:p_peak]
        threshold = isoelectric_value + 0.5 * np.std(start_segment)
        crossings = np.where(np.diff(np.sign(start_segment - threshold)))[0]
        p_start = crossings[-1] + start_search_start if crossings.size else start_search_start
        p_wave_starts.append(p_start)

        # P 波結束：由 P 峰向右推進，直到訊號再度靠近基線
        end_search_end = min(qrs_onset, p_peak + window_samples)
        end_segment = ecg_signal[p_peak:end_search_end]
        threshold = isoelectric_value + 0.5 * np.std(end_segment)
        crossings = np.where(np.diff(np.sign(end_segment - threshold)))[0]
        p_end = crossings[0] + p_peak if crossings.size else min(p_peak + int(0.04 * sampling_rate), qrs_onset - 1)
        p_wave_ends.append(p_end)

    return np.array(p_wave_starts), np.array(p_wave_ends)

# =============================
# Function: 計算所有病人 P 波持續時間
# =============================
def compute_p_wave_duration_for_all_patients(X_detrended, r_peaks_all_patients, sampling_rate=500):
    num_patients, _, num_leads = X_detrended.shape
    p_wave_durations_all = np.empty((num_patients, num_leads), dtype=object)

    for patient_idx in range(num_patients):
        for lead_idx in range(num_leads):
            if np.all(r_peaks_all_patients[patient_idx, lead_idx] == 'INCOMPLETE'):
                continue

            ecg_signal = X_detrended[patient_idx, :, lead_idx]
            r_peaks = r_peaks_all_patients[patient_idx, lead_idx]

            interval = int(0.070 * sampling_rate)
            qrs_regions = [(r - interval, r + interval) for r in r_peaks]
            q_peaks = find_q_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            s_peaks = find_s_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)

            qrs_onsets = find_qrs_onset(ecg_signal, q_peaks, sampling_rate)
            qrs_offsets = find_qrs_offset(ecg_signal, s_peaks, sampling_rate)

            p_peaks, isoelectric_values = detect_p_peaks_and_isoelectric_line(ecg_signal, qrs_onsets)
            p_wave_starts, p_wave_ends = detect_p_wave_boundaries(ecg_signal, p_peaks, isoelectric_values, qrs_onsets)

            ms_per_sample = 1000 / sampling_rate
            durations = [(end - start) * ms_per_sample for start, end in zip(p_wave_starts, p_wave_ends)]
            p_wave_durations_all[patient_idx, lead_idx] = durations

    return p_wave_durations_all

# =============================
# Function: PR segment = P 波結束點 到 QRS 起始點
# =============================
def compute_pr_segment_duration_for_all_patients(X_detrended, r_peaks_all_patients, sampling_rate=500):
    num_patients, _, num_leads = X_detrended.shape
    pr_segment_durations_all = np.empty((num_patients, num_leads), dtype=object)

    for patient_idx in range(num_patients):
        for lead_idx in range(num_leads):
            if np.all(r_peaks_all_patients[patient_idx, lead_idx] == 'INCOMPLETE'):
                continue

            ecg_signal = X_detrended[patient_idx, :, lead_idx]
            r_peaks = r_peaks_all_patients[patient_idx, lead_idx]

            interval = int(0.070 * sampling_rate)
            qrs_regions = [(r - interval, r + interval) for r in r_peaks]
            q_peaks = find_q_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            s_peaks = find_s_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)

            qrs_onsets = find_qrs_onset(ecg_signal, q_peaks, sampling_rate)
            qrs_offsets = find_qrs_offset(ecg_signal, s_peaks, sampling_rate)

            p_peaks, isoelectric_values = detect_p_peaks_and_isoelectric_line(ecg_signal, qrs_onsets)
            p_wave_starts, p_wave_ends = detect_p_wave_boundaries(ecg_signal, p_peaks, isoelectric_values, qrs_onsets)

            ms_per_sample = 1000 / sampling_rate
            durations = [(onset - end) * ms_per_sample for end, onset in zip(p_wave_ends, qrs_onsets)]
            pr_segment_durations_all[patient_idx, lead_idx] = durations

    return pr_segment_durations_all

# =============================
# Function: PR interval = 從 P 波起始點 到 QRS 起始點
# =============================
def compute_pr_interval_duration_for_all_patients(X_detrended, r_peaks_all_patients, sampling_rate=500):
    num_patients, _, num_leads = X_detrended.shape
    pr_interval_durations_all = np.empty((num_patients, num_leads), dtype=object)

    for patient_idx in range(num_patients):
        for lead_idx in range(num_leads):
            if np.all(r_peaks_all_patients[patient_idx, lead_idx] == 'INCOMPLETE'):
                continue

            ecg_signal = X_detrended[patient_idx, :, lead_idx]
            r_peaks = r_peaks_all_patients[patient_idx, lead_idx]

            interval = int(0.070 * sampling_rate)
            qrs_regions = [(r - interval, r + interval) for r in r_peaks]
            q_peaks = find_q_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            s_peaks = find_s_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)

            qrs_onsets = find_qrs_onset(ecg_signal, q_peaks, sampling_rate)
            p_peaks, isoelectric_values = detect_p_peaks_and_isoelectric_line(ecg_signal, qrs_onsets)
            p_wave_starts, p_wave_ends = detect_p_wave_boundaries(ecg_signal, p_peaks, isoelectric_values, qrs_onsets)

            ms_per_sample = 1000 / sampling_rate
            durations = [(onset - start) * ms_per_sample for start, onset in zip(p_wave_starts, qrs_onsets)]
            pr_interval_durations_all[patient_idx, lead_idx] = durations

    return pr_interval_durations_all
