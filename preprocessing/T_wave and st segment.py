def find_qrs_offset(ecg_signal, s_peaks, sampling_rate=500):
    window_size = int(0.02 * sampling_rate)  # 40ms 窗口
    qrs_offsets = []
    for s_peak in s_peaks:
        min_slope = float('inf')
        offset = s_peak + 1
        for i in range(s_peak + 1, s_peak + window_size):
            if i >= len(ecg_signal):
                break
            slope = compute_slope(ecg_signal, i, window_size)
            if slope < min_slope:
                min_slope = slope
                offset = i
        qrs_offsets.append(offset)
    return np.array(qrs_offsets)


# ✅ 偵測 T 波峰值與等電基線

def detect_t_peaks(ecg_signal, qrs_offsets, sampling_rate, search_window=0.35):
    t_peaks = []
    isoelectric_lines = []
    window_samples = int(search_window * sampling_rate)

    for offset in qrs_offsets:
        start = offset
        end = min(len(ecg_signal), offset + window_samples)
        segment = ecg_signal[start:end]

        local_max_idx = np.argmax(segment) + start
        local_min_idx = np.argmin(segment) + start
        isoelectric_value = ecg_signal[offset]
        isoelectric_lines.append(isoelectric_value)

        if abs(ecg_signal[local_max_idx] - isoelectric_value) > abs(ecg_signal[local_min_idx] - isoelectric_value):
            t_peaks.append(local_max_idx)
        else:
            t_peaks.append(local_min_idx)

    return t_peaks, isoelectric_lines


# ✅ 偵測 T 波邊界（起始與結束）

def detect_t_wave_boundaries(ecg_signal, t_peaks, isoelectric_values, qrs_offsets, sampling_rate=500, search_window=0.12):
    window_samples = int(search_window * sampling_rate)
    t_wave_starts = []
    t_wave_ends = []

    for t_peak, iso_val, qrs_offset in zip(t_peaks, isoelectric_values, qrs_offsets):
        # 起始點
        start_start = max(t_peak - window_samples, qrs_offset + 1)
        start_end = t_peak
        start_segment = ecg_signal[start_start:start_end]
        threshold = iso_val + 0.5 * np.std(start_segment)
        crossings = np.where(np.diff(np.sign(start_segment - threshold)))[0]
        t_start = crossings[-1] + start_start if crossings.size else start_start
        t_wave_starts.append(t_start)

        # 結束點
        end_start = t_peak
        end_end = t_peak + window_samples
        end_segment = ecg_signal[end_start:end_end]
        threshold = iso_val + 0.5 * np.std(end_segment)
        crossings = np.where(np.diff(np.sign(end_segment - threshold)))[0]
        t_end = crossings[0] + end_start if crossings.size else end_end
        t_wave_ends.append(min(len(ecg_signal) - 1, t_end))

    return np.array(t_wave_starts), np.array(t_wave_ends)


# ✅ 計算 ST segment 時長（QRS offset ~ T 波起點）

def compute_st_segment_duration_all_patients(X_detrended, r_peaks_all_patients, sampling_rate=500):
    num_patients, _, num_leads = X_detrended.shape
    st_durations_all = np.empty((num_patients, num_leads), dtype=object)

    for pid in range(num_patients):
        for lid in range(num_leads):
            if np.all(r_peaks_all_patients[pid, lid] == 'INCOMPLETE'):
                continue

            ecg_signal = X_detrended[pid, :, lid]
            r_peaks = r_peaks_all_patients[pid, lid]
            interval = int(0.07 * sampling_rate)
            qrs_regions = [(r - interval, r + interval) for r in r_peaks]

            q_peaks = find_q_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            s_peaks = find_s_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            qrs_offsets = find_qrs_offset(ecg_signal, s_peaks, sampling_rate)
            t_peaks, isoelectric_values = detect_t_peaks(ecg_signal, qrs_offsets, sampling_rate)
            t_starts, _ = detect_t_wave_boundaries(ecg_signal, t_peaks, isoelectric_values, qrs_offsets)

            ms_per_sample = 1000 / sampling_rate
            st_durations = [(start - offset) * ms_per_sample for offset, start in zip(qrs_offsets, t_starts)]
            st_durations_all[pid, lid] = st_durations

    return st_durations_all


# ✅ T 波持續時間

def compute_t_wave_duration_for_all_patients(X_detrended, r_peaks_all_patients, sampling_rate=500):
    num_patients, _, num_leads = X_detrended.shape
    t_wave_durations_all = np.empty((num_patients, num_leads), dtype=object)

    for pid in range(num_patients):
        for lid in range(num_leads):
            if np.all(r_peaks_all_patients[pid, lid] == 'INCOMPLETE'):
                continue

            ecg_signal = X_detrended[pid, :, lid]
            r_peaks = r_peaks_all_patients[pid, lid]
            interval = int(0.07 * sampling_rate)
            qrs_regions = [(r - interval, r + interval) for r in r_peaks]

            q_peaks = find_q_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            s_peaks = find_s_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            qrs_offsets = find_qrs_offset(ecg_signal, s_peaks, sampling_rate)
            t_peaks, isoelectric_values = detect_t_peaks(ecg_signal, qrs_offsets, sampling_rate)
            t_starts, t_ends = detect_t_wave_boundaries(ecg_signal, t_peaks, isoelectric_values, qrs_offsets)

            ms_per_sample = 1000 / sampling_rate
            durations = [(end - start) * ms_per_sample for start, end in zip(t_starts, t_ends)]
            t_wave_durations_all[pid, lid] = durations

    return t_wave_durations_all
