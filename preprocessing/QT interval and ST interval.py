# ✅ QT 間期計算

def compute_qt_interval_duration_for_all_patients(X_detrended, r_peaks_all_patients, sampling_rate=500):
    num_patients, _, num_leads = X_detrended.shape
    qt_interval_all = np.empty((num_patients, num_leads), dtype=object)

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
            qrs_onsets = find_qrs_onset(ecg_signal, q_peaks, sampling_rate)
            qrs_offsets = find_qrs_offset(ecg_signal, s_peaks, sampling_rate)
            t_peaks, isoelectric_values = detect_t_peaks(ecg_signal, qrs_offsets, sampling_rate)
            _, t_ends = detect_t_wave_boundaries(ecg_signal, t_peaks, isoelectric_values, qrs_offsets)

            ms_per_sample = 1000 / sampling_rate
            durations = [(end - start) * ms_per_sample for start, end in zip(qrs_onsets, t_ends)]
            qt_interval_all[pid, lid] = durations

    return qt_interval_all


def calculate_average_qt_durations(qt_interval_all):
    num_patients, num_leads = qt_interval_all.shape
    averages = np.empty((num_patients, num_leads))

    for i in range(num_patients):
        for j in range(num_leads):
            intervals = qt_interval_all[i, j]
            if intervals is not None and len(intervals) > 0:
                averages[i, j] = np.mean(intervals)
            else:
                averages[i, j] = np.nan
    return averages


def calculate_global_average_qt_interval_durations(average_qt_durations):
    return np.nanmean(average_qt_durations, axis=1)


# ✅ ST 間期計算

def compute_st_interval_duration_for_all_patients(X_detrended, r_peaks_all_patients, sampling_rate=500):
    num_patients, _, num_leads = X_detrended.shape
    st_interval_all = np.empty((num_patients, num_leads), dtype=object)

    for pid in range(num_patients):
        for lid in range(num_leads):
            r_peaks = r_peaks_all_patients[pid, lid]
            if isinstance(r_peaks, str) and r_peaks == 'INCOMPLETE':
                st_interval_all[pid, lid] = 'INCOMPLETE'
                continue

            ecg_signal = X_detrended[pid, :, lid]
            interval = int(0.07 * sampling_rate)
            qrs_regions = [(r - interval, r + interval) for r in r_peaks]

            q_peaks = find_q_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            s_peaks = find_s_peaks_within_region(ecg_signal, r_peaks, qrs_regions, sampling_rate)
            qrs_offsets = find_qrs_offset(ecg_signal, s_peaks, sampling_rate)
            t_peaks, isoelectric_values = detect_t_peaks(ecg_signal, qrs_offsets, sampling_rate)
            _, t_ends = detect_t_wave_boundaries(ecg_signal, t_peaks, isoelectric_values, qrs_offsets)

            ms_per_sample = 1000 / sampling_rate
            durations = [(end - start) * ms_per_sample for start, end in zip(qrs_offsets, t_ends)]
            st_interval_all[pid, lid] = durations

    return st_interval_all


def calculate_average_st_interval_durations(st_interval_all):
    num_patients, num_leads = st_interval_all.shape
    average_st = np.empty((num_patients, num_leads))

    for i in range(num_patients):
        for j in range(num_leads):
            intervals = st_interval_all[i, j]
            if intervals is not None and len(intervals) > 0:
                numeric_intervals = [x for x in intervals if not isinstance(x, str)]
                if len(numeric_intervals) > 0:
                    average_st[i, j] = np.mean(numeric_intervals)
                else:
                    average_st[i, j] = np.nan
            else:
                average_st[i, j] = np.nan

    return average_st


def calculate_global_average_st_interval_durations(average_st):
    return np.nanmean(average_st, axis=1)
