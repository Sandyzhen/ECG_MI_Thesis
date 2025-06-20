import numpy as np
def calculate_overall_average_intervals(average_intervals):
    num_patients, num_leads = average_intervals.shape

    overall_average_intervals = np.empty(num_patients)

    for patient_index in range(num_patients):
        patient_intervals = average_intervals[patient_index]
        overall_average_intervals[patient_index] = np.nanmean(patient_intervals)

    return overall_average_intervals


def calculate_overall_min_intervals(min_intervals):
    num_patients, num_leads = min_intervals.shape

    overall_min_intervals = np.full(num_patients, np.nan)


    for patient_index in range(num_patients):
        patient_intervals = min_intervals[patient_index]
        overall_min_intervals[patient_index] = np.nanmin(patient_intervals)

    return overall_min_intervals


def calculate_overall_max_intervals(max_intervals):
    num_patients, num_leads = max_intervals.shape

    overall_max_intervals = np.empty(num_patients)

    for patient_index in range(num_patients):
        patient_intervals = max_intervals[patient_index]
        overall_max_intervals[patient_index] = np.nanmax(patient_intervals)

    return overall_max_intervals


def calculate_overall_median_intervals(median_intervals):
    num_patients, num_leads = median_intervals.shape

    overall_median_intervals = np.empty(num_patients)

    for patient_index in range(num_patients):
        patient_intervals = median_intervals[patient_index]
        overall_median_intervals[patient_index] = np.nanmedian(patient_intervals)

    return overall_median_intervals


def calculate_overall_std_intervals(std_intervals):
    num_patients, num_leads = std_intervals.shape

    overall_std_intervals = np.empty(num_patients)

    for patient_index in range(num_patients):
        patient_intervals = std_intervals[patient_index]
        overall_std_intervals[patient_index] = np.nanstd(patient_intervals)

    return overall_std_intervals  # 這裏返回了一個數組



def calculate_overall_percentiles_25th_intervals(percentiles_25th_intervals):
    num_patients, num_leads = percentiles_25th_intervals.shape

    overall_percentiles_25th_intervals = np.empty(num_patients)

    for patient_index in range(num_patients):
        patient_intervals = percentiles_25th_intervals[patient_index]
        overall_percentiles_25th_intervals[patient_index] = np.nanpercentile(patient_intervals,25)

    return overall_percentiles_25th_intervals


def calculate_overall_percentiles_75th_intervals(percentiles_75th_intervals):
    num_patients, num_leads = percentiles_75th_intervals.shape

    overall_percentiles_75th_intervals = np.empty(num_patients)

    for patient_index in range(num_patients):
        patient_intervals = percentiles_75th_intervals[patient_index]
        overall_percentiles_75th_intervals[patient_index] = np.nanpercentile(patient_intervals,75)

    return overall_percentiles_75th_intervals

def calculate_overall_percentiles_99th_intervals(percentiles_99th_intervals):
    num_patients, num_leads = percentiles_99th_intervals.shape

    overall_percentiles_99th_intervals = np.empty(num_patients)

    for patient_index in range(num_patients):
        patient_intervals = percentiles_99th_intervals[patient_index]
        overall_percentiles_99th_intervals[patient_index] = np.nanpercentile(patient_intervals,99)

    return overall_percentiles_99th_intervals


import numpy as np
def calculate_overall_average_area(average_area):
    num_patients, num_leads = average_area.shape

    overall_average_area = np.empty(num_patients)

    for patient_index in range(num_patients):
        patient_area = average_area[patient_index]
        overall_average_area[patient_index] = np.nanmean(patient_area)

    return overall_average_area







# Calculate the overall average intervals
overall_average_qrs_intervals = calculate_overall_average_intervals(average_qrs_durations)
overall_average_pr_intervals = calculate_overall_average_intervals(average_pr_interval_durations)
overall_average_qt_intervals = calculate_overall_average_intervals(average_qt_durations)
overall_average_st_intervals = calculate_overall_average_intervals(average_st_interval_durations)
overall_average_rr_intervals = calculate_overall_average_intervals(mean_rr_interval_all_patients)
overall_average_p_wave_durations = calculate_overall_average_intervals(average_p_wave_durations)
overall_average_q_wave_durations = calculate_overall_average_intervals(average_q_durations)
overall_average_s_wave_durations = calculate_overall_average_intervals(average_s_durations)
overall_average_t_wave_durations = calculate_overall_average_intervals(average_t_durations)
overall_average_st_segments = calculate_overall_average_intervals(average_st_segment_durations)
overall_average_pr_segments = calculate_overall_average_intervals(average_pr_segment_durations)


# Calculate the overall average intervals
overall_max_qrs_intervals = calculate_overall_max_intervals(average_qrs_durations)
overall_max_pr_intervals = calculate_overall_max_intervals(average_pr_interval_durations)
overall_max_qt_intervals =  calculate_overall_max_intervals(average_qt_durations)
overall_max_st_intervals =  calculate_overall_max_intervals(average_st_interval_durations)
overall_max_rr_intervals =  calculate_overall_max_intervals(mean_rr_interval_all_patients)
overall_max_p_wave_durations = calculate_overall_max_intervals(average_p_wave_durations)
overall_max_q_wave_durations = calculate_overall_max_intervals(average_q_durations)
overall_max_s_wave_durations = calculate_overall_max_intervals(average_s_durations)
overall_max_t_wave_durations =calculate_overall_max_intervals(average_t_durations)
overall_max_st_segments = calculate_overall_max_intervals(average_st_segment_durations)
overall_max_pr_duration =  calculate_overall_max_intervals(average_pr_segment_durations)

# Calculate the overall average intervals
overall_min_qrs_intervals = calculate_overall_min_intervals(average_qrs_durations)
overall_min_pr_intervals = calculate_overall_min_intervals(average_pr_interval_durations)
overall_min_qt_intervals =  calculate_overall_min_intervals(average_qt_durations)
overall_min_st_intervals =  calculate_overall_min_intervals(average_st_interval_durations)
overall_min_rr_intervals =  calculate_overall_min_intervals(mean_rr_interval_all_patients)
overall_min_p_wave_durations = calculate_overall_min_intervals(average_p_wave_durations)
overall_min_q_wave_durations = calculate_overall_min_intervals(average_q_durations)
overall_min_s_wave_durations = calculate_overall_min_intervals(average_s_durations)
overall_min_t_wave_durations =calculate_overall_min_intervals(average_t_durations)
overall_min_st_segments = calculate_overall_min_intervals(average_st_segment_durations)
overall_min_pr_duration =  calculate_overall_min_intervals(average_pr_segment_durations)


# Calculate the overall average intervals
overall_median_qrs_intervals =calculate_overall_median_intervals(average_qrs_durations)
overall_median_pr_intervals =calculate_overall_median_intervals(average_pr_interval_durations)
overall_median_qt_intervals = calculate_overall_median_intervals(average_qt_durations)
overall_median_st_intervals = calculate_overall_median_intervals(average_st_interval_durations)
overall_median_rr_intervals = calculate_overall_median_intervals(mean_rr_interval_all_patients)
overall_median_p_wave_durations = calculate_overall_median_intervals(average_p_wave_durations)
overall_median_q_wave_durations = calculate_overall_median_intervals(average_q_durations)
overall_median_s_wave_durations =calculate_overall_median_intervals(average_s_durations)
overall_median_t_wave_durations =calculate_overall_median_intervals(average_t_durations)
overall_median_st_segments =calculate_overall_median_intervals(average_st_segment_durations)
overall_median_pr_duration = calculate_overall_median_intervals(average_pr_segment_durations)

overall_std_qrs_intervals =calculate_overall_std_intervals(average_qrs_durations)
overall_std_pr_intervals =calculate_overall_std_intervals(average_pr_interval_durations)
overall_std_qt_intervals = calculate_overall_std_intervals(average_qt_durations)
overall_std_st_intervals = calculate_overall_std_intervals(average_st_interval_durations)
overall_std_rr_intervals = calculate_overall_std_intervals(mean_rr_interval_all_patients)
overall_std_p_wave_durations =calculate_overall_std_intervals(average_p_wave_durations)
overall_std_q_wave_durations = calculate_overall_std_intervals(average_q_durations)
overall_std_s_wave_durations =calculate_overall_std_intervals(average_s_durations)
overall_std_t_wave_durations =calculate_overall_std_intervals(average_t_durations)
overall_std_st_segments =calculate_overall_std_intervals(average_st_segment_durations)
overall_std_pr_duration = calculate_overall_std_intervals(average_pr_segment_durations)


overall_percentiles_25th_qrs_intervals =calculate_overall_percentiles_25th_intervals(average_qrs_durations)
overall_percentiles_25th_pr_intervals =calculate_overall_percentiles_25th_intervals(average_pr_interval_durations)
overall_percentiles_25th_qt_intervals = calculate_overall_percentiles_25th_intervals(average_qt_durations)
overall_percentiles_25th_st_intervals =calculate_overall_percentiles_25th_intervals(average_st_interval_durations)
overall_percentiles_25th_rr_intervals = calculate_overall_percentiles_25th_intervals(mean_rr_interval_all_patients)
overall_percentiles_25th_p_wave_durations =calculate_overall_percentiles_25th_intervals(average_p_wave_durations)
overall_percentiles_25th_q_wave_durations = calculate_overall_percentiles_25th_intervals(average_q_durations)
overall_percentiles_25th_s_wave_durations =calculate_overall_percentiles_25th_intervals(average_s_durations)
overall_percentiles_25th_t_wave_durations =calculate_overall_percentiles_25th_intervals(average_t_durations)
overall_percentiles_25th_st_segments =calculate_overall_percentiles_25th_intervals(average_st_segment_durations)
overall_percentiles_25th_pr_duration =calculate_overall_percentiles_25th_intervals(average_pr_segment_durations)



overall_percentiles_75th_qrs_intervals =calculate_overall_percentiles_75th_intervals(average_qrs_durations)
overall_percentiles_75th_pr_intervals =calculate_overall_percentiles_75th_intervals(average_pr_interval_durations)
overall_percentiles_75th_qt_intervals = calculate_overall_percentiles_75th_intervals(average_qt_durations)
overall_percentiles_75th_st_intervals =calculate_overall_percentiles_75th_intervals(average_st_interval_durations)
overall_percentiles_75th_rr_intervals = calculate_overall_percentiles_75th_intervals(mean_rr_interval_all_patients)
overall_percentiles_75th_p_wave_durations =calculate_overall_percentiles_75th_intervals(average_p_wave_durations)
overall_percentiles_75th_q_wave_durations = calculate_overall_percentiles_75th_intervals(average_q_durations)
overall_percentiles_75th_s_wave_durations =calculate_overall_percentiles_75th_intervals(average_s_durations)
overall_percentiles_75th_t_wave_durations =calculate_overall_percentiles_75th_intervals(average_t_durations)
overall_percentiles_75th_st_segments =calculate_overall_percentiles_75th_intervals(average_st_segment_durations)
overall_percentiles_75th_pr_duration =calculate_overall_percentiles_75th_intervals(average_pr_segment_durations)


overall_percentiles_99th_qrs_intervals =calculate_overall_percentiles_99th_intervals(average_qrs_durations)
overall_percentiles_99th_pr_intervals =calculate_overall_percentiles_99th_intervals(average_pr_interval_durations)
overall_percentiles_99th_qt_intervals = calculate_overall_percentiles_99th_intervals(average_qt_durations)
overall_percentiles_99th_st_intervals =calculate_overall_percentiles_99th_intervals(average_st_interval_durations)
overall_percentiles_99th_rr_intervals = calculate_overall_percentiles_99th_intervals(mean_rr_interval_all_patients)
overall_percentiles_99th_p_wave_durations =calculate_overall_percentiles_99th_intervals(average_p_wave_durations)
overall_percentiles_99th_q_wave_durations = calculate_overall_percentiles_99th_intervals(average_q_durations)
overall_percentiles_99th_s_wave_durations =calculate_overall_percentiles_99th_intervals(average_s_durations)
overall_percentiles_99th_t_wave_durations =calculate_overall_percentiles_99th_intervals(average_t_durations)
overall_percentiles_99th_st_segments =calculate_overall_percentiles_99th_intervals(average_st_segment_durations)
overall_percentiles_99th_pr_duration =calculate_overall_percentiles_99th_intervals(average_pr_segment_durations)





import pandas as pd

# Create a dictionary with all the interval averages
data = {
    "heart rate":average_heart_rate_per_patient,
    "P wave duration": overall_average_p_wave_durations,
    "Q wave duration": overall_average_q_wave_durations,
    "S wave duration": overall_average_s_wave_durations,
    "T wave duration": overall_average_t_wave_durations,
    "PR interval": overall_average_pr_intervals,
    "QT interval": overall_average_qt_intervals,
    "ST interval": overall_average_st_intervals,
    "RR interval": overall_average_rr_intervals,
    "QRS interval": overall_average_qrs_intervals,
    "ST segment": overall_average_st_segments,
    "PR segement":overall_average_pr_segments,   
    "Max P wave duration": overall_max_p_wave_durations,
    "Max Q wave duration": overall_max_q_wave_durations,
    "Max S wave duration": overall_max_s_wave_durations,
    "Max T wave duration": overall_max_t_wave_durations,
    "Max PR interval": overall_max_pr_intervals,
    "Max QT interval": overall_max_qt_intervals,
    "Max ST interval": overall_max_st_intervals,
    "Max RR interval": overall_max_rr_intervals,
    "Max QRS interval": overall_max_qrs_intervals,
    "Max ST segment": overall_max_st_segments,
    "Max PR segement":overall_max_pr_duration,
    "Min P wave duration": overall_min_p_wave_durations,
    "Min Q wave duration": overall_min_q_wave_durations,
    "Min S wave duration": overall_min_s_wave_durations,
    "Min T wave duration": overall_min_t_wave_durations,
    "Min PR interval": overall_min_pr_intervals,
    "Min QT interval": overall_min_qt_intervals,
    "Min ST interval": overall_min_st_intervals,
    "Min RR interval": overall_min_rr_intervals,
    "Min QRS interval": overall_min_qrs_intervals,
    "Min ST segment": overall_min_st_segments,
    "Min PR segement":overall_min_pr_duration,
    "Median P wave duration": overall_median_p_wave_durations,
    "Median Q wave duration": overall_median_q_wave_durations,
    "Median S wave duration": overall_median_s_wave_durations,
    "Median T wave duration": overall_median_t_wave_durations,
    "Median PR interval": overall_median_pr_intervals,
    "Median QT interval": overall_median_qt_intervals,
    "Median ST interval": overall_median_st_intervals,
    "Median RR interval": overall_median_rr_intervals,
    "Median QRS interval": overall_median_qrs_intervals,
    "Median ST segment": overall_median_st_segments,
    "Median PR segement":overall_median_pr_duration,
    "Std P wave duration": overall_std_p_wave_durations,
    "Std Q wave duration": overall_std_q_wave_durations,
    "Std S wave duration": overall_std_s_wave_durations,
    "Std T wave duration": overall_std_t_wave_durations,
    "Std PR interval": overall_std_pr_intervals,
    "Std QT interval": overall_std_qt_intervals,
    "Std ST interval": overall_std_st_intervals,
    "Std RR interval": overall_std_rr_intervals,
    "Std QRS interval": overall_std_qrs_intervals,
    "Std ST segment": overall_std_st_segments,
    "Std PR segement":overall_std_pr_duration,
    "Perc25 P wave duration": overall_percentiles_25th_p_wave_durations,
    "Perc25 Q wave duration": overall_percentiles_25th_q_wave_durations,
    "Perc25 S wave duration": overall_percentiles_25th_s_wave_durations,
    "Perc25 T wave duration": overall_percentiles_25th_t_wave_durations,
    "Perc25 PR interval": overall_percentiles_25th_pr_intervals,
    "Perc25 QT interval": overall_percentiles_25th_qt_intervals,
    "Perc25 ST interval": overall_percentiles_25th_st_intervals,
    "Perc25 RR interval": overall_percentiles_25th_rr_intervals,
    "Perc25 QRS interval": overall_percentiles_25th_qrs_intervals,
    "Perc25 ST segment": overall_percentiles_25th_st_segments,
    "Perc25 PR segement":overall_percentiles_25th_pr_duration,
    "Perc75 P wave duration": overall_percentiles_75th_p_wave_durations,
    "Perc75 Q wave duration": overall_percentiles_75th_q_wave_durations,
    "Perc75 S wave duration": overall_percentiles_75th_s_wave_durations,
    "Perc75 T wave duration": overall_percentiles_75th_t_wave_durations,
    "Perc75 PR interval": overall_percentiles_75th_pr_intervals,
    "Perc75 QT interval": overall_percentiles_75th_qt_intervals,
    "Perc75 ST interval": overall_percentiles_75th_st_intervals,
    "Perc75 RR interval": overall_percentiles_75th_rr_intervals,
    "Perc75 QRS interval": overall_percentiles_75th_qrs_intervals,
    "Perc75 ST segment": overall_percentiles_75th_st_segments,
    "Perc75 PR segement":overall_percentiles_75th_pr_duration,
    "Perc99 P wave duration": overall_percentiles_99th_p_wave_durations,
    "Perc99 Q wave duration": overall_percentiles_99th_q_wave_durations,
    "Perc99 S wave duration": overall_percentiles_99th_s_wave_durations,
    "Perc99 T wave duration": overall_percentiles_99th_t_wave_durations,
    "Perc99 PR interval": overall_percentiles_99th_pr_intervals,
    "Perc99 QT interval": overall_percentiles_99th_qt_intervals,
    "Perc99 ST interval": overall_percentiles_99th_st_intervals,
    "Perc99 RR interval": overall_percentiles_99th_rr_intervals,
    "Perc99 QRS interval": overall_percentiles_99th_qrs_intervals,
    "Perc99 ST segment": overall_percentiles_99th_st_segments,
    "Perc99 PR segement":overall_percentiles_99th_pr_duration,
    "P wave area": overall_average_p_wave_area,
    "T wave area": overall_average_t_wave_area,
    "QRS wave area": overall_average_qrs_wave_area,
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)


df_filter_2=df_filter.drop(["height", "weight","nurse","site","device","strat_fold",
                            "recording_date","report","heart_axis","infarction_stadium1",
                            "infarction_stadium2","validated_by","second_opinion","initial_autogenerated_report",
                            "validated_by_human","baseline_drift","static_noise","burst_noise",
                            "electrodes_problems","extra_beats","pacemaker","filename_lr",
                            "filename_hr","scp_codes","diagnostic_superclass","diagnostic_superclass_2","patient_id"], axis=1)


df_filter_2= df_filter_2.reset_index(drop=True)

merged_df =df_filter_2.join(df)

merged_df.to_csv('~/ECG特徵值持續時間.csv', index=False)
