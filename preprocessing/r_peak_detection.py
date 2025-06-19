import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import biosppy

# ========== 頻譜比較：過濾前 vs. 去趨勢後 ==========

def plot_fft_comparison(X_raw, X_detrended, sampling_rate):
    sample_ecg_signal_original = X_raw[0, :, 0]
    sample_ecg_signal_filtered = X_detrended[0, :, 0]

    fft_original = fft(sample_ecg_signal_original)
    fft_filtered = fft(sample_ecg_signal_filtered)

    N = sample_ecg_signal_original.size
    T = 1.0 / sampling_rate
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(xf, 2.0/N * np.abs(fft_original[0:N//2]))
    plt.title('Original Signal Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.subplot(1, 2, 2)
    plt.plot(xf, 2.0/N * np.abs(fft_filtered[0:N//2]))
    plt.title('Filtered Signal Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

# ========== 原始與濾波後訊號圖 ==========

def plot_raw_and_filtered(X_raw, X_detrended):
    plt.figure(figsize=(20, 5))
    plt.plot(X_raw[0][:, 0], label='Original', linewidth=1.2)
    plt.title("Original ECG Signal")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(X_detrended[0][:, 0], label='Filtered', linewidth=1.2)
    plt.title("Filtered ECG Signal")
    plt.grid(True)
    plt.show()

# ========== R 峰偵測（Hamilton） ==========

def detect_qrs_all_patients(X_cleaned, sampling_rate):
    num_patients, num_timepoints, num_leads = X_cleaned.shape
    r_peaks_all_patients = np.empty((num_patients, num_leads), dtype=object)

    for i in range(num_patients):
        for j in range(num_leads):
            ecg_signal = X_cleaned[i, :, j]
            try:
                initial_r_peaks = biosppy.signals.ecg.hamilton_segmenter(
                    signal=ecg_signal, sampling_rate=sampling_rate)[0]
                corrected_r_peaks, = biosppy.signals.ecg.correct_rpeaks(
                    signal=ecg_signal, rpeaks=initial_r_peaks, sampling_rate=sampling_rate, tol=0.05)

                if len(corrected_r_peaks) <= 4:
                    r_peaks_all_patients[i, j] = 'INCOMPLETE'
                else:
                    corrected_r_peaks = np.delete(corrected_r_peaks, [0, -1])
                    r_peaks_all_patients[i, j] = corrected_r_peaks
            except Exception as e:
                print(f"Error in patient {i}, lead {j}: {e}")
                r_peaks_all_patients[i, j] = np.array([])

    return r_peaks_all_patients

# ========== 畫圖 + 標 R 峰 ==========

def plot_ecg_and_r_peaks(X_detrended, r_peaks_all_patients, patient_index, lead_index):
    ecg_signal = X_detrended[patient_index, :, lead_index]
    r_peaks = r_peaks_all_patients[patient_index, lead_index]

    if isinstance(r_peaks, str):
        print(f"No valid R-peaks for patient {patient_index}, lead {lead_index}")
        return

    plt.figure(figsize=(20, 6))
    plt.plot(ecg_signal, label='ECG signal')

    for r in r_peaks:
        plt.plot(r, ecg_signal[r], 'ro')
        plt.annotate('R', (r, ecg_signal[r]), textcoords="offset points", xytext=(-10,-10), ha='center')

    plt.title(f"Patient {patient_index + 1}, lead {lead_index + 1}")
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ========== 範例執行方式（請在主程式中呼叫） ==========

# sampling_rate = 500
# plot_fft_comparison(X, X_detrended, sampling_rate)
# plot_raw_and_filtered(X, X_detrended)
# r_peaks_all_patients = detect_qrs_all_patients(X_detrended, sampling_rate)
# plot_ecg_and_r_peaks(X_detrended, r_peaks_all_patients, patient_index=99, lead_index=2)
