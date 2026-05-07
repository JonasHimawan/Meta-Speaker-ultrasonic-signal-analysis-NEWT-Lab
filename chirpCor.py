# import numpy as np
# import sounddevice as sd
# import matplotlib.pyplot as plt
# from scipy import signal

# # ---- CONFIG ----
# duration = 5         # seconds
# sample_rate = 96000     # Hz
# left_freq = 20000    # Hz
# right_freq = 23000       # Hz

# f_start = 23000
# f_stop = 25000

# #output_device = 34  # IXO22 output (2 channels)
# #input_device = 33   # IXO22 input (2 channels)

# # ---- Generate time base ----
# t = np.linspace(0, duration, int(sample_rate*duration), endpoint=False)

# # ---- Define left and right channel signals ----
# #chan1 = np.sin(2*np.pi*left_freq*t)
# chan1 = signal.chirp(t, f_start, t[-1], f_stop, method='linear')

# chan2 = np.sin(2*np.pi*right_freq*t)

# expected = signal.chirp(t, f_start-right_freq, t[-1] , f_stop-right_freq, method='linear')


# # Stack into stereo array
# stereo_out = np.column_stack((chan1,chan2))



# # ---- Play and record ----
# recording = sd.playrec(stereo_out,sample_rate, channels=1, blocking=True, blocksize=1024)

# mic = recording[:, 0]


# # ---- FFT of raw microphone signal ----
# fft_vals = np.fft.fft(mic)
# freqs = np.fft.fftfreq(len(fft_vals), 1/sample_rate)

# # Only positive frequencies
# pos_freqs = freqs[:len(freqs)//2]
# pos_magnitude = np.abs(fft_vals[:len(fft_vals)//2])

# # ---- Plot frequency spectrum ----
# plt.figure(figsize=(10, 4))
# plt.plot(pos_freqs, pos_magnitude)
# plt.title("Microphone Frequency Spectrum")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Amplitude")
# plt.xlim(0, max(left_freq,right_freq,f_stop,f_start)+1000)  # show up to 30 kHz
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# # ----- Plot spectrogram ----
# plt.specgram(mic, NFFT=2048, Fs=sample_rate, noverlap=1024, cmap='plasma')
# plt.show()


# # ---- Plot Cross correlation ----
# corr = signal.correlate(mic, expected, mode='full')
# lags = np.arange(-len(mic)+1, len(mic))
# plt.figure(figsize=(10, 4))
# plt.plot(lags, np.abs(corr))
# plt.title("Cross-correlation between Mic Signal and Difference Signal")
# plt.xlabel("Lag")
# plt.ylabel("Correlation")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# NEW


import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy import signal


def collect_data(
    duration=5,
    sample_rate=96000,
    #left_freq=20000,
    right_freq=23000,
    f_start=23000,
    f_stop=25000,
    show_plots=False,
    save_prefix=None,
    bufferFreq=500,
):
    # ---- Generate time base ----
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # ---- Define left and right channel signals ----
    chan1 = signal.chirp(t, f_start, t[-1], f_stop, method="linear")
    chan2 = np.sin(2 * np.pi * right_freq * t)
    expected = signal.chirp(
        t, f_start - right_freq, t[-1], f_stop - right_freq, method="linear"
    )

    min_freq = min(right_freq, f_start, f_stop) - bufferFreq
    max_freq = max(right_freq, f_start, f_stop) + bufferFreq

    # Stack into stereo array
    stereo_out = np.column_stack((chan1, chan2))

    # ---- Play and record ----
    recording = sd.playrec(
        stereo_out,
        sample_rate,
        channels=1,
        blocking=True,
        blocksize=1024,
    )

    mic = recording[:, 0]

    # ---- FFT of raw microphone signal ----
    fft_vals = np.fft.fft(mic)
    freqs = np.fft.fftfreq(len(fft_vals), 1 / sample_rate)

    # Filter out frequencies outside the expected range
    valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs = freqs[valid_mask]
    fft_vals = fft_vals[valid_mask]

    pos_freqs = freqs[: len(freqs) // 2]
    pos_magnitude = np.abs(fft_vals[: len(fft_vals) // 2])

    #Reconstruct filtered time-domain signal
    filtered_mic = np.fft.ifft(fft_vals, n=len(mic)).real

    # ---- Cross correlation ----
    corr = signal.correlate(filtered_mic, expected, mode="full")
    lags = np.arange(-len(filtered_mic) + 1, len(filtered_mic))
    corr_abs = np.abs(corr)

    peak_corr = float(np.max(corr_abs))
    peak_lag = int(lags[np.argmax(corr_abs)])
    mic_rms = float(np.sqrt(np.mean(mic**2)))
    mic_peak = float(np.max(np.abs(mic)))

    if save_prefix:
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

        np.save(save_prefix + "_mic.npy", mic)
        np.save(save_prefix + "_corr.npy", corr)

    if show_plots:
        plt.figure(figsize=(10, 4))
        plt.plot(pos_freqs, pos_magnitude)
        plt.title("Microphone Frequency Spectrum")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.xlim(0, max_freq)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.specgram(mic, NFFT=2048, Fs=sample_rate, noverlap=1024, cmap="plasma")
        plt.title("Microphone Spectrogram")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(lags, corr_abs)
        plt.title("Cross-correlation between Mic Signal and Difference Signal")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "peak_corr": peak_corr,
        "peak_lag": peak_lag,
        "mic_rms": mic_rms,
        "mic_peak": mic_peak,
        "num_samples": len(mic),
    }


if __name__ == "__main__":
    result = collect_data(show_plots=True, save_prefix="data/test")
    print(result)
