import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy import signal

# ---- CONFIG ----
duration = 5             # seconds
sample_rate = 96000     # Hz
left_freq = 20000     # Hz
right_freq = 23000       # Hz

f_start = 22000
f_stop = 24000

#output_device = 34  # IXO22 output (2 channels)
#input_device = 33   # IXO22 input (2 channels)

# ---- Generate stereo signals ----
t = np.linspace(0, duration, int(sample_rate*duration), endpoint=False)

# ---- Define left and right channel signals ----
chan1 = np.sin(2*np.pi*left_freq*t)
#chan1 = signal.chirp(t, f_start, t[-1], f_stop, method='linear')

chan2 = np.sin(2*np.pi*right_freq*t)


# Stack into stereo array
stereo_out = np.column_stack((chan1,chan2))

'''
# ---- Play and record ----
sd.play(stereo_out, samplerate=sample_rate, device=output_device)
recording = sd.rec(int(sample_rate*duration),
                   samplerate=sample_rate,
                   channels=1,  # record mono
                   dtype='float32',
                   device=input_device)
sd.wait()
'''

# ---- Play and record ----
recording = sd.playrec(stereo_out,sample_rate, channels=1, blocking=True, blocksize=1024)

mic = recording[:, 0]



# ---- FFT of raw microphone signal ----
fft_vals = np.fft.fft(mic)
freqs = np.fft.fftfreq(len(fft_vals), 1/sample_rate)

# Only positive frequencies
pos_freqs = freqs[:len(freqs)//2]
pos_magnitude = np.abs(fft_vals[:len(fft_vals)//2])

# ---- Plot frequency spectrum ----
plt.figure(figsize=(10, 4))
plt.plot(pos_freqs, pos_magnitude)
plt.title("Microphone Frequency Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.xlim(0, max(left_freq,right_freq,f_stop,f_start)+1000)  # show up to 30 kHz
plt.grid(True)
plt.tight_layout()
plt.show()

