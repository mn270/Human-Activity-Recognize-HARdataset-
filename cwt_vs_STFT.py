"""
Show differences between WT and STFT
"""
from scipy import signal
import matplotlib.pyplot as plt
import  numpy as np
import pywt
waveletname =  'morl'
scales = range(1,200)

t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
t = np.linspace(-1, 1, 50, endpoint=False)
sig1  = np.sin(2 * np.pi * 16 * t)+100*np.sin(2 * np.pi *0.1 * t)
for i in range(50):
    sig[50+i] = sig1[i] + sig[50+i]
coeff, freq = pywt.cwt(sig, scales, waveletname, 1)
t = np.linspace(0, 200, 200, endpoint=False)
plt.plot(t,sig,color='k')
plt.title('Transformed signal')
plt.ylabel('Amplitude')
plt.xlabel('t [s]')
plt.figure()
plt.pcolormesh(coeff, cmap='plasma')
plt.title('Wavelet Transform (Morlett kernel)')
plt.ylabel('f [Hz]')
plt.xlabel('t [s]')
f, t, Zxx = signal.stft(sig, fs=400,nperseg = 8)
t = t*400
plt.figure()
plt.pcolormesh(t, f, np.abs(Zxx), cmap='plasma')
plt.title('Short Time Fourier Transform (STFT)')
plt.ylabel('f [Hz]')
plt.xlabel('t [s]')
plt.show()