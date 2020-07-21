"""
Create Butterworth filter to use it on Android app
"""
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

fs = 50
fn = 20

b, a = signal.butter(3, fn/(fs/2), 'low', analog=False)
w, h = signal.freqz(b, a,fs = fs)
print("B:")
print(b)
print("----------------------------------------")
print("A:")
print(a)
plt.title('Freuency response')
plt.plot(w, 20*np.log10(np.abs(h)))
plt.axis([0,21,-5,0.2])
plt.title('Freuency response third order Butterwortha filter (fn=20Hz)')
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid()
plt.show()

