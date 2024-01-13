import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)
y = np.sin(t) #+ np.sin(3*t) + np.sin(5*t)

# plt.plot(t, y, color = "r")

sp = np.fft.fft(y)
freq = np.fft.fftfreq(t.shape[-1])

# plt.scatter(freq, sp.real, s=10)
# plt.show()

y_inverse = np.fft.ifft(sp).real

plt.plot(t, y_inverse, color = "r")
plt.plot(t, y)
plt.show()