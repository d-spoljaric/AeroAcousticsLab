import numpy as np
import scipy as sp

mB = 4
s = 1

coeff = -1j*np.exp(-1j*mB)

def force_time(t):
    return np.cos(t)

def force_s(s):
    Fs_real = sp.integrate.quad(lambda t: np.real(force_time(t)*np.exp(1j*s*t)), 0, 1)
    Fs_imag = sp.integrate.quad(lambda t: np.imag(force_time(t)*np.exp(1j*s*t)), 0, 1)
    return Fs_real[0] + 1j*Fs_imag[0]

phi = np.pi/4
# Negative s
neg_s = force_s(-s)*np.exp(1j*(mB-(-s))*(phi-np.pi/2))
# Positive s
pos_s = force_s(s)*np.exp(1j*(mB-(s))*(phi-np.pi/2))

tot_s = neg_s+pos_s

print(neg_s)
print(pos_s)
print(tot_s)

print(coeff*tot_s)