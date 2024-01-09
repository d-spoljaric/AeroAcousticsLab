import numpy as np
import matplotlib.pyplot as plt

def compute_pressure(r: int|float, M: int|float, Fr: int|float, Mr: int|float, Fr_dot: int|float) -> int|float:
    """
    Returns scalar pressure value
    """
    global c
    frac1 = (-1/c)*(Fr_dot/(r*(1-Mr)**2))
    frac2 = -Fr/((r**2)*(1-Mr)**2)
    frac3 = (Mr-M**2)/(1-Mr)
    return (1/(4*np.pi))*(frac1 + frac2*(frac3+1))

def x_M(R: int | float, theta: int | float, phi: int | float) -> np.ndarray:
    '''
    Returns a row vector of the position of the observer with respect to the origin
    '''
    return R*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

def x_S(R: int | float, omega: int | float, t: int | float, angle_offset: int|float) -> np.ndarray:
    '''
    Returns a row vector of the position of the force with respect to the origin
    '''
    return R*np.array([np.cos(omega*t + angle_offset), np.sin(omega*t + angle_offset), 0])

def compute_r(x_m: np.ndarray, x_s: np.ndarray) -> np.ndarray:
    '''
    Returns a row vector of the positiion of the observer with respect to the force
    '''
    return x_m - x_s

c  = 340

mach_range = np.arange(0.1, 0.7, 0.1)
p_array = np.zeros(shape = mach_range.shape)
R0 = 100
R1=1

theta = np.pi/4
phi = np.pi/4
omega = 200
xM = x_M(R0, theta, phi)
xS = x_S(R1, omega, 0, 0)
r = compute_r(xM, xS)

F = np.array([0, 0, 500])
Fr_dot = 0
Fr = np.dot(F, r)

for count,mach in enumerate(mach_range):
    Mr = mach*R0*np.sin(theta)*np.sin(phi)/np.linalg.norm(r)
    p = compute_pressure(np.linalg.norm(r), mach, Fr, Mr, Fr_dot)
    p_array[count] = p
    
plt.plot(mach_range, p_array)
plt.show()