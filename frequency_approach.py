import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================== Given Simulation Constants ===================================
# ========== ISA Conditions at 0m =========
T = 288.15  # [K], temperature
p = 101325  # [N/m^2], pressure
rho = 1.225  # [kg/m^3], density
# =========================================
thrust_total = 2000  # [N]
r_thrust = 0.85  # [-], location of force as a fraction of propeller radius
M_tip = 0.3  # [-], propeller tip Mach number
M_force = M_tip * r_thrust  # [-], Mach number at force location
D = 1  # [m], propeller radius
R_1 = r_thrust * D / 2  # [m], magnitude of location of force application vector
B = 1  # [-], number of propeller blades
c = np.sqrt(1.4 * 287.15 * T)  # [m/s], speed of sound
vel_rad = M_tip * r_thrust * c / R_1  # [rad/s], radial velocity of the location of force application
blade_offset = 2 * np.pi / B  # [rad], constant angle offset for each blade of the propeller

# ============================== Chosen Simulation Constants ==================================
R_0 = 100  # [m], magnitude of observer location vector
observer_phi = np.pi / 4  # [rad], phi angle of observer position
# observer_theta_list = np.arange(0.00001, 105, 15)
observer_theta_list = np.linspace(0.00001, 90, 100)
# observer_theta_list = np.array([45])

periodic_force = 100 #[N], amplitude of periodic force
n = 1 # Determines the harmonic of the force

m_val = 10
s_val = 0

# ============================= Sound Functions ========================================
def compute_spl(p: np.ndarray) -> int | float:
    """
    Computes Sound Pressure Level (SPL) from array of pressure fluctuations
    """
    p_rms = compute_rms(p)
    p_ref = 2e-5
    return 10 * np.log10((p_rms / p_ref) ** 2)


def compute_pwl(spl: int | float, r: int | float) -> int | float:
    """
    Computes sound Power Level (PWL) from SPL and distance
    """
    return spl + 11 + 20 * np.log10(r)


# ============================ Helper Functions =====================================
def compute_rms(x: np.ndarray) -> int | float:
    return np.sqrt(np.mean(x ** 2))


def pressure(phi: float, theta: float, force_func: object, s: np.ndarray | float, m: int) -> float:
    global B
    global vel_rad
    global R_0
    global c
    global thrust_total
    global M_force

    theta = np.deg2rad(theta)

    if s == 0:
        coeff = 1j * m * (B ** 2) * vel_rad / (4 * np.pi * c * R_0)
        exp_term = np.exp(-1j * m * B * np.pi / 2) * np.exp(1j * m * B * (phi - vel_rad * R_0 / c))
        Fs = thrust_total/B
        bessel = sp.special.jv(m * B, m * B * M_force * np.sin(theta))
        p_mB = coeff * Fs * exp_term * bessel * np.cos(theta)
    else:
        Fs_real = (vel_rad / (2 * np.pi)) * \
                  sp.integrate.quad(lambda t: force_func(t, s) * np.real(np.exp(1j * s * vel_rad * t)), 0,
                                    2 * np.pi / vel_rad)[0]
        Fs_imag = (vel_rad / (2 * np.pi)) * \
                  sp.integrate.quad(lambda t: force_func(t, s) * np.imag(np.exp(1j * s * vel_rad * t)), 0,
                                    2 * np.pi / vel_rad)[0]
        Fs = (Fs_real + 1j * Fs_imag)/B

        coeff = 1j * m * (B ** 2) * vel_rad * np.exp(-1j * m * B * vel_rad * R_0 / c) / (4 * np.pi * c * R_0)
        exp_term = np.exp(1j * (m * B - s) * (phi - np.pi / 2))
        bessel = sp.special.jv(m * B - s, m * B * np.sin(theta))

        sigma_term = np.sum(Fs * exp_term * bessel * np.cos(theta))
        p_mB = coeff * sigma_term

    return p_mB


def force(t, s):
    global thrust_total
    global periodic_force
    global n
    if s == 0:
        return thrust_total
    else:
        period = (1 / n) * 2 * np.pi / vel_rad
        A = 2 * np.pi / period  # Do not edit this
        return thrust_total + periodic_force * np.cos(A * t)



# ============================== Main Loop ===============================================

# Simulation arrays
m_array = np.arange(-m_val, m_val+1,1)
s_array = np.arange(-s_val, s_val+1, 1)

p_mB_array = np.zeros(shape=(len(observer_theta_list), len(m_array)), dtype=np.complex_)
p_rms = np.zeros(shape = observer_theta_list.shape)
SPL_array = np.zeros(shape = observer_theta_list.shape)
PWL_array = np.zeros(shape = observer_theta_list.shape)


# Calculating p_mB for each theta and m
for count_theta, theta in tqdm(enumerate(observer_theta_list), desc="Looping over theta"):
    for count_m, m in tqdm(enumerate(m_array), desc="Looping over m"):
        for s in s_array:
            p_mB = pressure(observer_phi, theta, force, s, m)
            p_mB_array[count_theta, count_m] = p_mB

    # Reordering the p_mB array as required by numpy ifft method
    p_mB_temp = p_mB_array[count_theta, :]
    p_mB_ifft = np.zeros(shape=p_mB_temp.shape, dtype=np.complex_)
    zero_freq_idx = m_val

    p_mB_ifft[0] = p_mB_temp[zero_freq_idx]
    p_mB_ifft[1:m_val + 1] = p_mB_temp[zero_freq_idx + 1:]
    p_mB_ifft[m_val + 1:] = p_mB_temp[:zero_freq_idx]

    p_mB_array[count_theta, :] = p_mB_ifft

    # Perform inverse fourier transform
    p_t = np.real(np.fft.ifft(p_mB_array[count_theta, :]))

    # Compute p_rms
    p_rms[count_theta] = compute_rms(p_t)

    # Compute SPL and PWL
    SPL_array[count_theta] = compute_spl(p_t)

plt.plot(observer_theta_list, SPL_array)
plt.show()
