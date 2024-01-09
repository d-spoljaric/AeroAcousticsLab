import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import jv
from scipy.integrate import quad

# =============================== Given Simulation Constants ===================================
# ========== ISA Conditions at 0m =========
T = 288.15 # [K], temperature
p = 101325 # [N/m^2], pressure
rho = 1.225 # [kg/m^3], density
c = np.sqrt(1.4*287.15*T) # [m/s], speed of sound
# =========================================
thrust_total = 2000 # [N]
r_thrust = 0.85 # [-], location of force as a fraction of propeller radius
M_tip = 0.3 # [-], propeller tip Mach number
M_force = M_tip*r_thrust # [-], Mach number at force location
D = 1 # [m], propeller radius
R_1 = r_thrust*D/2 # [m], magnitude of location of force application vector
B = 4 # [-], number of propeller blades
vel_rad = M_tip*r_thrust*c/R_1 # [rad/s], radial velocity of the location of force application
#blade_offset = 2*np.pi/B # [rad], constant angle offset for each blade of the propeller 
T_blade = thrust_total/B # [N], thrust per blade

# ============================== Chosen Simulation Constants ==================================
R_0 = 100 # [m], magnitude of observer location vector
observer_phi = np.pi/4 # [rad], phi angle of observer position
dtheta_large = 2
dtheta_small = 0.05
theta_max_beg = 88
theta_max = 90
theta = [0, 15, 30, 45, 60, 75, 90]



m_array = np.arange(1,4+1,1)
#s_array = np.arange(-10,10+1,1)
s_array = [0]
time_array = np.linspace(0,2*np.pi/vel_rad,100)

def compute_spl(p: int|float) -> int|float:
    '''
    Computes Sound Pressure Level (SPL) from array of pressure fluctuations
    '''
    p_rms = compute_rms(p)
    p_ref = 2e-5
    return 10*np.log10((p_rms/p_ref)**2)

def compute_rms(x: int|float) -> int|float:
    return np.sqrt(np.mean(x**2))


def F_t(F: int|float, t: np.ndarray, Omega: int|float) -> np.ndarray:
#    return F * np.sin(Omega * t)
    return F


# ================================ Main ==========================================
if __name__ == "__main__":
    
    SPL_list = np.zeros(shape=(len(theta),len(m_array)))
    prms_list = np.zeros(shape=(len(theta),len(m_array)))

    p = np.zeros(shape=(len(theta),len(m_array)))
    temp = np.zeros(shape=(len(theta),len(m_array)))
    F_s = 0

    for i in tqdm(range(len(theta)), desc = "Looping Through Observer theta"):
        observer_theta = np.deg2rad(theta[i])
        for m in tqdm(range(len(m_array)), desc = "Looping throgh m"):
            p_InnerSigma = 0
            p_OuterSigma = (0+1j)*m_array[m]*B**2*vel_rad*np.exp(-(0+1j)*m_array[m]*B*vel_rad*R_0/c)/(4*np.pi*c*R_0)
            for s in tqdm(range(len(s_array)), desc = "Looping through s"):
                integral_real, error_real = quad(lambda t: np.real(F_t(thrust_total,t,vel_rad)), 0, 2 * np.pi / vel_rad)
                integral_imag, error_imag = quad(lambda t: np.imag(F_t(thrust_total,t,vel_rad)), 0, 2 * np.pi / vel_rad)
                F_s = vel_rad/2/np.pi*(integral_real + 1j * integral_imag)
#                F_s = vel_rad/2/np.pi*np.trapz(F_t(T_blade,time_array,vel_rad)) * np.exp((0+1j)*s_array[s]*vel_rad*time_array)
                p_temp = F_s*np.exp((0+1j)*(m_array[m]*B-s_array[s])*(observer_phi-np.pi/2))*jv(m_array[m]*B-s_array[s],m_array[m]*B*M_force*np.sin(observer_theta))*np.cos(observer_theta)
                p_InnerSigma = p_InnerSigma + p_temp
                a = 1

            print(f"Shape of F_s: {np.shape(F_s)}")
            print(f"Shape of p_temp: {np.shape(p_temp)}")
            p_final = p_OuterSigma * p_InnerSigma
            p[i,m] = p_OuterSigma * p_InnerSigma
            SPL_list[i,m] = compute_spl(p[i,m])
            prms_list[i,m] = compute_rms(p[i,m])

    
    for i in range(len(prms_list)):
        prms_list[i] = compute_rms(p[i, :])
    
    for i in range(len(SPL_list)):
        SPL_list[i] = compute_spl(prms_list[i])
        
    plt.plot(theta, prms_list)
    plt.show()