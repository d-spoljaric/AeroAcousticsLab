import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import jv

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
theta = [0]

F_s = T_blade # Since F is independent of time, the integration is always constant
s = 0

m = np.arange(-4,4+1,1)


def compute_spl(p: int|float) -> int|float:
    '''
    Computes Sound Pressure Level (SPL) from array of pressure fluctuations
    '''
    p_rms = compute_rms(p)
    p_ref = 2e-5
    return 10*np.log10((p_rms/p_ref)**2)

def compute_rms(x: int|float) -> int|float:
    return np.sqrt(np.mean(x**2))


# ================================ Main ==========================================
if __name__ == "__main__":
    SPL_list = np.zeros(shape=(len(theta),len(m)))
    prms_list = np.zeros(shape=(len(theta),len(m)))

    p = np.zeros(shape=(len(theta),len(m)))
    temp = np.zeros(shape=(len(theta),len(m)))

    for i in tqdm(range(len(theta)), desc = "Looping Through Observer theta"):
        observer_theta = np.deg2rad(theta[i])
        for k in tqdm(range(len(m)), desc = "Looping throgh m"):
            p[i,k] = (0+1j)*m[k]*B**2*vel_rad*np.exp(-(0+1j)*m[k]*B*vel_rad*R_0/c)/(4*np.pi*c*R_0)*F_s*np.exp((0+1j)*m[k]*B*(observer_phi-np.pi/2))*jv(m[k]*B,m[k]*B*M_force*np.sin(observer_theta))*np.cos(observer_theta)
            temp[i,k] = jv(m[k]*B,m[k]*B*M_force*np.sin(observer_theta))
            SPL_list[i,k] = compute_spl(p[i,k])
            prms_list[i,k] = compute_rms(p[i,k])

    print(SPL_list)

