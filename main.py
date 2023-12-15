import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================== Given Simulation Constants ===================================
# ========== ISA Conditions at 0m =========
T = 288.15 # [K], temperature
p = 101325 # [N/m^2], pressure
rho = 1.225 # [kg/m^3], density
# =========================================
thrust_total = 2000 # [N]
r_thrust = 0.85 # [-], location of force as a fraction of propeller radius
M_tip = 0.3 # [-], propeller tip Mach number
M_force = M_tip*r_thrust # [-], Mach number at force location
D = 1 # [m], propeller radius
R_1 = r_thrust*D/2 # [m], magnitude of location of force application vector
B = 4 # [-], number of propeller blades
c = np.sqrt(1.4*287.15*T) # [m/s], speed of sound
vel_rad = M_tip*r_thrust*c/R_1 # [rad/s], radial velocity of the location of force application
blade_offset = 2*np.pi/B # [rad], constant angle offset for each blade of the propeller 
T_blade = thrust_total/B # [N], thrust per blade

# ============================== Chosen Simulation Constants ==================================

t_total = 20 # [s], total simulation time
dt = 0.01 # [s], simulation time incremenets

R_0 = 100 # [m], magnitude of observer location vector
observer_theta = 0 # [rad], theta angle of observer position
observer_phi = np.pi/4 # [rad], phi angle of observer position

# ================================ Position Vectors and Scalars =========================================
def x_M(R: int | float, theta: int | float, phi: int | float) -> np.ndarray:
    '''
    Returns a row vector of the poisition of the observer with respect to the origin
    '''
    return R*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

def x_S(R: int | float, omega: int | float, t: int | float, angle_offset: int|float) -> np.ndarray:
    '''
    Returns a row vector of the poisition of the force with respect to the origin
    '''
    return R*np.array([np.cos(omega*t + angle_offset), np.sin(omega*t + angle_offset), 0])

def r(x_m: np.ndarray, x_s: np.ndarray) -> np.ndarray:
    '''
    Returns a row vector of the poisition of the observer with respect to the force
    '''
    return x_m - x_s

def r_phase(R0: int | float, R1: int | float, theta: int | float, phi: int | float, omega: int | float, t: int | float, angle_offset: int|float) -> int | float:
    return R0 - R1 * np.cos(theta)*np.cos(omega*t + angle_offset - phi)

# ============================== Force and Mach in direction of observer =======================
def F_r(F: int|float, theta: int|float) -> int|float:
    '''
    Returns force scalar in direction of observer
    '''
    return F*np.cos(theta)

def M_vec(M: int|float, omega: int|float, t: int|float, angle_offset: int|float) -> np.ndarray:
    '''
    Returns Mach vector
    '''
    return M*np.array([-np.sin(omega*t + angle_offset), np.cos(omega*t + angle_offset), 0])

# ============================= Observer and Retarded Time ====================================
t_observer= np.arange(0, t_total + dt, dt)

def compute_t_retarded(t: int | float, r: int | float) -> int | float:
    '''
    Returns retarded time corresponding to observer time. Note, 'r' corresponds to the the 'r_phase' function
    '''
    global c
    return t - r/c

# ============================= Time Domain Pressure ========================================
def pressure(r: int|float, M: int|float, Fr: int|float, Mr: int|float) -> int|float:
    """
    Returns scalar pressure value
    """
    frac1 = -Fr/((r**2)*(1-Mr)**2)
    frac2 = (Mr-M**2)/(1-Mr)
    return (1/(4*np.pi))*frac1*(frac2+1)

# ============================= Helper Functions ========================================


# ============================ Simulation =========================================

if __name__ == "__main__":
    p_total = np.zeros(shape=t_observer.shape)
    t_retarded = np.zeros(shape = t_observer.shape)
    for i in range(B):
        for j in range(len(t_observer)):
            omega_t_offset = 0 #blade_offset*i
            
            r_p = r_phase(R_0, R_1, observer_theta, observer_phi, vel_rad, t_observer[j], omega_t_offset)
            t_ret = compute_t_retarded(t_observer[j], r_p)
            t_retarded[j] += t_ret
            
            xM = x_M(R_0, observer_theta, observer_phi)
            xS = x_S(R_1, vel_rad, t_ret, omega_t_offset)
            r_t = r(xM, xS)
            
            Fr = F_r(T_blade, observer_theta)
            M = M_vec(M_force, vel_rad, t_ret, omega_t_offset)
            Mr = np.dot(M, r_t/np.linalg.norm(r_t))
            
            p_total[j] += pressure(np.linalg.norm(r_t), M_force, Fr, Mr)
    
    plt.plot(t_observer, p_total)
    plt.show()