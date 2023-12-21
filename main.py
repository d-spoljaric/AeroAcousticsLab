import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

rotations = 8
t_total =  rotations*2*np.pi/vel_rad# [s], total simulation time
dt = 0.00005 # [s], simulation time incremenets

## The simulation is split into 2 "phases". The first phase is simulated on a coarser set of theta values
## since and the second phase, where the gradient increases and the SPL plot becomes asymptotic, has a much 
## smaller step in theta. 

R_0 = 100 # [m], magnitude of observer location vector
observer_phi = np.pi/4 # [rad], phi angle of observer position
dtheta_large = 5 # [rad], large steps in theta for first phase of simulation
dtheta_small = 0.1 # [rad], small steps in theta for first phase of simulation
theta_max_beg = 85 # [rad], end of first phase and start of second phase of simulation
theta_min = 0 # [rad], minimum theta for simulation
theta_max = 90 # [rad], maximum theta for simulation
observer_theta_list_beg = np.arange(theta_min, theta_max_beg + dtheta_large, dtheta_large) # [rad], array of theta for first phase
observer_theta_list = np.append(observer_theta_list_beg, np.arange(observer_theta_list_beg[-1] + dtheta_small, theta_max, dtheta_small)) # [rad], array of all theta

# ================================ Position Vectors and Scalars =========================================
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

def r(x_m: np.ndarray, x_s: np.ndarray) -> np.ndarray:
    '''
    Returns a row vector of the positiion of the observer with respect to the force
    '''
    return x_m - x_s

def r_phase(R0: int | float, R1: int | float, theta: int | float, phi: int | float, omega: int | float, t: int | float, angle_offset: int|float) -> int | float:
    return R0 - R1 * np.sin(theta)*np.cos(omega*t + angle_offset - phi)

# ============================== Force and Mach in direction of observer =======================
def F_r(F: int|float, theta: int|float, r: int|float) -> int|float:
    '''
    Returns scalar of force in direction of observer
    '''
    global R_0
    return F*R_0*np.cos(theta)/r

def M_vec(M: int|float, omega: int|float, t: int|float, angle_offset: int|float) -> np.ndarray:
    '''
    Returns Mach vector
    '''
    return M*np.array([-np.sin(omega*t + angle_offset), np.cos(omega*t + angle_offset), 0])

# ============================= Observer and Retarded Time ====================================
start_time = 0
t_observer= np.arange(start_time, start_time + t_total + dt, dt)

def compute_t_retarded(t: int | float, r: int | float) -> int | float:
    '''
    Returns retarded time corresponding to observer time. Note, 'r' corresponds to the the 'r_phase' function
    '''
    global c
    return t - r/c

# ============================= Time Domain Pressure ========================================
def pressure(r: int|float, M: int|float, Fr: int|float, Mr: int|float, Fr_dot: int|float) -> int|float:
    """
    Returns scalar pressure value
    """
    global c
    frac1 = (-1/c)*(Fr_dot/(r*(1-Mr)**2))
    frac2 = -Fr/((r**2)*(1-Mr)**2)
    frac3 = (Mr-M**2)/(1-Mr)
    return (1/(4*np.pi))*(frac1 + frac2*(frac3+1))

# ============================= Sound Functions ========================================
def compute_spl(p: np.ndarray) -> int|float:
    '''
    Computes Sound Pressure Level (SPL) from array of pressure fluctuations
    '''
    p_rms = compute_rms(p)
    p_ref = 2e-5
    return 10*np.log10((p_rms/p_ref)**2)

def compute_pwl(SPL: int|float, r: int|float) -> int|float:
    '''
    Computes sound Power Level (PWL) from SPL and distance
    '''
    return SPL + 11 + 20*np.log10(r)

# ============================ Helper Functions =====================================
def compute_rms(x: np.ndarray) -> int|float:
    return np.sqrt(np.mean(x**2))

# ================================ Main ==========================================

constant_force = False
s = 2

if __name__ == "__main__":
    SPL_list = np.zeros(shape = observer_theta_list.shape)
    PWL_list = np.zeros(shape = observer_theta_list.shape)
    prms_list = np.zeros(shape = observer_theta_list.shape)
    
    for i in tqdm(range(len(observer_theta_list)), desc = "Looping Through Observer theta"):
        observer_theta = np.deg2rad(observer_theta_list[i])
        
        p_total = np.zeros(shape=t_observer.shape)
        p_blade = np.zeros(shape=(len(t_observer), B))
        t_retarded = np.zeros(shape = t_observer.shape)
        omega_t_blade = np.zeros(shape = (len(t_observer), B))
        f_blade = np.zeros(shape = (len(t_observer), B))
        
        for k in tqdm(range(B), desc = "Looping throgh blades"):    
            for j in range(len(t_observer)):
                omega_t_offset = blade_offset*k 
                r_p = r_phase(R_0, R_1, observer_theta, observer_phi, vel_rad, t_observer[j], omega_t_offset)
                t_ret = compute_t_retarded(t_observer[j], r_p)
                if k==0:
                    t_retarded[j] += t_ret
                        
                if constant_force:
                    force_blade = T_blade
                    force_blade_dot = 0
                else:
                    force_blade = T_blade*np.sin(t_ret*s*vel_rad/(2*np.pi))
                    force_blade_dot = T_blade*(s*vel_rad/(2*np.pi))*np.cos(t_ret*s*vel_rad/(2*np.pi))
                    
                f_blade[j, k] = force_blade

                xM = x_M(R_0, observer_theta, observer_phi)
                xS = x_S(R_1, vel_rad, t_ret, omega_t_offset)
                r_t = r(xM, xS)
                
                f_blade_vec = np.array([0, 0, force_blade])
                f_blade_dot_vec = np.array([0, 0, force_blade_dot])
                
                Fr = np.dot(f_blade_vec, r_t/np.linalg.norm(r_t))
                Fr_dot = np.dot(f_blade_dot_vec, r_t/np.linalg.norm(r_t))
                M = M_vec(M_force, vel_rad, t_ret, omega_t_offset)
                Mr = np.dot(M, r_t/np.linalg.norm(r_t))

                p = pressure(np.linalg.norm(r_t), M_force, Fr, Mr, Fr_dot)

                p_blade[j, k] = p
                
                p_total[j] += p
                omega_t_blade[j, k] = omega_t_offset+vel_rad*t_observer[j]
                a = 1
                # print(p_total)
                # input()
        
        
        # for i in range(B):
        #     plt.plot(t_observer, p_blade[:, i], label = f"Blade {i+1}")
        # plt.legend()
        # plt.minorticks_on()
        # plt.grid(True, which = "both")
        # plt.show()
                
        SPL_list[i] = compute_spl(p_total)
        # PWL_list[i] = compute_pwl(p_total)
        prms_list[i] = compute_rms(p_total)
    
    ax = plt.subplot(1, 1, 1)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.plot(observer_theta_list, SPL_list, color = "k", linewidth = 0.8)
    plt.minorticks_on()
    plt.grid(True, which = "both")
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel("SPL")
    plt.ylim([0, 1.05*np.max(SPL_list)])
    plt.show()
    
    ax2 = plt.subplot(1, 1, 1)
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.plot(observer_theta_list, prms_list, color = "k", linewidth = 0.8)
    plt.minorticks_on()
    plt.grid(True, which = "both")
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"$P_{rms}$ [Pa]")
    plt.ylim([0, 1.05*np.max(prms_list)])
    plt.show()
    
    
    
