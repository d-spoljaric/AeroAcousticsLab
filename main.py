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

R_0 = 100 # [m], magnitude of observer location vector
observer_phi_list = np.arange(0, 105, 15) # [rad], phi angle of observer position
observer_theta = np.pi/4 # [rad], theta angle of observer position

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
    return R0 - R1 * np.cos(theta)*np.cos(omega*t + angle_offset - phi)

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
def pressure(r: int|float, M: int|float, Fr: int|float, Mr: int|float) -> int|float:
    """
    Returns scalar pressure value
    """
    frac1 = -Fr/((r**2)*(1-Mr)**2)
    frac2 = (Mr-M**2)/(1-Mr)
    return (1/(4*np.pi))*frac1*(frac2+1)

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
if __name__ == "__main__":
    SPL_list = np.zeros(shape = observer_phi_list.shape)
    PWL_list = np.zeros(shape = observer_phi_list.shape)
    
    for i in tqdm(range(len(observer_phi_list)), desc = "Looping Through Observer Phi"):
        observer_phi = np.deg2rad(observer_phi_list[i])
        print(observer_phi)
        
        p_total = np.zeros(shape=t_observer.shape)
        p_blade = np.zeros(shape=(len(t_observer), B))
        t_retarded = np.zeros(shape = t_observer.shape)
        
        for i in tqdm(range(B), desc = "Looping throgh blades"):
            for j in tqdm(range(len(t_observer)), desc = "Looping through time"):
                omega_t_offset = blade_offset*i

                r_p = r_phase(R_0, R_1, observer_theta, observer_phi, vel_rad, t_observer[j], omega_t_offset)
                t_ret = compute_t_retarded(t_observer[j], r_p)
                if i==0:
                    t_retarded[j] += t_ret

                xM = x_M(R_0, observer_theta, observer_phi)
                xS = x_S(R_1, vel_rad, t_ret, omega_t_offset)
                r_t = r(xM, xS)

                Fr = F_r(T_blade, observer_theta, R_0)
                M = M_vec(M_force, vel_rad, t_ret, omega_t_offset)
                Mr = np.dot(M, r_t/np.linalg.norm(r_t))

                p = pressure(np.linalg.norm(r_t), M_force, Fr, Mr)

                p_blade[j, i] = p
                
                p_total[j] += p
                # print(p_total)
                # input()
                
                SPL_list[i] = compute_spl(p_total)
                # PWL_list[i] = compute_pwl(p_total)
            
        # for i in range(B):
        #     plt.plot(t_observer, p_blade[:, i], label = f"Blade {i+1}")
        # plt.legend()
        # plt.minorticks_on()
        # plt.grid(True, which = "both")
        # plt.show()
        
        # plt.plot(t_observer, p_total)
        # plt.show()
    
    ax = plt.subplot(1, 1, 1)
    # ax.get_yaxis().get_major_formatter().set_useOffset(False)
    # plt.scatter(observer_phi_list, PWL_list)
    plt.scatter(observer_phi_list, SPL_list)
    plt.minorticks_on()
    plt.grid(True, which = "both")
    plt.xlabel(r"\phi [rad]")
    plt.ylabel("SPL")
    plt.show()
 
 
