import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import symbols, sin, diff

# Operating conditions and given data
thrust = 2000
gamma = 0
gamma_rad = np.deg2rad(gamma)
m_tip = 0.3
diameter = 1
n_blades = 4
f_applied = 0.85*diameter/2
mach_applied = 0.85 * m_tip  # Should we include the radius here?
r0 = 100

# Assumed Data
density = 1.225
temperature = 288.15
gas_constant = 287.15
#theta = 89
# theta = np.linspace(45, 88, 1)
theta = np.arange(0, 105, 15)
#theta_rad = np.deg2rad(theta)
#print(theta_rad)
phi = 45
phi_rad = phi * np.pi / 180
#phi_rad = math.radians(phi)
alpha = (360 / n_blades) * (np.pi / 180)
p_ref = 2 * 10**-5

# Determining essential quantities
radius = diameter / 2
# F = thrust/n_blades
v_sound = np.sqrt(1.4*gas_constant*temperature)
v_angular = (m_tip * v_sound) / radius   # v = r (x) omega
r1 = mach_applied * v_sound / v_angular  # No radial force component
time = np.arange(0,0.1+0.00005, 0.00005)  # Time array

p_eachblade = np.array(np.zeros((n_blades, len(time))))
prms_eachblade = np.array(np.zeros((n_blades, len(theta))))
SPL_eachblade = np.array(np.zeros((n_blades, len(theta))))
PWL_eachblade = np.array(np.zeros((n_blades, len(theta))))
retardedtime_plot = np.array(np.zeros((n_blades, len(time))))
SPLavg_blade = np.zeros(len(theta))
PWLavg_blade = np.zeros(len(theta))

SPL_array = np.zeros(shape = len(theta))
a = 0

for k in range(len(theta)):

    x_m = r0 * np.array([np.sin(theta[k]*np.pi/180) * np.cos(phi_rad), np.sin(theta[k]*np.pi/180) * np.sin(phi_rad), np.cos(theta[k]*np.pi/180)])  # indices of elements are 0, 1, 2
    
    p_fluc_total = np.zeros(shape = len(time))

    for i in range(n_blades):
        for j in range(len(time)):

            # Calculation of r(t)
            rt_phase = r0 - r1 * np.sin(theta[k]*np.pi/180) * np.cos(v_angular*time[j] + i*alpha - phi_rad)
            rt_magnitude = r0
            retarded_time = time[j] - rt_phase / v_sound
            # retardedtime_plot[i, j] = np.append(retardedtime_plot, np.array[retarded_time])

            # Defining periodically varying force
            # x = symbols('x')
            # x = v_angular*retarded_time
            force_variation = thrust# * np.cos(v_angular * retarded_time)
            F = force_variation/n_blades
            # print(F)

            # Calculation of position of object wrt origin
            x_s = r1 * np.array([np.cos(v_angular*retarded_time + i*alpha), np.sin(v_angular*retarded_time + i*alpha), 0])

            # Position vector between observer and object
            r_vector = x_m - x_s
            r_magnitude = np.linalg.norm(r_vector)
            # print(r_magnitude)

            # Force vector
            # F_vector = F * np.array([-np.sin(gamma_rad) * np.sin(v_angular*retarded_time + i*alpha), np.sin(gamma_rad)*np.cos(v_angular*retarded_time + i*alpha), np.cos(gamma_rad)])
            F_r = (F / r_magnitude) * r0 * np.cos(theta[k]*np.pi/180)
            # F_idot = np.array([0, 0, -v_angular*np.sin(v_angular*retarded_time)])
            F_idot = np.array([0, 0, 0])
            F_rdot = np.dot(F_idot, r_vector / np.linalg.norm(r_vector))
            # print(F_dot)

            # F_rdot = np.array([0, 0, np.dot(F_dot, r_vector / np.linalg.norm(r_vector))])
            #F_r = (F / r_magnitude) * np.array([(x_m[0] - x_s[0]) * -np.sin(gamma) * np.sin(v_angular*retarded_time + i*alpha) + (x_m[1] - x_s[1]) * np.sin(gamma) * np.cos(v_angular*retarded_time + i*alpha), (x_m[2] - x_s[2]) * np.cos(gamma)])
            # F_rdot = d/dt(F_i) * r_vector
            # M_rdot = 0
            F_m = 0  # Will add the proper formula later on for a different gamma

            # Mach vector
            M_vector = mach_applied * np.array([-np.sin(v_angular*retarded_time + i*alpha), np.cos(v_angular*retarded_time + i*alpha), 0])
            M_r = np.dot(M_vector, r_vector / np.linalg.norm(r_vector))

            # Defining terms in the equation of p(x,t)
            #a = 1/v_sound
            #b = np.linalg.norm(r_vector)**2
            #c = 1 - M_r
            #d = M_r - mach_applied**2

            p = -( (F_rdot / (v_sound * np.linalg.norm(r_vector) * (1 - M_r)**2)) + ((F_r * (M_r - mach_applied**2)) / (np.linalg.norm(r_vector)**2 * (1 - M_r)**3)) + ((F_r - F_m) / (np.linalg.norm(r_vector)**2 * (1 - M_r)**2)) ) / (4 * np.pi)

            # Calculation of SPL and PWL
            # i = ith blade, k = kth theta, j = jth time step
            p_eachblade[i, j] = p
            p_fluc_total[j] += p
            

        plt.figure(0)
        plt.plot(time, p_eachblade[i, :])
        plt.legend(["1", "2", "3", "4", "5"])
        plt.xlabel("Time [s]")
        plt.ylabel("p(x,t)")

        prms_eachblade[i, k] = np.sqrt(np.mean(p_eachblade[i, :]*p_eachblade[i, :]))
        rms_value = prms_eachblade[i, k]
        #print(prms_eachblade[i, k])

        SPL_eachblade[i, k] = 20 * np.log10(prms_eachblade[i, k] / p_ref)
        PWL_eachblade[i, k] = SPL_eachblade[i, k] + 11 + 20 * np.log10(np.linalg.norm(r_vector))

    SPLavg_blade[k] = np.mean(SPL_eachblade[1, k])
    print(SPLavg_blade)
    PWLavg_blade[k] = np.mean(PWL_eachblade[1, k])
    
    prms = np.sqrt(np.mean(p_fluc_total**2))
    
    SPL_array[k] = 10*np.log10((prms/p_ref)**2)


plt.figure(1)
plt.scatter(theta, SPL_array)
plt.xlabel(r"$\theta$ [°]")
plt.ylabel("SPL - dB")
#plt.show()


plt.figure(2)
plt.scatter(theta, PWLavg_blade)
plt.xlabel(r"$\theta$ [°]")
plt.ylabel("PWL - dB")
#plt.show()

plt.show()