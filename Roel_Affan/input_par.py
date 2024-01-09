import numpy as np

thrust = 2000
gamma = 0 * np.pi / 180
diameter = 1
r_force = 0.85*diameter/2
n_blades = 4 # np.array([4, 5])
mach_tip = 0.3 # np.array([0.3, 0.6])
density = 1.225 # sea level
temperature = 293   # 20 degrees Celcius
specific_heat = 1.4
gas_const = 287.15  # air
r0 = 100
# phi = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi/180
# phi = np.arange(0, 1, 1) * np.pi / 180
# theta = np.arange(0, 360, 1) * np.pi / 180
phi = np.array([np.pi/4])
theta = np.array([0, 15, 30, 45, 60, 75, 90]) * np.pi/180

force_one_blade = thrust/n_blades
s = np.array([0])
fs = np.array([force_one_blade])
mach_force = mach_tip*2*r_force/diameter
v_sound = np.sqrt(temperature*specific_heat*gas_const)
v_angular = mach_tip*v_sound/(diameter/2)