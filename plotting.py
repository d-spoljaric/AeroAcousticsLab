import matplotlib.pyplot as plt
import numpy as np

def filter_angles(data: np.ndarray, angle_list: np.ndarray) -> np.ndarray:
    filtered_data = np.zeros(shape = (len(angle_list), 2))
    for count, angle in enumerate(angle_list):
        data_idx = np.where(np.abs(angle - data[:, 0]) == np.min(np.abs(angle-data[:, 0])))[0][0]
        filtered_data[count, 0] = data[data_idx, 0]
        filtered_data[count, 1] = data[data_idx, 1]
    return filtered_data

# Unfiltered


SPL_M03_B4_const = np.genfromtxt(r"Data\SPL_M03_B4_const.dat")
SPL_M06_B4_const = np.genfromtxt(r"Data\SPL_M06_B4_const.dat")
SPL_M03_B5_const = np.genfromtxt(r"Data\SPL_M03_B5_const.dat")
SPL_M06_B5_const = np.genfromtxt(r"Data\SPL_M06_B5_const.dat")

SPL_M03_B4_var = np.genfromtxt(r"Data\SPL_M03_B4_var.dat")
SPL_M06_B4_var = np.genfromtxt(r"Data\SPL_M06_B4_var.dat")
SPL_M03_B5_var = np.genfromtxt(r"Data\SPL_M03_B5_var.dat")
SPL_M06_B5_var = np.genfromtxt(r"Data\SPL_M06_B5_var.dat")

# Filtered or angles requested in assignment
directivity_plot_theta = np.arange(0, 105, 15)

SPL_M03_B4_const_filtered = filter_angles(SPL_M03_B4_const, directivity_plot_theta)
SPL_M06_B4_const_filtered = filter_angles(SPL_M06_B5_const, directivity_plot_theta)
SPL_M03_B5_const_filtered = filter_angles(SPL_M03_B5_const, directivity_plot_theta)
SPL_M06_B5_const_filtered = filter_angles(SPL_M06_B5_const, directivity_plot_theta)

SPL_M03_B4_var_filtered = filter_angles(SPL_M03_B4_var, directivity_plot_theta)
SPL_M06_B4_var_filtered = filter_angles(SPL_M06_B4_var, directivity_plot_theta)
SPL_M03_B5_var_filtered = filter_angles(SPL_M03_B5_var, directivity_plot_theta)
SPL_M06_B5_var_filtered = filter_angles(SPL_M06_B5_var, directivity_plot_theta)



ax = plt.subplot(1, 1, 1)
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.plot(SPL_M03_B4_const[:, 0], SPL_M03_B4_const[:, 1], color = "k", linewidth = 1, label = "M=0.3, B=4")
plt.plot(SPL_M06_B4_const[:, 0], SPL_M06_B4_const[:, 1], color = "r", linewidth = 1, label = "M=0.6, B=4")
plt.plot(SPL_M03_B5_const[:, 0], SPL_M03_B5_const[:, 1], color = "orange", linewidth = 1, label = "M=0.3, B=5")
plt.plot(SPL_M06_B5_const[:, 0], SPL_M06_B5_const[:, 1], color = "g", linewidth = 1, label = "M=0.6, B=5")
# plt.scatter(SPL_M03_B4_const_filtered[:, 0], SPL_M03_B4_const_filtered[:, 1], color = "k", label = "M=0.3, B=4")
# plt.scatter(SPL_M06_B4_const_filtered[:, 0], SPL_M06_B4_const_filtered[:, 1], color = "r", label = "M=0.6, B=4")
# plt.scatter(SPL_M03_B5_const_filtered[:, 0], SPL_M03_B5_const_filtered[:, 1], color = "orange", label = "M=0.3, B=5")
# plt.scatter(SPL_M06_B5_const_filtered[:, 0], SPL_M06_B5_const_filtered[:, 1], color = "k", label = "M=0.6, B=5")
plt.minorticks_on()
plt.grid(True, which = "both")
plt.xlabel(r"$\theta$ [rad]")
plt.ylabel("SPL")
# plt.ylim([0, 60])
plt.legend()
plt.show()

ax2 = plt.subplot(1, 1, 1)
ax2.get_yaxis().get_major_formatter().set_useOffset(False)
plt.plot(SPL_M03_B4_var[:, 0], SPL_M03_B4_var[:, 1], color = "k", linewidth = 1, label = "M=0.3, B=4")
plt.plot(SPL_M06_B4_var[:, 0], SPL_M06_B4_var[:, 1], color = "r", linewidth = 1, label = "M=0.6, B=4")
plt.plot(SPL_M03_B5_var[:, 0], SPL_M03_B5_var[:, 1], color = "orange", linewidth = 1, label = "M=0.3, B=5")
plt.plot(SPL_M06_B5_var[:, 0], SPL_M06_B5_var[:, 1], color = "g", linewidth = 1, label = "M=0.6, B=5")
# plt.scatter(SPL_M03_B4_var_filtered[:, 0], SPL_M03_B4_var_filtered[:, 1], color = "k", label = "M=0.3, B=4")
# plt.scatter(SPL_M06_B4_var_filtered[:, 0], SPL_M06_B4_var_filtered[:, 1], color = "r", label = "M=0.6, B=4")
# plt.scatter(SPL_M03_B5_var_filtered[:, 0], SPL_M03_B5_var_filtered[:, 1], color = "orange", label = "M=0.3, B=5")
# plt.scatter(SPL_M06_B5_var_filtered[:, 0], SPL_M06_B5_var_filtered[:, 1], color = "k", label = "M=0.6, B=5")
plt.minorticks_on()
plt.grid(True, which = "both")
plt.xlabel(r"$\theta$ [rad]")
plt.ylabel("SPL")
plt.ylim([0, 75])
plt.legend()
plt.show()