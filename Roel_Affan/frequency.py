import numpy as np
import input_par as inp
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Apply the default theme
sns.set_theme()


def frequency(input_par, phi, theta, m, s):
    '''
    This function is to plot the frequency domain of a dipole sound source as a result of a constant force on a rotor
    bladed. The equation is obtained from the Fundamentals of Aeroacoustics assignment slides. It contains several
    terms, which are called a, b, c, d and e to decrease the length of one line and for debugging purposes.
    Note that the equation has a sum over s_i. The individual parts are summed in p_part and at the end multiplied with
    the term 'a'.

    The pressure amplitude (real part) and phase (imaginary part) are inside the output 'p'.

    :param input_par: This is the input file where all the parameters are specified. (see input_par.py) for all these
                      parameters.
    :param phi: Angle with the horizontal position component and the x axis.
    :param theta: Angle of the position vector with the plane of rotation of the rotor.
    :param m: This is the harmonic number is just a number from -XX to +XX. It describes the shape of a harmonic like
              the shapes on a string of a guitar.
    :param s: This is the summing parameter. It contributes to the Bessel function in term 'd' and to term 'e'.
    :return: Pressure numbers 'p'. It is a complex number where the complex part is the phase and the real part the
             amplitude. If plotted against the frequency Bm (blade number times 'm') you get a even curve for the
             amplitude and odd for the phase. Therefore, the phase should sum to zero and the signal should remain as
             a purely real signal.
    '''

    # Part of the frequency pressure equation before the sum
    a = -1j * m * input_par.n_blades ** 2 * input_par.v_angular * np.exp(
        -1j * m * input_par.n_blades * input_par.v_angular * input_par.r0 /
        input_par.v_sound) / (4 * np.pi * input_par.v_sound * input_par.r0)
    p_part = 0  # to define p_part
    for s_i in range(len(s)):   # for every s, de sum part must be calculated and summed into p_part.

        b = input_par.fs[s_i] * np.exp(-1j * (m * input_par.n_blades - input_par.s[s_i]) * (phi - np.pi / 2))
        c = m * input_par.n_blades * input_par.mach_force * np.sin(theta)   # input of the bessel function
        d = sp.special.jn(m * input_par.n_blades - input_par.s[s_i], c)     # Bessel function
        e = -(m * input_par.n_blades - input_par.s[s_i]) * np.sin(input_par.gamma) / (m * input_par.n_blades * input_par.mach_force) + \
            np.cos(theta) * np.cos(input_par.gamma)
        # e = np.cos(theta) * np.cos(input_par.gamma)

        p_part += b * d * e     # Summing over s

    p = a * p_part

    return p

# def freq_to_spl(p_frec):
#     p_time = np.fft.ifft(p_frec)
#     p_rms = np.sqrt(np.mean((np.real(p_time) ** 2)))
#     p_ref = 2 * 10 ** -5
#     spl = 10 * np.log10(p_rms ** 2 / p_ref ** 2) # / 2
#     return spl

def freq_to_spl(p_freq):
    p_rms = np.sqrt(np.real(sum(p_freq**2))/ len(p_freq))  # calculate prms
    p_ref = 2 * 10 ** -5    # Defined reference pressure
    spl = 10 * np.log10(p_rms ** 2 / p_ref ** 2)    # final spl
    return spl


if __name__ == "__main__":
    # # ______________Frequency domain excecution_________________
    # Specify s range and m range. s are the steps of a sum so it should be with steps of 1.
    # m can vary with steps of 0.5.
    m = np.arange(-5*inp.n_blades, 5*inp.n_blades+1, 1) # this does not comply with the inverse FT
    m = np.arange(0, 5*inp.n_blades+1, 1)
    m = np.append(m, np.arange(-5*inp.n_blades, 0, 1))
    m = m[m != 0]  # m = 0 give a singularity so it should be removed
    p_out = {}
    # for phi_i in tqdm(inp.phi, desc="Calculating the pressure"):
    for phi_i in inp.phi:
        p_out["Phi=" + str(round(np.degrees(phi_i),0))] = {}    # for every phi
        for theta_i in tqdm(inp.theta, desc="Calculating pressure"):
            p_out["Phi=" + str(round(np.degrees(phi_i), 0))]["Theta=" + str(round(np.degrees(theta_i), 0))] = {}
            # p is calculated for many m so the individual p are stored in an array.
            p = np.array([])
            for m_i in m:
                p = np.append(p, np.array(frequency(inp, phi_i, theta_i, m_i, inp.s)))  # put the individual p in the array
            p_out["Phi=" + str(round(np.degrees(phi_i),0))]["Theta=" + str(round(np.degrees(theta_i), 0))]["Freq"] = p
            p_out["Phi=" + str(round(np.degrees(phi_i),0))]["Theta=" + str(round(np.degrees(theta_i), 0))]["SPL"] = freq_to_spl(p)
            print("Theta="+str(np.degrees(theta_i)))
            print("Pressure="+str(p))
            print("SPL="+str(freq_to_spl(p)))
            print()


    # To check if the imaginary part is zero, the sum of all these parts are taken.
    arr = np.imag(p)
    sum_imag = sum(np.imag(p))
    print("The sum of the imaginary part of the signal is " + str(sum_imag))
    input("Check sum")

    # Plot the phase and amplitude spectrum.
    fontsizee = 20  # This should make the tekst readable.
    for phi_i in range(len(inp.phi)):
    # for phi_i in range(1):
        plt.close('all')
        for plots_i in tqdm(range(0, len(inp.theta), 60), desc="Making figures"):
            fig_spec, ax_spec = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            # plots["fig"] = np.append(plots["fig"], plt.subplots(nrows=1, ncols=2)[0])
            # plots["ax"] = np.append(plots["ax"], plt.subplots(nrows=1, ncols=2)[1])
            ax_spec[0].scatter(m * inp.n_blades, np.imag(p_out["Phi=" + str(round(np.degrees(inp.phi[phi_i]),0))]
                                                      ["Theta=" + str(round(np.degrees(inp.theta[plots_i]), 0))]["Freq"]))  # Plot the phase spectrum

            ax_spec[0].set_ylabel("Phase", fontsize=fontsizee)
            ax_spec[0].set_xlabel("Frequency mB", fontsize=fontsizee)
            ax_spec[0].tick_params(axis='both', labelsize=fontsizee)

            p_ref = 2 * 10 ** -5
            amp = np.real(p_out["Phi=" + str(round(np.degrees(inp.phi[phi_i]),0))]
                                                      ["Theta=" + str(round(np.degrees(inp.theta[plots_i]), 0))]["Freq"])
            amp_db = 10 * np.log10(amp ** 2 / p_ref ** 2)
            ax_spec[1].scatter(m * inp.n_blades, amp)  # Plot the amplitude spectrum
            ax_spec[1].set_ylabel("Amplitude", fontsize=fontsizee)
            ax_spec[1].set_xlabel("Frequency mB", fontsize=fontsizee)
            ax_spec[1].tick_params(axis='both', labelsize=fontsizee)
            # save_spec = r'C:\Users\roelv\Documents\School\TU_Delft\MSC_1\Physics\AE4260A_Fundamentals_of_Aeroacoustics\Assignment\figures\spectrum_phi='+str(round(np.degrees(inp.phi[phi_i]),0))+'_theta='+str(round(np.degrees(inp.theta[plots_i]), 0))+'.png'
            fig_spec.tight_layout()
            # fig_spec.savefig(save_spec)

            # plt.close('all')
            spl = np.array([])
            for theta_i in inp.theta:
                spl = np.append(spl, p_out["Phi=" + str(round(np.degrees(inp.phi[phi_i]),0))]
                                                          ["Theta=" + str(round(np.degrees(theta_i), 0))]["SPL"])


            fig_spec_spl, ax_spec_spl = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
            ax_spec_spl.scatter(np.degrees(inp.theta), spl)  # Plot the amplitude spectrum
            ax_spec_spl.set_ylabel("SPL", fontsize=fontsizee)
            ax_spec_spl.set_xlabel(r'$\theta$ [deg]', fontsize=fontsizee)
            ax_spec_spl.tick_params(axis='both', labelsize=fontsizee)
            ax_spec_spl.set_ylim([-10, 60])
            # save_spl = r'C:\Users\roelv\Documents\School\TU_Delft\MSC_1\Physics\AE4260A_Fundamentals_of_Aeroacoustics\Assignment\figures\spl_phi='+str(round(np.degrees(inp.phi[phi_i]),0))+'.png'
            fig_spec_spl.tight_layout()
            # fig_spec_spl.savefig(save_spl)

            # plt.close('all')
            fig_polar, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
            ax_polar.plot(inp.theta, spl)
            # save_polar = r'C:\Users\roelv\Documents\School\TU_Delft\MSC_1\Physics\AE4260A_Fundamentals_of_Aeroacoustics\Assignment\figures\polar_spl_phi=' + str(
            # round(np.degrees(inp.phi[phi_i]), 0)) + '.png'
            # fig_polar.tight_layout()
            # fig_polar.savefig(save_polar)
        plt.show()