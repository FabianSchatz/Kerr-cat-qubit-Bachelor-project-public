from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from math import sqrt
from scipy.optimize import curve_fit

N = 20  # Fock state truncation (here 20 to make code run faster)
K = 2 * pi * 10 * 1e5  # Kerr nonlinearity
e2 = 2 * pi * 10 * 2.4**2 * 1e5  # squeezing drive strength
alpha = sqrt(e2 / K)  # sqrt of photon number
omega = 2 * pi * 5 * 1e9  # resonance frequency
Q = 2 * 1e5  # quality factor
kappa = omega / Q  # photon loss rate
kappa_thermal = kappa * 0.08  # photon gain rate (adjust parameter to see dependence)
kappa_eff = 2 * pi * 230  # a^dagger a noise rate (adjust parameter to see dependence)


a = destroy(N)  # destruction operator
x = (a + a.dag()) / 2  # position operator

H_cat = -K * a.dag() * a.dag() * a * a + e2 * (
    a.dag() ** 2 + a**2
)  # Kerr cat Hamiltonian after intialization


times = np.linspace(0, 1e-6 * 150, 400)  # time scale of system evolution

opt = {"nsteps": 7000}  # number of steps for numerical solving of the master equation

time_ev = mesolve(
    H_cat,
    coherent(N, alpha),  # +X state of the Kerr cat qubit
    tlist=times,
    c_ops=[
        sqrt(kappa) * a,
        sqrt(kappa_thermal) * a.dag(),
        sqrt(kappa_eff) * a.dag() * a,
    ],  # complete list of noise processes with corresponding noise rates
    e_ops=[x],
    args=None,
    options=opt,
)

plt.plot(
    times * 1e6, time_ev.expect[0]
)  # plot x expectation value versus time of evolution in microseconds


# fit an exponential (even though the decay of the x expectation value is not necessarily exponential, we find that this is a very good approximation)
def fitted_exponential(x, t):
    return alpha * np.exp(-x / t)  # fit function


popt, pcov = curve_fit(
    fitted_exponential, times, time_ev.expect[0]
)  # return fitted dephasing time
print(popt)  # print fitted dephasing time
y_fit = fitted_exponential(
    times, *popt
)  # return fitted exponential decay to see how well the time evolution matches with this fit

# plot appearance
plt.plot(times * 1e6, y_fit, "--")
plt.xlabel("time in microseconds")
plt.ylabel("expectation value of x")
plt.ylim(0, None)
plt.xlim(0, None)

# plot a textbox with the used parameters of the KCQ
textstr = f"K = {K/(2*pi)/1e6} Hz\ne2 = {e2/(2*pi)/1e6} Hz\nQ = {Q/1e6} Million"
plt.text(
    0.75,
    0.95,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.show()
