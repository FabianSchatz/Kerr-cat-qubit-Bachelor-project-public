import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy import pi
from numpy import tanh
from scipy import interpolate
from joblib import dump


# only short comments for this plot: comments in detail are in "Negativity in terms of K*Q qutip simulation and DEQ.py"
# only slight tweek to get the photon number instead of the negativity for optimal initialization
omega = 2 * np.pi * 5 * 1e9  # Fock state truncation


# DEQ model for the initialization process
def model(n, t, K, e2, kappa, time_param):
    alpha2_t = e2 / (2 * K) * (tanh((t - 4 * time_param / K) * K / time_param) + 1)
    increase = (
        n
        * pi**2
        * e2
        / (16 * alpha2_t**2 * time_param)
        * (1 - tanh((t - 4 * time_param / K) * K / time_param) ** 2)
    )
    decrease = -2 * alpha2_t * kappa * n
    return increase + decrease


# solve this differential equation to get end negativity for system parameters
def min_wigner_model(K, e2, kappa, time_param):
    t = np.linspace(3 * time_param / K, 6 * time_param / K, 1000)
    n0 = 1 / pi * np.exp(-(pi**2) * K / (4 * 0.24 * e2))
    n = odeint(model, n0, t, args=(K, e2, kappa, time_param))
    return n[-1]


# optimize over e2
def min_wigner_optimized_over_e2_model(K0, kappa0, time_param):
    list_min_wigner_e2 = [
        -min_wigner_model(K0, i, kappa0, time_param)
        for i in np.linspace(K0 * 0.4, 8 * K0, 400)
    ]
    return -min(list_min_wigner_e2), np.linspace(0.4 * K0, 8 * K0, 400)[
        list_min_wigner_e2.index(min(list_min_wigner_e2))
    ] / (2 * pi)


# generate an array for product K*Q and an array that gives the corresponding negativity
KQ_array = np.logspace(np.log10(7 * 10**9), np.log10(2 * 10**12), 100)
photon_number_KQ = [
    min_wigner_optimized_over_e2_model(
        2 * pi * 5 * 1e5, omega / (KQ_array[i] / (5 * 1e5)), 2
    )[1]
    / (5 * 1e5)
    for i in range(len(KQ_array))
]

# plot results
plt.plot(
    KQ_array,
    photon_number_KQ,
    color="black",
    label="Calculation for adiabatic evolution",
)

# plot appearance
plt.ylim(0, None)
plt.xscale("log")
plt.xlabel(r"$K\cdot Q$")
plt.ylabel("photon number" r"$\quad \alpha^2$")
plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="gray")
plt.tick_params(axis="both", which="both", direction="in", length=6, width=1)
plt.minorticks_on()
plt.show()
