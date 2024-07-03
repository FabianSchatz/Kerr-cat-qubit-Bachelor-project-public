from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from math import sqrt
from joblib import load
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

# load the function that gives the photon number for optimal initialization
n_KQ = load("n(KQ) qutip 100 values interpolation newest")


# to get alpha we take the sqrt, for values of K*Q smaller than 7e9 Hz we dont care about the dephasing time because the initialized state doesn't have Negativity in the Wigner function
# we set the value alpha value for K*Q = 7e9
def alpha(KQ):
    if KQ > 7 * 1e9:
        return sqrt(n_KQ(KQ))
    else:
        return sqrt(0.92)


N = 20  # Fock state truncation
omega = 2 * pi * 5 * 1e9  # resonance frequency


# goal: simulate decay time of the coherent alpha state to get dephasing time for KCQ
# therefore plot the x expectation value against time and fit an exponential decay
# function takes characteristic system parameters K in Hz and Q
# function assumes thermal nuumber and kappa_eff (rate for a^dagger * a noise) to be the same as in Grimm paper
def dephasing_time(K_Hz, Q):
    # calculate related system parameters from K and Q (alpha value such that we have maximum negativity)
    K = K_Hz * 2 * pi
    alpha_value = alpha(K_Hz * Q)
    e2 = K * alpha_value**2  # calculate the e2 of the optimal initialization
    kappa = omega / Q  # photon loss rate
    kappa_thermal = kappa * 0.08  # photon gain rate
    kappa_eff = 0  # no alpha^dagger a noise

    a = destroy(N)  # destruction operator
    x = (a + a.dag()) / 2  # x operator

    H_cat = -K * a.dag() * a.dag() * a * a + e2 * (
        a.dag() ** 2 + a**2
    )  # static Hamiltonian

    times = np.linspace(0, 1e-6 * 150, 400)  # time scale of system evolution

    opt = {
        "nsteps": 7000
    }  # number of steps in numerical solving of the master equation

    # master equation solver, coherent alpha state is the initial state, loss operators corresponding to the noise processes, e_ops=[x] gives the x expectation value for each time in times
    time_ev = mesolve(
        H_cat,
        coherent(N, alpha_value),
        tlist=times,
        c_ops=[
            sqrt(kappa) * a,
            sqrt(kappa_thermal) * a.dag(),
            sqrt(kappa_eff) * a.dag() * a,
        ],
        e_ops=[x],
        args=None,
        options=opt,
    )

    # fit exponential decay (at t=0 the x expectation value of the coherent state is alpha), c is the dephasing constant
    def fitted_exponential(x, c):
        return alpha_value * np.exp(-x * c)

    dephasing_time_value, _ = curve_fit(fitted_exponential, times, time_ev.expect[0])
    print(dephasing_time_value[0])
    return dephasing_time_value[0]


# make a K,Q grid
K = np.linspace(0.5e5, 10e5, 10)
Q = np.linspace(0.5e5, 10e5, 10)


# Calculate the dephasing times for each pair (K, Q)
decay_constants = np.zeros((len(K), len(Q)))
for i in range(len(K)):
    for j in range(len(Q)):
        decay_constants[i, j] = dephasing_time(K[i], Q[j])

# Plot the contour plot with the given levels for the dephasing time
plt.figure(figsize=(8, 6))
viridis_r = plt.cm.get_cmap("viridis").reversed()
contour = plt.contour(
    K / 1e6,
    Q / 1e6,
    decay_constants.T,
    levels=np.array([3e3, 4e3, 5e3, 6e3, 1e4, 1.5e4, 3e4, 6e4]),
    cmap=viridis_r,
    vmin=3e3,
    vmax=6.5e4,
)

# plot appearance
# plt.colorbar(contour, label=" KCQ Dephasing Time in " r"$\mu s$")
plt.xlabel("K in MHz")
plt.ylabel("Q in " r"$10^6$")

plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="black")
plt.tick_params(axis="both", which="both", direction="in", length=6, width=1)
plt.minorticks_on()

# Legend description
legend_description = Line2D(
    [0], [0], linestyle="", label=r"$\text{Decay constant in } 1/s $"
)

# Contour legend labels
legend_labels = [
    "3000",
    "4000",
    "5000",
    "6000",
    "10000",
    "15000",
    "30000",
    "60000",
]
legend_colors = [
    contour.collections[i].get_edgecolor() for i in range(len(legend_labels))
]
legend_handles = [
    Line2D([0], [0], linestyle="-", color=legend_colors[i])
    for i in range(len(legend_labels))
]

# Add description to the legend handles and labels
legend_handles.insert(0, legend_description)
legend_labels.insert(0, r"$\text{Decay constant in } 1/s $")

plt.legend(legend_handles, legend_labels)

plt.show()
