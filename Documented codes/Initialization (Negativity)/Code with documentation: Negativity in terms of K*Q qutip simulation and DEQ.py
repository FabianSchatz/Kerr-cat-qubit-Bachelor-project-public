import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy import pi
from numpy import tanh
from scipy import interpolate
from joblib import dump
from joblib import load

omega = 2 * np.pi * 5 * 1e9  # resonance frequency


# derivation of the differential equation for the negativity during turn up is explained in detail in Bachelor's thesis
def DEQ_Negativity(n, t, K, e2, kappa, time_param):
    # alpha2_t means alpha(t)**2
    alpha2_t = e2 / (2 * K) * (tanh((t - 4 * time_param / K) * K / time_param) + 1)
    # differential equation for increasing negativity due to increasing alpha
    increase = (
        n
        * pi**2
        * e2
        / (16 * alpha2_t**2 * time_param)
        * (1 - tanh((t - 4 * time_param / K) * K / time_param) ** 2)
    )
    # differential equation for decreasing negativity due to photon loss
    decrease = -2 * alpha2_t * kappa * n
    # complete differential equation
    dNegdt = increase + decrease
    return dNegdt


# solve DEQ for the system parameters (no optimizing over e2 yet)
# starting value of the numerical integration is explained in detail in Bachelor's thesis, but short explanation inside function
# return end negativity
def max_wigner_DEQ(K, e2, kappa, time_param):
    # numerical integration starts at time 3 * time_param / K, if it starts earlier, negativity values are very small and numerical integration fails (gives wrong result)
    t = np.linspace(3 * time_param / K, 6 * time_param / K, 1000)
    # calculation of Wigner negativity at the start of the integration, can be calculated because the start state is almost the cat state as there for small alpha photon loss is negligible
    n0 = 1 / pi * np.exp(-(pi**2) * K / (4 * 0.24 * e2))
    # numerical integration
    n = odeint(DEQ_Negativity, n0, t, args=(K, e2, kappa, time_param))
    return n[-1]


# optimize negativity over e2 (corresponding to photon number of the end state)
def max_wigner_optimized_over_e2_DEQ(K0, kappa0, time_param):
    # create a list of all negativity values for the e2's
    list_max_wigner_optimized_over_e2 = [
        max_wigner_DEQ(K0, i, kappa0, time_param)
        for i in np.linspace(K0 * 0.8, 7 * K0, 50)
    ]
    # return the max negativity value and the e2 in Hz for which we get this negativity
    return max(list_max_wigner_optimized_over_e2), np.linspace(0.8 * K0, 7 * K0, 50)[
        list_max_wigner_optimized_over_e2.index(max(list_max_wigner_optimized_over_e2))
    ] / (2 * pi)


# define the axis K*Q
KQ_array = np.logspace(np.log10(7 * 10**9), np.log10(2 * 10**12), 100)
# like for the qutip simulation, fix a K and go over kappa instead of K*Q (because optimized negativity is only dependent on K*Q)
K_fixed = 5e5
kappa_array = omega / (KQ_array / (K_fixed))

# generate the optimized negativity values for given K and kappa and therefore given K*Q
optimized_negativity_values_DEQ = [
    max_wigner_optimized_over_e2_DEQ(2 * pi * K_fixed, kappa_array[i], 2)[0]
    for i in range(len(kappa_array))
]

# plot the optimized negativity values against K*Q
plt.plot(
    KQ_array,
    optimized_negativity_values_DEQ,
    color="black",
    label="Calculation for adiabatic evolution",
)


# plot for comparison the Negativity values we got from the qutip simulation
# load the function that gives Negativity in terms of K*Q for the qutip simulation
Negativity_KQ_qutip = load("Negativity(KQ) 100 values newest")
plt.plot(KQ_array, Negativity_KQ_qutip(KQ_array), color="red", label="qutip simulation")

# plot appearance
plt.xscale("log")
plt.xlabel("K*Q in Hz")
plt.ylabel("Maximum Wigner negativity")

plt.legend()

plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="gray")
plt.tick_params(axis="both", which="both", direction="in", length=6, width=1)
plt.minorticks_on()

plt.show()
