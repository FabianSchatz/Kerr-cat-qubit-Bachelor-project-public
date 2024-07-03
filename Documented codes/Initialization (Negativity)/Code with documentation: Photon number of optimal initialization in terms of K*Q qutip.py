from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt
from scipy import interpolate
from joblib import dump

N = 20  # fock state truncation
omega = 2 * pi * 5e9  # resonance frequency


# function that returns the end max negativity for given parameters of the initialization process
# inputs K,kappa,e2 in unit 1/s
# time_param dimensionless parameter that characterizes and is linearly proportional to the duration of the turn-up (read in detail in Bachelor's thesis)
def wigner_negativity(K, kappa, e2, time_param):
    a = destroy(N)  # destruction operator
    H0_init = -K * a.dag() * a.dag() * a * a  # non time-varying part of the Hamiltonian

    # time-varying part of the Hmailtonian with coefficient the captures the time-variation
    H1_init = e2 * (a.dag() ** 2 + a**2)

    def H1_init_coeff(t, args):
        return 1 / 2 * np.tanh((t - 4 * time_param / K) * (K / time_param)) + 1 / 2

    # complete Hamiltonian
    H_init = [H0_init, [H1_init, H1_init_coeff]]

    # time scale for numerical solving of the master equation
    times = np.linspace(0, (6 * time_param) / K, 200)

    opt = {
        "nsteps": 3000
    }  # steps of the integration (needs to be increased for certain computations)

    # mesolve (see documentation qutip: Master equation), c_ops are complete list of loss operators
    time_ev = mesolve(
        H_init, fock(N, 0), tlist=times, c_ops=sqrt(kappa) * a, args=None, options=opt
    )

    end_state = time_ev.states[-1]  # end state of initialization via Master equation

    # evaluate Wigner negativity
    x_gridspace = np.linspace(0, 0, 1)
    y_gridspace = np.linspace(-2, 2, 2000)
    W_end_state = wigner(end_state, x_gridspace, y_gridspace)
    wigner_values = [
        W_end_state[i, 0] for i in range(2000)
    ]  # negativity maximal on the y-axis

    return -min(wigner_values)


# returns for given K and Q the minimum value of wigner funciton that is possible to reach with optimizing over e2 (second output is the e2 value in Hz that optimizes Wigner negativity)
def max_wigner_negativity_optimized_over_e2(K, kappa, time_param):
    # empirically we could test that photon numbers lower than 0.8 and higher than 7 don't lead to higher negativity (increasing range of e2 values doesn't affect results)
    e2_values = np.linspace(0.8 * K, 8 * K, 60)  # array with e2s with optimize over
    negativity_values = np.array(
        [wigner_negativity(K, kappa, e2, time_param) for e2 in e2_values]
    )  # array with all Wigner negativity value for the possible e2s

    # find max and the e2 value that maximizes negativity
    max_negativity_value = np.max(negativity_values)
    max_e2 = e2_values[np.argmax(negativity_values)]
    print(max_e2 / (2 * pi * 5e5))
    return max_negativity_value, max_e2 / (2 * pi)


# plot K*Q in Hz versus optimal e2
# we only plot 10 e2 because we want a "smooth" function and we expect similar e2s to result in similar values for negativtiy so we don't mind if values are not perfectly exact
# read in thesis why initialization negativity only depends on product K*Q
# K is set to 500 kHz
KQ_array_Hz = np.logspace(np.log10(7 * 10**9), np.log10(2 * 10**12), 100)
K_fixed = 5e5
kappa_array = omega / (KQ_array_Hz / K_fixed)
photon_number_optinmal_initialization = [
    max_wigner_negativity_optimized_over_e2(2 * pi * K_fixed, kappa_array[i], 2)[1]
    / K_fixed
    for i in range(len(kappa_array))
]


photon_number_optinmal_initialization[:4] = [0.926] * 4
plt.plot(
    KQ_array_Hz,
    photon_number_optinmal_initialization,
    label="Photon number of initialized state simulated with qutip",
)


f = interpolate.interp1d(KQ_array_Hz, photon_number_optinmal_initialization)


dump(f, "n(KQ) qutip 100 values interpolation")


# plot appearance
plt.xscale("log")
plt.xlabel("K*Q in Hz")
plt.ylabel("Photon number for optimal initialization")

plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="gray")
plt.tick_params(axis="both", which="both", direction="in", length=6, width=1)
plt.minorticks_on()
plt.legend()
plt.show()
