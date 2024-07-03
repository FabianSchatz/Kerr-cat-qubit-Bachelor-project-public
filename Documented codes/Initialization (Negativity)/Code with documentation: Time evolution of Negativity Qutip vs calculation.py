from qutip import *
import numpy as np
from numpy import tanh, pi, sqrt
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# definition of parameters
N = 40  # Fock state truncation
K = 2 * pi * 4 * 1e5  # Kerr nonlinearity
e2 = 2 * pi * 10 * 1e5  # end value of squeezing drive
alpha = sqrt(e2 / K)  # sqrt of photon number
omega = 2 * pi * 5 * 1e9  # resonance frequency
Q = 10 * 1e5  # quality factor
kappa = omega / Q  # photon loss rate

time_param = 10  # time parameter that characterizes the duration of the initialization


def min_wigner(K0, kappa0, e20, time_param):
    a = destroy(N)  # destruction operator

    H0_init = (
        -K0 * a.dag() * a.dag() * a * a
    )  # non time-varying part of the Hamiltonian
    # time-varying part of Hamiltonian with time variation is captured in the coefficient
    H1_init = e20 * (a.dag() ** 2 + a**2)

    def H1_init_coeff(t, args):
        return 1 / 2 * tanh((t - 4 * time_param / K0) * (K0 / time_param)) + 1 / 2

    # complete Hamiltonian
    H_init = [H0_init, [H1_init, H1_init_coeff]]

    # time scale for initialization
    times = np.linspace(0, 6 * time_param / K0, 400)

    # solve Master equation with qutip.mesolve (c_ops is complete list of loss operators, H_init is time-varying Hamiltonian, fock(N,0) is the 0 Fock state to start the initialization process)
    opt = {"nsteps": 3000}  # number of steps of numerical master equation solving
    time_ev = mesolve(
        H_init, fock(N, 0), tlist=times, c_ops=sqrt(kappa0) * a, args=None, options=opt
    )

    # calculate the Wigner function an a grid
    gridspace = np.linspace(-5, 5, 100)
    W_states = [wigner(state, gridspace, gridspace) for state in time_ev.states]

    # make a list that has an entry for every time step in times with the max negativity at that time step
    neg_t = []
    for W in W_states:
        wigner_values = [W[i, 50] for i in range(100)]
        neg_t.append(-min(wigner_values))

    return neg_t


# plot the negativity vs time for this qutip simulation
times = np.linspace(0, 6 * time_param / K, 400)
plt.plot(
    times * K / time_param,
    min_wigner(K, kappa, e2, time_param),
    label="qutip simulation",
)


# differential equation explained in "Negativity in terms of K*Q qutip simulation and DEQ"
def Negativity_DEQ(n, t):
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


t = np.linspace(3 * time_param / K, 6 * time_param / K, 1000)
n0 = 1 / pi * np.exp(-(pi**2) * K / (4 * 0.24 * e2))

n = odeint(Negativity_DEQ, n0, t)


# plot the negativity vs time for the DEQ calculation
plt.plot(
    t * K / time_param,
    n,
    color="red",
    label="Calculation for\nadiabatic evolution (DEQ)",
)

# textbox with relevant parameters
textstr = f"K = {K/(2*pi)/1e6} Hz\ne2 = {e2/(2*pi)/1e6} Hz\nQ = {Q/1e6} Million \ntime_parameter ={time_param}"
plt.text(
    0.05,
    0.75,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.5),
)

# plot appearance
plt.xlabel(r"$\frac{time\cdot K}{p_t}$")
plt.ylabel("Maximum Wigner negativity")
plt.title(f"Time evolution Wigner function squeezing drive turn on")
plt.legend()
plt.show()
