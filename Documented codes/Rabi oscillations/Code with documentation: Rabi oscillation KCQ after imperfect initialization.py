from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from math import sqrt

from matplotlib.animation import FuncAnimation

# this should be a guide how to implement Rabi oscillations after a imperfect initializaiton of the KCQ
# parameters
N = 30  # Fock state truncation
K = 4 * pi * 1e5  # Kerr nonlinearity
e2 = 4 * pi * 3 * 1e5  # squeezing drive strength
alpha = sqrt(e2 / K)  # sqrt of photon number
omega = 5 * 1e9  # resonance frequency
Q = 10 * 1e5  # quality factor
kappa = omega / Q  # photon loss rate


a = destroy(N)  # destruction operator

H0_init = -K * a.dag() * a.dag() * a * a  # non time-varying part of the Hamiltonian

H1_init = e2 * (a.dag() ** 2 + a**2)  # time varying part of the Hamiltonian


# coefficient for the time-varying part of the Hamiltonian
def H1_init_coeff(t, args):
    return 1 / 2 * math.tanh((t - 50 / K) * (K / 20)) + 1 / 2


# complete Hamiltonian
H_init = [H0_init, [H1_init, H1_init_coeff]]
# time scale
times = np.linspace(0, 100 / K, 2000)

opt = Options(nsteps=3000)  # steps of numerical solving of master equation

# solving master equation numrically for intialization process
time_ev = mesolve(
    H_init, fock(N, 0), tlist=times, c_ops=sqrt(kappa) * a, args=None, options=opt
)

end_state = time_ev.states[
    -1
]  # return end state as the start state for rabi oscillations


ex = 1e6 / 2
H_drive = (
    -K * a.dag() * a.dag() * a * a
    + e2 * (a.dag() ** 2 + a**2)
    + ex * a.dag()
    + np.conjugate(ex) * a
)  # Hmailtonian with Rabi drive term


times = np.linspace(0, 8 * 10 ** (-6), 800)

opt = Options(nsteps=3000)
# solve master equation
time_ev_drive = mesolve(
    H_drive, end_state, times, c_ops=sqrt(kappa) * a, args=None, options=opt
)

# plot for the evolution after the initialization
fig, ax = plt.subplots()


def update(frame):
    state = time_ev_drive.states[frame]
    gridspace = np.linspace(-5, 5, 100)
    W = wigner(state, gridspace, gridspace)  # generate Wigner function
    ax.clear()  # clear plot to ensure only one frame is plotted at a time
    ax.contourf(
        gridspace, gridspace, W, 100, cmap="seismic", vmin=-0.317, vmax=0.317
    )  # plot Wigner function, 100 levels of colors from vmin to vmax for the colorscale
    # plot appearance
    ax.set_title("Wigner function at t = {:.1e}".format(times[frame]))
    ax.set_xlabel("Position")
    ax.set_ylabel("Momentum")
    ax.set_aspect("equal")


# generate animation
ani = FuncAnimation(fig, update, frames=len(times), repeat=False, interval=10)
# plt.show()
ani.save("Amimation Rabi oscillations KCQ.mp4", writer="ffmpeg")
