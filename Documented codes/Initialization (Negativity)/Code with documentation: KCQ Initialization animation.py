from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from math import sqrt

from matplotlib.animation import FuncAnimation


# show initialization process in phase space with Wigner function
# left figure show animated initialization and right figure show perfect initialized without loss

# parameter as always
N = 50  # Fock state truncation
K = 2 * pi * 1 * 1e5  # Kerr nonlinearity
e2 = 2 * pi * 4 * 1e5  # squeezing drive strength
alpha = sqrt(
    e2 / K
)  # sqrt of photon number for the coherent eigenstate of the Hamiltonian
omega = 2 * pi * 5 * 1e9  # resonance frequency
Q = 1 * 1e6  # quality factor
kappa = omega / Q  # photon loss rate


a = destroy(N)  # destruction operator
time_param = 2  # time parameter (dimensionless) characterizing the duration of the squeezing drive turn up

# time-varying Hamiltonian (here time_param = 2)
H0_init = -K * a.dag() * a.dag() * a * a  # non time-varying part of Hamiltonian

H1_init = e2 * (a.dag() ** 2 + a**2)  # time-varying part of Hamiltonian


# coefficient that captures time dependence
def H1_init_coeff(t, args):
    return 1 / 2 * math.tanh((t - 4 * time_param / K) * (K / (2 * time_param))) + 1 / 2


# complete Hamiltonian
H_init = [H0_init, [H1_init, H1_init_coeff]]

# solving master equation
times = np.linspace(0, 6 * time_param / K, 200)

opt = {"nsteps": 3000}
time_ev = mesolve(
    H_init, fock(N, 0), tlist=times, c_ops=sqrt(kappa) * a, args=None, options=opt
)


# Plot
fig, axs = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.3)


# make frame for each time step
def update(frame):
    state = time_ev.states[frame]
    gridspace = np.linspace(-5, 5, 100)
    W = wigner(state, gridspace, gridspace)  # generate Wigner function
    axs[0].clear()  # clear plot to ensure only one frame is plotted at a time
    c = axs[0].contourf(
        gridspace, gridspace, W, 100, cmap="seismic", vmin=-1 / pi, vmax=1 / pi
    )  # plot the Wigner function (three dimensional graph), vmin/vmax is lower/upper limit for the colorscale
    # plot appearance
    axs[0].set_title("Wigner function at t = {:.1e}".format(times[frame]))
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Momentum")
    axs[0].set_aspect("equal")


# make colorbar
gridspace = np.linspace(-5, 5, 100)
W_even_cat = wigner(
    1 / sqrt(2) * (coherent(N, alpha) + coherent(N, -alpha)), gridspace, gridspace
)

c = axs[1].contourf(
    gridspace, gridspace, W_even_cat, 100, cmap="seismic", vmin=-1 / pi, vmax=1 / pi
)

cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
cbar = plt.colorbar(c, cax=cbar_ax)

# plot appearance
axs[0].set_position([0.07, 0.3, 0.4, 0.4])
axs[1].set_position([0.48, 0.3, 0.4, 0.4])
axs[1].set_aspect("equal")

# initialize animation
ani = FuncAnimation(fig, update, frames=len(times), repeat=False, interval=20)
plt.show()
