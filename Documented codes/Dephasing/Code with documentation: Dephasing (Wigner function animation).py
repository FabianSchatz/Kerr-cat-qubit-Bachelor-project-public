from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from math import sqrt

from matplotlib.animation import FuncAnimation

# parameter
N = 30  # Fock state truncation
K = 2 * pi * 1 * 1e5  # Kerr nonlinearity
e2 = 2 * pi * 1 * 1e5  # squeezing drive strength
alpha = sqrt(e2 / K)  # sqrt of photon number
omega = 2 * pi * 5 * 1e9  # resonance frequency
Q = 1 * 1e6  # quality factor
kappa = omega / Q  # photon loss rate
kappa_thermal = kappa * 0.08  # photon gain rate
kappa_eff = 2 * pi * 230  # a^dagger a noise rate


a = destroy(N)  # destruction operator


H_cat = -K * a.dag() * a.dag() * a * a + e2 * (
    a.dag() ** 2 + a**2
)  # static Hamiltonian


times = np.linspace(0, 100 / K, 200)  # time scale

opt = {"nsteps": 3000}  # steps for numerical solving of master equation

# master equation solver, with 3 types of noise
time_ev = mesolve(
    H_cat,
    coherent(N, alpha),
    tlist=times,
    c_ops=[
        sqrt(kappa) * a,
        sqrt(kappa_thermal) * a.dag(),
        sqrt(kappa_eff) * a.dag() * a,
    ],
    args=None,
    options=opt,
)


# Plot
fig, axs = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.3)


# plot Wigner function the state in each time frame
def update(frame):
    state = time_ev.states[frame]
    gridspace = np.linspace(-5, 5, 100)
    W = wigner(state, gridspace, gridspace)  # generate Wigner function
    axs[
        0
    ].clear()  # ensures that only one frame is plotted at a time, clears the subplot
    c = axs[0].contourf(
        gridspace, gridspace, W, 100, cmap="seismic", vmin=-1 / pi, vmax=1 / pi
    )  # makes the coloured three dimensional graph plot, vmin/vmax sets the lower/upper limit for the colourscale, 100 means 100 levels for the colorscale, cmap= colormap

    # plot appearance
    axs[0].set_title("Wigner function at t = {:.1e}".format(times[frame]))
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Momentum")
    axs[0].set_aspect("equal")


# have the original state as comparison
gridspace = np.linspace(-5, 5, 100)
W_even_cat = wigner(
    1 / sqrt(2) * (coherent(N, alpha) + coherent(N, -alpha)), gridspace, gridspace
)

c = axs[1].contourf(
    gridspace,
    gridspace,
    wigner(coherent(N, alpha), gridspace, gridspace),
    100,
    cmap="seismic",
    vmin=-1 / pi,
    vmax=1 / pi,
)

# colorbar
cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
cbar = plt.colorbar(c, cax=cbar_ax)
axs[0].set_position([0.07, 0.3, 0.4, 0.4])
axs[1].set_position([0.48, 0.3, 0.4, 0.4])
axs[1].set_aspect("equal")

# initialize animation
ani = FuncAnimation(fig, update, frames=len(times), repeat=False, interval=10)
plt.show()
