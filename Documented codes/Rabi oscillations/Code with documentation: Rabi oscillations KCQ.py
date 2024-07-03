from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt

from matplotlib.animation import FuncAnimation


# parameters (here: from Grimm paper)
N = 30
K = 2 * pi * 6700000
e2 = 2 * pi * 17750000
alpha = sqrt(e2 / K)
ex = 10738186  # I calculated this value from parameters they used in Grimm paper, but don't know exactly anymore
# however, this can be just be a guide on how to implement rabi oscillations in qutip

a = destroy(N)  # desruction operator

H = -K * a.dag() * a.dag() * a * a + e2 * (
    a.dag() ** 2 + a**2
)  # Hamiltonian without Rabi oscillation drive

H_drive = (
    -K * a.dag() * a.dag() * a * a
    + e2 * (a.dag() ** 2 + a**2)
    + ex * a.dag()
    + np.conjugate(ex) * a
)  # Hamiltonian with Rabi oscillation drive

# Cat_alpha state initialization
state_init = 1 / sqrt(2) * (coherent(N, alpha) + coherent(N, -alpha))


times = np.linspace(0, 0.8 * 10 ** (-6), 800)  # time scale of evolution

opt = Options(nsteps=3000)

time_ev_drive = mesolve(H_drive, state_init, times, options=opt)


fig, ax = plt.subplots()


# make animation frames
# for every time of the evolution, plot the corresponding state
def update(frame):
    state = time_ev_drive.states[frame]
    gridspace = np.linspace(-5, 5, 100)
    W = wigner(state, gridspace, gridspace)  # make Wigner function
    ax.clear()
    ax.contourf(
        gridspace, gridspace, W, levels=100, cmap="seismic", vmin=-1 / pi, vmax=1 / pi
    )  # make the colorscale, 100 levels form -1/pi to 1/pi
    # plot appearance
    ax.set_title("Wigner function at t = {:.1e}".format(times[frame]))
    ax.set_xlabel("Position")
    ax.set_ylabel("Momentum")
    ax.set_aspect("equal")


# animation initialization
ani = FuncAnimation(fig, update, frames=len(times), repeat=False, interval=10)
plt.show()
