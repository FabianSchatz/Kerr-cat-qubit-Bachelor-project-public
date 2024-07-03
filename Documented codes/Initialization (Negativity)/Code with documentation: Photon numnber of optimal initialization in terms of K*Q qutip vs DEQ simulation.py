import numpy as np
from joblib import load
import matplotlib.pyplot as plt

# import functions that give the photon number of the optimal initialization
n_qutip = load(
    "n(KQ) qutip 100 values interpolation test"
)  # for the simulation with qutip
n_calculation = load(
    "n(K*Q) for optimal initialization"
)  # for the calculation with the the DEQ

KQ_array = np.logspace(
    np.log10(7 * 10**9), np.log10(2 * 10**12), 500
)  # a axis of product of K and Q in Hz

# plotting and plot appearance
plt.plot(KQ_array, n_calculation(KQ_array), label="Calculation with DEQ")
plt.plot(KQ_array, n_qutip(KQ_array), label="qutip simulation")
plt.xlabel("K*Q in Hz")
plt.ylabel("photon number of optimal initialization")
plt.xscale("log")
plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="gray")
plt.tick_params(axis="both", which="both", direction="in", length=6, width=1)
plt.minorticks_on()
plt.ylim(0, None)
plt.legend()
plt.show()
