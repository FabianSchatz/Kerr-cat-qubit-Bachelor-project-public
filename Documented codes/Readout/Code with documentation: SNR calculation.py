import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt


K = 2 * pi * 3 * 1e5  # Kerr nonlinearity
e2 = 2 * pi * 4 * 1e5  # squeezing drive strength
alpha = sqrt(e2 / K)  # sqrt of photon number of KCQ
g_cr = 2 * pi * 1e5  # readout drive strength

kappa_b = 2 * pi * 1e6  # photon loss rate of the readout cavity

end_time = 10 * 1e-6  # max readout integration time for this scenario
time_steps = 1000  # 1000 points to generate a contiuous graph

t_readout = np.linspace(
    0, end_time, time_steps
)  # create x axis (readout integration time)
t_readout = t_readout[
    1:
]  # formula is not defined for integration time 0, so we remove the first value on the time-axis

# check energy gap condition
print(2 * K * alpha * kappa_b / g_cr**2)  # should be >>1


# compute SNR with the formula derived in my Bachelor's thesis
def SNR_boxcar_formula(g_cr, alpha, kappa_b, time):
    return (
        4
        * sqrt(2 * time / kappa_b)
        * g_cr
        * alpha
        * (1 - 2 / (kappa_b * time) * (1 - exp(-0.5 * kappa_b * time)))
    )


# generate SNR array (y values)
SNR_boxcar = SNR_boxcar_formula(g_cr, alpha, kappa_b, t_readout)

# plot data and plot appearance
plt.plot(t_readout * 1e6, SNR_boxcar)
plt.xlabel("readout integration time in " r"$\mu s$")
plt.ylabel("SNR")
plt.grid()
plt.xlim(0, None)
plt.ylim(0, None)
plt.show()
