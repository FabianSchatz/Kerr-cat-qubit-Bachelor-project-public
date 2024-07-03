import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp
from qutip import *

# parameters
NUMBER_OF_TRAJECTORIES = 10  # number of trajectories for the monte carlo solver, for high enough number of trajectories we get the same result as for the master equation solver
# because we use two system in the tensor product, master equation is too computational intensive and we use monte carlo solver
N = 20  # fock state truncation
K = 2 * pi * 3 * 1e5  # Kerr nonlinearity
e2 = 2 * pi * 4 * 1e5  # squeezing drive strength
omega = 2 * pi * 5 * 1e9  # resonance frequency
Q = 1 * 1e5  # quality factor
alpha = sqrt(e2 / K)  # sqrt of photon number


g_cr = 2 * pi * 1 * 1e5  # readout drive strength
kappab = 2 * pi * 1e6  # photon loss rate of readout cavity

# operators
a = destroy(N)  # destruction operator
I = identity(N)  # identity operator
x = tensor(I, (a + a.dag()) / 2)  # x operator of readout cavity
y = tensor(I, 1j * (a - a.dag()))  # y operator of readout cavity


# Hamiltonian
H_stabilization = tensor(
    -K * a.dag() * a.dag() * a * a + e2 * (a.dag() ** 2 + a**2), identity(N)
)  # Hamiltonian of KCQ
H_readout = (
    1j * g_cr * (tensor(a, a.dag()) - tensor(a.dag(), a))
)  # Readout drive Hamiltonian

H_total = H_stabilization + H_readout  # complete Hamiltonian

rho_0 = tensor(coherent(N, alpha), fock(N, 0))  # initial state
end_time = 4 * 1e-6  # end time
n_time_steps = 100  # number of time steps
times = np.linspace(0, end_time, n_time_steps)  # time scale for monte carlo solver

# monte carlo solver returns for a^dagger+a of the reaout system, the signal we want to integrate
stoc_solution_x = mcsolve(
    H_total,
    rho_0,
    times,
    c_ops=[tensor(I, sqrt(kappab) * a)],
    e_ops=[2 * x],
    ntraj=NUMBER_OF_TRAJECTORIES,
)

dt = end_time / n_time_steps  # time step dt

# signal is composed from stochastic part a_in and a from readout cavity (theory in detail in Bachelor's thesis)
dB = np.sqrt(dt) * (np.random.randn(n_time_steps)) / sqrt(2)
# devide by sqrt(2), because dB in our process is a complex (two-dim) random normal distribution, but we only care about the real part of a_in, so we divide by sqrt(2)
a_in_x = dB / dt
readout_2x = stoc_solution_x.expect[0]  # a+a^dagger from readout cavity
# signal is a_out+a_out^dagger
# in the following line signal is integrated over a time dt
signal = (
    sqrt(kappab) * readout_2x + 2 / sqrt(2) * a_in_x
) * dt  # list with all the integrated signals a_out from t to t+dt ([integrated signal from 0 to dt, integrated signal from dt to 2dt, ...])
integrated_signal = sqrt(kappab) * np.cumsum(
    signal
)  # cumulative sum gives integral of the signal from 0 until that time

plt.plot(
    times * 1e6, integrated_signal
)  # plot the integrated signal (measurement outcome) after time t vs t
plt.xlabel(
    "time in microseconds"
)  # time in microseconds, that's why times*1e6 in line above
plt.ylabel("Integrated signal")
plt.show()  # show one example path


# make the whole process of creating a path a function, so we can repeat it multiple times
def end_signal():
    dB_x = np.sqrt(dt) * (
        np.random.randn(n_time_steps) / sqrt(2)
    )  # we know that dB*dB^dagger =dt, that's where sqrt(dt) comes from, essential in time step dt variance increase of dt is acquired
    # np.random.randn(n_time_steps) creates a 1-dim array of values, where each value it taken from a normal distribution with standard deviation=variance=1
    # devide by sqrt(2), because dB in our process is a complex (two-dim) random normal distribution, but we only care about the real part of a_in, so we divide by sqrt(2)
    a_in_x = dB_x / dt
    readout_2x = stoc_solution_x.expect[0]  # a+a^dagger from readout cavity
    # signal is a_out+a_out^dagger
    # in the following line signal is integrated over a time dt

    signal = (
        sqrt(kappab) * readout_2x + 2 / sqrt(2) * a_in_x
    ) * dt  # list with all the integrated signals a_out from t to t+dt ([integrated signal from 0 to dt, integrated signal from dt to 2dt, ...])
    # 2*a_in_x because we integrate a_out +a_out^dagger = "a_out_x*2" noise
    # divide by sqrt(2) because we model quantum noise with classical (for quantum noise increments we have variance=(dB+dB^dagger)(dB+dB^dagger)=dB*dB^dagger+...=dt) and if we model dB=1/sqrt2(dW+idW) where dW and dB have same distribution than variance=(dB+dB^dagger)(dB+dB^dagger)=2dWdW=2dt, to correct that we include this factor of sqrt(2)
    integrated_signal = sqrt(kappab) * np.cumsum(
        signal
    )  # cumulative sum gives integral of the signal from 0 until that time
    return integrated_signal[-1]  # return the last value in the list which gives the


signal_array = [end_signal() for i in range(10000)]
mu = np.mean(signal_array)
print(mu)
SNR = 3 / sqrt(2)  # calculated SNR with quantum Langevin equation
sigma = mu / SNR

# make histogram of the outcomes
bins = np.arange(mu - 4 * sigma, mu + 4 * sigma, 8 * sigma / 50)
plt.hist(signal_array, bins=bins, density=True, edgecolor="black")
x_array = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

# print expected outcome from the calculation
plt.plot(
    x_array,
    1 / sqrt(2 * pi * sigma**2) * exp(-((x_array - mu) ** 2) / (2 * sigma**2)),
    color="orange",
)
# plot appearance
plt.xlabel("signal")
plt.ylabel("probability")
plt.show()
