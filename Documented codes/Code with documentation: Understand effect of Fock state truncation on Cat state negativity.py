from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi

# in this plot we want to find the maximum negativity for alpha Cat states dependent on alpha
# we can derive a formula (done in my Bachelors thesis) and we can plot the Wigner function in Qutip

N = 30  # Fock state truncation


# evaluate Wigner negativity by plotting the Wigner function
def Negativity_even_cat(alpha0):
    # grid for plotting the Wigner function
    y_gridspace = np.linspace(-10, 10, 500)
    x_gridspace = np.linspace(0, 0, 1)
    # even cat state with correct normalization
    even_cat = (
        1
        / sqrt(2 + 2 * np.exp(-2 * alpha0**2))
        * (coherent(N, alpha0) + coherent(N, -alpha0))
    )
    W_even_cat = wigner(even_cat, x_gridspace, y_gridspace)
    # evaluate Wigner function values on the axis Re(a)=0
    wigner_values = [W_even_cat[i, 0] for i in range(500)]
    # return minimal value (corresponds to maximal Negativity)
    return min(wigner_values)


# plot maximal WWigner negativity value against corresponding alpha
alpha_array = np.linspace(0, 5, 100)
plt.plot(
    alpha_array,
    [-Negativity_even_cat(alpha_array[i]) for i in range(len(alpha_array))],
    label=f"Qutip simulation negativity \n(Fock state truncation at N={N})",
)
# plot appearance
plt.xlabel("alpha")
plt.ylabel("max negativity")
plt.title("Maximum negativity of alpha Cat states")


# calculate Wigner negativity with formula from thesis
def max_negativity(alpha):
    y_max = np.pi / (4 * alpha)
    return (1 / pi) * np.e ** (-2 * y_max**2)


# for this method plot again Negativity against alpha
alpha_array = np.linspace(0, 5, 200)
plt.plot(alpha_array, max_negativity(alpha_array), color="r", label="Exact negativity")


plt.legend()
plt.show()
