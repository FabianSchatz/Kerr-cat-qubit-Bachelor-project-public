import numpy as np
import xarray as xr
import hvplot.xarray
import panel as pn
from joblib import load
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Load the fitted function
# function is dumped in file: "Documentation: Negativity in terms of K*Q"
Negativity_KQ_small_range = load("Negativity(KQ) 100 values newest")


# Negativity is only plotted for values of K*Q bigger than 7*1e9 Hz, but we need to set Negativity for smaller values (it is negligible so we set it 0)
def Negativity_KQ(KQ):
    if KQ > 7 * 1e9:
        return Negativity_KQ_small_range(KQ)
    else:
        return 0


# define axes
K = np.linspace(2.5 * 1e3, 1e6, 100)
Q = np.linspace(5 * 1e4, 2 * 1e6, 100)

# every point on the K, Q grid gives as a value K*Q, with that we compute the Negativity with the above created function
Neg_values = [
    [Negativity_KQ(K[i] * Q[j]) for i in range(len(K))] for j in range(len(Q))
]

# create a contour plot (height lines)
# we divide K and Q by 1e5 because we give Q in 10°5 and K in 100 kHz
# levels define the Negativity of the lines of equal Negativity, "what K and Q we need to reach a certain Negativity value"
contour_plot = plt.contour(
    K / 1e5,
    Q / 1e5,
    Neg_values,
    levels=np.array([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]),
    cmap="viridis",
)

# making the labels for the legend
legend_labels = [
    "Neg=0.025",
    "Neg=0.05",
    "Neg=0.075",
    "Neg=0.1",
    "Neg=0.125",
    "Neg=0.15",
    "Neg=0.175",
    "Neg=0.2",
]

# making the color lines for each label (legend_handle), by getting the color from the height lines with get_edgecolor()
legend_colors = [
    contour_plot.collections[i].get_edgecolor() for i in range(len(legend_labels))
]
legend_handles = [
    Line2D([0], [0], linestyle="-", color=legend_colors[i])
    for i in range(len(legend_labels))
]

# creating the legend
plt.legend(legend_handles, legend_labels)


# Grid
plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="gray")
plt.tick_params(axis="both", which="both", direction="in", length=6, width=1)
plt.minorticks_on()

plt.xlabel("K in " r"100 kHz")
plt.ylabel("Q in " r"$ 10^5$")
plt.show()
