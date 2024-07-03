import numpy as np
import xarray as xr
import hvplot.xarray
import panel as pn
from joblib import load

# Load the function that gives Negativity in terms of product K*Q
Negativity_KQ = load("Negativity(KQ) 100 values")


# make the function compatible to arrays such that it just acts on every element
def Negativity_KQ_for_arrays(x):
    array = []
    for i in range(len(x)):
        if x[i] > 7 * 1e9:
            array.append(Negativity_KQ(x[i]))
        else:
            array.append(0)
    return array


# Create K values
K_values = np.linspace(2.5 * 1e3, 1e6, 1000)

# Create Q values
Q_values = np.linspace(5 * 1e4, 2 * 1e6, 1000)

# create Negativity values: always fix one K at a time and sweep over Q to create the Negativity values for each pair
Negativity_values = []
for K in K_values:
    Negativity_values_for_K = Negativity_KQ_for_arrays(K * Q_values)
    Negativity_values.append(Negativity_values_for_K)


# Stack the Negativity values into a 2D array in order to use it for function xr.DataArray
Negativity_values_array = np.stack(Negativity_values, axis=0)

# create Data array
Data_array = xr.DataArray(
    data=Negativity_values_array,
    dims=["K", "Q"],
    coords={"K": K_values, "Q": Q_values},
    name="Negativity_values",
)

# Create a slider widget for the parameter
w_parameter = pn.widgets.FloatSlider(name="K", start=2.5 * 1e3, end=1 * 1e6)


# Define the interactive plot for given Q
def Negativity_Q(K):
    index = np.abs(
        K_values - K
    ).argmin()  # finds the index of the value in K_values that is closest to K
    selected_K_data = Data_array.isel(K=index)
    return selected_K_data.hvplot.line(
        x="Q", ylabel="Negativity", width=500, height=300
    )


# Combine the widget and the plot
interactive_plot = pn.interact(Negativity_Q, K=w_parameter)

# show interactive plot
pn.panel(interactive_plot).show()
