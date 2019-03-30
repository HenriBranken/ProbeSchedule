import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# Define some constants
# ----------------------------------------------------------------------------------------------------------------------
# 1. `n_neighbours` is the width of the Gaussian used in generating the weighted moving average.
# 2. `delta_x` is the step size to be used when generating a list of x values.
# ----------------------------------------------------------------------------------------------------------------------
n_neighbours = 2
delta_x = 1
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define some helper functions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. find_nearest_index(model_array, raw_value):
#    Give the index in the model_array at which the corresponding value is the one that resembles raw_value the best.
# 2. gaussian(x, amp=1, mean=0, sigma=10):
#    A Gaussian series which is to be used as a list of weights.  These weights are used in calculating a moving
#    wieghted average.
# 3. weighted_moving_average(x, y, step_size=1.0, width=10):
#    Generate a smoothed weighted moving average of a raw scatter plot of (x, y) values.  Experiment with the width
#    parameter to avoid overfitting the raw (x, y) data.
# 4. get_r_squared(x_raw, y_raw, x_fit, y_fit):
#    Determine the goodness of the fit (x_fit, y_fit) compared to the raw scatter (x_raw, y_raw).
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def find_nearest_index(model_array, raw_value):
    model_array = np.asarray(model_array)
    idx = (np.abs(model_array - raw_value)).argmin()
    return idx


def gaussian(x, amp=1, mean=0, sigma=10):
    return amp*np.exp(-(x - mean)**2 / (2*sigma**2))


def weighted_moving_average(x, y, step_size=1.0, width=10):
    x_min, x_max = math.floor(min(x)), math.ceil(max(x))
    num = int((x_max - x_min) // step_size + 1)
    bin_coords = np.linspace(start=x_min, stop=x_max, num=num, endpoint=True)
    bin_avgs = np.zeros(len(bin_coords))

    for index in range(len(bin_coords)):
        weights = gaussian(x=x, mean=bin_coords[index], sigma=width)
        bin_avgs[index] = np.average(y, weights=weights)

    return bin_coords, bin_avgs


def get_r_squared(x_raw, y_raw, x_fit, y_fit):
    indices = []
    for x in x_raw:
        indices.append(find_nearest_index(x_fit, x))
    y_proxies = []
    for i in indices:
        y_proxies.append(y_fit[i])
    y_bar = np.mean(y_raw)
    ssreg = np.sum((y_proxies - y_bar)**2)
    sstot = np.sum((y_raw - y_bar)**2)
    return ssreg/sstot
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Fit a "model" to gdd versus kcp
# ======================================================================================================================
# 1. Load the `kcp_vs_gdd_df` dataframe
# 2. Create column "delta_kcp" which calculates the difference between consecutive "daily_trend_kcp" entries.
# 3. Extract the beginning and end flat kcp portions.
# 4. Define x_raw and y_raw.
# 5. Get a fit to x_raw and y_raw.
# 6. Pad x_smoothed with values in the range 0 <= x_val < x_smoothed[0].  This ensures x_smoothed starts at 0.
# 7. Determine the goodness of the fit.
# ======================================================================================================================
# 1. Load the `kcp_vs_gdd_df` dataframe
kcp_vs_gdd_df = pd.read_excel("./data/kcp_vs_smoothed_cumul_gdd.xlsx", header=0, usecols="B,C")

# 2. Create column "delta_kcp" which calculates the difference between consecutive "daily_trend_kcp" entries.
kcp_vs_gdd_df["delta_kcp"] = kcp_vs_gdd_df["daily_trend_kcp"] - kcp_vs_gdd_df["daily_trend_kcp"].shift(-1)
kcp_vs_gdd_df["delta_kcp"].fillna(method="ffill", inplace=True)

# 3. Extract the beginning and end flat kcp portions.
beginning_flat_kcp = kcp_vs_gdd_df["daily_trend_kcp"].values[0]
end_flat_kcp = kcp_vs_gdd_df["daily_trend_kcp"].values[-1]

# 4. Define x_raw and y_raw.
independent_var = kcp_vs_gdd_df["smoothed_cumul_gdd"].values
dependent_var = kcp_vs_gdd_df["daily_trend_kcp"].values

# 5. Get a fit to x_raw and y_raw.
x_smoothed, y_smoothed = weighted_moving_average(x=independent_var, y=dependent_var, step_size=delta_x,
                                                 width=n_neighbours)
x_begin = x_smoothed[0]

# 6. Pad x_smoothed with values in the range 0 <= x_val < x_smoothed[0].  This ensures x_smoothed starts at 0.
x_prepend = np.arange(start=0, stop=x_begin, step=delta_x)
y_prepend = np.array([y_smoothed[0]] * len(x_prepend))

x_smoothed = np.insert(x_smoothed, 0, values=x_prepend)
y_smoothed = np.insert(y_smoothed, 0, values=y_prepend)

# 7. Determine the goodness of the fit.
r_squared = get_r_squared(independent_var, dependent_var, x_smoothed, y_smoothed)
print("The goodness of the fit is: {:.4f}.".format(r_squared))
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Wrap up the results
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Make a plot of empirical kcp versus cumulative gdd.
# 2. Make a plot of smoothed kcp (y_smoothed) versus cumulative gdd (x_smoothed).
# 3. Save the plot to "./figures/kcp_versus_gdd.png"
# 4. Create a DataFrame where `x_smoothed` is the index column, and `y_smoothed` is a data column.
# 5. Save the newly created dataframe to "./data/fit_of_kcp_versus_cumulative_gdd.xlsx".
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1 & 2.  Make plots of empirical and fitted data.
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x=independent_var, y=dependent_var, c="blue", alpha=0.3, edgecolors="black", label="Emperical")
ax.plot(x_smoothed, y_smoothed, color="indianred", linewidth=2.5, alpha=1.0, label="Weighted Moving Average")
ax.grid()
ax.set_xlabel("Smoothed Cumulative GDD")
ax.set_ylabel("Crop Coefficient, $k_{cp}$")
ax.set_title("$k_{cp}$ versus GDD")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_xticks(np.arange(start=0, stop=max(x_smoothed) + 1, step=200))
plt.legend(loc="best")
plt.tight_layout()

# 3. Save the plot to "./figures/kcp_versus_gdd.png"
plt.savefig("./figures/kcp_versus_gdd_smoothed.png")

# 4. Create a DataFrame where `x_smoothed` is the index column, and `y_smoothed` is a data column.
df = pd.DataFrame(data={"kcp": y_smoothed}, index=x_smoothed, copy=True)
df.index.rename("cumulative_gdd", inplace=True)

# 5. Save the newly created dataframe to "./data/fit_of_kcp_versus_cumulative_gdd.xlsx".
df.to_excel("./data/fit_of_kcp_vs_cumulative_gdd.xlsx", sheet_name="golden_delicious",
            float_format="%.7f", header=True, index=True, index_label="cumulative_gdd")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
