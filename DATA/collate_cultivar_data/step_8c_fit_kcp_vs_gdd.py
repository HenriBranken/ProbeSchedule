import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import helper_functions as h

# -----------------------------------------------------------------------------
# Define some constants
# -----------------------------------------------------------------------------
# 1. `delta_x` is the step size to be used when generating a list of x values.
# -----------------------------------------------------------------------------
delta_x = 1
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define some helper functions
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. interpolation_1d
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interpolation_1d(x, y, step_size=1.0):
    x_min, x_max = math.floor(x[0]), math.ceil(x[-1])
    num = int((x_max - x_min)/step_size + 1)
    x_linspaced = np.linspace(start=x_min, stop=x_max, num=num, endpoint=True)
    y_fit = np.interp(x_linspaced, x, y)
    return x_linspaced, y_fit


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Fit a "model" to gdd versus kcp
# =============================================================================
# 1. Load the `kcp_vs_gdd_df` dataframe
# 2. Create column "delta_kcp" which calculates the difference between
#    consecutive "daily_trend_kcp" entries.
# 3. Extract the beginning and end flat kcp portions.
# 4. Define x_raw and y_raw.
# 5. Get a fit to x_raw and y_raw.
# 6. Pad x_smoothed with values in the range 0 <= x_val < x_smoothed[0].
#    This ensures x_smoothed starts at 0.
# 7. Determine the goodness of the fit.
# =============================================================================
# 1. Load the `kcp_vs_gdd_df` dataframe
kcp_vs_gdd_df = pd.read_excel("./data/kcp_vs_smoothed_cumul_gdd.xlsx",
                              header=0, usecols="B,C")

# 3. Extract the beginning flat portion value.
beginning_flat_kcp = kcp_vs_gdd_df["daily_trend_kcp"].values[0]

# 4. Define x_raw and y_raw.
independent_var = kcp_vs_gdd_df["smoothed_cumul_gdd"].values
dependent_var = kcp_vs_gdd_df["daily_trend_kcp"].values

# 5. Get a fit to x_raw and y_raw.
x_smoothed, y_smoothed = interpolation_1d(x=independent_var, y=dependent_var,
                                          step_size=delta_x)
x_begin = x_smoothed[0]

# 6. Pad x_smoothed with values in the range 0 <= x_val < x_smoothed[0].
# This ensures x_smoothed starts at 0.
x_prepend = np.arange(start=0, stop=x_begin, step=delta_x)
y_prepend = np.array([y_smoothed[0]] * len(x_prepend))

x_smoothed = np.insert(x_smoothed, 0, values=x_prepend)
y_smoothed = np.insert(y_smoothed, 0, values=y_prepend)

# 7. Determine the goodness of the fit.
r_squared = h.get_r_squared(x_raw=independent_var, y_raw=dependent_var,
                            x_fit=x_smoothed, y_fit=y_smoothed)
print("The goodness of the fit is: {:.3f}.".format(r_squared))
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Wrap up the results
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Make a plot of empirical kcp versus cumulative gdd.
# 2. Make a plot of smoothed kcp versus cumulative gdd.
# 3. Save the plot to "./figures/kcp_versus_gdd.png"
# 4. Create a DataFrame where `x_smoothed` is the index column, and
#    `y_smoothed` is a data column.
# 5. Save the newly created dataframe to
#    "./data/fit_of_kcp_versus_cumulative_gdd.xlsx".
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1 & 2.  Make plots of empirical and fitted data.
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x=independent_var, y=dependent_var, c="blue", alpha=0.3,
           edgecolors="black", label="Emperical")
ax.plot(x_smoothed, y_smoothed, color="indianred", linewidth=2.5, alpha=1.0,
        label="1D Interpolation")
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

# 4. Create a DataFrame where `x_smoothed` is the index column, and
#    `y_smoothed` is a data column.
df = pd.DataFrame(data={"kcp": y_smoothed}, index=x_smoothed, copy=True)
df.index.rename("cumulative_gdd", inplace=True)

# 5. Save the newly created dataframe to
#    "./data/fit_of_kcp_versus_cumulative_gdd.xlsx".
df.to_excel("./data/fit_of_kcp_vs_cumulative_gdd.xlsx",
            sheet_name="golden_delicious", float_format="%.7f", header=True,
            index=True, index_label="cumulative_gdd")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
