import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# Define some constants
# ----------------------------------------------------------------------------------------------------------------------
epsilon_standard = 0.00001
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define some helper functions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.    get_r_squared(x_values, y_values, degree):
#       Get the coefficient of determination, as well as the coefficient(s) of the fitting polynomial.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.
def get_r_squared(x_values, y_values, degree):  # also known as the coefficient of determination
    coeffs = np.polyfit(x_values, y_values, degree)  # Get the coefficients of the polynomial fit to the scatter plot.
    pol = np.poly1d(coeffs)  # Create a poly1d object from the coefficients of the polynomial.
    yhat = pol(x_values)
    ybar = np.sum(y_vals)/len(y_vals)  # The mean of all the kcp values in the scatter plot.
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((y_values - ybar)**2)
    rsquared = ssreg / sstot
    return coeffs, rsquared
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Find a polynomial trend that accurately fits the empirical data
# ======================================================================================================================
# 1.    Load the `kcp_vs_gdd_df` dataframe which contains the columns:
#       ["datetimestamp", "smoothed_cumul_gdd", "daily_trend_kcp"]
# 2.    Create column "delta_kcp" which calculates the difference between consecutive "daily_trend_kcp" entries.
# 3.    Extract the beginning and end flat kcp portions.
# 4.    Find the instances/samples/entries where "delta_kcp" != 0.  This section will look like a parabola.
# 5.    Initialise some variables, and run a while loop that determines the best-fitting polynomial to the empirical
#       data.
# 6.    Generate a domain of x values, `linspaced_gdd`, for which there are no flat portions.
# 7.    For `linspaced_gdd` calculate a corresponding trend line, `predicted_kcp`, according to the best-fitting
#       polynomial parameters.
# ======================================================================================================================
# 1. Load the `kcp_vs_gdd_df` dataframe
kcp_vs_gdd_df = pd.read_excel("./data/kcp_vs_smoothed_cumul_gdd.xlsx", header=0, usecols="B,C")
# 2. Create column "delta_kcp" which calculates the difference between consecutive "daily_trend_kcp" entries.
kcp_vs_gdd_df["delta_kcp"] = kcp_vs_gdd_df["daily_trend_kcp"] - kcp_vs_gdd_df["daily_trend_kcp"].shift(-1)
kcp_vs_gdd_df["delta_kcp"].fillna(method="ffill", inplace=True)

# 3. Extract the beginning and end flat kcp portions.
beginning_flat_kcp = kcp_vs_gdd_df["daily_trend_kcp"].values[0]
end_flat_kcp = kcp_vs_gdd_df["daily_trend_kcp"].values[-1]

# 4. Find the instances/samples/entries where "delta_kcp" != 0.
condition = kcp_vs_gdd_df["delta_kcp"] != 0
good_indices = kcp_vs_gdd_df[condition].index

filtered_df = kcp_vs_gdd_df.iloc[good_indices, [0, 1]]
x_vals, y_vals = filtered_df["smoothed_cumul_gdd"].values, filtered_df["daily_trend_kcp"].values
beginning_gdd = math.ceil(x_vals[0])
end_gdd = math.floor(x_vals[-1])

# 5. Initialise some variables, and run a while loop that determines the best-fitting polynomial
saved_trend_lines = []  # this will store all the various polynomial fits.
fit_coeffs = []  # instantiate an empty list that will contain the coefficients of the np.polyfit executions
fit_statistics = []
epsilon = 999.0  # A dummy initialisation to initiate the while-loop.
highest_order = 2  # Performing a linear fit (y = mx + c), where highest_order = 1, makes no sense for this case study.
polynomial_orders = []
index = 0
while epsilon > epsilon_standard:
    z, fit_statistic = get_r_squared(x_values=x_vals, y_values=y_vals, degree=highest_order)
    fit_coeffs.append(list(z))
    fit_statistics.append(fit_statistic)
    p = np.poly1d(z)  # It is convenient to use poly1d objects when dealing with polynomials
    saved_trend_lines.append(p(x_vals))
    if index == 0:
        epsilon = 999.0
    else:
        epsilon = np.abs(fit_statistics[index] - fit_statistics[index - 1])
    polynomial_orders.append(highest_order)
    index += 1
    highest_order += 1

if len(polynomial_orders) >= 2:
    # We do not need the last elements in `polynomial_orders`, `saved_trend_lines`, `fit_coeffs`, `fit_statistics`.
    # We do not need it because the last fit_statistics differs from the previous one by less than `epsilon_standard`.
    # This means there is a negligible difference between the last 2 polynomial fits.  Hence, remove the last one.
    del polynomial_orders[-1]
    del saved_trend_lines[-1]
    del fit_coeffs[-1]
    del fit_statistics[-1]

print(polynomial_orders)
polynomial_order = polynomial_orders[-1]
trend_line = saved_trend_lines[-1]
fit_coeffs = fit_coeffs[-1]
best_fitting_pol = np.poly1d(fit_coeffs)
fit_statistic = fit_statistics[-1]

# 6. Generate a domain of x values, `linspaced_gdd`, for which there are no flat portions.
num = (end_gdd - beginning_gdd) // 1 + 1
linspaced_gdd = np.linspace(start=beginning_gdd, stop=end_gdd, num=num, endpoint=True, dtype=np.int64)

# 7. For `linspaced_gdd` calculate a corresponding trend line, `predicted_kcp`.
predicted_kcp = best_fitting_pol(linspaced_gdd)
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Generate flat portions for the kcp_vs_gdd curve where kcp, as pointed out by the empirical data, remains constant.
# ----------------------------------------------------------------------------------------------------------------------
# 1.    Propagate a flat kcp portion on the left side (in the direction to the left) until gdd = 0.
#       Start this propagation at the point where `predicted_kcp` starts to dip below `beginning_flat_kcp`.
# 2.    Propagate a flat kcp portion on the right side, in the direction to the right.
#       Start this propagation at the point where `predicted_kcp` starts to dip below `end_flat_kcp`.
# 3.    Stitch all the prepend, linspaced, and append portions together for gdd and kcp.
#       Finally, we end up with 2 lists:  `stitched_gdd` and `stitched_kcp`.
# ----------------------------------------------------------------------------------------------------------------------
# 1. Propagate a flat kcp portion on the left side.
arr = np.where(predicted_kcp < beginning_flat_kcp)[0]
if len(arr) != 0:
    diff = 1
    i = 0
    while diff == 1:
        try:
            diff = arr[i + 1] - arr[i]
            i += 1
            index = arr[i]
        except IndexError:
            index = arr[-1]
            break
    predicted_kcp[:index + 1] = beginning_flat_kcp

# 2. Propagate a flat kcp portion on the right side.
arr = np.where(predicted_kcp < end_flat_kcp)[0]
if len(arr) != 0:
    diff = 1
    i = 0
    while diff == 1:
        try:
            diff = arr[i + 1] - arr[i]
            i += 1
            index = arr[i]
        except IndexError:
            index = arr[0]
            break
    predicted_kcp[index:] = end_flat_kcp

prepend_gdd = np.arange(start=0, stop=beginning_gdd, step=1, dtype=np.int64)
prepend_kcp = [predicted_kcp[0]] * len(prepend_gdd)

append_gdd = np.arange(start=end_gdd + 1, stop=end_gdd + 1 + len(prepend_gdd), step=1, dtype=np.int64)
append_kcp = [predicted_kcp[-1]] * len(append_gdd)

# 3. Stitch all the prepend, linspaced, and append portions together for gdd and kcp.
stitched_gdd = []
stitched_gdd.extend(prepend_gdd)
stitched_gdd.extend(list(linspaced_gdd))
stitched_gdd.extend(append_gdd)

stitched_kcp = []
stitched_kcp.extend(prepend_kcp)
stitched_kcp.extend(list(predicted_kcp))
stitched_kcp.extend(append_kcp)
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Wrap up the results
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.    Make a plot of empirical kcp versus cumulative gdd.
# 2.    Make a plot of stitched kcp versus cumulative gdd.
# 3.    Save the plot to "./figures/kcp_versus_gdd.png"
# 4.    Create a DataFrame where `stitched_gdd` is the index column, and `stitched_kcp` is a data column.
# 5.    Save the newly created dataframe to "./data/fit_of_kcp_versus_cumulative_gdd.xlsx".
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1 & 2.  Make plots of empirical and fitted data.
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x=x_vals, y=y_vals, c="blue", alpha=0.3, edgecolors="black", label="Emperical")
ax.plot(stitched_gdd, stitched_kcp, color="indianred", linewidth=2.5, alpha=1.0, label="Polynomial fit")
ax.grid()
ax.set_xlabel("Smoothed Cumulative GDD")
ax.set_ylabel("Crop Coefficient, $k_{cp}$")
ax.set_title("$k_{cp}$ versus GDD")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_xticks(np.arange(start=0, stop=max(stitched_gdd) + 1, step=200))
plt.legend(loc="best")
plt.tight_layout()

# 3. Save the plot to "./figures/kcp_versus_gdd.png"
plt.savefig("./figures/kcp_versus_gdd.png")

# 4. Create a DataFrame where `stitched_gdd` is the index column, and `stitched_kcp` is a data column.
df = pd.DataFrame(data={"kcp": stitched_kcp}, index=stitched_gdd, copy=True)
df.index.rename("cumulative_gdd", inplace=True)

# 5. Save the newly created dataframe to "./data/fit_of_kcp_versus_cumulative_gdd.xlsx".
df.to_excel("./data/fit_of_kcp_vs_cumulative_gdd.xlsx", sheet_name="golden_delicious",
            float_format="%.7f", header=True, index=True, index_label="cumlative_gdd")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
