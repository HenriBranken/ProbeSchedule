import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
import datetime
import pandas as pd
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
from scipy.signal import argrelextrema

# ----------------------------------------------------------------------------------------------------------------------
# Declare important constants
# ----------------------------------------------------------------------------------------------------------------------
epsilon_standard = 0.001
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the serialised data that were saved in `main.py`, and unpickle it.
# `data_to_plot` only consists of cleaned (datetime, kcp) samples for each probe.
with open("data/data_to_plot", "rb") as f:
    data_to_plot = pickle.load(f)

with open("./data/probe_ids.txt", "r") as f:
    probe_ids = f.readlines()
probe_ids = [x.rstrip() for x in probe_ids]

cco_df = pd.read_excel("./data/reference_crop_coeff.xlsx", sheet_name=0, header=0, index_col=0,
                       parse_dates=True)

with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Define helper functions
#   1.  rectify_trend(fitted_trend_values)
#   2.  simplify_trend(fitted_trend_values)
# ======================================================================================================================
def rectify_trend(fitted_trend_values):
    """
    Manipulate all data-points from fitted_trend_values that are less than or equal to 0.
    :param fitted_trend_values:  The polynomial trend values fitted to the stacked scatter plot.
    :return:  A trend where all data-points are greater than 0.  The values in the curve that were less than or equal to
    0 were manipulated to be equal to the nearest positive value.  These are flat segments that run parallel to the
    x-axis (and are also typically very close to the x-axis).
    """
    if all(ii > 0 for ii in fitted_trend_values):
        return fitted_trend_values
    else:
        negative_indices = np.where(fitted_trend_values <= 0)[0]  # identify all the "negative" indices
        last_positive_index = min(negative_indices) - 1  # identify the last possible index having a positive value
        first_positive_index = max(negative_indices) + 1
        if last_positive_index < 0:
            fitted_trend_values[negative_indices] = fitted_trend_values[first_positive_index]
        else:
            fitted_trend_values[negative_indices] = fitted_trend_values[last_positive_index]
        return fitted_trend_values


def simplify_trend(fitted_trend_values):
    """
    We only want the polynomial trend to have 1 extreme point: the local maximum.  Local minima are not allowed.
    Therefore, when moving away from the local maximum (in either the left or right direction), if a local minimum is
    found that "wings up" as you further move away from the local maximum, then points beyond that local minimum are
    manipulated to remain constant, and thus run parallel to the x-axis.  These can be easily identified from the graph
    as flat portions at the left and/or right edges running parallel to the x-axis.
    :param fitted_trend_values:  The polynomial trend values fitted to the stacked scatter plot.  These trend values
    have also already been processed once by the `rectify_trend` function.
    :return:  Return a trend line where portions "winging up" from local minima have been flattened out.  Therefore we
    return a simplified version of the polynomial fit where oscillations have been flattened out.
    """
    loc_maxima_index = argrelextrema(fitted_trend_values, np.greater)[0]
    loc_minima_indices = argrelextrema(fitted_trend_values, np.less)[0]
    if len(loc_minima_indices) >= 1:
        number_of_minima = len(loc_minima_indices)
        for j in range(number_of_minima):
            minima_index = loc_minima_indices[j]
            if minima_index < loc_maxima_index[0]:
                fitted_trend_values[0: minima_index] = fitted_trend_values[minima_index]
            else:
                fitted_trend_values[minima_index:] = fitted_trend_values[minima_index]
        return fitted_trend_values
    else:
        return fitted_trend_values


def get_r_squared(x_vals, y_vals, degree):  # also known as the coefficient of determination
    """
    The coefficient of determination (denoted as R^2) is a key output of regression analysis.  It is interpreted as the
    proportion of the variance in the dependent variable that is predictable from the independent variable.  An R^2 of 0
    means that the dependent variable cannot be predicted from the independent variable.  An R^2 of 1 means that the
    dependent variable can be predicted without error from the independent variable.  An R^2 between 0 and 1 indicates
    the extent to which the dependent variable is predictable.  An R^2 of 0.10 means that 10 percent of the variance in
    Y is predictable from X, and so on...
    :param x_vals:  x represents the number of days into the Season.  Equivalent to n.
    :param y_vals:  y represents the calculated kcp values.
    :param degree:  The highest order in the polynomial.
    :return:  Return the coefficient of determination.  In this case study we expect it to be in the range (0; 1).  The
    greater the R^2 value, the better the fit to the scatter plot.
    """
    coeffs = np.polyfit(x_vals, y_vals, degree)  # Get the coefficients of the polynomial fit to the scatter plot.
    pol = np.poly1d(coeffs)  # Create a poly1d object from the coefficients of the polynomial.
    yhat = pol(x_vals)  # The 'fitted/predicted' kcp values.
    ybar = np.sum(y_vals)/len(y_vals)  # The mean of all the kcp values in the scatter plot.
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((y - ybar)**2)
    rsquared = ssreg / sstot
    return coeffs, rsquared
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Stack all the data from all the different probes
# Sort the data (ascending) according to the datetime column
# Save the sorted data in an Excel file
# Also store the numerical data columns `days` and `kcp` in a numpy array and save to file
# ----------------------------------------------------------------------------------------------------------------------
# Get the starting year
sample_dates = data_to_plot[0][0]
sample_dates.sort()

# Stack all the data from all the different probes by using a for-loop
concatenated_data = np.empty(shape=(1, 2))  # initialise a placeholder to which data values will be stacked
for i in range(len(data_to_plot)):
    date_arr = np.reshape(np.array(data_to_plot[i][0]), newshape=(-1, 1))
    kcp_arr = np.reshape(np.array(data_to_plot[i][1]), newshape=(-1, 1))
    arr = np.hstack((date_arr, kcp_arr))
    concatenated_data = np.vstack((concatenated_data, arr))
concatenated_data = np.delete(concatenated_data, 0, axis=0)  # delete the first dummy row

# Initialise a pandas DataFrame called `df`
df = pd.DataFrame(index=concatenated_data[:, 0], data=concatenated_data[:, 1], columns=["kcp"])
df.index.name = "datetimeStamp"

# Get the starting month for the cultivar Season
df.sort_index(inplace=True)
starting_month = df.index[0].month

# Calculate the date offset for each sample from the beginning of the cultivar Season
df["offset"] = df.index - datetime.datetime(year=starting_year, month=starting_month, day=1)

# Convert the time-delta objects of the `offset` column to integer type (therefore representing the number of days)
df["days"] = df["offset"].dt.days

# Sort the whole DataFrame by the `days` column in ascending order
df.sort_values(by="days", axis=0, inplace=True)
df.to_excel("data/kcp_vs_days.xlsx", sheet_name="sheet_1", header=True, index=True, index_label=True,
            columns=["days", "kcp"])

# Store `days` and `kcp` columns in numpy array (of type float).
# We want to use this array later for polynomial fitting.
days = np.reshape(df.loc[:, "days"].values, newshape=(-1, 1)).astype(dtype=float)  # float conversion for pol. fitting.
kcp = np.reshape(df.loc[:, "kcp"].values, newshape=(-1, 1))
data_np_array = np.hstack((days, kcp))
np.save("data/kcp_vs_days_data", data_np_array)
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Perform polynomial fits to `data_np_array`
# Make plots of `kcp` versus `Days into the Season` of the Polynomial fits
# Here, the x-axis is simply an integer: the offset (in units of days) since the beginning of the season
# ======================================================================================================================
x = np.array(data_np_array[:, 0], dtype=float)
y = np.array(data_np_array[:, 1], dtype=float)

saved_trend_lines = []  # this will store all the various polynomial fits.
# Set the plotting meta-data, etc...
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("$n$ Days into the Season")
ax.set_ylabel("$k_{cp}$")
ax.set_title("$k_{cp}$ versus $n$ days")
loc = plticker.MultipleLocator(base=30)  # The spacing between the major-gridlines must be 30 (calendar days)
ax.xaxis.set_major_locator(loc)
ax.grid(True)
ax.set_xlim(left=0, right=366)
ax.set_ylim(bottom=0, top=KCP_MAX)
ax.scatter(x, y, c="magenta", marker=".", edgecolors="black", alpha=0.5, label="Cleaned Probe Data")

fit_coeffs = []  # instantiate an empty list that will contain the coefficients of the np.polyfit executions
linspaced_x = np.arange(start=0, stop=366, step=1)  # linspaced_x = [0, 1, 2, ..., 365].  This represents number of days
fit_statistics = []

epsilon = 999.0  # A dummy initialisation to initiate the while-loop.
highest_order = 2  # Performing a linear fit (y = mx + c), where highest_order = 1, makes no sense for this case study.
polynomial_orders = []
index = 0
while epsilon > epsilon_standard:
    z, fit_statistic = get_r_squared(x_vals=x, y_vals=y, degree=highest_order)
    fit_coeffs.append(list(z))
    fit_statistics.append(fit_statistic)
    p = np.poly1d(z)  # It is convenient to use poly1d objects when dealing with polynomials
    trend_values = rectify_trend(p(linspaced_x))
    trend_values = simplify_trend(trend_values)
    saved_trend_lines.append(trend_values)
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

for i in range(len(polynomial_orders)):
    highest_order = polynomial_orders[i]
    ax.plot(linspaced_x, saved_trend_lines[i], label="Order-{} Polynomial Fit".format(highest_order), alpha=0.55)
ax.legend()
plt.tight_layout()
plt.savefig("figures/fit_kcp_versus_days.png")
plt.close()
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create another plot where the x-axis is of type datetime.
# In this new plot we have kcp versus Month of the Year, i.e.: July/01, August/01, ..., June/01
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
starting_date = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
datetime_linspaced = []  # these dates will be used to plot the trend as inferred from the polynomial
for i in linspaced_x:
    datetime_linspaced.append(starting_date + datetime.timedelta(days=float(i)))

datetime_clouded = []  # These correspond to the scatter plot.  There may be some samples for which the date overlaps.
for i in x:  # x is defined in the polynomial fits block.  Represents number of days into the season.
    datetime_clouded.append(starting_date + datetime.timedelta(days=float(i)))

fig, ax = plt.subplots(figsize=(10, 5))
major_xticks = pd.date_range(start=datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1),
                             end=datetime.datetime(year=starting_year + 1, month=BEGINNING_MONTH, day=1), freq="MS")
ax.set_xticks(major_xticks)
ax.set_xlabel("Date (Month of the Year)")
ax.set_ylabel("$k_{cp}$")
ax.set_title("$k_{cp}$ versus Month of the Year")
ax.grid(True)
ax.set_xlim(left=datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1),
            right=datetime.datetime(year=starting_year+1, month=BEGINNING_MONTH, day=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))  # Month/01
ax.set_ylim(bottom=0.0, top=KCP_MAX)
for i in range(len(saved_trend_lines)):
    trend_line = saved_trend_lines[i]
    ax.plot(datetime_linspaced, trend_line, alpha=0.55, label="Order-{} Polynomial fit".format(polynomial_orders[i]))
ax.scatter(datetime_clouded, y, c="magenta", marker=".", edgecolors="black", alpha=0.5, label="Cleaned Probe Data")
ax.scatter(cco_df.index, cco_df["cco"].values, c="red", marker=".", alpha=0.5, label="Reference $k_{cp}$")
ax.legend()
fig.autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig("figures/fit_kcp_versus_month.png")
plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Write the r-squared statistics to file
# Each line in the file corresponds to the highest order used in the polynomial fit, (and the associated R^2 statistic)
# ----------------------------------------------------------------------------------------------------------------------
with open("./data/statistics_trend_lines.txt", "w") as f:
    f.write("highest_polynomial_order | statistic\n")
    for i, p in enumerate(polynomial_orders):
        f.write("{:.0f} | {:.8f}\n".format(p, fit_statistics[i]))
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Extract the trend line that generates the greatest r-squared (coefficient of determination, R^2).
# Save the data related to the best-fitting trend line.
# ======================================================================================================================
prized_index = np.where(fit_statistics == max(fit_statistics))[0][0]

datetimestamp = np.reshape(datetime_linspaced, newshape=(-1, 1))
kcp = np.reshape(saved_trend_lines[prized_index], newshape=(-1, 1))
np.save("data/daily_trend_of_kcp_vs_datetime", np.hstack((datetimestamp, kcp)))
# ======================================================================================================================
