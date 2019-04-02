import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
from scipy.signal import argrelextrema

# ----------------------------------------------------------------------------------------------------------------------
# Declare important constants
# ----------------------------------------------------------------------------------------------------------------------
n_neighbours_list = list(np.arange(start=1, stop=25+1, step=1))
delta_x = 1
x_limits = [0, 365]
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the serialised data that were saved in `step_1_perform_cleaning.py`; unpickle the serialised data.
# `data_to_plot` only consists of cleaned (datetime, kcp) samples for each probe.
cleaned_df = pd.read_excel("./data/stacked_cleaned_data_for_overlay.xlsx", header=0, index_col=[0, 1],
                           parse_dates=True)
outer_index = list(cleaned_df.index.get_level_values("probe_id").unique())
inner_index = list(cleaned_df.index.get_level_values("datetimeStamp").unique())
cleaned_df.index = cleaned_df.index.droplevel(0)
cleaned_df.sort_index(axis=0, level="datetimeStamp", ascending=True, inplace=True)

# Generate a list of all the probe-id names.
with open("./data/probe_ids.txt", "r") as f:
    probe_ids = f.readlines()
probe_ids = [x.rstrip() for x in probe_ids]

# Extract the starting year.  The year of the most "historic/past" sample.
with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())

# Instantiate a pandas DataFrame containing the reference crop coefficient values.
cco_df = pd.read_excel("./data/reference_crop_coeff.xlsx", sheet_name=0, header=0, index_col=0,
                       parse_dates=True)
cco_df["days"] = cco_df.index - datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
cco_df["days"] = cco_df["days"].dt.days  # we use the dt.days attribute
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Define some helper functions
# 1. rectify_trend(fitted_trend_values)
# 2. simplify_trend(fitted_trend_values)
# ======================================================================================================================
def rectify_trend(fitted_trend_values):
    """
    Manipulate all data-points from fitted_trend_values that are less than or equal to 0.
    :param fitted_trend_values:  The polynomial trend values fitted to the stacked scatter plot.
    :return:  A trend where all data-points are greater than 0.  The values in the curve that were less than or equal to
    0 were manipulated to be equal to the nearest positive value.  These are flat segments that run parallel to the
    x-axis (and are also typically very close to the x-axis).
    """
    if all(j > 0 for j in fitted_trend_values):
        return fitted_trend_values
    else:
        negative_indices = np.where(fitted_trend_values <= 0)[0]  # identify all the indices where the fit is negative
        diff_arr = np.ediff1d(negative_indices, to_begin=1)
        if all(diff_arr == 1):
            if negative_indices[0] == 0:
                fitted_trend_values[negative_indices] = fitted_trend_values[negative_indices[-1] + 1]
            else:
                fitted_trend_values[negative_indices] = fitted_trend_values[negative_indices[0] - 1]
        else:
            special_index = np.where(diff_arr != 1)[0][0]
            index_where_left_neg_portion_ends = negative_indices[special_index - 1]
            fitted_trend_values[0: index_where_left_neg_portion_ends + 1] = \
                fitted_trend_values[index_where_left_neg_portion_ends + 1]
            index_where_right_neg_portion_starts = negative_indices[special_index]
            fitted_trend_values[index_where_right_neg_portion_starts:] = \
                fitted_trend_values[index_where_right_neg_portion_starts - 1]
        return fitted_trend_values


def find_nearest_index(model_array, raw_value):
    model_array = np.asarray(model_array)
    idx = (np.abs(model_array - raw_value)).argmin()
    return idx


def gaussian(x, amp=1, mean=0, sigma=10):
    return amp*np.exp(-(x - mean)**2 / (2*sigma**2))


def weighted_moving_average(x, y, step_size=1.0, width=10, x_lims=None):
    if x_lims:
        x_min, x_max = x_lims[0], x_lims[1]
    else:
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
    for j in indices:
        y_proxies.append(y_fit[j])
    y_bar = np.mean(y_raw)
    ssreg = np.sum((y_proxies - y_bar)**2)
    sstot = np.sum((y_raw - y_bar)**2)
    return ssreg/sstot


def get_n_local_extrema(y_fit):
    args_loc_minima = argrelextrema(y_fit, np.less)
    num_loc_minima = len(args_loc_minima[0])
    args_loc_maxima = argrelextrema(y_fit, np.greater)
    num_loc_maxima = len(args_loc_maxima[0])
    num_loc_extrema = num_loc_minima + num_loc_maxima
    return num_loc_extrema


def get_prized_index(n_bumps_list):
    some_list = np.where(np.asarray(n_bumps_list) == 1)[0]
    if len(some_list) > 0:
        return some_list[0]
    else:
        print("There is not an index where n_bumps == 1.")
        sys.exit(1)
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Calculate the offset of each date from the beginning of the season
# Convert the offsets, which are of type timedelta, to integer values, representing the number of days
# Sort the DataFrame by the "days" column
# ----------------------------------------------------------------------------------------------------------------------
# Calculate the number_of_days offset for each sample from the beginning of the cultivar Season
cleaned_df["offset"] = cleaned_df.index - datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)

# Convert the time-delta objects of the `offset` column to integer type (therefore representing the number of days)
cleaned_df["days"] = cleaned_df["offset"].dt.days

# Sort the whole DataFrame by the `days` column in ascending order
cleaned_df.sort_values(by="days", axis=0, inplace=True)
# At the bottom of the file, we save cleaned_df to an Excel file.
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Perform polynomial fits to `data_np_array`
# Make plots of `kcp` versus `Days into the Season` of the Polynomial fits
# Here, the x-axis is simply an integer: the offset (in units of days) since the beginning of the season
# ======================================================================================================================
independent_var = cleaned_df["days"].values
dependent_var = cleaned_df["kcp"].values  # the crop coefficient, kcp

saved_trend_lines = []  # this will store all the various trend lines.
r_squared_stats = []
num_bumps = []

for n_neighbours in n_neighbours_list:
    x_smoothed, y_smoothed = weighted_moving_average(x=independent_var, y=dependent_var, step_size=delta_x,
                                                     width=n_neighbours, x_lims=x_limits)
    y_smoothed = rectify_trend(fitted_trend_values=y_smoothed)
    saved_trend_lines.append(zip(x_smoothed, y_smoothed))
    r_squared_stats.append(get_r_squared(x_raw=independent_var, y_raw=dependent_var, x_fit=x_smoothed,
                                         y_fit=y_smoothed))
    num_bumps.append(get_n_local_extrema(y_smoothed))
prized_index = get_prized_index(num_bumps)
trend_line = saved_trend_lines[prized_index]
unpack = [list(t) for t in zip(*trend_line)]
x_smoothed, y_smoothed = unpack[0], unpack[1]

_, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Number of days into the Season")
ax.set_ylabel("$k_{cp}$")
ax.set_title("$k_{cp}$ versus number of days into the season")
ax.grid(True)
ax.set_xlim(left=x_limits[0], right=x_limits[1])
ax.set_ylim(bottom=0.0, top=KCP_MAX)
major_xticks = np.arange(start=x_limits[0], stop=x_limits[1], step=30, dtype=np.int64)
ax.set_xticks(major_xticks)
ax.plot(x_smoothed, y_smoothed, alpha=0.70, label="n_neighbours = {}.".format(n_neighbours_list[prized_index]))
ax.scatter(independent_var, dependent_var, c="magenta", marker=".", edgecolors="black", alpha=0.5,
           label="Cleaned Probe Data")
ax.scatter(cco_df["days"].values, cco_df["cco"].values, c="yellow", marker=".", alpha=0.5, label="Reference $k_{cp}$")
ax.legend()
plt.tight_layout()
plt.savefig("./figures/WMA_kcp_versus_days.png")
plt.cla()
plt.clf()
plt.close()
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create another plot where the x-axis is of type datetime.
# In this new plot we have kcp versus Month of the Year, i.e.: July/01, August/01, ..., June/01, July/01
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
starting_date = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
linspaced_x = list(np.arange(start=x_limits[0], stop=x_limits[1]+1, step=1))
datetime_linspaced = [starting_date + datetime.timedelta(days=float(i)) for i in linspaced_x]

datetime_clouded = [starting_date + datetime.timedelta(days=float(i)) for i in independent_var]

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

ax.plot(datetime_linspaced, y_smoothed, alpha=0.70, label="n_neighbours = {}.".format(n_neighbours_list[prized_index]))

ax.scatter(datetime_clouded, dependent_var, c="magenta", marker=".", edgecolors="black", alpha=0.5,
           label="Cleaned Probe Data")
ax.scatter(cco_df.index, cco_df["cco"].values, c="yellow", marker=".", alpha=0.3, label="Reference $k_{cp}$, \"cco\"")
ax.legend()
fig.autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig("./figures/WMA_kcp_versus_month.png")
plt.cla()
plt.clf()
plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Write the r-squared statistics to file
# Each line in the file corresponds to the highest order used in the polynomial fit, (and the associated R^2 statistic)
# ----------------------------------------------------------------------------------------------------------------------
with open("./data/statistics_trend_lines.txt", "w") as f:
    f.write("n_neighbours | statistic\n")
    for i, nn in enumerate(n_neighbours_list):
        f.write("{:.0f} | {:.7f}\n".format(nn, r_squared_stats[i]))
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Save prized_index
# Save prized number of neighbours (corresponding to prized_index)
# Save the data related to the best-fitting trend line
# ======================================================================================================================
with open("./data/prized_index.txt", "w") as f:
    f.write("{:.0f}".format(prized_index))

with open("./data/prized_n_neighbours.txt", "w") as f:
    f.write("{:.0f}".format(n_neighbours_list[prized_index]))

df = pd.DataFrame(data={"datetimeStamp": datetime_linspaced,
                        "WMA_kcp_trend": y_smoothed},
                  index=datetime_linspaced, columns=["WMA_kcp_trend"], copy=True)
df.to_excel("./data/WMA_kcp_trend_vs_datetime.xlsx", float_format="%.7f", columns=["WMA_kcp_trend"], header=True,
            index=True, index_label="datetimeStamp")

# Save the (sorted) data of the cleaned kcp data to an Excel file
cleaned_df.to_excel("data/kcp_vs_days.xlsx", sheet_name="sheet_1", header=True, index=True, index_label=True,
                    columns=["days", "kcp"], float_format="%.7f")
# ======================================================================================================================
