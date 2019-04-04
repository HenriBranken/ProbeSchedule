import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import sys
import math
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
import datetime
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
pd.options.mode.chained_assignment = None  # default='warn'


# ======================================================================================================================
# Define some other important constants
# ======================================================================================================================
n_neighbours_list = list(np.arange(start=1, stop=25+1, step=1))
delta_x = 1
x_limits = [0, 365]
quantile = 0.97
n_iterations = 5
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Import the cleaned (scatterplot) data of kcp versus datetimeStamp.
# 2. Import the (first/beginning) kcp vs datetimestamp trend generated by a WMA.
# 3.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Extract the starting year.  The year of the most "historic/past" sample.
with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())
start_date = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)

# 1.
# Load the cleaned (scatterplot) data of kcp versus datetimeStamp from `./data/stacked_cleaned_data_for_overlay.xlsx`
# that was saved in `step_1_perform_cleaning.py`.
# We are interested in the `datetimeStamp` index, and the newly created `days` column.
cleaned_df = pd.read_excel("./data/stacked_cleaned_data_for_overlay.xlsx", header=0, index_col=[0, 1],
                           parse_dates=True)
outer_index = list(cleaned_df.index.get_level_values("probe_id").unique())
inner_index = list(cleaned_df.index.get_level_values("datetimeStamp").unique())
cleaned_df.index = cleaned_df.index.droplevel(0)
cleaned_df.sort_index(axis=0, level="datetimeStamp", ascending=True, inplace=True)  # sort in ascending order.
cleaned_df["days"] = cleaned_df.index - start_date
cleaned_df["days"] = cleaned_df["days"].dt.days
x_scatter = cleaned_df["days"].values
y_scatter = cleaned_df["kcp"].values

# 2.
# Import the (first/beginning) kcp vs datetimestamp trend generated by a WMA.  This was done in
# `step_3_weighted_moving_average.py`.  We need to import the sheet `day_frequency` from `binned_kcp_data.xlsx`.
# We are interested in the `datetimestamp` index, and the `day_averaged_kcp` column.
WMA_kcp_vs_date_df = pd.read_excel("./data/binned_kcp_data.xlsx", sheet_name="day_frequency", header=0,
                                   index_col=0, parse_dates=True, squeeze=False)
x_WMA = WMA_kcp_vs_date_df["season_day"].values
y_WMA = WMA_kcp_vs_date_df["day_averaged_kcp"].values

# 3.
# Instantiate a pandas DataFrame containing the reference crop coefficient values.
cco_df = pd.read_excel("./data/reference_crop_coeff.xlsx", sheet_name=0, header=0, index_col=0,
                       parse_dates=True)
cco_df["days"] = cco_df.index - datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
cco_df["days"] = cco_df["days"].dt.days  # we use the dt.days attribute
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Define all the helper functions.
# ----------------------------------------------------------------------------------------------------------------------
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


def get_offsets_from_trendline(x_raw, y_raw, x_fit, y_fit):
    offsets = []
    indices = []
    for x in x_raw:
        indices.append(find_nearest_index(x_fit, x))
    y_proxies = []
    for idx in indices:
        y_proxies.append(y_fit[idx])
    for j in range(len(y_raw)):
        offsets.append(np.abs(y_raw[j] - y_proxies[j]))
    return offsets


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


def create_interim_df(x_raw, y_raw, dev):
    dataframe = pd.DataFrame({"x": x_raw, "y": y_raw, "dev": dev}, copy=True)
    dataframe.index.name = "i"
    dataframe.sort_values(by="dev", axis=0, ascending=False, inplace=True)
    return dataframe


def create_new_generation_xy_scatter(dataframe, threshold):
    dataframe["to_be_removed"] = dataframe["dev"] >= threshold
    dataframe = dataframe[~dataframe["to_be_removed"]]
    dataframe.sort_values(by="x", axis=0, ascending=True, inplace=True)
    return dataframe["x"].values, dataframe["y"].values


def create_xy_df(x_vals, y_vals, iteration, status):
    if status == "scatter":
        dataframe = pd.DataFrame(data={"x_scatter": x_vals, "y_scatter": y_vals, "iteration": iteration}, copy=True)
        dataframe.index.name = "i"
    else:
        dataframe = pd.DataFrame(data={"x_WMA": x_vals, "y_WMA": y_vals, "iteration": iteration}, copy=True)
        dataframe.index.name = "i"
    return dataframe


def get_xs_and_ys(dataframe, iteration, status="scatter"):
    x_var, y_var = "x_" + status, "y_" + status
    sub_df = dataframe.loc[(iteration, ), [x_var, y_var]]
    return sub_df[x_var].values, sub_df[y_var].values
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Initialise the "starting points" corresponding to iteration i = 0.
# ======================================================================================================================
r_squared_stat = get_r_squared(x_raw=x_scatter, y_raw=y_scatter, x_fit=x_WMA, y_fit=y_WMA)
xy_scatter_df = create_xy_df(x_vals=x_scatter, y_vals=y_scatter, iteration=int(0), status="scatter")
xy_WMA_df = create_xy_df(x_vals=x_WMA, y_vals=y_WMA, iteration=int(0), status="WMA")
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data Munging.  In the for-loop we iteratively remove the "farthest" outliers.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
r_squared_stat_list = [r_squared_stat]
xy_scatter_dfs = [xy_scatter_df]
xy_WMA_dfs = [xy_WMA_df]

for i in range(n_iterations):
    deviations = get_offsets_from_trendline(x_raw=x_scatter, y_raw=y_scatter, x_fit=x_WMA, y_fit=y_WMA)
    interim_df = create_interim_df(x_raw=x_scatter, y_raw=y_scatter, dev=deviations)
    values = np.quantile(interim_df["dev"].values, q=quantile)
    x_scatter, y_scatter = create_new_generation_xy_scatter(dataframe=interim_df, threshold=values)
    xy_scatter_dfs.append(create_xy_df(x_vals=x_scatter, y_vals=y_scatter, iteration=int(i+1), status="scatter"))

    saved_trend_lines = []  # this will store all the various trend lines.
    num_bumps = []

    for n_neighbours in n_neighbours_list:
        x_WMA, y_WMA = weighted_moving_average(x=x_scatter, y=y_scatter, step_size=delta_x,
                                               width=n_neighbours, x_lims=x_limits)
        y_WMA = rectify_trend(fitted_trend_values=y_WMA)
        saved_trend_lines.append(zip(x_WMA, y_WMA))
        num_bumps.append(get_n_local_extrema(y_WMA))
    prized_index = get_prized_index(num_bumps)
    trend_line = saved_trend_lines[prized_index]
    unpack = [list(t) for t in zip(*trend_line)]
    x_WMA, y_WMA = unpack[0], unpack[1]
    xy_WMA_dfs.append(create_xy_df(x_vals=x_WMA, y_vals=y_WMA, iteration=int(i+1), status="WMA"))
    r_squared_stat = get_r_squared(x_raw=x_scatter, y_raw=y_scatter, x_fit=x_WMA, y_fit=y_WMA)
    r_squared_stat_list.append(r_squared_stat)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Stack all the dataframes together into a MultiIndex DataFrame
# ----------------------------------------------------------------------------------------------------------------------
xy_scatter_df = pd.concat(xy_scatter_dfs, ignore_index=False, sort=False, copy=True)
xy_scatter_df = xy_scatter_df.set_index(["iteration", xy_scatter_df.index])

xy_WMA_df = pd.concat(xy_WMA_dfs, ignore_index=False, sort=False, copy=True)
xy_WMA_df = xy_WMA_df.set_index(["iteration", xy_WMA_df.index])
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Plot all the results in an array of figures
# ======================================================================================================================
num_cols = 2
num_rows = int(math.ceil(len(r_squared_stat_list) / num_cols))
season_xticks = list(np.arange(start=0, stop=366, step=30))

fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7.07, 10))
axs = axs.flatten()
plt.subplots_adjust(wspace=0.05)
fig.suptitle("$k_{cp}$ versus Number of Days.  Outliers removed in iterative cycles.")
fig.autofmt_xdate()

for i, ax in enumerate(axs):
    ax.set_ylim(bottom=0.0, top=KCP_MAX)
    ax.grid(True)
    x_scatter, y_scatter = get_xs_and_ys(dataframe=xy_scatter_df, iteration=i, status="scatter")
    x_WMA, y_WMA = get_xs_and_ys(dataframe=xy_WMA_df, iteration=i, status="WMA")
    ax.scatter(x_scatter, y_scatter, marker=".", color="magenta", s=20, edgecolors="black", linewidth=1, alpha=0.5,
               label="Scatter plot")
    ax.scatter(cco_df["days"].values, cco_df["cco"].values, marker=".", color="yellow", s=15, alpha=0.5,
               label="Reference $k_{cp}$")
    ax.plot(x_WMA, y_WMA, linewidth=1.5, alpha=0.75, label="WMA trend")
    ax.tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                   labelsize="small", axis="x")
    ax.set_xticks(season_xticks)
    ax.set_xticklabels(season_xticks, rotation=40, ha="right")
    ax.set_xlim(left=0, right=366)
    ax.set_ylim(bottom=0, top=KCP_MAX)
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
    ax.legend(prop={"size": 6}, loc=6)
    ax.annotate(s="$R^2$ = {:.3f}\nIteration {}".format(r_squared_stat_list[i], i), xycoords="axes fraction",
                xy=(0.01, 0.87), fontsize=9)
for k in [0, 2, 4]:
    axs[k].set_ylabel("$k_{cp}$")
for k in [4, 5]:
    axs[k].set_xlabel("Days")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./figures/remove_outliers_henri_method.png")
plt.close()
# ======================================================================================================================
