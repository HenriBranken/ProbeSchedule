import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import math
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
import datetime
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
# import sys
# import matplotlib.dates as mdates


# ======================================================================================================================
# Define some other important constants
# ======================================================================================================================
n_neighbours_list = list(np.arange(start=30, stop=1-1, step=-1))
delta_x = 1
x_limits = [0, 365]
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]
pol_degree = 4  # It is not recommended to go above 4 because then the oscillations start to grow wildly out of control.


class NoProperWMATrend(Exception):
    pass
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Extract the starting year.  The year of the most "historic/past" sample.
with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())
start_date = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)


# Load the cleaned (scatterplot) data of kcp versus datetimeStamp from `./data/stacked_cleaned_data_for_overlay.xlsx`
# that was saved in `step_1_perform_cleaning.py`.
# We are interested in the `datetimeStamp` index, and the newly created `days` column.
probe_set_df = pd.read_excel("./data/stacked_cleaned_data_for_overlay.xlsx", header=0, index_col=[0, 1],
                             parse_dates=True)
outer_index = list(probe_set_df.index.get_level_values("probe_id").unique())
inner_index = list(probe_set_df.index.get_level_values("datetimeStamp").unique())
scatter_df = probe_set_df.copy(deep=True)
scatter_df.index = scatter_df.index.droplevel(0)
scatter_df.sort_index(axis=0, level="datetimeStamp", ascending=True, inplace=True)  # sort in ascending order.
scatter_df["days"] = scatter_df.index - start_date
scatter_df["days"] = scatter_df["days"].dt.days
x_scatter = scatter_df["days"].values
y_scatter = scatter_df["kcp"].values


# Import the (first/beginning) kcp vs datetimestamp smoothed trend from `binned_kcp_data.xlsx`.  We need to import the
# sheet `day_frequency` from `binned_kcp_data.xlsx`.  We are interested in the `season_day` column, and the
# `day_averaged_kcp` column.
smoothed_kcp_vs_date_df = pd.read_excel("./data/binned_kcp_data.xlsx", sheet_name="day_frequency", header=0,
                                        index_col=0, parse_dates=True, squeeze=False)
x_smoothed = smoothed_kcp_vs_date_df["season_day"].values
y_smoothed = smoothed_kcp_vs_date_df["day_averaged_kcp"].values


# Instantiate a pandas DataFrame containing the reference crop coefficient values.
cco_df = pd.read_excel("./data/reference_crop_coeff.xlsx", sheet_name=0, header=0, index_col=0,
                       parse_dates=True)
cco_df["days"] = cco_df.index - datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
cco_df["days"] = cco_df["days"].dt.days  # we use the dt.days attribute


# Load all the probes
with open("../probe_ids.txt", "r") as f2:
    probe_ids = [x.rstrip() for x in f2.readlines()]
n_iterations = int(len(probe_ids) - 1)


# Determine the mode that was used in `step_3_smoothed_version.py`.
with open("./data/mode.txt", "r") as f:
    mode = f.readline().rstrip()  # mode is either "Polynomial-fit" or "WMA"


# ----------------------------------------------------------------------------------------------------------------------
# Define all the helper functions.
# ----------------------------------------------------------------------------------------------------------------------
def rectify_trend(fitted_trend_values):
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
    nearest_index = (np.abs(model_array - raw_value)).argmin()
    return nearest_index


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
    y_better = rectify_trend(fitted_trend_values=bin_avgs)
    return bin_coords, y_better


def get_r_squared(x_raw, y_raw, x_fit, y_fit):
    indices = []
    y_proxies = []  # value of fit line, y_fit, whose x-coord is closest to the x-coord of y_raw.
    for x in x_raw:
        indices.append(find_nearest_index(x_fit, x))
    for j in indices:
        y_proxies.append(y_fit[j])
    # y_bar = np.mean(y_raw)  # only useful for linear regression
    ssres = 0
    for k in range(len(x_raw)):
        ssres += (y_proxies[k] - y_raw[k])**2
    # ssreg = np.sum((y_proxies - y_bar)**2)  # only useful for linear regression
    # sstot = np.sum((y_raw - y_bar)**2)  # only useful for linear regression
    return np.sqrt(ssres/len(y_raw))


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
        return some_list[-1]
    else:
        raise NoProperWMATrend("No index at which the number of extrema is 1.")


def create_xy_df(x_vals, y_vals, iteration, status):
    if status == "scatter":
        dataframe = pd.DataFrame(data={"x_scatter": x_vals, "y_scatter": y_vals, "iteration": iteration}, copy=True)
        dataframe.index.name = "i"
    else:
        dataframe = pd.DataFrame(data={"x_smoothed": x_vals, "y_smoothed": y_vals, "iteration": iteration}, copy=True)
        dataframe.index.name = "i"
    return dataframe


def get_xs_and_ys(dataframe, iteration, status="scatter"):
    x_var, y_var = "x_" + status, "y_" + status
    sub_df = dataframe.loc[(iteration, ), [x_var, y_var]]
    return sub_df[x_var].values, sub_df[y_var].values


def collapse_dataframe(multi_index_df, tbr_probe_list):
    df = multi_index_df.copy(deep=True)
    for pr in tbr_probe_list:
        df.drop(index=pr, level=0, inplace=True)
    df.index = df.index.droplevel(0)
    df.sort_index(axis=0, level="datetimeStamp", ascending=True, inplace=True)  # sort in ascending order.
    df["days"] = df.index - start_date
    df["days"] = df["days"].dt.days
    return df["days"].values, df["kcp"].values


def extract_probe_df(multi_index_df, probe):
    df = multi_index_df.loc[(probe, ), ["kcp"]]
    df["days"] = df.index - start_date
    df["days"] = df["days"].dt.days
    df.sort_values("days", ascending=True, inplace=True)
    return df[["days", "kcp"]]


# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Define polynomial helper functions.
# These helper functions come into play when the Weighted Moving Average approach fails and gives a ZeroDivisionError.
# As a result, we cannot use the WMA approach, but have to perform a polynomial fit.
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def get_final_polynomial_fit(x_raw, y_raw, step_size=1.0, degree=pol_degree, x_lims=None):
    if x_lims:
        x_min, x_max = x_lims[0], x_lims[1]
    else:
        x_min, x_max = math.floor(min(x_raw)), math.ceil(max(x_raw))
    num = int((x_max - x_min) // step_size + 1)
    bin_coords = np.linspace(start=x_min, stop=x_max, num=num, endpoint=True)
    coeffs = np.polyfit(x_raw, y_raw, degree)  # Get the coefficients of the polynomial fit to the scatter plot.
    pol = np.poly1d(coeffs)  # Create a poly1d object from the coefficients of the polynomial.
    bin_avgs = pol(bin_coords)
    y_better = rectify_trend(fitted_trend_values=bin_avgs)
    y_best = simplify_trend(fitted_trend_values=y_better)
    return bin_coords, y_best


def simplify_trend(fitted_trend_values):
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
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


# ======================================================================================================================
# Initialise the "starting points" corresponding to iteration i = 0.
# ======================================================================================================================
r_squared_stat = get_r_squared(x_raw=x_scatter, y_raw=y_scatter, x_fit=x_smoothed, y_fit=y_smoothed)
xy_scatter_df = create_xy_df(x_vals=x_scatter, y_vals=y_scatter, iteration=int(0), status="scatter")
xy_smoothed_df = create_xy_df(x_vals=x_smoothed, y_vals=y_smoothed, iteration=int(0), status="smoothed")
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data Munging.  In the for-loop we iteratively remove "bad" probes
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
r_squared_stat_list = [r_squared_stat]  # how good the trend fits the scatter plot
xy_scatter_dfs = [xy_scatter_df]
xy_smoothed_dfs = [xy_smoothed_df]
removed_probes = []

# assert that the number of flagging iterations is less than the number of probes.
assert len(probe_ids) > n_iterations, "Number of iterations is greater than the number of probes.\n" \
                                      "Decrease n_iterations such that n_iterations < len(probe_ids)."

q = 0
for i in range(n_iterations):
    try:
        probes_r_squared = []
        for p in probe_ids:
            probe_df = extract_probe_df(multi_index_df=probe_set_df, probe=p)
            val = get_r_squared(x_raw=probe_df["days"].values, y_raw=probe_df["kcp"].values,
                                x_fit=x_smoothed, y_fit=y_smoothed)
            probes_r_squared.append(val)
        max_arg_index = np.where(probes_r_squared == max(probes_r_squared))[0][0]
        removed_probes.append(probe_ids.pop(max_arg_index))

        x_scatter, y_scatter = collapse_dataframe(multi_index_df=probe_set_df, tbr_probe_list=removed_probes)
        xy_scatter_dfs.append(create_xy_df(x_vals=x_scatter, y_vals=y_scatter, iteration=int(i + 1), status="scatter"))

        if mode == "WMA":
            saved_trend_lines = []  # this will store all the various trend lines associated with different n_neighbours
            num_bumps = []
            tracker = 0
            for n_neighbours in n_neighbours_list:
                try:
                    x_smoothed, y_smoothed = weighted_moving_average(x=x_scatter, y=y_scatter, step_size=delta_x,
                                                                     width=n_neighbours, x_lims=x_limits)
                    saved_trend_lines.append(zip(x_smoothed, y_smoothed))
                    num_bumps.append(get_n_local_extrema(y_smoothed))
                    tracker += 1
                except ZeroDivisionError:
                    n_neighbours_list = n_neighbours_list[:tracker]
                    break  # exit the for-loop.  We have reached the point where n_neighbours is too small.
            try:
                prized_index = get_prized_index(num_bumps)
                trend_line = saved_trend_lines[prized_index]
                unpack = [list(t) for t in zip(*trend_line)]
                x_smoothed, y_smoothed = unpack[0], unpack[1]
            except NoProperWMATrend as e:
                print(e)
                print("{:.>80}".format("Cannot perform Exponentially-Weighted-Moving-Average."))
                print("{:.>80}".format("Proceeding with Polynomial Fit."))
                mode = "Polynomial-fit"  # Switch over to "Polynomial-fit" mode.  No longer using "WMA" mode.
                x_smoothed, y_smoothed = get_final_polynomial_fit(x_raw=x_scatter, y_raw=y_scatter, step_size=delta_x,
                                                                  degree=pol_degree, x_lims=x_limits)
        else:  # i.e. we know for sure that mode is "Polynomial-fit".
            x_smoothed, y_smoothed = get_final_polynomial_fit(x_raw=x_scatter, y_raw=y_scatter, step_size=delta_x,
                                                              degree=pol_degree, x_lims=x_limits)

        xy_smoothed_dfs.append(create_xy_df(x_vals=x_smoothed, y_vals=y_smoothed, iteration=int(i + 1),
                                            status="smoothed"))
        r_squared_stat = get_r_squared(x_raw=x_scatter, y_raw=y_scatter, x_fit=x_smoothed, y_fit=y_smoothed)
        r_squared_stat_list.append(r_squared_stat)
        q += 1
    except IndexError:
        print("Had to exit the for-loop prematurely.")
        break  # exit the for-loop
n_iterations = q  # we re-assign n_iterations


# ----------------------------------------------------------------------------------------------------------------------
# Stack all the dataframes together into a MultiIndex DataFrame
# ----------------------------------------------------------------------------------------------------------------------
xy_scatter_df = pd.concat(xy_scatter_dfs, ignore_index=False, sort=False, copy=True)
xy_scatter_df = xy_scatter_df.set_index(["iteration", xy_scatter_df.index])

xy_smoothed_df = pd.concat(xy_smoothed_dfs, ignore_index=False, sort=False, copy=True)
xy_smoothed_df = xy_smoothed_df.set_index(["iteration", xy_smoothed_df.index])
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Plot all the results in an array of figures
# ======================================================================================================================
num_cols = 2
num_rows = int(math.ceil(q / num_cols))
season_xticks = list(np.arange(start=x_limits[0], stop=x_limits[1], step=30))

fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7.07, 10))
axs = axs.flatten()
plt.subplots_adjust(wspace=0.05)
fig.suptitle("$k_{cp}$ versus Number of Days.  One \"Bad\" probe removed per iteration.")

for i, ax in enumerate(axs):
    print("The value of i is: {}.".format(i))
    ax.set_ylim(bottom=0.0, top=KCP_MAX)
    ax.grid(True)
    x_scatter, y_scatter = get_xs_and_ys(dataframe=xy_scatter_df, iteration=i, status="scatter")
    x_smoothed, y_smoothed = get_xs_and_ys(dataframe=xy_smoothed_df, iteration=i, status="smoothed")
    ax.scatter(x_scatter, y_scatter, marker=".", color="magenta", s=20, edgecolors="black", linewidth=1, alpha=0.5,
               label="Scatter plot")
    ax.scatter(cco_df["days"].values, cco_df["cco"].values, marker=".", color="yellow", s=15, alpha=0.5,
               label="Reference $k_{cp}$")
    ax.plot(x_smoothed, y_smoothed, linewidth=1.5, alpha=0.75, label="Smoothed")
    ax.tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                   labelsize="small", axis="x")
    ax.set_xticks(season_xticks)
    ax.set_xticklabels(season_xticks, rotation=40, ha="right")
    ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.set_ylim(bottom=0, top=KCP_MAX)
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
    ax.legend(prop={"size": 6}, loc=6)
    ax.annotate(s="$R^2$ = {:.3f}\n"
                  "Iteration {}\n"
                  "{} removed next".format(r_squared_stat_list[i], i, removed_probes[i]),
                xycoords="axes fraction", xy=(0.01, 0.80), fontsize=9)
    if i == n_iterations:
        break
axs[-1].set_xlabel("$n$ Days into the Season")
axs[-2].set_xlabel("$n$ Days into the Season")
axs[-2].set_ylabel("$k_{cp}$")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./figures/remove_outliers_jacobus_method.png")
plt.close()
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Investigate `r_squared_stat_list` to determine the index at which we should truncate.
# Re-normalise `removed_probes`.
# Re-normalise `probe_ids`.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
consecutive_differences = np.diff(r_squared_stat_list)
idx_first_pos_value = np.where(consecutive_differences > 0)[0][0]  # incidentally equal to the number of "bad" probes
length_of_bad_probes = len(removed_probes)
while length_of_bad_probes != idx_first_pos_value:
    probe_ids.append(removed_probes.pop())
    length_of_bad_probes = len(removed_probes)
probe_ids.sort()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Create a figure array showing the trend against the remaining "good" probes.
# In addition, annotate the R^2 of the trend against each probe scatter plot.
# We show both the latest trend (after removing all the bad probes), and the original trend (where no probes were
# removed).
# ----------------------------------------------------------------------------------------------------------------------
x_smoothed = xy_smoothed_dfs[idx_first_pos_value]["x_smoothed"].values
y_smoothed = xy_smoothed_dfs[idx_first_pos_value]["y_smoothed"].values
x_smoothed_old, y_smoothed_old = xy_smoothed_dfs[0]["x_smoothed"].values, xy_smoothed_dfs[0]["y_smoothed"].values
probes_r_squared = []
probes_r_squared_old = []
for p in probe_ids:
    probe_df = extract_probe_df(multi_index_df=probe_set_df, probe=p)
    probes_r_squared.append(get_r_squared(x_raw=probe_df["days"].values, y_raw=probe_df["kcp"].values,
                                          x_fit=x_smoothed, y_fit=y_smoothed))
    probes_r_squared_old.append(get_r_squared(x_raw=probe_df["days"].values, y_raw=probe_df["kcp"].values,
                                              x_fit=x_smoothed_old, y_fit=y_smoothed_old))

ncols = 1
nrows = math.ceil(len(probe_ids) / ncols)
fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(7.07, 10))
axs = axs.flatten()
fig.suptitle("$k_{cp}$ vs Days.  The Probe Scatter, Original Trend, and Latest Trend are compared.")

zipped_meta = ((m, c) for m, c in zip(marker_list, color_list))

for i, p in enumerate(probe_ids):
    meta = next(zipped_meta)
    axs[i].set_ylim(bottom=0.0, top=KCP_MAX)
    axs[i].grid(True)
    probe_df = extract_probe_df(multi_index_df=probe_set_df, probe=p)
    x_scatter, y_scatter = probe_df["days"].values, probe_df["kcp"].values
    axs[i].scatter(x_scatter, y_scatter, marker=meta[0], color=meta[1], s=20, edgecolors="black", linewidth=1,
                   alpha=0.5, label=p)
    axs[i].scatter(cco_df["days"].values, cco_df["cco"].values, marker=".", color="yellow", s=15, alpha=0.5,
                   label="Reference $k_{cp}$")
    axs[i].plot(x_smoothed, y_smoothed, linewidth=1.5, alpha=0.75,
                label="Iteration {:.0f}: $R^2 = {:.3f}$".format(idx_first_pos_value, probes_r_squared[i]))
    axs[i].plot(x_smoothed_old, y_smoothed_old, linewidth=1.5, alpha=0.75, ls="-.",
                label="Iteration 0: $R^2 = {:.3f}$".format(probes_r_squared_old[i]))
    axs[i].tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                       labelsize="small", axis="x")
    axs[i].set_xticks(season_xticks)
    axs[i].set_xticklabels(season_xticks, rotation=40, ha="right")
    axs[i].set_xlim(left=x_limits[0], right=x_limits[1])
    axs[i].set_ylim(bottom=0, top=KCP_MAX)
    axs[i].set_ylabel("$k_{cp}$")
    for tick in axs[i].get_xticklabels():
        tick.set_visible(True)
    axs[i].legend(prop={"size": 8}, loc="best")
axs[-1].set_xlabel("Days")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./figures/remaining_good_probes.png")
plt.close()
# ----------------------------------------------------------------------------------------------------------------------
