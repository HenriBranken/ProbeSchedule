import pandas as pd
import numpy as np
import os
import math
from cleaning_operations import KCP_MAX
from itertools import cycle
import matplotlib.pyplot as plt
import helper_functions as hf
import helper_meta_data as hm
import helper_data as hd
pd.options.mode.chained_assignment = None  # default='warn'


# =============================================================================
# Define some other important constants
# =============================================================================
n_neighbours_list = hm.n_neighbours_list
delta_x = hm.delta_x
x_limits = hm.x_limits
marker_color_meta = hm.marker_color_meta[:]
marker_color_meta = cycle(marker_color_meta)
ALLOWED_TAIL_DEVIATION = hm.ALLOWED_TAIL_DEVIATION
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Get the starting year: the year of the most "historic/past" sample.
starting_year = hm.starting_year
season_start_date = hm.season_start_date

# Load the cleaned (scatterplot) data of kcp versus datetimeStamp from
# "./data/stacked_cleaned_data_for_overlay.xlsx" that was saved in
# `step_1_perform_cleaning.py`.  We are interested in the `datetimeStamp`
# index, and the newly created `days` column.
cleaned_multi_df = hd.cleaned_multi_df
outer_index = hd.outer_index[:]
inner_index = hd.inner_index[:]

scatter_df = cleaned_multi_df.copy(deep=True)
scatter_df.index = scatter_df.index.droplevel(0)
scatter_df.sort_index(axis=0, level="datetimeStamp", ascending=True,
                      inplace=True)
scatter_df["days"] = scatter_df.index - season_start_date
scatter_df["days"] = scatter_df["days"].dt.days
x_scatter = scatter_df["days"].values
y_scatter = scatter_df["kcp"].values


# Import the (first/beginning) kcp vs datetimestamp smoothed trend from
# "binned_kcp_data.xlsx".  We need to import the sheet `day_frequency` from
# "binned_kcp_data.xlsx".  We are interested in the `season_day` column, and
# the `day_averaged_kcp` column.
smoothed_kcp_vs_date_df = pd.read_excel("./data/binned_kcp_data.xlsx",
                                        sheet_name="day_frequency", header=0,
                                        index_col=0, parse_dates=True,
                                        squeeze=False)
x_smoothed = smoothed_kcp_vs_date_df["season_day"].values
y_smoothed = smoothed_kcp_vs_date_df["day_averaged_kcp"].values


# Get the DataFrame holding the reference crop coefficients.
cco_df = hd.cco_df.copy(deep=True)

# Get all the probe_ids.
probe_ids = hm.probe_ids[:]
n_iterations = int(len(probe_ids) - 1)


# Determine the final mode that was used in `step_3_smoothed_version.py`.
# mode is either "Polynomial-fit" or "WMA".  Preferably it should be "WMA".
with open("./data/mode.txt", "r") as f:
    mode = f.readline().rstrip()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Define some helper functions
# -----------------------------------------------------------------------------
def collapse_dataframe(multi_index_df, tbr_probe_list, starting_date):
    # Return a new Single_Index DataFrame where all the probes specified in
    # `tbr_probe_list` are REMOVED.  We basically get back a pruned DataFrame
    # containing the data of the remaining "healthy probes".
    df = multi_index_df.copy(deep=True)
    for pr in tbr_probe_list:
        df.drop(index=pr, level=0, inplace=True)
    df.index = df.index.droplevel(0)
    df.sort_index(axis=0, level="datetimeStamp", ascending=True, inplace=True)
    df["days"] = df.index - starting_date
    df["days"] = df["days"].dt.days
    return df["days"].values, df["kcp"].values


def extract_probe_df(multi_index_df, probe, starting_date):
    df = multi_index_df.loc[(probe, ), ["kcp"]]
    df["days"] = df.index - starting_date
    df["days"] = df["days"].dt.days
    df.sort_values("days", ascending=True, inplace=True)
    return df[["days", "kcp"]]


def check_tails(y_trendline, leading_ref=cco_df["cco"].values[0],
                trailing_ref=cco_df["cco"].values[-1],
                allowed_dev=ALLOWED_TAIL_DEVIATION, suppress_check=False):
    leading_val, trailing_val = y_trendline[0], y_trendline[-1]
    leading_dev = np.absolute(leading_val - leading_ref)/leading_ref
    trailing_dev = np.absolute(trailing_val - trailing_ref)/trailing_ref
    if suppress_check:
        return True
    if (leading_dev <= allowed_dev) and (trailing_dev <= allowed_dev):
        return True
    else:
        return False


def check_improvement(latest_stat, stat_list, suppress_check=False):
    if suppress_check:
        return True
    if latest_stat <= stat_list[-1]:
        return True
    else:
        return False
# -----------------------------------------------------------------------------


# =============================================================================
# Initialise the "starting points" corresponding to iteration i = 0.
# =============================================================================
r_squared_stat = hf.get_r_squared(x_raw=x_scatter, y_raw=y_scatter,
                                  x_fit=x_smoothed, y_fit=y_smoothed)
xy_scatter_df = hf.create_xy_df(x_vals=x_scatter, y_vals=y_scatter,
                                iteration=int(0), status="scatter")
xy_smoothed_df = hf.create_xy_df(x_vals=x_smoothed, y_vals=y_smoothed,
                                 iteration=int(0), status="smoothed")
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data Munging.  In the for-loop we iteratively remove "bad" probes.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Populate lists with our "starting points" that we created in the above block:
r_squared_stat_list = [r_squared_stat]
xy_scatter_dfs = [xy_scatter_df]
xy_smoothed_dfs = [xy_smoothed_df]
removed_probes = []

# Assert that the number of flagging iterations is less than the num_probes:
assert len(probe_ids) > n_iterations, "Number of iterations is greater than " \
                                      "the number of probes.\n" \
                                      "Decrease n_iterations such that " \
                                      "n_iterations < len(probe_ids)."

q = 0
# Cycle through the screening logic `n_iterations` times.
for i in range(n_iterations):
    try:
        # Populate probes_r_squared list with r-squares between probe scatter
        # data and the current smoothed trend line:
        probes_r_squared = []
        # Loop over all the probes:
        for p in probe_ids:
            # Extract an individual probe:
            probe_df = extract_probe_df(multi_index_df=cleaned_multi_df,
                                        probe=p,
                                        starting_date=season_start_date)
            # Compute r-squared between probe scatter and current trend line:
            val = hf.get_r_squared(x_raw=probe_df["days"].values,
                                   y_raw=probe_df["kcp"].values,
                                   x_fit=x_smoothed, y_fit=y_smoothed)
            # Append r-squared `val` to the growing probes_r_squared list:
            probes_r_squared.append(val)
        # Extract the probe index whose probe scatter data is the most poorly
        # fit by the smoothed trendline.  Call this probe the "worst probe":
        max_arg_index = np.where(probes_r_squared == max(probes_r_squared))[0]
        max_arg_index = max_arg_index[0]
        # Append the "worst probe" to `removed_probes`.
        removed_probes.append(probe_ids.pop(max_arg_index))
        # Generate a "pruned" (xy) scatter dataset, and assign to `x_scatter`
        # and `y_scatter`:
        some_tuple = collapse_dataframe(multi_index_df=cleaned_multi_df,
                                        tbr_probe_list=removed_probes,
                                        starting_date=season_start_date)
        # We have finally garnered our `x_scatter` and `y_scatter` data.
        x_scatter, y_scatter = some_tuple
        # Determine the mode; whether it be "WMA" or "Polynomial-fit".
        if mode == "WMA":
            # This will store all the various trend lines associated with
            # different values of `n_neighbours`:
            saved_trend_lines = []
            num_bumps = []
            tracker = 0
            for n_neighbours in n_neighbours_list:
                try:
                    # Generate new smoothed trendline based on `n_neighbours`
                    # hyperparameter:
                    x_smoothed, y_smoothed = \
                        hf.weighted_moving_average(x=x_scatter, y=y_scatter,
                                                   step_size=delta_x,
                                                   width=n_neighbours,
                                                   x_lims=x_limits,
                                                   append_=True)
                    # Append smoothed trend line to `saved_trend_lines`.
                    saved_trend_lines.append(zip(x_smoothed, y_smoothed))
                    # Append n_bumps to `num_bumps` list.
                    num_bumps.append(hf.get_n_local_extrema(y_smoothed))
                    tracker += 1
                # I.e. out scatter is too sparse to generate a WMA trendline:
                except ZeroDivisionError:
                    # Truncate `n_neighbours_list` to contain only the
                    # `n_neighbours` that we have looped over in the for-loop.
                    n_neighbours_list = n_neighbours_list[:tracker]
                    # Exit the for-loop; `n_neighbours` has become too small.
                    break
            try:
                # Get index at which the smoothed trend line has only 1 bump.
                prized_index = hf.get_prized_index(num_bumps)
                # Extract the associated smoothed trend_line having 1 bump.
                trend_line = saved_trend_lines[prized_index]
                # Extract `x_smoothed` and `y_smoothed` from `trend_line`.
                some_tuple = [list(t) for t in zip(*trend_line)]
                x_smoothed, y_smoothed = some_tuple[0], some_tuple[1]
            # I.e. there is no trendline just having 1 bump as desired.
            # Therefore we have to switch over to "Polynomial-fit" mode.
            except hf.NoProperWMATrend as e:
                print(e)
                print("{:.>80}".format("Cannot perform Gaussian-Weighted-"
                                       "Moving-Average."))
                print("{:.>80}".format("Proceeding with Polynomial Fit."))
                mode = "Polynomial-fit"  # Switch over to "Polynomial-fit" mode
                # Perform 4th-order pol-fit to `x_scatter` and `y_scatter`.
                x_smoothed, y_smoothed = \
                    hf.get_final_polynomial_fit(x_raw=x_scatter,
                                                y_raw=y_scatter,
                                                step_size=delta_x,
                                                degree=hf.pol_degree,
                                                x_lims=x_limits)
        # I.e. we know for sure that mode is "Polynomial-fit" and we do not
        # have to bother with "WMA" trendlines.
        else:
            # Perform a 4th-order polynomial fit to the scatter data.
            x_smoothed, y_smoothed = \
                hf.get_final_polynomial_fit(x_raw=x_scatter,
                                            y_raw=y_scatter,
                                            step_size=delta_x,
                                            degree=hf.pol_degree,
                                            x_lims=x_limits)
        # At this point we have finally garnered our new `x_smoothed` and
        # `y_smoothed` dataset.
        r_squared_stat = hf.get_r_squared(x_raw=x_scatter,
                                          y_raw=y_scatter,
                                          x_fit=x_smoothed,
                                          y_fit=y_smoothed)
        perf_status = check_improvement(latest_stat=r_squared_stat,
                                        stat_list=r_squared_stat_list,
                                        suppress_check=False)
        if perf_status:
            tail_status = check_tails(y_trendline=y_smoothed,
                                      suppress_check=False)
            # I.e., if `tail_status` is True:
            if tail_status:
                xy_smoothed_dfs.append(hf.create_xy_df(x_vals=x_smoothed,
                                                       y_vals=y_smoothed,
                                                       iteration=int(i + 1),
                                                       status="smoothed"))
                xy_scatter_dfs.append(hf.create_xy_df(x_vals=x_scatter,
                                                      y_vals=y_scatter,
                                                      iteration=int(i + 1),
                                                      status="scatter"))
                r_squared_stat_list.append(r_squared_stat)
                q += 1
            # I.e., `tail_status` is False.  The tail standards are NOT met:
            else:
                probe_ids.append(removed_probes.pop())
                print("\"Tail\"-standards are NOT satisfied.  "
                      "Have to exit the for-loop prematurely.")
                break
        else:
            probe_ids.append(removed_probes.pop())
            print("There is no improvement in the fit of the trendline "
                  "r-squared statistic.\n"
                  "Exiting the for-loop prematurely.")
            break
    # I.e. we cannot cycle through the screening logic `n_iterations` times:
    except IndexError:
        print("Cannot generate a new trendline.  "
              "Have to exit the for-loop prematurely.")
        break  # exit the for-loop
n_iterations = q  # we re-assign n_iterations


# -----------------------------------------------------------------------------
# Stack all the dataframes together into a MultiIndex DataFrame
# -----------------------------------------------------------------------------
xy_scatter_df = pd.concat(xy_scatter_dfs, ignore_index=False, sort=False,
                          copy=True)
xy_scatter_df = xy_scatter_df.set_index(["iteration", xy_scatter_df.index])

xy_smoothed_df = pd.concat(xy_smoothed_dfs, ignore_index=False, sort=False,
                           copy=True)
xy_smoothed_df = xy_smoothed_df.set_index(["iteration", xy_smoothed_df.index])
# -----------------------------------------------------------------------------


# =============================================================================
# Print out some basic results of the probe-screening algorithm:
# =============================================================================
print("\n" + "."*80)
print("Number of successful iterations/removals = {:.0f}.".format(q))
print("len(xy_scatter_dfs) = {:.0f}.".format(len(xy_scatter_dfs)))
# print(len(xy_scatter_df.index.unique(level="iteration")))
print("len(xy_smoothed_dfs) = {:.0f}.".format(len(xy_smoothed_dfs)))
# print(len(xy_smoothed_df.index.unique(level="iteration")))
print("len(r_squared_stat_list) = {:.0f}.".format(len(r_squared_stat_list)))
print("len(removed_probes) = {:.0f}.".format(len(removed_probes)))
print("len(probe_ids) = {:.0f}.".format(len(probe_ids)))
print("num_probes = len(probe_ids) + len(removed_probes) "
      "= {:.0f}.".format(len(probe_ids) + len(removed_probes)))
print("removed_probes = {}.".format(removed_probes))
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Save the results.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if not os.path.exists("./data/probe_screening/"):
    os.makedirs("data/probe_screening")

directory = "./data/probe_screening/"
files = os.listdir(directory)
for file in files:
    if file.endswith("_dfs.xlsx"):
        os.remove(os.path.join(directory, file))
        print("Removed the file named: {}.".format(file))
    if os.path.exists("./data/probe_screening/r_squared_stat_list.txt"):
        os.remove("./data/probe_screening/r_squared_stat_list.txt")
        print("Removed the file named: {}.".format("r_squared_stat_list.txt"))

writer_sc = pd.ExcelWriter("./data/probe_screening/xy_scatter_dfs.xlsx",
                           engine="xlsxwriter")
writer_sm = pd.ExcelWriter("./data/probe_screening/xy_smoothed_dfs.xlsx",
                           engine="xlsxwriter")
for i in range(q + 1):
    xy_scatter_dfs[i].to_excel(writer_sc, sheet_name="iter_{:.0f}".format(i),
                               columns=["x_scatter", "y_scatter"],
                               header=True, index=True, index_label="j",
                               float_format="%.7f")
    xy_smoothed_dfs[i].to_excel(writer_sm, sheet_name="iter_{:.0f}".format(i),
                                columns=["x_smoothed", "y_smoothed"],
                                header=True, index=True, index_label="j",
                                float_format="%.7f")
writer_sc.save()
writer_sm.save()

with open("./data/probe_screening/r_squared_stat_list.txt", "w") as f:
    for val in r_squared_stat_list:
        f.write(str(val) + "\n")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


"""
# =============================================================================
# Plot all the results in an array of figures
# =============================================================================
num_cols = 2
num_rows = int(math.ceil(q / num_cols))
season_xticks = list(np.arange(start=x_limits[0], stop=x_limits[1], step=30))

fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7.07, 10))
axs = axs.flatten()
plt.subplots_adjust(wspace=0.05)
fig.suptitle("$k_{cp}$ versus Number of Days.  "
             "One \"Bad\" probe removed per iteration.")

for i, ax in enumerate(axs):
    print("The value of i is: {}.".format(i))
    ax.set_ylim(bottom=0.0, top=KCP_MAX)
    ax.grid(True)
    x_scatter, y_scatter = hf.get_xs_and_ys(dataframe=xy_scatter_df,
                                            iteration=i, status="scatter")
    x_smoothed, y_smoothed = hf.get_xs_and_ys(dataframe=xy_smoothed_df,
                                              iteration=i, status="smoothed")
    ax.scatter(x_scatter, y_scatter, marker=".", color="magenta", s=20,
               edgecolors="black", linewidth=1, alpha=0.5,
               label="Scatter plot")
    ax.scatter(cco_df["season_day"].values, cco_df["cco"].values, marker=".",
               color="yellow", s=15, alpha=0.5, label="Reference $k_{cp}$")
    ax.plot(x_smoothed, y_smoothed, linewidth=1.5, alpha=0.75,
            label="Smoothed")
    ax.tick_params(which="major", bottom=True, labelbottom=True,
                   colors="black", labelcolor="black", labelsize="small",
                   axis="x")
    ax.set_xticks(season_xticks)
    ax.set_xticklabels(season_xticks, rotation=40, ha="right")
    ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.set_ylim(bottom=0, top=KCP_MAX)
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
    ax.legend(prop={"size": 6}, loc=6)
    ax.annotate(s="$R^2$ = {:.3f}\n"
                  "Iteration {}\n"
                  "{} removed next".format(r_squared_stat_list[i], i,
                                           removed_probes[i]),
                xycoords="axes fraction", xy=(0.01, 0.80), fontsize=9)
    if i == n_iterations:
        break
axs[-1].set_xlabel("$n$ Days into the Season")
axs[-2].set_xlabel("$n$ Days into the Season")
axs[-2].set_ylabel("$k_{cp}$")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./figures/remove_outliers_jacobus_method.png")
plt.close()
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Investigate `r_squared_stat_list` to get the idx at which we should truncate.
# Re-normalise `removed_probes`.
# Re-normalise `probe_ids`.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
consecutive_differences = np.diff(r_squared_stat_list)
if (consecutive_differences <= 0).all():
    idx_first_pos_value = np.where(consecutive_differences <= 0)[0][-1]
else:
    idx_first_pos_value = np.where(consecutive_differences > 0)[0][0]
# incidentally equal to the number of "bad" probes
length_of_bad_probes = len(removed_probes)
while length_of_bad_probes != idx_first_pos_value:
    probe_ids.append(removed_probes.pop())
    length_of_bad_probes = len(removed_probes)
probe_ids.sort()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Create a figure array showing the trend against the remaining "good" probes.
# In addition, annotate the R^2 of the trend against each probe scatter plot.
# We show both the latest trend (after removing all the bad probes), and the
# original trend (where no probes were removed).
# -----------------------------------------------------------------------------
x_smoothed = xy_smoothed_dfs[idx_first_pos_value]["x_smoothed"].values
y_smoothed = xy_smoothed_dfs[idx_first_pos_value]["y_smoothed"].values
x_smoothed_old = xy_smoothed_dfs[0]["x_smoothed"].values
y_smoothed_old = xy_smoothed_dfs[0]["y_smoothed"].values
probes_r_squared = []
probes_r_squared_old = []
for p in probe_ids:
    probe_df = extract_probe_df(multi_index_df=cleaned_multi_df, probe=p,
                                starting_date=season_start_date)
    probes_r_squared.append(hf.get_r_squared(x_raw=probe_df["days"].values,
                                             y_raw=probe_df["kcp"].values,
                                             x_fit=x_smoothed,
                                             y_fit=y_smoothed))
    probes_r_squared_old.append(hf.get_r_squared(x_raw=probe_df["days"].values,
                                                 y_raw=probe_df["kcp"].values,
                                                 x_fit=x_smoothed_old,
                                                 y_fit=y_smoothed_old))

ncols = 1
nrows = math.ceil(len(probe_ids) / ncols)
fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(7.07, 10))
axs = axs.flatten()
fig.suptitle("$k_{cp}$ vs Days.  The Probe Scatter, Original Trend, "
             "and Latest Trend are compared.")

for i, p in enumerate(probe_ids):
    meta = next(marker_color_meta)
    axs[i].set_ylim(bottom=0.0, top=KCP_MAX)
    axs[i].grid(True)
    probe_df = extract_probe_df(multi_index_df=cleaned_multi_df, probe=p,
                                starting_date=season_start_date)
    x_scatter, y_scatter = probe_df["days"].values, probe_df["kcp"].values
    axs[i].scatter(x_scatter, y_scatter, marker=meta[0], color=meta[1], s=20,
                   edgecolors="black", linewidth=1, alpha=0.5, label=p)
    axs[i].scatter(cco_df["season_day"].values, cco_df["cco"].values,
                   marker=".", color="yellow", s=15, alpha=0.5,
                   label="Reference $k_{cp}$")
    axs[i].plot(x_smoothed, y_smoothed, linewidth=1.5, alpha=0.75,
                label="Iteration {:.0f}: "
                      "$R^2 = {:.3f}$".format(idx_first_pos_value,
                                              probes_r_squared[i]))
    axs[i].plot(x_smoothed_old, y_smoothed_old, linewidth=1.5, alpha=0.75,
                ls="-.",
                label="Iteration 0: "
                      "$R^2 = {:.3f}$".format(probes_r_squared_old[i]))
    axs[i].tick_params(which="major", bottom=True, labelbottom=True,
                       colors="black", labelcolor="black", labelsize="small",
                       axis="x")
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
# -----------------------------------------------------------------------------
"""
