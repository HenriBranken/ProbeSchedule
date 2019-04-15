import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
from cleaning_operations import KCP_MAX
import helper_functions as hf
import helper_meta_data as hm


# -----------------------------------------------------------------------------
# Declare important constants
# -----------------------------------------------------------------------------
n_neighbours_list = hm.n_neighbours_list
delta_x = hm.delta_x
x_limits = hm.x_limits
mode = hm.mode  # The `default` at which we start out.
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create a DataFrame containing all the cleaned (date, kcp) data samples.
# Probe-id information is not necessary in this dataframe.
cleaned_df = pd.read_excel("./data/stacked_cleaned_data_for_overlay.xlsx",
                           header=0, index_col=[0, 1], parse_dates=True)
outer_index = list(cleaned_df.index.get_level_values("probe_id").unique())
inner_index = list(cleaned_df.index.get_level_values("datetimeStamp").unique())
cleaned_df.index = cleaned_df.index.droplevel(0)
cleaned_df.sort_index(axis=0, level="datetimeStamp", ascending=True,
                      inplace=True)

# Generate a list of all the probe-id names.
probe_ids = hm.probe_ids

# Extract the starting year.  The year of the most "historic/past" sample.
starting_year = hm.starting_year

# Create pandas DataFrame containing the reference crop coefficient values.
cco_df = pd.read_excel("./data/reference_crop_coeff.xlsx", sheet_name=0,
                       header=0, index_col=0, parse_dates=True)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Calculate the offset of each date from the beginning of the season.
# Convert the offsets, which are of type timedelta, to integer values.
# Sort the DataFrame by the "days" column.
# -----------------------------------------------------------------------------
# Calculate the number_of_days offset for each sample from the beginning.
cleaned_df["offset"] = cleaned_df.index - hm.season_start_date

# Convert the time-delta objects of the `offset` column to integer type.
cleaned_df["days"] = cleaned_df["offset"].dt.days

# Sort the whole DataFrame by the `days` column in ascending order
cleaned_df.sort_values(by="days", axis=0, inplace=True)
# At the bottom of the file, we save cleaned_df to an Excel file.
# -----------------------------------------------------------------------------


# =============================================================================
# Perform weighted-moving-average trends to the data
# Make plots of `kcp` versus `Days into the Season` of the WMA trends
# Here, the x-axis is simply an integer.
# =============================================================================
independent_var = cleaned_df["days"].values
dependent_var = cleaned_df["kcp"].values  # the crop coefficient, kcp

saved_trend_lines = []  # this will store all the various trend lines.
r_squared_stats = []
num_bumps = []

tracker = 0
for n_neighbours in n_neighbours_list:
    try:
        x_smoothed, y_smoothed = hf.weighted_moving_average(x=independent_var,
                                                            y=dependent_var,
                                                            step_size=delta_x,
                                                            width=n_neighbours,
                                                            x_lims=x_limits,
                                                            append_=True)
        saved_trend_lines.append(zip(x_smoothed, y_smoothed))
        r_squared_stats.append(hf.get_r_squared(x_raw=independent_var,
                                                y_raw=dependent_var,
                                                x_fit=x_smoothed,
                                                y_fit=y_smoothed))
        num_bumps.append(hf.get_n_local_extrema(y_smoothed))
        tracker += 1
    except ZeroDivisionError:
        n_neighbours_list = n_neighbours_list[:tracker]  # truncate
        break  # exit the for-loop.
try:
    prized_index = hf.get_prized_index(num_bumps)
    trend_line = saved_trend_lines[prized_index]
    unpack = [list(t) for t in zip(*trend_line)]
    x_smoothed, y_smoothed = unpack[0], unpack[1]
    with open("./data/prized_index.txt", "w") as f:
        f.write("{:.0f}".format(prized_index))
    with open("./data/prized_n_neighbours.txt", "w") as f:
        f.write("{:.0f}".format(n_neighbours_list[prized_index]))
except hf.NoProperWMATrend as e:
    print(e)
    print("{:.>80}".format("Cannot perform WMA."))
    print("{:.>80}".format("Proceeding with Polynomial Fit."))
    r_squared_stats = []
    mode = "Polynomial-fit"  # Switch over to the new mode.
    x_smoothed, y_smoothed = hf.get_final_polynomial_fit(x_raw=independent_var,
                                                         y_raw=dependent_var,
                                                         step_size=delta_x,
                                                         degree=hf.pol_degree,
                                                         x_lims=x_limits)
    r_squared_stats.append(hf.get_r_squared(x_raw=independent_var,
                                            y_raw=dependent_var,
                                            x_fit=x_smoothed,
                                            y_fit=y_smoothed))

# We need to safe-check the reasonableness of the WMA trendline if mode is
# still equal to "WMA".
if mode == "WMA":
    print("We are still in \"WMA\" mode.")
    r_squared_check = hf.get_r_squared(x_raw=independent_var,
                                       y_raw=dependent_var,
                                       x_fit=x_smoothed, y_fit=y_smoothed)
    print("r_squared_check = {:.4f}.".format(r_squared_check))

    x_pol, y_pol = hf.get_final_polynomial_fit(x_raw=independent_var,
                                               y_raw=dependent_var,
                                               step_size=delta_x,
                                               degree=hf.pol_degree,
                                               x_lims=x_limits)
    r_squared_pol = hf.get_r_squared(x_raw=independent_var,
                                     y_raw=dependent_var,
                                     x_fit=x_pol, y_fit=y_pol)
    print("r_squared_pol = {:.4f}.".format(r_squared_pol))
    if r_squared_check > r_squared_pol:
        mode = "Polynomial-fit"
        print("We have switched over to \"Polynomial-fit\" mode.")
        r_squared_stats = []
        some_tup = hf.get_final_polynomial_fit(x_raw=independent_var,
                                               y_raw=dependent_var,
                                               step_size=delta_x,
                                               degree=hf.pol_degree,
                                               x_lims=x_limits)
        x_smoothed, y_smoothed = some_tup
        r_squared_stats.append(hf.get_r_squared(x_raw=independent_var,
                                                y_raw=dependent_var,
                                                x_fit=x_smoothed,
                                                y_fit=y_smoothed))


_, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Number of days into the Season")
ax.set_ylabel("$k_{cp}$")
ax.set_title("Smoothed $k_{cp}$ versus number of days into the season")
ax.grid(True)
ax.set_xlim(left=x_limits[0], right=x_limits[1])
ax.set_ylim(bottom=0.0, top=KCP_MAX)
major_xticks = np.arange(start=x_limits[0], stop=x_limits[1], step=30,
                         dtype=np.int64)
ax.set_xticks(major_xticks)
ax.plot(x_smoothed, y_smoothed, alpha=0.70, label=mode)
ax.scatter(independent_var, dependent_var, c="magenta", marker=".",
           edgecolors="black", alpha=0.5, label="Cleaned Probe Data")
ax.scatter(cco_df["season_day"].values, cco_df["cco"].values, c="yellow",
           marker=".", alpha=0.5, label="Reference $k_{cp}$")
ax.legend()
plt.tight_layout()
plt.savefig("./figures/Smoothed_kcp_versus_days.png")
plt.close()
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create another plot where the x-axis is of type datetime.
# In this new plot we have kcp versus Month of the Year.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
season_start_date = hm.season_start_date

linspaced_x = list(np.arange(start=x_limits[0], stop=x_limits[1]+1, step=1))

datetime_linspaced = []
for i in linspaced_x:
    days = float(i)
    datetime_linspaced.append(season_start_date +
                              datetime.timedelta(days=days))

datetime_clouded = []
for i in independent_var:
    days = float(i)
    datetime_clouded.append(season_start_date +
                            datetime.timedelta(days=days))

fig, ax = plt.subplots(figsize=(10, 5))
major_xticks = pd.date_range(start=season_start_date, end=hm.season_end_date,
                             freq="MS")
ax.set_xticks(major_xticks)
ax.set_xlabel("Date (Month of the Season)")
ax.set_ylabel("$k_{cp}$")
ax.set_title("Smoothed $k_{cp}$ versus Month of the Season")
ax.grid(True)
ax.set_xlim(left=season_start_date, right=hm.season_end_date)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))  # Month/01
ax.set_ylim(bottom=0.0, top=KCP_MAX)

ax.plot(datetime_linspaced, y_smoothed, alpha=0.70, label=mode)

ax.scatter(datetime_clouded, dependent_var, c="magenta", marker=".",
           edgecolors="black", alpha=0.5, label="Cleaned Probe Data")
ax.scatter(cco_df.index, cco_df["cco"].values, c="yellow", marker=".",
           alpha=0.3, label="Reference $k_{cp}$, \"cco\"")
ax.legend()
fig.autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig("./figures/Smoothed_kcp_versus_monthf.png")
plt.cla()
plt.clf()
plt.close()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Write the r-squared statistics to file
# Each line in the file corresponds to the n_neighbours and r_squared.
# -----------------------------------------------------------------------------
if mode == "WMA":
    with open("./data/statistics_wma_trend_lines.txt", "w") as f:
        f.write("n_neighbours | r_squared_statistic\n")
        for i, nn in enumerate(n_neighbours_list):
            f.write("{:.0f} | {:.7f}\n".format(nn, r_squared_stats[i]))
else:
    with open("./data/statistics_polynomial_fit.txt", "w") as f:
        f.write("highest_order | r_squared_statistic\n")
        f.write("{:.0f} | {:.7f}\n".format(hf.pol_degree, r_squared_stats[0]))
# -----------------------------------------------------------------------------


# =============================================================================
# Save the `mode`.
# Save the data related to the best-fitting trend line.
# Save the sorted data of the cleaned kcp data.
# =============================================================================
with open("./data/mode.txt", "w") as f:
    f.write(mode)

df = pd.DataFrame(data={"datetimeStamp": datetime_linspaced,
                        "smoothed_kcp_trend": y_smoothed},
                  index=datetime_linspaced, columns=["smoothed_kcp_trend"],
                  copy=True)

df.to_excel("./data/smoothed_kcp_trend_vs_datetime.xlsx", float_format="%.7f",
            columns=["smoothed_kcp_trend"], header=True, index=True,
            index_label="datetimeStamp")

# Save the (sorted) data of the cleaned kcp data to an Excel file
cleaned_df.to_excel("data/kcp_vs_days.xlsx", sheet_name="sheet_1", header=True,
                    index=True, index_label=True, columns=["days", "kcp"],
                    float_format="%.7f")
# =============================================================================
