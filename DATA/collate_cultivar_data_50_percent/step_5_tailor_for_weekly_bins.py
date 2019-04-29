import matplotlib.pyplot as plt
import pandas as pd
from cleaning_operations import KCP_MAX
import numpy as np
import helper_meta_data as hm
import helper_data as hd
from helper_functions import safe_removal


# -----------------------------------------------------------------------------
# Import the DAILY data, where kcp (the dependent variable) has daily bins.
# Also perform other imports for the plotting figure.
# -----------------------------------------------------------------------------
# Let us rather start from the beginning, and import the daily_averaged_kcp
# data from the `day_frequency` sheet in the `binned_kcp_data.xlsx` excel file.
# This way, this module incorporates the entire process from beginning to end.
pruned_kcp_df = pd.read_excel("./probe_screening/pruned_kcp_vs_days.xlsx",
                              header=0, index_col=0)
starting_year = hm.starting_year
season_day = list(pruned_kcp_df["x_smoothed"].values)
daily_kcp = list(pruned_kcp_df["y_smoothed"].values)

cco_df = hd.cco_df.copy(deep=True)

screened_kcp_df = pd.read_excel("./probe_screening/screened_kcp_vs_days.xlsx",
                                header=0, index_col=0)
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Declare some constants
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x_limits = hm.x_limits[:]  # beginning and end dates represented as integers.
season_start_date = hm.season_start_date
season_end_date = hm.season_end_date
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Remove old files that were generated in the previous execution of the script.
# -----------------------------------------------------------------------------
file_list = ["./figures/weekly_binned_kcp.png",
             "./figures/weekly_vs_cco.png",
             "./data/projected_weekly_data.xlsx"]
safe_removal(file_list=file_list)
# -----------------------------------------------------------------------------


# =============================================================================
# Create weekly-averaged kcp data with weekly bins.
# The dependent variable ranges in the interval [1; 52].
# =============================================================================
# Dictionary comprehension:
sweek_kcp_dict = {k: [] for k in np.arange(start=1, stop=52 + 1, step=1)}
# Gather daily-kcp data into weekly bins:
for i, s_day in enumerate(season_day):
    season_week = min((s_day // 7) + 1, 52)
    sweek_kcp_dict[season_week].append(daily_kcp[i])

# Perform averaging for every season week bin:
for k in sweek_kcp_dict.keys():
    sweek_kcp_dict[k] = np.average(sweek_kcp_dict[k])
# =============================================================================


# -----------------------------------------------------------------------------
# Project the weekly-averaged data onto daily bins.
# -----------------------------------------------------------------------------
datetime_stamp = pd.date_range(start=season_start_date, end=season_end_date,
                               freq="D")
# Make sure that datetime_stamp and season_day are of the same length;
# remove redundant elements at the ends of the lists (perform truncation);
# remove ints for which int > 364.
while len(season_day) > len(datetime_stamp):
    print("Performing a .pop() on season_day and daily_kcp.")
    season_day.pop()
    daily_kcp.pop()

projected_kcp = []
projected_week = []

# Populate the lists `projected_kcp` and `projected_week`.  Print out the data
# associated with each day:
for i in range(x_limits[1]):  # 0, 1, 2, ..., 363, 364
    projected_week.append(min((i//7) + 1, 52))
    projected_kcp.append(sweek_kcp_dict.get(projected_week[i],
                                            sweek_kcp_dict[52]))
    dtstamp = datetime_stamp[i].strftime("%Y-%m-%d")
    # print("i = {:>3}, s_day = {:>3}, s_week = {:>2},"
    #       " kcp = {:.4f}, datetime_stamp = {}.".format(i, season_day[i],
    #                                                    projected_week[i],
    #                                                    projected_kcp[i],
    #                                                    dtstamp))

# Perform a sanity check to ensure that the lists/iterables:
# 1) projected_week
# 2) season_day
# 3) projected_kcp
# 4) datetime_stamp
# are all of the SAME LENGTH.
print("len(projected_week) = {:.0f}.".format(len(projected_week)))
print("len(season_day) = {:.0f}.".format(len(season_day)))
print("len(projected_kcp) = {:.0f}.".format(len(projected_kcp)))
print("len(datetime_stamp) = {:.0f}.".format(len(datetime_stamp)))
len_1 = len(projected_week)
len_2 = len(season_day)
len_3 = len(projected_kcp)
len_4 = len(datetime_stamp)
assert len_1 == len_2 == len_3 == len_4, "Iterables are NOT of the same length"

# Create a DataFrame, `projected_df`, storing the projected data, and the
# `season_day`.
dict_for_df = {"projected_week": projected_week, "season_day": season_day,
               "projected_kcp": projected_kcp}
projected_df = pd.DataFrame(data=dict_for_df, index=datetime_stamp, copy=True)
projected_df.index.name = "datetime_stamp"
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Show a plot of the weekly data that is projected onto daily bins.
# The figure is paused for 4 seconds, after which the rest of the script is
# executed.
# Figure is also saved at "./probe_screening/weekly_binned_kcp.png".
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
_, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Season Day")
ax.set_ylabel("$k_{cp}$")
ax.set_title("Weekly-binned $k_{cp}$ versus Season Day")
ax.grid(True)
ax.set_xticks(np.arange(start=x_limits[0], stop=x_limits[1] + 1, step=30))
ax.plot(projected_df["season_day"].values,
        projected_df["projected_kcp"].values, lw=2, color="goldenrod",
        alpha=0.6, label="Weekly-binned $k_{cp}$")
ax.plot(season_day, daily_kcp, lw=1.5, color="olivedrab", alpha=0.5,
        label="\"Daily\" $k_{cp}$")
ax.scatter(cco_df["season_day"].values, cco_df["cco"].values,
           marker=".", color="yellow", label="cco", alpha=0.5)
ax.scatter(screened_kcp_df["x_scatter"].values,
           screened_kcp_df["y_scatter"].values, marker=".",
           color="cornflowerblue", edgecolors="black",
           label="\"Screened\" $k_{cp}$", alpha=0.7)
ax.set_xlim(left=x_limits[0], right=x_limits[1])
ax.set_ylim(bottom=0, top=KCP_MAX)
plt.legend()
plt.tight_layout()
# plt.pause(interval=4)
plt.savefig("./figures/weekly_binned_kcp.png")
plt.close()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Exclusively compare the weekly-binned data, and the reference cco.
# -----------------------------------------------------------------------------
_, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Season Day")
ax.set_ylabel("$k_{cp}$")
ax.set_title("Old vs New standards")
ax.grid(True)
ax.set_xticks(np.arange(start=x_limits[0], stop=x_limits[1] + 1, step=30))
ax.set_xlim(left=x_limits[0], right=x_limits[1])
ax.set_ylim(bottom=0, top=KCP_MAX)
ax.plot(projected_df["season_day"].values,
        projected_df["projected_kcp"].values, lw=3, color="goldenrod",
        alpha=1.0, label="Weekly-binned $k_{cp}$")
ax.plot(cco_df["season_day"].values, cco_df["cco"].values,
        color="lightpink", label="cco", lw=3)
ax.fill_between(cco_df["season_day"].values, cco_df["cco"].values,
                projected_df["projected_kcp"].values,
                where=cco_df["cco"].values <=
                projected_df["projected_kcp"].values,
                facecolor="lawngreen", interpolate=True,
                label="cco <= new kcp")
ax.fill_between(cco_df["season_day"].values, cco_df["cco"].values,
                projected_df["projected_kcp"].values,
                where=cco_df["cco"].values >=
                projected_df["projected_kcp"].values,
                facecolor="darkred", interpolate=True,
                label="cco >= new kcp")
plt.legend()
plt.tight_layout()
plt.savefig("./figures/weekly_vs_cco.png")
plt.close()
# -----------------------------------------------------------------------------


# =============================================================================
# Save the projected data to "./probe_screening/projected_weekly_data.xlsx".
# =============================================================================
projected_df.to_excel("./data/projected_weekly_data.xlsx",
                      float_format="%.7f", header=True, index=True,
                      index_label="datetime_stamp")
# =============================================================================
