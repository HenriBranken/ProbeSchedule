import matplotlib.pyplot as plt
import pandas as pd
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
import datetime
import numpy as np
import helper_meta_data as hm


# -----------------------------------------------------------------------------
# Import the DAILY data, where kcp (the dependent variable) has daily bins.
# -----------------------------------------------------------------------------
# Let us rather start from the beginning, and import the daily_averaged_kcp
# data from the `day_frequency` sheet in the `binned_kcp_data.xlsx` excel file.
# This way, this module incorporates the entire process from beginning to end.
pruned_kcp_df = pd.read_excel("./probe_screening/pruned_kcp_vs_days.xlsx",
                              header=0, index_col=0)

starting_year = hm.starting_year
season_day = list(pruned_kcp_df["x_smoothed"].values)
daily_kcp = pruned_kcp_df["y_smoothed"].values
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Declare some constants
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x_limits = hm.x_limits[:]  # the beginning date, and the end date.
season_start_date = hm.season_start_date
season_end_date = hm.season_end_date
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Create weekly-averaged kcp data with weekly bins.
# The dependent variable ranges in the interval [1; 52].
# =============================================================================
# Dictionary comprehension
sweek_kcps_dict = {k: [] for k in np.arange(start=1, stop=52 + 1, step=1)}
# Gather daily-kcp data into weekly bins:
for i, s_day in enumerate(season_day):
    season_week = min((s_day // 7) + 1, 52)
    sweek_kcps_dict[season_week].append(daily_kcp[i])
# Perform averaging for every season week bin:
for k in sweek_kcps_dict.keys():
    sweek_kcps_dict[k] = np.average(sweek_kcps_dict[k])
# =============================================================================


# -----------------------------------------------------------------------------
# Project the weekly-averaged data onto daily bins.
# -----------------------------------------------------------------------------
datetime_stamp = pd.date_range(start=season_start_date, end=season_end_date,
                               freq="D")

while len(season_day) > len(datetime_stamp):
    season_day.pop()

projected_kcp = []
projected_week = []
# season_day = []

for i in range(x_limits[1]):
    # datetime_stamp.append(season_start_date + datetime.timedelta(days=i))
    projected_week.append(min((i//7) + 1, 52))
    projected_kcp.append(sweek_kcps_dict.get(projected_week[i],
                                             sweek_kcps_dict[52]))
    # season_day.append(i)
    # season_day.append(i)
    # dtstamp = datetime_stamp[i].strftime("%Y-%m-%d")
    # print("i = {:>3}, s_day = {:>3}, s_week = {:>2},"
    #       " kcp = {:.4f}, datetime_stamp = {}.".format(i, season_day[i],
    #                                                    projected_week[i],
    #                                                    projected_kcp[i],
    #                                                    dtstamp))

print("len(projected_week) = {:.0f}.".format(len(projected_week)))
print("len(season_day) = {:.0f}.".format(len(season_day)))
print("len(projected_kcp) = {:.0f}.".format(len(projected_kcp)))
print("len(datetime_stamp) = {:.0f}.".format(len(datetime_stamp)))
dict_for_df = {"projected_week": projected_week, "season_day": season_day,
               "projected_kcp": projected_kcp}
projected_df = pd.DataFrame(data=dict_for_df, index=datetime_stamp, copy=True)
projected_df.index.name = "datetime_stamp"
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Show a plot of the weekly data that is projected onto daily bins.
# The figure is paused for 4 seconds, after which the rest of the script is
# executed.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig, ax = plt.subplots(figsize=(10, 7.07))
ax.set_xlabel("Season Day")
ax.set_ylabel("Weekly-binned $k_{cp}$")
ax.set_title("Weekly-binned $k_{cp}$ versus Season Day")
ax.grid(True)
ax.set_xticks(np.arange(start=x_limits[0], stop=x_limits[1]+1, step=30))
ax.plot(projected_df["season_day"].values,
        projected_df["projected_kcp"].values, lw=2)
ax.set_xlim(left=x_limits[0], right=x_limits[1])
ax.set_ylim(bottom=0, top=KCP_MAX)
plt.tight_layout()
plt.pause(interval=4)
plt.savefig("./probe_screening/")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Save the projected data to "./data/projected_weekly_data.xlsx".
# =============================================================================
projected_df.to_excel("./probe_screening/projected_weekly_data.xlsx",
                      float_format="%.7f", header=True, index=True,
                      index_label="datetime_stamp")
# =============================================================================
