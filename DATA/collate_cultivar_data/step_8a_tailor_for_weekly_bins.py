import matplotlib.pyplot as plt
import pandas as pd
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
import datetime
import numpy as np
import calendar


# -----------------------------------------------------------------------------
# Import the DAILY data, where kcp (the dependent variable) has daily bins.
# -----------------------------------------------------------------------------
# Let us rather start from the beginning, and import the daily_averaged_kcp
# data from the `day_frequency` sheet in the `binned_kcp_data.xlsx` excel file.
# This way, this module incorporates the entire process from beginning to end.
binned_kcp_data_df = pd.read_excel("./data/binned_kcp_data.xlsx",
                                   sheet_name="day_frequency", header=0,
                                   index_col=0, parse_date=True)

with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())

# Remove season_day 366.
to_check_for_date = datetime.datetime(year=starting_year+1,
                                      month=BEGINNING_MONTH, day=1)
condition = binned_kcp_data_df.index != to_check_for_date
binned_kcp_data_df = binned_kcp_data_df[condition]

season_day = binned_kcp_data_df["season_day"].values
daily_kcp = binned_kcp_data_df["day_averaged_kcp"].values
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Declare some constants
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x_limits = [0, 365]  # the beginning date, and the end date.
starting_date = datetime.datetime(year=starting_year, month=BEGINNING_MONTH,
                                  day=1)

last_day_of_the_month = calendar.monthrange(year=starting_year+1,
                                            month=BEGINNING_MONTH-1)[1]
ending_date = datetime.datetime(year=starting_year+1, month=BEGINNING_MONTH-1,
                                day=last_day_of_the_month)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Create weekly-averaged kcp data with weekly bins.
# The dependent variable ranges in the interval [1; 52].
# =============================================================================
# Dictionary comprehension
sweek_kcps_dict = {k: [] for k in np.arange(start=1, stop=52+1, step=1)}
for i, s_day in enumerate(season_day):
    season_week = min(((s_day - 1)//7) + 1, 52)
    sweek_kcps_dict[season_week].append(daily_kcp[i])
for k in sweek_kcps_dict.keys():  # perform averaging for every season week.
    sweek_kcps_dict[k] = np.average(sweek_kcps_dict[k])
# =============================================================================


# -----------------------------------------------------------------------------
# Project the weekly-averaged data onto daily bins.
# -----------------------------------------------------------------------------
datetime_stamp = pd.date_range(start=starting_date, end=ending_date, freq="D",
                               normalize=True)
projected_kcp = []
projected_week = []

for i in range(x_limits[1]):
    projected_week.append(min((i//7) + 1, 52))
    projected_kcp.append(sweek_kcps_dict.get(projected_week[i],
                                             sweek_kcps_dict[52]))
    dtstamp = datetime_stamp[i].strftime("%Y-%m-%d")
    print("i = {:>3}, s_day = {:>3}, s_week = {:>2},"
          " kcp = {:.4f}, datetime_stamp = {}.".format(i, season_day[i],
                                                       projected_week[i],
                                                       projected_kcp[i],
                                                       dtstamp))

dict_for_df = {"projected_week": projected_week, "season_day": season_day,
               "projected_kcp": projected_kcp}
projected_df = pd.DataFrame(data=dict_for_df, index=datetime_stamp, copy=True)
projected_df.index.name = "datetime_stamp"
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Show a plot of the weekly data that is projected onto daily bins
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig, ax = plt.subplots(figsize=(10, 7.07))
ax.set_xlabel("Season Day")
ax.set_ylabel("Weekly-binned $k_{cp}$")
ax.set_title("Weekly-binned $k_{cp}$ versus Season Day")
ax.grid(True)
ax.set_xticks(np.arange(start=x_limits[0], stop=x_limits[1]+1, step=30))
ax.plot(projected_df["season_day"].values,
        projected_df["projected_kcp"].values, lw=2)
ax.set_xlim(left=projected_df["season_day"].values[0],
            right=projected_df["season_day"].values[-1])
ax.set_ylim(bottom=0, top=KCP_MAX)
plt.tight_layout()
plt.pause(interval=4)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Save the projected data to `./data/projected_data.xlsx`.
# =============================================================================
projected_df.to_excel("./data/projected_weekly_data.xlsx", float_format="%.7f",
                      header=True, index=True, index_label="datetime_stamp")
# =============================================================================
