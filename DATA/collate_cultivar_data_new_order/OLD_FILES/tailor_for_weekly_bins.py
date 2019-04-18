import matplotlib.pyplot as plt
import pandas as pd
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
import datetime
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Import the necessary data
# ----------------------------------------------------------------------------------------------------------------------
weekly_df = pd.read_excel("./data/binned_kcp_data.xlsx", sheet_name="week_frequency", header=0, index_col=0,
                          parse_dates=True)
season_week = weekly_df["season_week"].values
weekly_kcp = weekly_df["weekly_averaged_kcp"].values

with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Declare the necessary constants
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x_limits = [0, 365]
starting_date = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
from_season_week_to_kcp_dict = dict(zip(season_week, weekly_kcp))
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Project the weekly-averaged data onto daily bins.
# ======================================================================================================================
datetime_stamp = []
repeated_kcp = []
season_day = []
s_week = []

for i in range(x_limits[1]):
    datetime_stamp.append(starting_date + datetime.timedelta(days=i))
    season_day.append(i + 1)
    s_week.append(min((i//7) + 1, 52))  # s_week can never exceed 52.  We cap its maximum value at 52.
    repeated_kcp.append(from_season_week_to_kcp_dict.get(s_week[i], from_season_week_to_kcp_dict[52]))
    print("i = {:>3}, s_day = {:>3}, s_week = {:>2},"
          " kcp = {:.4f}, datetime_stamp = {}.".format(i, season_day[i], s_week[i], repeated_kcp[i],
                                                       datetime_stamp[i].strftime("%Y-%m-%d")))

dict_for_df = {"season_week": s_week, "season_day": season_day, "repeated_kcp": repeated_kcp}
projected_df = pd.DataFrame(data=dict_for_df, index=datetime_stamp, copy=True)
projected_df.index.name = "datetime_stamp"
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Show a plot of the weekly data that is projected onto daily bins
# ----------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7.07))
ax.set_xlabel("Season Day")
ax.set_ylabel("Weekly-binned $k_{cp}$")
ax.set_title("Weekly-binned $k_{cp}$ versus Season Day")
ax.grid(True)
ax.set_xticks(np.arange(start=x_limits[0], stop=x_limits[1], step=30))
ax.plot(projected_df["season_day"].values, projected_df["repeated_kcp"].values, lw=2)
ax.set_xlim(left=projected_df["season_day"].values[0], right=projected_df["season_day"].values[-1])
ax.set_ylim(bottom=0, top=KCP_MAX)
plt.tight_layout()
plt.pause(interval=3)
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Save the projected data to `./data/projected_data.xlsx`.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
projected_df.to_excel("./data/projected_data.xlsx", float_format="%.7f", header=True, index=True,
                      index_label="datetime_stamp")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
