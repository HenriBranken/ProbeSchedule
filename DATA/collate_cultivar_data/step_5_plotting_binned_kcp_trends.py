import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from cleaning_operations import KCP_MAX, BEGINNING_MONTH
import datetime
pd.set_option('display.max_columns', 6)

# ----------------------------------------------------------------------------------------------------------------------
# Load the necessary data
# ----------------------------------------------------------------------------------------------------------------------
#   1.  kcp_trend_vs_datetime is the data associated with the best polynomial fit
#   2.  kcp_vs_days --> cleaned probe data
#   3.  kcp_vs_day_df --> kcp as a function of day of the year/season
#   4.  kcp_vs_week_df --> kcp as a function of week of the year/season
#   5.  kcp_vs_month_df --> kcp as a function of month of the year/season
#   6.  probe_ids --> List of probe-ids used in the analysis
# ----------------------------------------------------------------------------------------------------------------------
# Extract the data associated with the weighted moving average generated in `step_3_weighted_moving_average.py`
kcp_vs_days_df = pd.read_excel("./data/WMA_kcp_trend_vs_datetime.xlsx", header=0, names=["WMA_kcp_trend"],
                               index_col=0, parse_dates=True)
base_datetimestamp = kcp_vs_days_df.index
base_daily_kcp = kcp_vs_days_df["WMA_kcp_trend"].values

# base_datetimestamp = kcp_trend_vs_datetime[:, 0]
# base_daily_kcp = kcp_trend_vs_datetime[:, 1]

with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())
starting_week = base_datetimestamp[0].isocalendar()[1]  # the calendar week
starting_date = base_datetimestamp[0]

# kcp_vs_days is the cleaned scatter_plot data
kcp_vs_days = pd.read_excel("data/kcp_vs_days.xlsx", header=0, names=["days", "kcp"], index_col=0, parse_dates=True)
kcp_vs_day_df = pd.read_excel("data/binned_kcp_data.xlsx", sheet_name="day_frequency", header=0,
                              names=["season_day", "calendar_day", "day_averaged_kcp"], index_col=0,
                              squeeze=True, parse_dates=True)
kcp_vs_week_df = pd.read_excel("data/binned_kcp_data.xlsx", sheet_name="week_frequency", header=0,
                               names=["season_week", "calendar_week", "weekly_averaged_kcp"], index_col=0,
                               squeeze=True, parse_dates=True)
kcp_vs_month_df = pd.read_excel("data/binned_kcp_data.xlsx", sheet_name="month_frequency", header=0,
                                names=["season_month", "calendar_month", "monthly_averaged_kcp"], index_col=0,
                                squeeze=True, parse_dates=True)

with open("./data/probe_ids.txt", "r") as f:
    probe_ids = f.readlines()
probe_ids = [x.rstrip() for x in probe_ids]

cco_df = pd.read_excel("./data/reference_crop_coeff.xlsx", sheet_name=0, header=0, index_col=0,
                       parse_dates=True)
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Code block in which we extract the lowest possible n_neighbours associated with one local maximum in the trend line.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
with open("./data/prized_index.txt", "r") as f:
    prized_index = int(f.readline().rstrip())

with open("./data/prized_n_neighbours.txt", "r") as f:
    prized_n_neighbours = int(f.readline().rstrip())
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Replicate/repeat kcp values for weekly and monthly bins accordingly
# ======================================================================================================================
repeated_weekly_kcp_vs_day = []
repeated_monthly_kcp_vs_day = []
for d in base_datetimestamp:
    season_week = (d - starting_date).days // 7 + 1
    if season_week == 53:
        season_week = 52
    condition = kcp_vs_week_df["season_week"] == season_week
    repeated_weekly_kcp_vs_day.append(kcp_vs_week_df[condition]["weekly_averaged_kcp"][0])
    calendar_month = d.month
    condition = kcp_vs_month_df["calendar_month"] == calendar_month
    repeated_monthly_kcp_vs_day.append(kcp_vs_month_df[condition]["monthly_averaged_kcp"][0])
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot the data
# Compare the scatter plot of the cleaned data with the:
#   a.  Best trend
#   b.  Weekly-binned kcp data
#   c.  Monthly-binned kcp data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(kcp_vs_days.index, kcp_vs_days["kcp"], c="magenta", marker=".", edgecolors="black", alpha=0.5,
           label="Cleaned Probe Data")
ax.plot(base_datetimestamp, base_daily_kcp, linewidth=2, alpha=1.0,
        label="n_neighbours = {}".format(prized_n_neighbours))
ax.plot(base_datetimestamp, repeated_weekly_kcp_vs_day, linewidth=2, label="Weekly-binned $k_{cp}$", alpha=0.55)
ax.plot(base_datetimestamp, repeated_monthly_kcp_vs_day, linewidth=2, label="Monthly-binned $k_{cp}$", alpha=0.55)
ax.scatter(cco_df.index, cco_df["cco"].values, c="yellow", marker=".", alpha=0.5, label="Reference $k_{cp}$")
ax.set_xlabel("Date (Month of the Year)")
ax.set_ylabel("$k_{cp}$")
ax.set_title("Different binning strategies for $k_{cp}$ as a function of time")
ax.set_xlim(left=base_datetimestamp[0], right=base_datetimestamp[-1])
ax.set_ylim(bottom=0.0, top=KCP_MAX)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))  # Month/01
major_xticks = pd.date_range(start=datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1),
                             end=datetime.datetime(year=starting_year + 1, month=BEGINNING_MONTH, day=1), freq="MS")
# Notice that the `MS` alias stands for `Month Start Frequency`.  Jul/01, Aug/01, etc...
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases gives a full list of
# frequency aliases.
ax.set_xticks(major_xticks)
ax.legend()
ax.grid()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("./figures/kcp_binning_strategies.png")
plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
