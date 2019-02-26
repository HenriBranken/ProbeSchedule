import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
pd.set_option('display.max_columns', 6)

# ----------------------------------------------------------------------------------------------------------------------
# Load the necessary data
#   1.  kcp_trend_vs_datetime is the data associated with the quadratic fit (2nd-order polynomial)
#   2.  kcp_vs_days --> cleaned probe data
#   3.  kcp_vs_day_df --> kcp as a function of day of the year/season
#   4.  kcp_vs_week_df --> kcp as a function of week of the year/season
#   5.  kcp_vs_month_df --> kcp as a function of month of the year/season
# ----------------------------------------------------------------------------------------------------------------------
kcp_trend_vs_datetime = np.load("data/daily_trend_of_kcp_vs_datetime.npy")

base_datetimestamp = kcp_trend_vs_datetime[:, 0]
base_daily_kcp = kcp_trend_vs_datetime[:, 1]

starting_year = base_datetimestamp[0].year
starting_week = base_datetimestamp[0].isocalendar()[1]
starting_date = base_datetimestamp[0]

kcp_vs_days = pd.read_excel("data/kcp_vs_days.xlsx", sheet_name="sheet_1", header=0, names=["days", "kcp"],
                            index_col=0, squeeze=True, parse_dates=True)
kcp_vs_day_df = pd.read_excel("data/binned_kcp_data.xlsx", sheet_name="day_frequency", header=0,
                              names=["season_day", "calendar_day", "day_averaged_kcp"], index_col=0,
                              squeeze=True, parse_dates=True)
kcp_vs_week_df = pd.read_excel("data/binned_kcp_data.xlsx", sheet_name="week_frequency", header=0,
                               names=["season_week", "calendar_week", "weekly_averaged_kcp"], index_col=0,
                               squeeze=True, parse_dates=True)
kcp_vs_month_df = pd.read_excel("data/binned_kcp_data.xlsx", sheet_name="month_frequency", header=0,
                                names=["season_month", "calendar_month", "monthly_averaged_kcp"], index_col=0,
                                squeeze=True, parse_dates=True)
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Replicate kcp values for weekly and monthly bins accordingly
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
#   a.  Polynomial fit
#   b.  Weekly-binned kcp data
#   c.  Monthly-binned kcp data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(kcp_vs_days.index, kcp_vs_days["kcp"], c="magenta", marker=".",
           edgecolors="black", alpha=0.5, label="Cleaned Probe Data")
ax.plot(base_datetimestamp, base_daily_kcp, linewidth=1, label="Polynomial trend")
ax.plot(base_datetimestamp, repeated_weekly_kcp_vs_day, linewidth=1, label="Weekly-binned $k_{cp}$")
ax.plot(base_datetimestamp, repeated_monthly_kcp_vs_day, linewidth=1, label="Monthly-binned $k_{cp}$")
ax.set_xlabel("Date (Month of the Year)")
ax.set_ylabel("$k_{cp}$")
ax.set_title("Different binning strategies for $k_{cp}$ as a function of time")
ax.set_xlim(left=base_datetimestamp[0], right=base_datetimestamp[-1])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))  # Month/01
ax.legend()
ax.grid()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/kcp_binning_strategies.png")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
