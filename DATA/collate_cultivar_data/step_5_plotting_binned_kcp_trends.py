import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from cleaning_operations import KCP_MAX
import helper_meta_data as hm
import helper_data as hd
pd.set_option('display.max_columns', 6)

# -----------------------------------------------------------------------------
# Load the necessary data
# -----------------------------------------------------------------------------
# 1. kcp_trend_vs_datetime is the data associated with the smoothed version
# 2. kcp_vs_days --> cleaned probe data
# 3. kcp_vs_day_df --> kcp as a function of day of the year/season
# 4. kcp_vs_week_df --> kcp as a function of week of the year/season
# 5. kcp_vs_month_df --> kcp as a function of month of the year/season
# -----------------------------------------------------------------------------
# Extract the data associated with the smoothed trend line.
kcp_vs_days_df = pd.read_excel("./data/smoothed_kcp_trend_vs_datetime.xlsx",
                               header=0, names=["smoothed_kcp_trend"],
                               index_col=0, parse_dates=True)
base_datetimestamp = kcp_vs_days_df.index
base_daily_kcp = kcp_vs_days_df["smoothed_kcp_trend"].values

starting_year = hm.starting_year
starting_week = base_datetimestamp[0].isocalendar()[1]  # the calendar week
season_start_date = hm.season_start_date
season_end_date = hm.season_end_date
season_xticks = hd.season_xticks

# `kcp_vs_days` is the cleaned scatter_plot data.
kcp_vs_days = pd.read_excel("data/kcp_vs_days.xlsx", header=0,
                            names=["days", "kcp"], index_col=0,
                            parse_dates=True)
# `kcp_vs_day_df` is the smoothed trend line.
kcp_vs_day_df = pd.read_excel("data/binned_kcp_data.xlsx",
                              sheet_name="day_frequency", header=0,
                              names=["season_day", "calendar_day",
                                     "day_averaged_kcp"], index_col=0,
                              squeeze=True, parse_dates=True)
# `kcp_vs_week_df` is the weekly-binned kcp data.
kcp_vs_week_df = pd.read_excel("data/binned_kcp_data.xlsx",
                               sheet_name="week_frequency", header=0,
                               names=["season_week", "calendar_week",
                                      "weekly_averaged_kcp"], index_col=0,
                               squeeze=True, parse_dates=True)
# `kcp_vs_month_df` is the monthly-binned kcp data.
kcp_vs_month_df = pd.read_excel("data/binned_kcp_data.xlsx",
                                sheet_name="month_frequency", header=0,
                                names=["season_month", "calendar_month",
                                       "monthly_averaged_kcp"], index_col=0,
                                squeeze=True, parse_dates=True)

# Get the reference crop coefficients as a function of datestamp/season_day.
cco_df = hd.cco_df.copy(deep=True)

# Extract the final `mode`, whether it be "WMA" or "Polynomial-fit".
with open("./data/mode.txt", "r") as f:
    mode = f.readline().rstrip()
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Code block in which we extract the lowest possible n_neighbours, IF
# mode == "WMA".
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if mode == "WMA":
    with open("./data/prized_index.txt", "r") as f:
        prized_index = int(f.readline().rstrip())
    with open("./data/prized_n_neighbours.txt", "r") as f:
        prized_n_neighbours = int(f.readline().rstrip())
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Project `kcp values` for weekly and monthly bins accordingly
# =============================================================================
repeated_weekly_kcp_vs_day = []
repeated_monthly_kcp_vs_day = []
for d in base_datetimestamp:
    # Extract the season week:
    season_week = min(((d - season_start_date).days // 7) + 1, 52)
    condition = kcp_vs_week_df["season_week"] == season_week
    # Find kcp value corresponding to season week:
    to_append = kcp_vs_week_df[condition]["weekly_averaged_kcp"][0]
    # Append associated kcp value to `repeated_weekly_kcp_vs_day` list:
    repeated_weekly_kcp_vs_day.append(to_append)
    # Repeat the procedure for monthly bins....................................
    # Extract the calendar month
    calendar_month = d.month
    condition = kcp_vs_month_df["calendar_month"] == calendar_month
    # Fin the kcp value corresponding to calendar_month:
    to_append = kcp_vs_month_df[condition]["monthly_averaged_kcp"][0]
    # Append associated kcp value to `repeated_monthly_kcp_vs_day` list:
    repeated_monthly_kcp_vs_day.append(to_append)
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot the data.
# Compare the scatter plot of the cleaned kcp data with the:
# a. Smoothed trendline.
# b. Projected Weekly-binned kcp data.
# c. Projected Monthly-binned kcp data.
# The figure is stored at "./figures/kcp_binning_strategies.png".
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(kcp_vs_days.index, kcp_vs_days["kcp"], c="magenta", marker=".",
           edgecolors="black", alpha=0.5, label="Cleaned Probe Data")
ax.plot(base_datetimestamp, base_daily_kcp, linewidth=2, alpha=1.0, label=mode)
ax.plot(base_datetimestamp, repeated_weekly_kcp_vs_day, linewidth=2,
        label="Weekly-binned $k_{cp}$", alpha=0.55)
ax.plot(base_datetimestamp, repeated_monthly_kcp_vs_day, linewidth=2,
        label="Monthly-binned $k_{cp}$", alpha=0.55)
ax.scatter(cco_df.index, cco_df["cco"].values, c="yellow", marker=".",
           alpha=0.5, label="Reference $k_{cp}$")
ax.set_xlabel("Date (Month of the Year/Season)")
ax.set_ylabel("$k_{cp}$")
ax.set_title("Different binning strategies for $k_{cp}$ as a function of time")
ax.set_xlim(left=season_start_date, right=season_end_date)
ax.set_ylim(bottom=0.0, top=KCP_MAX)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))  # Month/01
ax.set_xticks(season_xticks)
ax.legend()
ax.grid()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("./figures/kcp_binning_strategies.png")
plt.close()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
