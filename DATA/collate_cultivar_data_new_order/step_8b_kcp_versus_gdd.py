import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
import numpy as np
import helper_functions as hf
import helper_meta_data as hm
import helper_data as hd


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define some constants
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. `color_ls_meta` is a list of tuples containing values to be used when
#    plotting figures.  These values pertain to the color and linestyle of a
#    plot.
# 2. cycle makes an iterator from the iterable that reverts to the beginning if
#    the iterable is exhausted.  This enables us, in theory, to infinitely loop
#    through the iterable.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
color_ls_meta = hm.color_ls_meta[:]
color_ls_meta = cycle(color_ls_meta)
meta = next(color_ls_meta)

starting_year = hm.starting_year

CULTIVAR = hm.CULTIVAR
WEEKLY_BINNED_VERSION = hm.WEEKLY_BINNED_VERSION  # This defaults to `True`.
# Alternative option is `None`.
# Specifies whether to use weekly-binned kcp data or not.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Import the necessary data to perform data crunching on.
# -----------------------------------------------------------------------------
# 1. We will import the excel file at `./data/processed_probe_data.xlsx`
#    In this excel file we need the column "cumulative_gdd".
#    We may use any sheet/probe_id in this excel file.  "cumulative_gdd" is
#    identical for each sheet/probe_id.  The index_col is "date".
# 2. We will import the excel file at `./data/binned_kcp_data.xlsx`
#    We need the sheet "day_frequency".
#    The columns in "day_frequency" are: "season_day", "calendar_day",
#    "day_averaged_kcp".
#    The index_col in "day_frequency" is:  "datetimestamp".
# "cumulative_gdd" is to become our new x-axis.
# "day_averaged_kcp" is to become our new y-axis.
# The common denominator between these two features is "season_day".
# -----------------------------------------------------------------------------
# 1.
processed_eg_df = hd.processed_eg_df.copy(deep=True)

# 2.
kcp_trend_df = pd.read_excel("./data/binned_kcp_data.xlsx",
                             sheet_name="day_frequency", header=0, index_col=0,
                             squeeze=False, parse_dates=True)
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data Crunching.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Wrap the dates of processed_eg_df.index
# 2. For each wrapped date, get the corresponding "season_day".
#    This is achieved by applying the `map` function on a pandas Series.
# 3. Drop all the samples for which cumulative_gdd is NaN.
#    This is done inplace.
# 4. Sort the DataFrame by the season_day column.
# 5. Get Numpy representations of "season_day" and "cumulative_gdd".
# 6. Get a weighted moving average of the "cumulative_gdd" data.
# 7. Plotting of "cumulative_gdd" scatter plot and "smoothed_cumul_gdd"
#    line plot.
# 8. Save DataFrame giving "season_day" and "smoothed_gdd".  "wrapped_date"
#    serves as the index.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.
processed_eg_df["wrapped_dates"] = hf.date_wrapper(processed_eg_df.index,
                                                   starting_year=starting_year)

# 2.
stamp_season_dict = dict(zip(kcp_trend_df.index,
                             kcp_trend_df["season_day"].values))
series_to_be_assigned = \
    processed_eg_df["wrapped_dates"].map(stamp_season_dict, na_action="ignore")
processed_eg_df["season_day"] = series_to_be_assigned

# 3.
processed_eg_df.dropna(subset=["cumulative_gdd"], inplace=True)

# 4.
processed_eg_df.sort_values(by=["season_day"], ascending=True, inplace=True)

# 5.
x_vals = processed_eg_df["season_day"].values
y_vals = processed_eg_df["cumulative_gdd"].values

# 6.
x_plt, y_plt = hf.weighted_moving_average(x=x_vals, y=y_vals, width=10)

# 7.
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x=processed_eg_df["season_day"].values,
           y=processed_eg_df["cumulative_gdd"].values, marker="^", c="blue",
           edgecolors="black", alpha=0.30, s=20,
           label="Cumulative GDD's of multiple seasons")
ax.plot(x_plt, y_plt, alpha=1, color=meta[0], linestyle=meta[1],
        label="Smoothed Cumulative GDD")
ax.set_xlabel("Season Day $(n)$")
ax.set_ylabel("Cumulative GDD")
ax.set_title("(Smoothed) Cumulative GDD versus Season Day")
ax.set_xlim(left=0, right=366)
ax.set_ylim(bottom=0)
ax.grid()
plt.xticks(np.arange(start=0, stop=max(x_plt) + 1, step=30))
plt.legend()
plt.tight_layout()
plt.savefig("./figures/smoothed_cumul_GDD.png")
plt.cla()
plt.clf()
plt.close()

# 8.
x_plt = x_plt.astype(int)
day_gdd_dict = dict(zip(x_plt, y_plt))
series_to_be_assigned = processed_eg_df["season_day"].map(day_gdd_dict)
processed_eg_df["smoothed_cumul_gdd"] = series_to_be_assigned
smoothed_cumul_gdd_df = \
    pd.DataFrame(data=processed_eg_df[["season_day", "smoothed_cumul_gdd"]],
                 index=processed_eg_df["wrapped_dates"], copy=True)
# smoothed_cumul_gdd_df.drop_duplicates(inplace=True)
smoothed_cumul_gdd_df.to_excel("./data/smoothed_cumul_gdd_vs_season_day.xlsx",
                               header=True, index=True,
                               index_label="wrapped_date", verbose=True,
                               float_format="%.7f")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Finally connect kcp (dependent variable) with GDD (independent variable).
# =============================================================================
# 1. Make a dictionary where "season_day" are the keys, and "day_averaged_kcp"
#    are the values.
# 2. Add "daily_trend_kcp" column to `smoothed_cumul_gdd_df` by mapping
#    "season_day" to "day_averaged_kcp" via the `sday_kcp_dict` dictionary.
# 3. Generate plot of kcp versus smoothed cumulative GDD.
# 4. Save .xlsx file of kcp versus smoothed cumulative GDD.
# =============================================================================
if WEEKLY_BINNED_VERSION:
    projected_df = pd.read_excel("./data/projected_weekly_data.xlsx", header=0,
                                 index_col=0, parse_dates=True)
    season_day = projected_df["season_day"].values
    daily_kcp = projected_df["projected_kcp"].values
else:
    season_day = kcp_trend_df["season_day"].values
    daily_kcp = kcp_trend_df["day_averaged_kcp"].values
# 1.
sday_kcp_dict = dict(zip(season_day, daily_kcp))

# 2.
series_to_be_assigned = smoothed_cumul_gdd_df["season_day"].map(sday_kcp_dict)
smoothed_cumul_gdd_df["daily_trend_kcp"] = series_to_be_assigned

# 3.
x_plt = smoothed_cumul_gdd_df["smoothed_cumul_gdd"].values
y_plt = smoothed_cumul_gdd_df["daily_trend_kcp"].values

_, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plt, y_plt, label=CULTIVAR)
ax.set_xlabel("(Smoothed) Cumulative GDD")
ax.set_ylabel("$k_{cp}$")
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.grid()
ax.set_title("$k_{cp}$ versus (Smoothed) Cumulative GDD")
plt.legend()
plt.xticks(np.arange(start=0, stop=max(x_plt) + 1, step=200))
plt.tight_layout()
plt.savefig("./figures/kcp_versus_GDD.png")
plt.cla()
plt.clf()
plt.close()

# 4.
smoothed_cumul_gdd_df.to_excel("./data/kcp_vs_smoothed_cumul_gdd.xlsx",
                               columns=["smoothed_cumul_gdd",
                                        "daily_trend_kcp"], header=True,
                               index=True, index_label="datetimestamp",
                               float_format="%.7f")
# =============================================================================
