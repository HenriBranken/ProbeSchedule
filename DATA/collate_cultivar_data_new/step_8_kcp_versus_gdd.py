import matplotlib.pyplot as plt
from itertools import cycle
import datetime
from cleaning_operations import BEGINNING_MONTH
import pandas as pd
import numpy as np

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define some constants
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.    `meta` is a list of tuples containing values to be used when plotting figures.  These values pertain to the
#       color and linestyle of a plot.
# 2.    cycle makes an iterator from the iterable that reverts to the beginning if the iterable is exhausted.
#       This enables us, in theory, to infinitely loop through the iterable.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
meta = [("goldenrod", "-"), ("green", "--"), ("blue", ":"), ("silver", "-."),
        ("burlywood", "-"), ("lightsalmon", "--"), ("chartreuse", ":")]
metacycle = cycle(meta)
meta = next(metacycle)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Define some helper functions
# ======================================================================================================================
# 1.    get_starting_year(dataframe):
#       Get the most historic ("youngest") year from the datetime index of the dataframe.
# 2.    date_wrapper(dataframe):
#       Wrap all the dates into the interval
#       [starting_year/BEGINNING_MONTH/01; starting_year+1/BEGINNING_MONTH-1/last date of (BEGINNING_MONTH-1)]
# 3.    gaussian(x, amp=1, mean=0, sigma=1):
#       Generate a Gaussian over the x datapoints having an amplitude of 1, a mean of <mean>, and a sigma of 10.
#       In a for loop, we loop over all the x-points and iteratively set the mean to each x value.
# 4.    weighted_moving_average(x, y, step_size=1, width=10):
#       Generate a weighted moving average of the (x, y) dataset.
#       Basically, we want a smoothed trend line running through the (x, y) scatter plot.
#       Later in the code, we use this function to generate a smoothed trend of the cumulative gdd data.
# ======================================================================================================================
def get_starting_year(dataframe):
    d = min(dataframe.index)
    return int(d.year)


def date_wrapper(dataframe):
    wrapped_dates = []
    for d in dataframe.index:
        if BEGINNING_MONTH <= d.month <= 12:
            hacked_date = datetime.datetime(year=get_starting_year(dataframe), month=d.month, day=d.day)
        else:
            hacked_date = datetime.datetime(year=get_starting_year(dataframe) + 1, month=d.month, day=d.day)
        wrapped_dates.append(hacked_date)
    return wrapped_dates


def gaussian(x, amp=1, mean=0, sigma=10):
    return amp*np.exp(-(x - mean)**2 / (2*sigma**2))


def weighted_moving_average(x, y, step_size=1, width=10):
    num = (np.max(x) - np.min(x)) // step_size + 1
    bin_coords = np.linspace(start=np.min(x), stop=np.max(x), num=num, endpoint=True)
    bin_avgs = np.zeros(len(bin_coords))

    for index in range(0, len(bin_coords)):
        weights = gaussian(x=x, mean=bin_coords[index], sigma=width)
        bin_avgs[index] = np.average(y, weights=weights)

    return bin_coords, bin_avgs
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Import the necessary data to perform data crunching on.
# ----------------------------------------------------------------------------------------------------------------------
# 1.    We will import the excel file at ./data/processed_probe_data.xlsx
#       In this excel file we need the column "cumulative_gdd".
#       We may use any sheet/probe_id in this excel file.  "cumulative_gdd" is identical for each sheet/probe_id.
#       The index_col is "date".
# 2.    We will import the excel file at ./data/binned_kcp_data.xlsx
#       We need the sheet "day_frequency".
#       The columns in "day_frequency" are: "season_day", "calendar_day", "day_averaged_kcp".
#       The index_col in "day_frequency" is:  "datetimestamp".
# 3.    "cumulative_gdd" is to become our new x-axis.
#       "day_averaged_kcp" is to become our y-axis.
#       The common denominator between these two features is "season_day".
# ----------------------------------------------------------------------------------------------------------------------
processed_df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=0, header=0, index_col=0,
                             usecols="A,R", squeeze=False, parse_dates=["date"])
kcp_trend_df = pd.read_excel("./data/binned_kcp_data.xlsx", sheett_name="day_frequency", header=0, index_col=0,
                             squeeze=False, parse_dates=True)
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data Crunching.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.    Wrap the dates of processed_df.index
# 2.    For each wrapped date, get the corresponding "season_day".
#       This is achieved by applying the `map` function on a pandas Series.
# 3.    Drop all the samples for which cumulative_gdd is NaN.
#       This is done inplace.
# 4.    Sort the DataFrame by the season_day column.
# 5.    Drop duplicate ("season_day", "cumulative_gdd") pairs.
# 6.    Get Numpy representations of "season_day" and "cumulative_gdd".
# 7.    Get a weighted average of the "cumulative_gdd" data.
# 8.    Plotting of "cumulative_gdd" scatter plot and "smoothed_cumul_gdd" line plot.
# 9.    Save DataFrame giving "season_day" and "smoothed_gdd".  "wrapped_date" serves as the index.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
processed_df["wrapped_dates"] = date_wrapper(processed_df)
stamp_season_dict = dict(zip(kcp_trend_df.index, kcp_trend_df["season_day"].values))
processed_df["season_day"] = processed_df["wrapped_dates"].map(stamp_season_dict, na_action="ignore")
processed_df.dropna(subset=["cumulative_gdd"], inplace=True)
processed_df.sort_values(by=["season_day"], ascending=True, inplace=True)
processed_df.drop_duplicates(subset=["season_day", "cumulative_gdd"], inplace=True)

x_vals, y_vals = processed_df["season_day"].values, processed_df["cumulative_gdd"].values

x_plt, y_plt = weighted_moving_average(x=x_vals, y=y_vals, width=10)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x=processed_df["season_day"].values, y=processed_df["cumulative_gdd"].values,
           marker="^", c="blue", edgecolors="black", alpha=0.30, s=20, label="Cumulative GDD's of multiple seasons")
ax.plot(x_plt, y_plt, alpha=1, color=meta[0], linestyle=meta[1], label="Smoothed Cumulative GDD")
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

x_plt = x_plt.astype(int)
day_gdd_dict = dict(zip(x_plt, y_plt))
processed_df["smoothed_cumul_gdd"] = processed_df["season_day"].map(day_gdd_dict)
smoothed_cumul_gdd_df = pd.DataFrame(data=processed_df[["season_day", "smoothed_cumul_gdd"]],
                                     index=processed_df["wrapped_dates"], copy=True)
smoothed_cumul_gdd_df.drop_duplicates(inplace=True)
smoothed_cumul_gdd_df.to_excel("./data/smoothed_cumul_gdd_vs_season_day.xlsx", header=True, index=True,
                               index_label="wrapped_date", verbose=True)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Finally connect kcp (dependent variable) with GDD (independent variable).
# ======================================================================================================================
# 1.    Make a dictionary where "season_day" are the keys, and "day_averaged_kcp" are the values.
# 2.    Add "daily_trend_kcp" column to `smoothed_cumul_gdd_df` by mapping "season_day" to "day_averaged_kcp" via the
#       `sday_kcp_dict` dictionary.
# 3.    Generate plot of kcp versus smoothed cumulative GDD.
# 4.    Save .xlsx file of kcp versus smoothed cumulative GDD.
# ======================================================================================================================
season_day = kcp_trend_df["season_day"].values
daily_kcp = kcp_trend_df["day_averaged_kcp"].values
sday_kcp_dict = dict(zip(season_day, daily_kcp))

smoothed_cumul_gdd_df["daily_trend_kcp"] = smoothed_cumul_gdd_df["season_day"].map(sday_kcp_dict)

x_plt = smoothed_cumul_gdd_df["smoothed_cumul_gdd"].values
y_plt = smoothed_cumul_gdd_df["daily_trend_kcp"].values

_, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plt, y_plt, label="Golden Delicious Apples")
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

smoothed_cumul_gdd_df.to_excel("./data/kcp_vs_smoothed_cumul_gdd.xlsx",
                               columns=["smoothed_cumul_gdd", "daily_trend_kcp"], header=True, index=True,
                               index_label="datetimestamp")
# ======================================================================================================================
