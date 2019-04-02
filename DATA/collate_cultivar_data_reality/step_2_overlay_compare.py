import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
import pandas as pd
from itertools import cycle
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# ======================================================================================================================
# Declare some necessary constants
# ======================================================================================================================
# 1. Create some meta data that will be used in the upcoming scatter plots
#    This meta data stores `marker` and `color` constants.
# ======================================================================================================================
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]
zipped_meta = cycle([(m, c) for m, c in zip(marker_list, color_list)])
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Load the cleaned data garnered in `step_1_perform_cleaning.py`.
# 2. Load the list of Probe-IDs stored at `./data/probe_ids.txt`.
# 3. Load the Reference Crop Coefficients stored at `./data/reference_crop_coeff.xlsx`
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.
# Load the cleaned data garnered in `step_1_perform_cleaning.py`.
cleaned_multi_df = pd.read_excel("./data/stacked_cleaned_data_for_overlay.xlsx", header=0, index_col=[0, 1],
                                 parse_dates=True)
outer_index = list(cleaned_multi_df.index.get_level_values("probe_id").unique())
inner_index = list(cleaned_multi_df.index.get_level_values("datetimeStamp").unique())

# 2.
# Get a list of all the Probe-IDs involved for the cultivar
with open("./data/probe_ids.txt", "r") as f2:
    probe_ids = f2.readlines()
probe_ids = [x.strip() for x in probe_ids]

# 3.
# Load the Reference Crop Coefficients stored at `./data/reference_crop_coeff.xlsx`
cco_df = pd.read_excel("./data/reference_crop_coeff.xlsx", sheet_name=0, header=0, index_col=0,
                       parse_dates=True)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Define all the helper functions
# ----------------------------------------------------------------------------------------------------------------------
# Get the year of the oldest (the most "historic/past") data point
def get_starting_year():
    inner_level = list(cleaned_multi_df.index.get_level_values("datetimeStamp").unique())
    return int(inner_level[0].year)


# Extract dates and kcp values from tuple
# Sort according to datetime
# Return sorted dates and associated kcp values that also got sorted in the process
# Note that wrapping has already occurred previously by calling cleaning_operations.get_final_dates(df) in `main.py`
def get_dates_and_kcp(dataframe, probe_id):
    sub_df = dataframe.loc[(probe_id, ), ["kcp"]]
    return sub_df.index, sub_df["kcp"].values


def date_wrapper(date_iterable):
    new_dates = []
    for datum in date_iterable:
        extract_month = datum.month
        if BEGINNING_MONTH <= extract_month <= 12:
            new_dates.append(datetime.datetime(year=starting_year, month=extract_month, day=datum.day))
        else:
            new_dates.append(datetime.datetime(year=starting_year + 1, month=extract_month, day=datum.day))
    return new_dates


def get_labels(begin, terminate):
    return [x for x in pd.date_range(start=begin, end=terminate, freq="MS")]
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Plot all the cleaned kcp data and save the plotted figure
# ======================================================================================================================
if not os.path.exists("./figures"):
    os.makedirs("figures")

starting_year = get_starting_year()  # extract the starting year

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Month of the Season")
ax.set_ylabel("$k_{cp}$")
ax.set_title("$k_{cp}$ versus Month of the Season")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax.set_xlim(left=datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1),
            right=datetime.datetime(year=starting_year+1, month=BEGINNING_MONTH, day=1))
major_xticks = pd.date_range(start=datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1),
                             end=datetime.datetime(year=starting_year + 1, month=BEGINNING_MONTH, day=1), freq="MS")
# Notice that the `MS` alias stands for `Month Start Frequency`.  E.g.: Jul/01, Aug/01, etc...
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
# gives a full list of frequency aliases.
ax.set_xticks(major_xticks)
ax.set_ylim(bottom=0.0, top=KCP_MAX)
ax.grid(True)
for i, p in enumerate(outer_index):  # iterate over the datasets corresponding to the different probes
    dates, kcp = get_dates_and_kcp(dataframe=cleaned_multi_df, probe_id=p)
    meta = next(zipped_meta)
    ax.scatter(dates, kcp, marker=meta[0], color=meta[1], s=60, edgecolors="black", linewidth=1, alpha=0.5,
               label=probe_ids[i])
ax.scatter(cco_df.index, cco_df["cco"], color="yellow", marker=".", linewidth=1, alpha=1.0,
           label="Reference $k_{cp}$")
ax.legend()
fig.autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig("./figures/overlay.png")
plt.cla()
plt.clf()
plt.close()
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Make a plot of etcp for each probe on a single set of axes.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Reset the cycle
zipped_meta = cycle([(m, c) for m, c in zip(marker_list, color_list)])

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Date")
ax.set_ylabel("$ET_{cp}$ [mm]")
ax.set_title("$ET_{cp}$ versus Date")
beginning_dates = []
end_dates = []
for p in probe_ids:
    meta = next(zipped_meta)
    df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(p), header=0, index_col=0,
                       parse_dates=True)
    condition = df["binary_value"] == 0
    useful_dates = date_wrapper(df[condition].index)
    ax.scatter(useful_dates, df.loc[useful_dates, "etcp"], marker=meta[0], color=meta[1], s=60, label=p,
               alpha=0.5, edgecolors="black")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
major_xticks = get_labels(begin=datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1),
                          terminate=datetime.datetime(year=starting_year+1, month=BEGINNING_MONTH, day=1))
ax.set_xticks(major_xticks)
ax.set_xlim(left=datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1),
            right=datetime.datetime(year=starting_year+1, month=BEGINNING_MONTH, day=1))
ax.legend()
ax.grid()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("./figures/etcp_versus_date.png")
plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Save new data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Save the `starting_year` to `./data/starting_year.txt`
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
with open("./data/starting_year.txt", "w") as f:
    f.write(str(starting_year))
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
