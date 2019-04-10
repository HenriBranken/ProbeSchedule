import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
import pandas as pd
from cleaning_operations import description_dict
import math
import numpy as np


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the cleaned data garnered in `step_1_perform_cleaning.py`.
cleaned_multi_df = pd.read_excel("./data/stacked_cleaned_data_for_overlay.xlsx", header=0, index_col=[0, 1],
                                 parse_dates=True)
outer_index = list(cleaned_multi_df.index.get_level_values("probe_id").unique())
inner_index = list(cleaned_multi_df.index.get_level_values("datetimeStamp").unique())

# Get a list of all the Probe-IDs involved for the cultivar
with open("../probe_ids.txt", "r") as f2:
    probe_ids = [x.rstrip() for x in f2.readlines()]
num_probes = len(probe_ids)

# Create a folder for each probe-id in the `figures` parent folder.
for p in probe_ids:
    if not os.path.exists("./figures/{}/".format(p)):
        os.makedirs("./figures/{}/".format(p))

processed_eg_df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=0, header=0, index_col=0, squeeze=True,
                                parse_dates=True)

# Extract the starting year of the crop data, and also declare the starting-date for the crop data.
with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())
start_date = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)

Smoothed_kcp_trend_vs_datetime_df = pd.read_excel("./data/Smoothed_kcp_trend_vs_datetime.xlsx", sheet_name=0,
                                                  index_col=0, squeeze=False, parse_dates=True)
Smoothed_kcp_trend_vs_datetime_df["days"] = Smoothed_kcp_trend_vs_datetime_df.index - start_date
Smoothed_kcp_trend_vs_datetime_df["days"] = Smoothed_kcp_trend_vs_datetime_df["days"].dt.days
x_smoothed = Smoothed_kcp_trend_vs_datetime_df["days"].values
y_smoothed = Smoothed_kcp_trend_vs_datetime_df["Smoothed_kcp_trend"].values
x_smoothed_dates = Smoothed_kcp_trend_vs_datetime_df.index.values
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Define/Declare some "constants"
# ======================================================================================================================
season_begin_date = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
season_end_date = datetime.datetime(year=starting_year + 1, month=BEGINNING_MONTH, day=1)
api_start_date, api_end_date = min(processed_eg_df.index), max(processed_eg_df.index)
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]

num_cols = 2
num_rows = int(math.ceil(len(probe_ids) / num_cols))
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Define all the helper functions
# ----------------------------------------------------------------------------------------------------------------------
def get_dates_and_kcp(dataframe, probe_id):
    sub_df = dataframe.loc[(probe_id, ), ["kcp"]]
    sub_df["days"] = sub_df.index - start_date
    sub_df["days"] = sub_df["days"].dt.days
    return sub_df.index, sub_df["days"].values, sub_df["kcp"].values


def get_labels(begin, terminate, freq="MS"):
    return [x for x in pd.date_range(start=begin, end=terminate, freq=freq)]


def find_nearest_index(model_array, raw_value):
    model_array = np.asarray(model_array)
    ii = (np.abs(model_array - raw_value)).argmin()
    return ii


def get_r_squared(x_raw, y_raw, x_fit, y_fit):
    indices = []
    for x in x_raw:
        indices.append(find_nearest_index(x_fit, x))
    y_proxies = []
    for j in indices:
        y_proxies.append(y_fit[j])
    y_bar = np.mean(y_raw)
    ssreg = np.sum((y_proxies - y_bar)**2)
    sstot = np.sum((y_raw - y_bar)**2)
    return ssreg/sstot


season_xticks = get_labels(begin=season_begin_date, terminate=season_end_date)
api_xticks = get_labels(begin=api_start_date, terminate=api_end_date, freq="QS")
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define the vline date marking the beginning of a new season in the figures.
# A new season is just 1 year apart from the previous season (obviously).
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
vline_dates = []
for d in processed_eg_df.index:
    if (d.month == BEGINNING_MONTH) and (d.day == 1):
        new_season_date = datetime.datetime(year=d.year, month=d.month, day=d.day)
        vline_dates.append(new_season_date)

vline_date = vline_dates[0]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Plot array of cleaned kcp for all the probes
# ======================================================================================================================
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7.07, 10))
axs = axs.flatten()
fig.delaxes(axs[-1])
plt.subplots_adjust(wspace=0.05)
fig.suptitle("$k_{cp}$ versus Date")
fig.autofmt_xdate()

zipped_meta = ((m, c) for m, c in zip(marker_list, color_list))  # here we used generator comprehension

for idx, ax in enumerate(axs):
    meta = next(zipped_meta)
    ax.set_ylim(bottom=0.0, top=KCP_MAX)
    ax.grid(True)
    dates, days, kcp = get_dates_and_kcp(dataframe=cleaned_multi_df, probe_id=probe_ids[idx])
    r_squared = get_r_squared(x_raw=days, y_raw=kcp, x_fit=x_smoothed, y_fit=y_smoothed)
    # print("r^2 = {:.4f}.".format(r_squared))
    ax.scatter(dates, kcp, marker=meta[0], color=meta[1], s=20, edgecolors="black", linewidth=1, alpha=0.5,
               label=probe_ids[idx])
    ax.plot(x_smoothed_dates, y_smoothed, linewidth=1.5, alpha=0.75, label="Smoothed")
    ax.tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                   labelsize="small", axis="x")
    ax.set_xticks(season_xticks)
    ax.set_xticklabels(season_xticks, rotation=40, ha="right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(left=season_begin_date, right=season_end_date)
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
    ax.legend(prop={"size": 6}, loc=6)
    ax.annotate(s="$R^2$ = {:.3f}".format(r_squared), xycoords="axes fraction", xy=(0.01, 0.87), fontsize=9)
    if idx == (num_probes - 1):
        break
for idx in [0, 2, 4, 6]:
    axs[idx].set_ylabel("$k_{cp}$")
for idx in [5, 6]:
    axs[idx].set_xlabel("Month of the Season")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./figures/array_kcp.png")
plt.close()
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Make array of Total Irrigation Plots
# ----------------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7.07, 10))
axs = axs.flatten()
fig.delaxes(axs[-1])
plt.subplots_adjust(wspace=0.05)
fig.suptitle("Total Irrigation versus Date")
fig.autofmt_xdate()
zipped_meta = ((m, c) for m, c in zip(marker_list, color_list))

for idx, ax in enumerate(axs):
    df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[idx], header=0, index_col=0,
                       parse_dates=True)
    ax.grid(True)
    ax.bar(df.index, df["total_irrig"].values, color="green", label="{}".format(probe_ids[idx]))
    indices_irr = df["description"].str.contains(description_dict["irr_desc"], na=False)
    ax.scatter(indices_irr.index[indices_irr], df.loc[indices_irr, ["total_irrig"]], color="black", marker="o",
               s=5, alpha=0.7, edgecolors="black")
    for v in vline_dates:
        ax.axvline(x=v, linewidth=2, color="magenta", alpha=0.4, ls="--")
    ax.plot([], [], linewidth=2, color="magenta", label="New Season", alpha=0.4, ls="--")
    ax.tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                   labelsize="small", axis="x")
    ax.set_xticks(api_xticks)
    ax.set_xticklabels(api_xticks, rotation=40, ha="right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
    ax.set_xlim(left=api_start_date, right=api_end_date)
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
    ax.legend(prop={"size": 6})
    if idx == (num_probes - 1):
        break
for idx in [0, 2, 4, 6]:
    axs[idx].set_ylabel("Irrigation")
for idx in [5, 6]:
    axs[idx].set_xlabel("Date")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./figures/array_irrigation.png")
plt.close()
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Array of water profile plots
# ======================================================================================================================
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7.07, 10))
axs = axs.flatten()
fig.delaxes(axs[-1])
plt.subplots_adjust(wspace=0.05)
fig.suptitle("Profile Reading versus Date")
fig.autofmt_xdate()

for idx, ax in enumerate(axs):
    df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[idx], header=0, index_col=0,
                       parse_dates=True)
    ax.grid(True)
    ax.plot(df.index, df["profile"], label=probe_ids[idx], lw=1, alpha=0.6)
    indices_data_blip = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
    indices_large_dips = df["description"].str.contains(description_dict["large_profile_dip_desc"], na=False)
    ax.scatter(x=indices_data_blip.index[indices_data_blip], y=df.loc[indices_data_blip, "profile"], s=30,
               color="black", marker="*", label="Blips", edgecolors="red")
    ax.scatter(x=indices_large_dips.index[indices_large_dips], y=df.loc[indices_large_dips, "profile"], s=30,
               color="black", marker="X", label="Dips", edgecolors="green")
    for v in vline_dates:
        ax.axvline(x=v, linewidth=2, color="magenta", alpha=0.4, ls="--")
    ax.plot([], [], linewidth=1, color="magenta", label="New Season", alpha=0.4, ls="--")
    ax.tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                   labelsize="small", axis="x")
    ax.set_xticks(api_xticks)
    ax.set_xticklabels(api_xticks, rotation=40, ha="right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
    ax.set_xlim(left=api_start_date, right=api_end_date)
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
    ax.legend(prop={"size": 6})
    if idx == (num_probes - 1):
        break
for idx in [0, 2, 4, 6]:
    axs[idx].set_ylabel("Profile")
for idx in [5, 6]:
    axs[idx].set_xlabel("Date")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./figures/array_profile.png")
plt.close()
# ======================================================================================================================
