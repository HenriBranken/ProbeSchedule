import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
import pandas as pd
from cleaning_operations import description_dict
import math


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the cleaned data garnered in `step_1_perform_cleaning.py`.
cleaned_multi_df = pd.read_excel("./data/stacked_cleaned_data_for_overlay.xlsx", header=0, index_col=[0, 1],
                                 parse_dates=True)
outer_index = list(cleaned_multi_df.index.get_level_values("probe_id").unique())
inner_index = list(cleaned_multi_df.index.get_level_values("datetimeStamp").unique())

# Get a list of all the Probe-IDs involved for the cultivar
with open("data/probe_ids.txt", "r") as f2:
    probe_ids = f2.readlines()
probe_ids = [x.strip() for x in probe_ids]

for p in probe_ids:
    if not os.path.exists("./figures/{}/".format(p)):
        os.makedirs("./figures/{}/".format(p))

processed_eg_df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=0, header=0, index_col=0, squeeze=True,
                                parse_dates=True)

with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())
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
num_rows = int(math.ceil(len(probe_ids) / 2))
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Define all the helper functions
# ----------------------------------------------------------------------------------------------------------------------
# Extract dates and kcp values from tuple
# Sort according to datetime
# Return sorted dates and associated kcp values that also got sorted in the process
# Note that wrapping has already occurred previously by calling cleaning_operations.get_final_dates(df) in `main.py`
def get_dates_and_kcp(dataframe, probe_id):
    sub_df = dataframe.loc[(probe_id, ), ["kcp"]]
    return sub_df.index, sub_df["kcp"].values


def get_labels(begin, terminate, freq="MS"):
    return [x for x in pd.date_range(start=begin, end=terminate, freq=freq)]


season_xticks = get_labels(begin=season_begin_date, terminate=season_end_date)
api_xticks = get_labels(begin=api_start_date, terminate=api_end_date, freq="QS")
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define the vline date marking the beginning of a new season in the figures.
# A new season is just 1 year apart.
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
    dates, kcp = get_dates_and_kcp(dataframe=cleaned_multi_df, probe_id=probe_ids[idx])
    ax.scatter(dates, kcp, marker=meta[0], color=meta[1], s=20, edgecolors="black", linewidth=1, alpha=0.5,
               label=probe_ids[idx])
    ax.tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                   labelsize="small", axis="x")
    ax.set_xticks(season_xticks)
    ax.set_xticklabels(season_xticks, rotation=40, ha="right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(left=season_begin_date, right=season_end_date)
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
    ax.legend(prop={"size": 6})
    if idx == 6:
        break

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
    if idx == 6:
        break

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
    if idx == 6:
        break

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./figures/array_profile.png")
plt.close()
# ======================================================================================================================
