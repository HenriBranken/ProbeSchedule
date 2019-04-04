import os
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import matplotlib.dates as mdates
import datetime
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from cleaning_operations import description_dict
register_matplotlib_converters()


# ======================================================================================================================
# Define some constants
# ======================================================================================================================
# Create some meta data that will be used in the upcoming scatter plots
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]
zipped_meta = [(m, c) for m, c in zip(marker_list, color_list)]
zipped_meta = cycle(zipped_meta)
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the cleaned data garnered in `step_1_perform_cleaning.py`.
cleaned_multi_df = pd.read_excel("./data/stacked_cleaned_data_for_overlay.xlsx", header=0, index_col=[0, 1],
                                 parse_dates=True)
outer_index = list(cleaned_multi_df.index.get_level_values("probe_id").unique())
inner_index = list(cleaned_multi_df.index.get_level_values("datetimeStamp").unique())

# Get a list of all the Probe-IDs involved for the cultivar
with open("./data/probe_ids.txt", "r") as f2:
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


# ----------------------------------------------------------------------------------------------------------------------
# Define all the helper functions
# ----------------------------------------------------------------------------------------------------------------------
def get_dates_and_kcp(dataframe, probe_id):
    sub_df = dataframe.loc[(probe_id, ), ["kcp"]]
    return sub_df.index, sub_df["kcp"].values


def get_labels(begin, terminate):
    return [x for x in pd.date_range(start=begin, end=terminate, freq="MS")]


def date_wrapper(date_iterable):
    new_dates = []
    for datum in date_iterable:
        extract_month = datum.month
        if BEGINNING_MONTH <= extract_month <= 12:
            new_dates.append(datetime.datetime(year=starting_year, month=extract_month, day=datum.day))
        else:
            new_dates.append(datetime.datetime(year=starting_year + 1, month=extract_month, day=datum.day))
    return new_dates
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define the vline date marking the beginning of a new season in the figures.
# A new season is just 1 year apart.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
eg_df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(probe_ids[0]), header=0, index_col=0,
                      squeeze=True, parse_dates=True)
vline_dates = []
for d in eg_df.index:
    if (d.month == BEGINNING_MONTH) and (d.day == 1):
        new_season_date = datetime.datetime(year=d.year, month=d.month, day=d.day)
        vline_dates.append(new_season_date)

vline_date = vline_dates[0]
beginning_datetime = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
end_datetime = datetime.datetime(year=starting_year + 1, month=BEGINNING_MONTH, day=1)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Plot the cleaned probe data for each probe on a separate figure.
# ======================================================================================================================
if not os.path.exists("./figures"):
    os.makedirs("figures")

for i, p in enumerate(outer_index):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Month of the Season")
    ax.set_ylabel("$k_{cp}$")
    ax.set_title("$k_{cp}$ versus Month of the Season.")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
    ax.set_xlim(left=beginning_datetime, right=end_datetime)
    major_xticks = get_labels(begin=beginning_datetime, terminate=end_datetime)
    ax.set_xticks(major_xticks)
    ax.set_ylim(bottom=0.0, top=KCP_MAX)
    ax.grid(True)
    dates, kcp = get_dates_and_kcp(dataframe=cleaned_multi_df, probe_id=p)
    meta = next(zipped_meta)
    ax.scatter(dates, kcp, marker=meta[0], color=meta[1], s=60, edgecolors="black", linewidth=1, alpha=0.5,
               label=probe_ids[i])
    ax.legend()
    fig.autofmt_xdate()  # rotate and align the tick labels so they look better
    plt.tight_layout()
    plt.savefig("./figures/{}/cleaned_probe_data.png".format(probe_ids[i]))
    plt.close()
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot the Profile Readings (versus date) for each probe on a separate plot.
# For each plot, indicate the discrete points where the following occur:
#   1.  Data Blips --> DATA_BLIP_DESC
#   2.  Large Dips --> LARGE_PROFILE_DIP_DESC
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for p in probe_ids:
    df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(p), header=0, index_col=0,
                       squeeze=True, parse_dates=True)
    condition = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
    data_blip_dates = df[condition].index
    condition = df["description"].str.contains(description_dict["large_profile_dip_desc"], na=False)
    large_dip_dates = df[condition].index
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["profile"], label="Daily Profile reading", lw=1, alpha=0.6)
    ax.scatter(x=data_blip_dates,
               y=df.loc[data_blip_dates, "profile"], s=50, color="black", marker="*",
               label="Data blips", edgecolors="red")
    ax.scatter(x=large_dip_dates,
               y=df.loc[large_dip_dates, "profile"], s=50, color="black", marker="X",
               label="'Large' Dips", edgecolors="green")
    major_xticks = get_labels(begin=df.index[0], terminate=df.index[-1])
    ax.set_xticks(major_xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
    ax.set_xlabel("(Running) Date")
    ax.set_ylabel("Profile Reading")
    ax.set_title("Profile Reading versus Date for Probe {}.".format(p))
    for v in vline_dates:
        ax.axvline(x=v, linewidth=3, color="magenta", alpha=0.4, ls="--")
    ax.plot([], [], linewidth=3, color="magenta", label="New Season", alpha=0.4, ls="--")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.savefig("./figures/{}/profile.png".format(p))
    plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# For each probe, make a plot of the irrigation.
# ======================================================================================================================
for p in probe_ids:
    df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(p), header=0, index_col=0,
                       squeeze=True, parse_dates=True)
    indices_irr = df["description"].str.contains(description_dict["irr_desc"], na=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df.index, df["total_irrig"], color="magenta", label="Irrigation")
    ax.scatter(indices_irr.index[indices_irr], df.loc[indices_irr, ["total_irrig"]], label="Flagged Irr. events",
               color="black", marker="o", s=5, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Irrigation [mm]")
    ax.set_title("Total Irrigation versus Time for Probe {}.".format(p))
    for v in vline_dates:
        ax.axvline(x=v, color="blue", ls="--", linewidth=3, alpha=0.4)
    ax.plot([], [], color="blue", ls="--", linewidth=3, alpha=0.4, label="New Season")
    major_xticks = get_labels(begin=df.index[0], terminate=df.index[-1])
    ax.set_xticks(major_xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
    ax.grid()
    ax.legend()
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.savefig("figures/{}/irrigation.png".format(p))
    plt.close()
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Plot the Rain events versus the Date.  They are identical for the different probes on a farm.
# Therefore only one probe_id is needed.
# ----------------------------------------------------------------------------------------------------------------------
p = probe_ids[0]
df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(p), header=0, index_col=0,
                   squeeze=True, parse_dates=True)
indices_rain = df["description"].str.contains(description_dict["rain_desc"], na=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["rain"], label="Daily Rain Reading", lw=1, alpha=0.6)
ax.scatter(indices_rain.index[indices_rain], y=df.loc[indices_rain, "rain"], s=30,
           color="black", marker="^", label=description_dict["rain_desc"], edgecolors="blue")
major_xticks = get_labels(begin=df.index[0], terminate=df.index[-1])
ax.set_xticks(major_xticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
for v in vline_dates:
    ax.axvline(x=v, linewidth=3, color="magenta", alpha=0.4, ls="--")
ax.plot([], [], linewidth=3, color="magenta", label="New Season", alpha=0.4, ls="--")
ax.set_xlabel("Date")
ax.set_ylabel("Rain [mm]")
ax.set_title("Rain reading versus Date.")
ax.grid()
ax.legend()
plt.tight_layout()
fig.autofmt_xdate()
plt.savefig("figures/rain.png")
plt.close()
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EvapoTranspiration versus Date.
# Only 1 graph is sufficient, therefore we only use 1 probeID.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
interim_df = pd.DataFrame(data={"eto": df["eto"], "etc": df["etc"], "description": df["description"]}, index=df.index,
                          copy=True)
interim_df.index.name = "datetimeStamp"

condition = interim_df["description"].str.contains(description_dict["etc_bad_desc"], na=False)
bad_etc_dates = df[condition].index
interim_df.loc[bad_etc_dates, "etc"] = np.nan
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(interim_df.index, interim_df["eto"], label="(Imputed) $ET_0$", lw=2, alpha=0.6, color="green")
ax.plot(interim_df.index, interim_df["etc"], label="Flagged $ET_c$", lw=2, alpha=0.6, color="blue")
major_xticks = get_labels(begin=df.index[0], terminate=df.index[-1])
ax.set_xticks(major_xticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
for v in vline_dates:
    ax.axvline(x=v, linewidth=3, color="magenta", alpha=0.4, ls="--")
ax.plot([], [], linewidth=3, color="magenta", alpha=0.4, linestyle="--", label="New Season")
ax.set_xlabel("Date")
ax.set_ylabel("Evapotranspiration")
ax.set_title("$ET_c$ and $ET_o$ after flagging and imputing Koue-Bokkeveld data.")
ax.grid()
ax.legend()
plt.tight_layout()
fig.autofmt_xdate()
plt.savefig("./figures/et.png")
plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Heat Units and GDD versus Time
# Only 1 graph is sufficient, therefore we only use 1 probeID.
# For this graph we use the gdd_cumulative function(s) helper function defined above.
# ======================================================================================================================
fig = plt.figure()
fig.set_size_inches(10, 5)

ax1 = fig.add_subplot(111)
color = "blue"
ax1.set_title("Heat Units and GDD versus Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Heat Units", color=color)
pl1 = ax1.bar(processed_eg_df.index, processed_eg_df["interpolated_hu"], color=color, label="Interpolated H.U.",
              alpha=0.5)
ax1.tick_params(axis="y", labelcolor=color)
for v in vline_dates:
    ax1.axvline(x=v, linewidth=3, linestyle="--", color="magenta", alpha=0.4)
ax1.plot([], [], linewidth=3, linestyle="--", color="magenta", alpha=0.4, label="New Season")
ax1.set_xlim(left=processed_eg_df.index[0], right=processed_eg_df.index[-1])
major_xticks = get_labels(begin=processed_eg_df.index[0], terminate=processed_eg_df.index[-1])
ax1.set_xticks(major_xticks)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
fig.autofmt_xdate()

ax2 = ax1.twinx()
color = "green"
ax2.set_ylabel("Cumulative GDD", color=color)
pl2 = ax2.plot(processed_eg_df.index, processed_eg_df["cumulative_gdd"], color=color, label="Cumulative GDD", lw=2,
               alpha=1.0)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim(bottom=0)
ax2.set_xticks(major_xticks)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
fig.autofmt_xdate()

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

fig.tight_layout()
plt.savefig("figures/GDD_heat_units_vs_time.png")
plt.close()
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Compare original heat_units versus the interpolated_hu
# ----------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col")

color = "red"
ax[0].bar(processed_eg_df.index, processed_eg_df["heat_units"], color=color, label="Base Heat Units", alpha=0.5)
for v in vline_dates:
    ax[0].axvline(x=v, linestyle="--", linewidth=2, color="cyan", alpha=0.5)
ax[0].plot([], [], linestyle="--", linewidth=2, color="cyan", alpha=0.5, label="New Season")
ax[0].set_ylabel("Heat Units")
ax[0].legend()

color = "green"
ax[1].bar(processed_eg_df.index, processed_eg_df["interpolated_hu"], color=color, label="Interpolated H.U.", alpha=0.5)
for v in vline_dates:
    ax[1].axvline(x=v, linestyle="--", linewidth=2, color="cyan", alpha=0.5)
ax[1].plot([], [], linestyle="--", linewidth=2, color="cyan", alpha=0.5, label="New Season")
ax[1].set_xlim(left=processed_eg_df.index[0], right=processed_eg_df.index[-1])
ax[1].set_ylabel("Heat Units")
ax[1].set_xlabel("Date")
ax[1].legend()

fig.autofmt_xdate()
fig.suptitle("Comparison between base Heat Units and interpolated Heat Units")
plt.savefig("./figures/heat_units_comparison.png")
plt.close()
# ----------------------------------------------------------------------------------------------------------------------
