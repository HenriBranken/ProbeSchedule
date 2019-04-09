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
t_base = 10.0
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

# Make a directory for each probe in the `figures` parent directory.
for p in probe_ids:
    if not os.path.exists("./figures/{}/".format(p)):
        os.makedirs("./figures/{}/".format(p))

processed_eg_df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=0, header=0, index_col=0, squeeze=True,
                                parse_dates=True)

# Get the starting year of the crop data
with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())

# Get the mode of the fitting procedure used in `step_3_smoothed_version.py`.
with open("./data/mode.txt", "r") as f:
    mode = f.readline().rstrip()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Define all the helper functions
# ----------------------------------------------------------------------------------------------------------------------
def get_dates_and_kcp(dataframe, probe_id):
    sub_df = dataframe.loc[(probe_id, ), ["kcp"]]
    return sub_df.index, sub_df["kcp"].values


def get_labels(begin, terminate, freq="MS"):
    return [x for x in pd.date_range(start=begin, end=terminate, freq=freq)]


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
api_start_date, api_end_date = min(processed_eg_df.index), max(processed_eg_df.index)
api_xticks = get_labels(begin=api_start_date, terminate=api_end_date, freq="QS")
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
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
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


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Make a plot of all the temperature curves:
# 1. T_min
# 2. T_max
# 3. T_24hour_avg
# 4. (T_min + T_max)/2.0
# 5. hline of T_base (which is 10 degrees Celsius for Apples).
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
t_min = processed_eg_df["T_min"].values
t_max = processed_eg_df["T_max"].values
t_24h_avg = processed_eg_df["T_24hour_avg"].values
t_avg = (t_min + t_max)/2.0
date_stamp = processed_eg_df.index

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 7.5))
fig.autofmt_xdate()
axs = axs.flatten()
plt.subplots_adjust(hspace=0.35)

axs[0].set_ylabel("Temperature, [Celsius]")
axs[0].grid(True)
axs[0].set_title("Average Temperature versus Date")
axs[0].plot(date_stamp, t_avg, color="forestgreen", label="0.5*(T_min + T_max)", lw=1)
axs[0].fill_between(date_stamp, 10, t_avg, where=t_avg <= 10, facecolor="powderblue", interpolate=True,
                    label="Heat Units = 0")
axs[0].fill_between(date_stamp, 10, t_avg, where=t_avg >= 10, facecolor="lightcoral", interpolate=True,
                    label="Heat Units > 0")
axs[0].axhline(y=10, color="black", label="Base Temperature", lw=2.5, ls="--")
for v in vline_dates:
    axs[0].axvline(x=v, color="magenta", ls="--", lw=2.5, alpha=0.5)
axs[0].plot([], [], color="magenta", ls="--", lw=2.5, alpha=0.7, label="New Season")
axs[0].tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                   labelsize="small", axis="x")
for tick in axs[0].get_xticklabels():
    tick.set_visible(True)
axs[0].set_xticks(api_xticks)
axs[0].set_xticklabels(api_xticks, rotation=40, ha="right")
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
axs[0].set_xlim(left=api_start_date, right=api_end_date)
axs[0].legend()

axs[1].bar(processed_eg_df.index, processed_eg_df["interpolated_hu"], color="darkgoldenrod", label="Heat Units",
           alpha=0.5)
for v in vline_dates:
    axs[1].axvline(x=v, linestyle="--", linewidth=2.5, color="magenta", alpha=0.5)
axs[1].plot([], [], linestyle="--", linewidth=2.5, color="magenta", alpha=0.5, label="New Season")
axs[1].tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                   labelsize="small", axis="x")
axs[1].grid(True)
axs[1].set_xticks(api_xticks)
axs[1].set_xticklabels(api_xticks, rotation=40, ha="right")
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
axs[1].set_xlim(left=api_start_date, right=api_end_date)
for tick in axs[1].get_xticklabels():
    tick.set_visible(True)
axs[1].set_ylabel("Heat Units")
axs[1].set_xlabel("Date")
axs[1].set_title("Heat Units versus Date")
axs[1].legend()

plt.tight_layout()
plt.savefig("./figures/temp_and_heat_units.png")
plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
