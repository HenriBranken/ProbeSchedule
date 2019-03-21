import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from cleaning_operations import BEGINNING_MONTH, KCP_MAX
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from cleaning_operations import description_dict
register_matplotlib_converters()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the serialised data that were saved in `main.py`, and unpickle it
with open("data/data_to_plot", "rb") as f:
    data_to_plot = pickle.load(f)

# Get a list of all the Probe-IDs involved for the cultivar
with open("data/probe_ids.txt", "r") as f2:
    probe_ids = f2.readlines()
probe_ids = [x.strip() for x in probe_ids]

for probe_id in probe_ids:
    if not os.path.exists("./figures/{}/".format(probe_id)):
        os.makedirs("./figures/{}/".format(probe_id))

processed_eg_df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=0, header=0, index_col=0, squeeze=True,
                                parse_dates=True)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Define all the helper functions
# ----------------------------------------------------------------------------------------------------------------------
# Get the year of the oldest (the most "historic") data point
def get_starting_year():
    date_values, _ = separate_dates_from_kcp(data_to_plot[0])
    return date_values[0].year


# Extract dates and kcp values from tuple
# Sort according to datetime
# Return sorted dates and associated kcp values that also got sorted in the process
# Note that wrapping has already occurred previously by calling cleaning_operations.get_final_dates(df) in `main.py`
def separate_dates_from_kcp(tuple_object):
    date_values = tuple_object[0]
    kcp_values = tuple_object[1]
    df_temp = pd.DataFrame({"date_values": date_values, "kcp_values": kcp_values})
    df_temp.sort_values(by="date_values", axis=0, inplace=True)
    unified = df_temp.values
    return unified[:, 0], unified[:, 1]  # column 0 is sorted dates, and column 1 stores kcp values


def get_labels(begin, terminate):
    return [x for x in pd.date_range(start=begin, end=terminate, freq="MS")]
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
beginning_datetime = datetime.datetime(year=get_starting_year(), month=BEGINNING_MONTH, day=1)
end_datetime = datetime.datetime(year=get_starting_year() + 1, month=BEGINNING_MONTH, day=1)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Plot the cleaned probe data for each probe on a separate figure.
# ======================================================================================================================
if not os.path.exists("./figures"):
    os.makedirs("figures")

starting_year = get_starting_year()  # extract the starting year

# Create some meta data that will be used in the upcoming scatter plots
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]
zipped_meta = ((m, c) for m, c in zip(marker_list, color_list))  # here we used generator comprehension


for i in range(len(data_to_plot)):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Month of the Year")
    ax.set_ylabel("$k_{cp}$")
    ax.set_title("$k_{cp}$ versus Month of the Year.")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
    ax.set_xlim(left=beginning_datetime, right=end_datetime)
    major_xticks = get_labels(begin=beginning_datetime, terminate=end_datetime)
    ax.set_xticks(major_xticks)
    ax.set_ylim(bottom=0.0, top=KCP_MAX)
    ax.grid(True)
    dates, kcp = separate_dates_from_kcp(data_to_plot[i])  # extract the data from the zipped object, and sort by date
    meta = next(zipped_meta)
    ax.scatter(dates, kcp, color=meta[1], marker=meta[0], s=60, edgecolors="black", linewidth=1, alpha=0.5,
               label=probe_ids[i])
    ax.legend()
    fig.autofmt_xdate()  # rotate and align the tick labels so they look better
    plt.tight_layout()
    plt.savefig("figures/{}/cleaned_probe_data.png".format(probe_ids[i]))
    plt.close()
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot the Profile Readings (versus date) for each probe on a separate plot.
# For each plot, indicate the discrete points where the following occur:
#   1.  Data Blips --> DATA_BLIP_DESC
#   2.  Large Dips --> LARGE_PROFILE_DIP_DESC
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for probe_id in probe_ids:
    df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(probe_id), header=0, index_col=0,
                       squeeze=True, parse_dates=True)
    indices_data_blip = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
    indices_large_dips = df["description"].str.contains(description_dict["large_profile_dip_desc"], na=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["profile"], label="Daily Profile reading", lw=1, alpha=0.6)
    ax.scatter(x=indices_data_blip.index[indices_data_blip], y=df.loc[indices_data_blip, "profile"], s=50,
               color="black", marker="*", label="Data blips", edgecolors="red")
    ax.scatter(x=indices_large_dips.index[indices_large_dips], y=df.loc[indices_large_dips, "profile"], s=50,
               color="black", marker="X", label="'Large' Dips", edgecolors="green")
    major_xticks = get_labels(begin=df.index[0], terminate=df.index[-1])
    ax.set_xticks(major_xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
    ax.set_xlabel("(Running) Date")
    ax.set_ylabel("Profile Reading")
    ax.set_title("Profile Reading versus Date for Probe {}.".format(probe_id))
    for v in vline_dates:
        ax.axvline(x=v, linewidth=3, color="magenta", alpha=0.4, ls="--")
    ax.plot([], [], linewidth=3, color="magenta", label="New Season", alpha=0.4, ls="--")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.savefig("figures/{}/profile.png".format(probe_id))
    plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# For each probe, make a plot of the irrigation.
# ======================================================================================================================
for probe_id in probe_ids:
    df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(probe_id), header=0, index_col=0,
                       squeeze=True, parse_dates=True)
    indices_irr = df["description"].str.contains(description_dict["irr_desc"], na=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df.index, df["total_irrig"], color="magenta", label="Irrigation")
    ax.scatter(indices_irr.index[indices_irr], df.loc[indices_irr, ["total_irrig"]], label="Flagged Irr. events",
               color="black", marker="o", s=5, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Irrigation [mm]")
    ax.set_title("Total Irrigation versus Time for Probe {}.".format(probe_id))
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
    plt.savefig("figures/{}/irrigation.png".format(probe_id))
    plt.close()
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Plot the Rain events versus the Date.  They are identical for the different probes on a farm.
# Therefore only one probe_id is needed.
# ----------------------------------------------------------------------------------------------------------------------
probe_id = probe_ids[0]
df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(probe_id), header=0, index_col=0,
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
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["eto"], label="(Imputed) $ET_0$", lw=2, alpha=0.6, color="green")
ax.plot(df.index, df["etc"], label="Flagged $ET_c$", lw=2, alpha=0.6, color="blue")
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
plt.savefig("figures/et.png")
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


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Make a plot of etcp for each probe on a single set of axes.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]
zipped_meta = ((m, c) for m, c in zip(marker_list, color_list))  # here we used generator comprehension

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Date")
ax.set_ylabel("$ET_{cp}$ [mm]")
ax.set_title("$ET_{cp}$ versus Date")
beginning_dates = []
end_dates = []
for probe_id in probe_ids:
    meta = next(zipped_meta)
    df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(probe_id), header=0, index_col=0,
                       parse_dates=True)
    condition = df["binary_value"] == 0
    useful_dates = df[condition].index
    beginning_dates.append(useful_dates[0])
    end_dates.append(useful_dates[-1])
    ax.scatter(useful_dates, df.loc[useful_dates, "etcp"], marker=meta[0], color=meta[1], s=30, label=probe_id,
               alpha=0.5, edgecolors="black")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
beginning_date, end_date = min(beginning_dates), max(end_dates)
major_xticks = get_labels(begin=beginning_date, terminate=end_date)
ax.set_xticks(major_xticks)
ax.set_xlim(left=beginning_date, right=end_date)
ax.legend()
ax.grid()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("./figures/etcp_versus_date.png")
plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
