import os
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import matplotlib.dates as mdates
from cleaning_operations import KCP_MAX
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from cleaning_operations import description_dict
import helper_functions as hf
import helper_meta_data as hm
import helper_data as hd
from helper_functions import safe_removal
register_matplotlib_converters()


# =============================================================================
# Define some constants
# =============================================================================
# Create some meta data that will be used in the upcoming scatter plots
marker_color_meta = hm.marker_color_meta[:]
marker_color_meta = cycle(marker_color_meta)
t_base = hm.temperature_base
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the cleaned data garnered in `step_1_perform_cleaning.py`.
cleaned_multi_df = hd.cleaned_multi_df.copy(deep=True)
outer_index = hd.outer_index[:]
inner_index = hd.inner_index[:]

# Get a list of all the Probe-IDs involved for the cultivar
probe_ids = hm.probe_ids[:]

# Make a directory for each probe in the "./figures/" parent directory.
for p in probe_ids:
    if not os.path.exists("./figures/{}/".format(p)):
        os.makedirs("./figures/{}/".format(p))

processed_eg_df = hd.processed_eg_df.copy(deep=True)

# Get the starting year of the crop data
starting_year = hm.starting_year
season_start_date = hm.season_start_date
season_end_date = hm.season_end_date

# Get the mode of the fitting procedure used in `step_3_smoothed_version.py`,
# whether it be "WMA" or "Polynomial-fit".
with open("./data/mode.txt", "r") as f:
    mode = f.readline().rstrip()

# Extract the data of the smoothed kcp trendline.
fn = "./probe_screening/pruned_kcp_vs_days.xlsx"
smoothed_kcp_trend_df = pd.read_excel(fn, index_col=0, squeeze=False)
x_smoothed = smoothed_kcp_trend_df["x_smoothed"].values
y_smoothed = smoothed_kcp_trend_df["y_smoothed"].values
x_smoothed_dates = list(pd.date_range(start=season_start_date,
                                      end=season_end_date, freq="D"))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define the vline dates marking the beginning of a new season in the figures.
# The beginning of different seasons are just 1 year apart.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
vline_dates = hd.vline_dates[:]
season_start_date = hm.season_start_date
season_end_date = hm.season_end_date
season_xticks = hd.season_xticks[:]
api_start_date = hm.api_start_date
api_end_date = hm.api_end_date
api_xticks = hd.api_xticks[:]
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Remove old figures.
# -----------------------------------------------------------------------------
file_list = ["./figures/rain.png", "./figures/et.png",
             "./figures/gdd_heat_units_vs_time.png",
             "./figures/temp_and_heat_units.png"]
safe_removal(file_list=file_list)

for p in probe_ids:
    files = os.listdir("./figures/{:s}/".format(p))
    for f in files:
        if f.endswith(".png"):
            os.remove("./figures/{:s}/{:s}".format(p, f))
            print("Removed the file: \"{}\".".format(f))
# -----------------------------------------------------------------------------


# =============================================================================
# Plot the cleaned probe data for each probe on a separate figure.
# Figures are saved at "./figures/<probe_id>/cleaned_probe_data.png"
# =============================================================================
print("Plotting cleaned probe data for each probe on a separate set of axes.")
if not os.path.exists("./figures"):
    os.makedirs("figures")

for i, p in enumerate(outer_index):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Month of the Season")
    ax.set_ylabel("$k_{cp}$")
    ax.set_title("$k_{cp}$ versus Month of the Season.")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
    ax.set_xlim(left=season_start_date, right=season_end_date)
    ax.set_xticks(season_xticks)
    ax.set_ylim(bottom=0.0, top=KCP_MAX)
    ax.grid(True)
    dates, kcp = hf.get_dates_and_kcp(dataframe=cleaned_multi_df, probe_id=p)
    season_day = pd.Series(dates - season_start_date).dt.days.values
    r_squared = hf.get_r_squared(x_raw=season_day, y_raw=kcp, x_fit=x_smoothed,
                                 y_fit=y_smoothed)
    dates, kcp = hf.get_dates_and_kcp(dataframe=cleaned_multi_df, probe_id=p)
    ax.scatter(dates, kcp, marker="o", color="lightseagreen", s=20,
               edgecolors="black", linewidth=1, alpha=0.75, label=p)
    ax.plot(x_smoothed_dates, y_smoothed, linewidth=1.5, alpha=0.75,
            label="Smoothed Trend", color="mediumvioletred")
    ax.legend()
    ax.annotate(s="$R^2$ = {:.3f}".format(r_squared), xycoords="axes fraction",
                xy=(0.01, 0.93))
    fig.autofmt_xdate()  # rotate and align the tick labels so they look better
    plt.tight_layout()
    plt.savefig("./figures/{}/cleaned_probe_data.png".format(probe_ids[i]))
    plt.close()
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot the Profile Readings (versus date) for each probe on a separate plot.
# For each plot, indicate the discrete points where the following occur:
# 1. Data Blips --> DATA_BLIP_DESC
# 2. Large Dips --> LARGE_PROFILE_DIP_DESC
# Figures are saved at "./figures/<probe_id>/profile.png".
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("Plotting Profile Readings for each probe on a separate set of axes.")
for p in probe_ids:
    df = pd.read_excel("./data/processed_probe_data.xlsx",
                       sheet_name="{}".format(p), header=0, index_col=0,
                       squeeze=True, parse_dates=True)
    DESC = description_dict["data_blip_desc"]
    condition = df["description"].str.contains(DESC, na=False)
    data_blip_dates = df[condition].index
    DESC = description_dict["large_profile_dip_desc"]
    condition = df["description"].str.contains(DESC, na=False)
    large_dip_dates = df[condition].index
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["profile"], label="Daily Profile reading", lw=1.5,
            alpha=0.6)
    ax.scatter(x=data_blip_dates, y=df.loc[data_blip_dates, "profile"], s=100,
               color="red", marker="*", label="Data blips", edgecolors="k",
               alpha=0.5)
    ax.scatter(x=large_dip_dates,
               y=df.loc[large_dip_dates, "profile"], s=100, color="green",
               marker="X", label="'Large' Dips", edgecolors="black",
               alpha=0.5)
    ax.set_xticks(api_xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
    ax.set_xlabel("Date")
    ax.set_ylabel("Profile Reading")
    ax.set_title("Profile Reading versus Date for Probe {}.".format(p))
    for v in vline_dates:
        ax.axvline(x=v, linewidth=3, color="magenta", alpha=0.4, ls="--")
    ax.plot([], [], linewidth=3, color="magenta", label="New Season",
            alpha=0.4, ls="--")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.savefig("./figures/{}/profile.png".format(p))
    plt.close()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# For each probe, make a plot of the irrigation.
# Figures are saved at "./figures/<probe_id>/irrigation.png".
# =============================================================================
print("Plot the irrigation for each probe on a separate set of axes.")
for p in probe_ids:
    df = pd.read_excel("./data/processed_probe_data.xlsx",
                       sheet_name="{}".format(p), header=0, index_col=0,
                       squeeze=True, parse_dates=True)
    indices_irr = df["description"].str.contains(description_dict["irr_desc"],
                                                 na=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df.index, df["total_irrig"].values, color="green", label=p,
           width=1.6, edgecolor="black", linewidth=0.5, alpha=0.5)
    ax.scatter(indices_irr.index[indices_irr],
               df.loc[indices_irr, ["total_irrig"]],
               label="Flagged Irr. events", color="blue", marker="o",
               s=20, alpha=0.7, edgecolors="black")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Irrigation [mm]")
    ax.set_title("Total Irrigation versus Time for Probe {}.".format(p))
    for v in vline_dates:
        ax.axvline(x=v, color="magenta", ls="--", linewidth=3, alpha=0.4)
    ax.plot([], [], color="magenta", ls="--", linewidth=3, alpha=0.4,
            label="New Season")
    ax.set_xticks(api_xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
    ax.grid()
    ax.legend()
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.savefig("./figures/{}/irrigation.png".format(p))
    plt.close()
# =============================================================================


# -----------------------------------------------------------------------------
# Plot the Rain events versus the Date.
# They are identical for the different probes on a farm.
# Therefore only one probe_id is needed.
# The figure is saved at "./figures/rain.png".
# -----------------------------------------------------------------------------
print("Plot the Rain events.")
p = probe_ids[0]
df = pd.read_excel("./data/processed_probe_data.xlsx",
                   sheet_name="{}".format(p), header=0, index_col=0,
                   squeeze=True, parse_dates=True)
indices_rain = df["description"].str.contains(description_dict["rain_desc"],
                                              na=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["rain"], label="Daily Rain Reading", lw=1, alpha=0.6)
ax.scatter(indices_rain.index[indices_rain], y=df.loc[indices_rain, "rain"],
           s=30, color="black", marker="^",
           label=description_dict["rain_desc"], edgecolors="blue")
ax.set_xticks(api_xticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
for v in vline_dates:
    ax.axvline(x=v, linewidth=3, color="magenta", alpha=0.4, ls="--")
ax.plot([], [], linewidth=3, color="magenta", label="New Season", alpha=0.4,
        ls="--")
ax.set_xlabel("Date")
ax.set_ylabel("Rain [mm]")
ax.set_title("Rain reading versus Date.")
ax.grid()
ax.legend()
plt.tight_layout()
fig.autofmt_xdate()
plt.savefig("figures/rain.png")
plt.close()
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EvapoTranspiration versus Date.
# Only 1 graph is sufficient, therefore we only use 1 probe_id.
# The figure is saved at "./figures/et.png".
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("Plot the evapotranspiration data.")
interim_df = pd.DataFrame(data={"eto": df["eto"],
                                "etc": df["etc"],
                                "description": df["description"]},
                          index=df.index, copy=True)
interim_df.index.name = "datetimeStamp"
DESC = description_dict["etc_bad_desc"]
condition = interim_df["description"].str.contains(DESC, na=False)
bad_etc_dates = df[condition].index
interim_df.loc[bad_etc_dates, "etc"] = np.nan
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(interim_df.index, interim_df["eto"], label="(Imputed) $ET_0$", lw=2,
        alpha=0.6, color="green")
ax.plot(interim_df.index, interim_df["etc"], label="Flagged $ET_c$", lw=2,
        alpha=0.6, color="blue")
ax.set_xticks(api_xticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
for v in vline_dates:
    ax.axvline(x=v, linewidth=3, color="magenta", alpha=0.4, ls="--")
ax.plot([], [], linewidth=3, color="magenta", alpha=0.4, linestyle="--",
        label="New Season")
ax.set_xlabel("Date")
ax.set_ylabel("Evapotranspiration")
ax.set_title("$ET_c$ and $ET_o$ after flagging and imputation.")
ax.grid()
ax.legend()
plt.tight_layout()
fig.autofmt_xdate()
plt.savefig("./figures/et.png")
plt.close()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Heat Units and GDD versus Time
# Only 1 graph is sufficient, therefore we only use 1 probeID.
# The figure is saved at "./figures/GDD_heat_units_vs_time.png".
# =============================================================================
print("Plot Heat Units and GDD data.")
fig = plt.figure()
fig.set_size_inches(10, 5)

ax1 = fig.add_subplot(111)
color = "blue"
ax1.set_title("Heat Units and GDD versus Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Heat Units", color=color)
pl1 = ax1.bar(processed_eg_df.index, processed_eg_df["interpolated_hu"],
              color=color, label="Interpolated H.U.", alpha=0.6, width=1.0)
ax1.tick_params(axis="y", labelcolor=color)
for v in vline_dates:
    ax1.axvline(x=v, linewidth=3, linestyle="--", color="magenta", alpha=0.4)
ax1.plot([], [], linewidth=3, linestyle="--", color="magenta", alpha=0.4,
         label="New Season")
ax1.set_xlim(left=api_start_date, right=api_end_date)
ax1.set_xticks(api_xticks)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
fig.autofmt_xdate()

ax2 = ax1.twinx()
color = "green"
ax2.set_ylabel("Cumulative GDD", color=color)
pl2 = ax2.plot(processed_eg_df.index, processed_eg_df["cumulative_gdd"],
               color=color, label="Cumulative GDD", lw=2, alpha=1.0)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim(bottom=0)
ax2.set_xticks(api_xticks)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
fig.autofmt_xdate()

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

fig.tight_layout()
plt.savefig("figures/gdd_heat_units_vs_time.png")
plt.close()
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Make a plot of the data:
# 1. (T_min + T_max)/2.0
# 2. hline of T_base (which is 10 degrees Celsius for Apples).
# 3. Heat Units.
# The figure is saved at "./figures/temp_and_heat_units.png".
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("Plot temperature and heat-units data.")
t_min = processed_eg_df["T_min"].values
t_max = processed_eg_df["T_max"].values
t_avg = (t_min + t_max)/2.0
date_stamp = processed_eg_df.index

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 7.07))
fig.autofmt_xdate()
axs = axs.flatten()
plt.subplots_adjust(hspace=0.35)

axs[0].set_ylabel("Temperature, [Celsius]")
axs[0].grid(True)
axs[0].set_title("Average Temperature versus Date")
axs[0].plot(date_stamp, t_avg, color="forestgreen",
            label="0.5*(T_min + T_max)", lw=1)
axs[0].fill_between(date_stamp, 10, t_avg, where=t_avg <= 10,
                    facecolor="powderblue", interpolate=True,
                    label="Heat Units = 0")
axs[0].fill_between(date_stamp, 10, t_avg, where=t_avg >= 10,
                    facecolor="lightcoral", interpolate=True,
                    label="Heat Units > 0")
axs[0].axhline(y=10, color="black", label="Base Temperature", lw=2.5, ls="--")
for v in vline_dates:
    axs[0].axvline(x=v, color="magenta", ls="--", lw=2.5, alpha=0.5)
axs[0].plot([], [], color="magenta", ls="--", lw=2.5, alpha=0.7,
            label="New Season")
axs[0].tick_params(which="major", bottom=True, labelbottom=True,
                   colors="black", labelcolor="black", labelsize="small",
                   axis="x")
for tick in axs[0].get_xticklabels():
    tick.set_visible(True)
axs[0].set_xticks(api_xticks)
axs[0].set_xticklabels(api_xticks, rotation=40, ha="right")
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
axs[0].set_xlim(left=api_start_date, right=api_end_date)
axs[0].legend()

condition = processed_eg_df["heat_units"] == 0
zero_hu_dates = processed_eg_df[condition].index
axs[1].bar(processed_eg_df.index, processed_eg_df["interpolated_hu"],
           color="lightcoral", label="Heat Units", alpha=1.0, width=1.0)
for v in vline_dates:
    axs[1].axvline(x=v, linestyle="--", linewidth=2.5, color="magenta",
                   alpha=0.5)
for zero_hu_date in zero_hu_dates:
    axs[1].axvline(x=zero_hu_date, linestyle="-", linewidth=1,
                   color="powderblue", alpha=1.0)
axs[1].plot([], [], linestyle="--", linewidth=2.5, color="magenta", alpha=0.5,
            label="New Season")
axs[1].plot([], [], linestyle="-", linewidth=1, color="powderblue", alpha=1.0,
            label="ZERO Heat Units")
axs[1].tick_params(which="major", bottom=True, labelbottom=True,
                   colors="black", labelcolor="black", labelsize="small",
                   axis="x")
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
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
