import os
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cleaning_operations import KCP_MAX
import pandas as pd
from cleaning_operations import description_dict
import math
import helper_functions as hf
import helper_meta_data as hm
import helper_data as hd


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the cleaned data garnered in `step_1_perform_cleaning.py`.
cleaned_multi_df = hd.cleaned_multi_df.copy(deep=True)
outer_index = hd.outer_index[:]
inner_index = hd.inner_index[:]

# Get a list of all the Probe-IDs involved for the cultivar
probe_ids = hm.probe_ids
num_probes = len(probe_ids)

# Create a folder for each probe-id in the `figures` parent folder, if it
# does not already exist.
for p in probe_ids:
    if not os.path.exists("./figures/{}/".format(p)):
        os.makedirs("./figures/{}/".format(p))

processed_eg_df = hd.processed_eg_df.copy(deep=True)

# Get the `starting_year`, `season_start_date`, `season_end_date`.
starting_year = hm.starting_year
season_start_date = hm.season_start_date
season_end_date = hm.season_end_date

# Extract the data of the smoothed kcp trendline.
fn = "./data/smoothed_kcp_trend_vs_datetime.xlsx"
skcp_vs_dt_df = pd.read_excel(fn, sheet_name=0, index_col=0, squeeze=False,
                              parse_dates=True)
skcp_vs_dt_df["days"] = skcp_vs_dt_df.index - season_start_date
skcp_vs_dt_df["days"] = skcp_vs_dt_df["days"].dt.days
x_smoothed = skcp_vs_dt_df["days"].values
y_smoothed = skcp_vs_dt_df["smoothed_kcp_trend"].values
x_smoothed_dates = skcp_vs_dt_df.index.values
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Define/Declare some "constants"
# =============================================================================
api_start_date = hm.api_start_date
api_end_date = hm.api_end_date
season_xticks = hd.season_xticks[:]
api_xticks = hd.api_xticks[:]
marker_color_meta = hm.marker_color_meta[:]
marker_color_meta = cycle(marker_color_meta)
vline_dates = hd.vline_dates[:]

num_cols = 2
num_rows = int(math.ceil(len(probe_ids) / num_cols))
# =============================================================================


# =============================================================================
# Plot array of cleaned kcp for all the probes along with the smoothed trend.
# In each plot, annotate the r-squared value.
# The figure is saved at "./figures/array_kcp.png".
# TODO: Populate the directory "./figures/kcp/kcp_<probe_id>.png".
# =============================================================================
print("Plot the cleaned kcp data, and original trendline, for each probe.")
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7.07, 10))
axs = axs.flatten()
fig.delaxes(axs[-1])
plt.subplots_adjust(wspace=0.05)
fig.suptitle("$k_{cp}$ versus Date")
fig.autofmt_xdate()

for idx, ax in enumerate(axs):
    meta = next(marker_color_meta)
    ax.set_ylim(bottom=0.0, top=KCP_MAX)
    ax.grid(True)
    dates, kcp = hf.get_dates_and_kcp(dataframe=cleaned_multi_df,
                                      probe_id=probe_ids[idx])
    days = pd.Series(dates - season_start_date).dt.days.values
    r_squared = hf.get_r_squared(x_raw=days, y_raw=kcp, x_fit=x_smoothed,
                                 y_fit=y_smoothed)
    ax.scatter(dates, kcp, marker=meta[0], color=meta[1], s=20,
               edgecolors="black", linewidth=1, alpha=0.5,
               label=probe_ids[idx])
    ax.plot(x_smoothed_dates, y_smoothed, linewidth=1.5, alpha=0.75,
            label="Smoothed")
    ax.tick_params(which="major", bottom=True, labelbottom=True,
                   colors="black", labelcolor="black", labelsize="small",
                   axis="x")
    ax.set_xticks(season_xticks)
    ax.set_xticklabels(season_xticks, rotation=40, ha="right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(left=season_start_date, right=season_end_date)
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
    ax.legend(prop={"size": 6}, loc=6)
    ax.annotate(s="$R^2$ = {:.3f}".format(r_squared), xycoords="axes fraction",
                xy=(0.01, 0.87), fontsize=9)
    if idx == (num_probes - 1):
        break
for idx in [0, 2, 4, 6]:
    axs[idx].set_ylabel("$k_{cp}$")
for idx in [5, 6]:
    axs[idx].set_xlabel("Month of the Season")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./figures/array_kcp.png")
plt.close()
# =============================================================================


# -----------------------------------------------------------------------------
# Make array of Total Irrigation Plots.
# This figure is saved at "./figures/array_irrigation.png".
# TODO: Populate directory "./figures/irrigation/irrigation_<probe_id>.png".
# -----------------------------------------------------------------------------
print("Produce a total irrigation plot for each probe.")
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7.07, 10))
axs = axs.flatten()
fig.delaxes(axs[-1])
plt.subplots_adjust(wspace=0.05)
fig.suptitle("Total Irrigation versus Date")
fig.autofmt_xdate()
marker_color_meta = hm.marker_color_meta[:]
marker_color_meta = cycle(marker_color_meta)

for idx, ax in enumerate(axs):
    df = pd.read_excel("./data/processed_probe_data.xlsx",
                       sheet_name=probe_ids[idx], header=0, index_col=0,
                       parse_dates=True)
    ax.grid(True)
    ax.bar(df.index, df["total_irrig"].values, color="green",
           label="{}".format(probe_ids[idx]))
    indices_irr = df["description"].str.contains(description_dict["irr_desc"],
                                                 na=False)
    ax.scatter(indices_irr.index[indices_irr],
               df.loc[indices_irr, ["total_irrig"]], color="black", marker="o",
               s=5, alpha=0.7, edgecolors="black")
    for v in vline_dates:
        ax.axvline(x=v, linewidth=2, color="magenta", alpha=0.4, ls="--")
    ax.plot([], [], linewidth=2, color="magenta", label="New Season",
            alpha=0.4, ls="--")
    ax.tick_params(which="major", bottom=True, labelbottom=True,
                   colors="black", labelcolor="black", labelsize="small",
                   axis="x")
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
# -----------------------------------------------------------------------------


# =============================================================================
# Array of water profile plots.
# The figure is saved at "./figures/array_profile.png".
# TODO: Populate directory "./figures/profile/profile_<probe_id>.png".
# =============================================================================
print("Produce a water profile plot for each probe.")
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7.07, 10))
axs = axs.flatten()
fig.delaxes(axs[-1])
plt.subplots_adjust(wspace=0.05)
fig.suptitle("Profile Reading versus Date")
fig.autofmt_xdate()

for idx, ax in enumerate(axs):
    df = pd.read_excel("./data/processed_probe_data.xlsx",
                       sheet_name=probe_ids[idx], header=0, index_col=0,
                       parse_dates=True)
    ax.grid(True)
    ax.plot(df.index, df["profile"], label=probe_ids[idx], lw=1, alpha=0.6)
    DESC = description_dict["data_blip_desc"]
    indices_data_blip = df["description"].str.contains(DESC, na=False)
    DESC = description_dict["large_profile_dip_desc"]
    indices_large_dips = df["description"].str.contains(DESC, na=False)
    ax.scatter(x=indices_data_blip.index[indices_data_blip],
               y=df.loc[indices_data_blip, "profile"], s=30,
               color="black", marker="*", label="Blips", edgecolors="red")
    ax.scatter(x=indices_large_dips.index[indices_large_dips],
               y=df.loc[indices_large_dips, "profile"], s=30,
               color="black", marker="X", label="Dips", edgecolors="green")
    for v in vline_dates:
        ax.axvline(x=v, linewidth=2, color="magenta", alpha=0.4, ls="--")
    ax.plot([], [], linewidth=1, color="magenta", label="New Season",
            alpha=0.4, ls="--")
    ax.tick_params(which="major", bottom=True, labelbottom=True,
                   colors="black", labelcolor="black", labelsize="small",
                   axis="x")
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
# =============================================================================
