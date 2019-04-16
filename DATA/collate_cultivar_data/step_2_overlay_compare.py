import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import cycle
from cleaning_operations import KCP_MAX
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from helper_functions import date_wrapper, get_dates_and_kcp, get_labels
import helper_meta_data as hm
import helper_data as hd
register_matplotlib_converters()


# =============================================================================
# Declare some necessary "constants"
# =============================================================================
# 1. Create some meta data that will be used in the upcoming scatter plots.
#    This meta data stores `marker` and `color` values.
# =============================================================================
marker_color_meta = hm.marker_color_meta[:]
marker_color_meta = cycle(marker_color_meta)
starting_year = hm.starting_year
season_start_date = hm.season_start_date
season_end_date = hm.season_end_date
major_xticks = hd.season_xticks
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Load the cleaned data garnered in `step_1_perform_cleaning.py`.
# 2. Load the list of Probe-IDs stored at `./data/probe_ids.txt`.
# 3. Load the Reference Crop Coefficients `./data/reference_crop_coeff.xlsx`
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.
# Load the cleaned data garnered in `step_1_perform_cleaning.py`.
cleaned_multi_df = hd.cleaned_multi_df.copy(deep=True)
outer_index = hd.outer_index[:]  # the outer_index is actually a list of probe
# IDs.

# 2.
# Get a list of all the Probe-IDs involved for the cultivar
probe_ids = hm.probe_ids

# 3.
# Load the Reference Crop Coefficients `./data/reference_crop_coeff.xlsx`
cco_df = hd.cco_df.copy(deep=True)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Plot all the cleaned kcp data and save the plotted figure.
# Saved at "./figures/overlay.png".
# =============================================================================
# Make sure that the "./figures" directory exists.
if not os.path.exists("./figures"):
    os.makedirs("figures")

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Month of the Season")
ax.set_ylabel("$k_{cp}$")
ax.set_title("$k_{cp}$ versus Month of the Season")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax.set_xlim(left=season_start_date, right=season_end_date)

ax.set_xticks(major_xticks)
ax.set_ylim(bottom=0.0, top=KCP_MAX)
ax.grid(True)
for i, p in enumerate(outer_index):  # iterate over the different probes
    dates, kcp = get_dates_and_kcp(dataframe=cleaned_multi_df, probe_id=p)
    meta = next(marker_color_meta)
    ax.scatter(dates, kcp, marker=meta[0], color=meta[1], s=60,
               edgecolors="black", linewidth=1, alpha=0.5, label=probe_ids[i])
ax.scatter(cco_df.index, cco_df["cco"], color="yellow", marker=".",
           linewidth=1, alpha=1.0, label="Reference $k_{cp}$")
ax.legend()
fig.autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig("./figures/overlay.png")
plt.cla()
plt.clf()
plt.close()
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot all the cleaned etcp data and save the figure.
# Saved at "./figures/etcp_versus_date.png".
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Reset the cycle iterable
marker_color_meta = hm.marker_color_meta[:]
marker_color_meta = cycle(marker_color_meta)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Date")
ax.set_ylabel("$ET_{cp}$ [mm]")
ax.set_title("$ET_{cp}$ versus Date")
beginning_dates = []
end_dates = []
for p in probe_ids:
    meta = next(marker_color_meta)
    df = pd.read_excel("./data/processed_probe_data.xlsx",
                       sheet_name="{}".format(p), header=0, index_col=0,
                       parse_dates=True)
    condition = df["binary_value"] == 0
    useful_dates = df[condition].index
    useful_etcp = df.loc[useful_dates, "etcp"].values
    useful_dates = date_wrapper(date_iterable=useful_dates,
                                starting_year=starting_year)
    ax.scatter(useful_dates, useful_etcp, marker=meta[0], color=meta[1], s=60,
               label=p, alpha=0.5, edgecolors="black")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax.set_xticks(major_xticks)
ax.set_xlim(left=season_start_date, right=season_end_date)
ax.legend()
ax.grid()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("./figures/etcp_versus_date.png")
plt.close()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
