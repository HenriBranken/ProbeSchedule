import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cleaning_operations import KCP_MAX
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from helper_functions import date_wrapper, safe_removal
import helper_meta_data as hm
import helper_data as hd
register_matplotlib_converters()


# =============================================================================
# Declare some necessary "constants"
# =============================================================================
# 1. Create some meta data that will be used in the upcoming scatter plots.
#    This meta data stores `marker` and `color` values.
# =============================================================================
starting_year = hm.starting_year
season_start_date = hm.season_start_date
season_end_date = hm.season_end_date
major_xticks = hd.season_xticks
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Load the cleaned data garnered in `step_1_perform_cleaning.py`.
# 2. Load the Reference Crop Coefficients "./data/reference_crop_coeff.xlsx"
# 3. Create a DataFrame containing wrapped dates and cleaned etcp values.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.
# Load the cleaned data garnered in `step_1_perform_cleaning.py`.
cleaned_df = hd.cleaned_df.copy(deep=True)

# 2.
# Load the Reference Crop Coefficients `./data/reference_crop_coeff.xlsx`
cco_df = hd.cco_df.copy(deep=True)

# 3.
# Create a DataFrame where the index is a wrapped date, and the only column is
# "etcp".  We only keep "etcp" entries that are non-NaN and are associated with
# a "binary_value" of 0.  For neatness, sort the "wrapped_date" index as well.
# Having duplicate dates in the index makes sense.
processed_dict = pd.read_excel("./data/processed_probe_data.xlsx",
                               sheet_name=None, header=0, index_col=0,
                               parse_dates=True)
keys = processed_dict.keys()
dfs = list()
for k in keys:
    dfs.append(processed_dict[k])
processed_df = pd.concat(dfs)
processed_df = processed_df[["etcp", "binary_value"]]
condition = processed_df["binary_value"] == 0
processed_df = processed_df[condition]
dates = processed_df.index.to_series()
wrapped_dates = date_wrapper(date_iterable=dates, starting_year=starting_year)
processed_df["wrapped_date"] = wrapped_dates
processed_df.set_index(keys=["wrapped_date"], drop=True, inplace=True)
processed_df.sort_index(axis=0, ascending=True, inplace=True)
processed_df = processed_df[["etcp"]]
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Remove old files that were generated in a previous execution of this script.
# -----------------------------------------------------------------------------
file_list = ["./figures/kcp_overlay.png",
             "./figures/etcp_overlay.png"]
safe_removal(file_list=file_list)
# -----------------------------------------------------------------------------


# =============================================================================
# Plot all the cleaned kcp data and save the plotted figure.
# Saved at "./figures/kcp_overlay.png".
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
ax.scatter(cleaned_df.index, cleaned_df["y_scatter"].values,
           color="rebeccapurple", edgecolors="lightpink", marker=".",
           alpha=0.7, s=100, label="Cleaned kcp Data")
ax.scatter(cco_df.index, cco_df["cco"].values, color="yellow", marker=".",
           alpha=0.6, label="Reference $k_{cp}$, \"cco\"")
ax.legend()
fig.autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig("./figures/kcp_overlay.png")
plt.close()
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot all the cleaned etcp data and save the figure.
# Saved at "./figures/etcp_overlay.png".
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Date")
ax.set_ylabel("$ET_{cp}$ [mm]")
ax.set_title("$ET_{cp}$ versus Date")
ax.scatter(processed_df.index, processed_df["etcp"].values,
           color="darkgreen", edgecolors="yellow", marker=".",
           alpha=0.7, s=100, label="Cleaned etcp Data")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax.set_xticks(major_xticks)
ax.set_xlim(left=season_start_date, right=season_end_date)
ax.legend()
ax.grid()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("./figures/etcp_overlay.png")
plt.close()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
