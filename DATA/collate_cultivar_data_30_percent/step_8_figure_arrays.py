import os
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cleaning_operations import KCP_MAX
import pandas as pd
from cleaning_operations import description_dict
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
probe_ids = hm.probe_ids[:]
num_probes = len(probe_ids)

processed_eg_df = hd.processed_eg_df.copy(deep=True)

# Get the `starting_year`, `season_start_date`, `season_end_date`.
starting_year = hm.starting_year
season_start_date = hm.season_start_date
season_end_date = hm.season_end_date

# Extract the data of the smoothed kcp trendline.
fn = "./probe_screening/pruned_kcp_vs_days.xlsx"
smoothed_kcp_trend_df = pd.read_excel(fn, index_col=0, squeeze=False)
x_smoothed = smoothed_kcp_trend_df["x_smoothed"].values
y_smoothed = smoothed_kcp_trend_df["y_smoothed"].values
x_smoothed_dates = list(pd.date_range(start=season_start_date,
                                      end=season_end_date, freq="D"))
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
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create the necessary directory structures.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
sub_dirs = ["kcp", "irrigation", "profile"]
for sub_dir in sub_dirs:
    if not os.path.exists("./figures/{:s}".format(sub_dir)):
        os.makedirs("figures/{:s}".format(sub_dir))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Remove Old Figures.
# -----------------------------------------------------------------------------
for sub_dir in sub_dirs:
    files = os.listdir("./figures/{:s}".format(sub_dir))
    for file in files:
        if file.endswith(".png"):
            os.remove("./figures/{:s}/{:s}".format(sub_dir, file))
            print("Removed the file: \"{}\".".format(file))
# -----------------------------------------------------------------------------


# =============================================================================
# Populate the directory "./figures/kcp/kcp_<probe_id>.png".
# =============================================================================
for i, p in enumerate(probe_ids):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylim(bottom=0.0, top=KCP_MAX)
    ax.set_xlabel("Month of the Season")
    ax.set_ylabel("$k_{cp}$")
    ax.set_title("Cleaned kcp Data for {:s}.".format(p))
    ax.grid(True)
    dates, kcp = hf.get_dates_and_kcp(dataframe=cleaned_multi_df, probe_id=p)
    season_day = pd.Series(dates - season_start_date).dt.days.values
    r_squared = hf.get_r_squared(x_raw=season_day, y_raw=kcp, x_fit=x_smoothed,
                                 y_fit=y_smoothed)
    ax.scatter(dates, kcp, marker="o", color="lightseagreen", s=20,
               edgecolors="black", linewidth=1, alpha=0.75, label=p)
    ax.plot(x_smoothed_dates, y_smoothed, linewidth=1.5, alpha=0.75,
            label="Smoothed Trend", color="mediumvioletred")
    ax.set_xticks(season_xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
    ax.set_xlim(left=season_start_date, right=season_end_date)
    ax.legend()
    ax.annotate(s="$R^2$ = {:.3f}".format(r_squared), xycoords="axes fraction",
                xy=(0.01, 0.93))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("./figures/kcp/kcp_{:s}.png".format(p))
    plt.close()
# =============================================================================


# -----------------------------------------------------------------------------
# Populate directory "./figures/irrigation/irrigation_<probe_id>.png".
# -----------------------------------------------------------------------------
for i, p in enumerate(probe_ids):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Irrigation")
    ax.set_title("Irrigation Data for {:s}.".format(p))
    ax.grid(True)
    df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=p,
                       header=0, index_col=0, parse_dates=True)
    ax.bar(df.index, df["total_irrig"].values, color="green", label=p,
           width=1.6, edgecolor="black", linewidth=0.5, alpha=0.5)
    indices_irr = df["description"].str.contains(description_dict["irr_desc"],
                                                 na=False)
    ax.scatter(indices_irr.index[indices_irr],
               df.loc[indices_irr, ["total_irrig"]], color="blue", marker="o",
               s=20, alpha=0.7, edgecolors="black", label="Flagged Irr. Event")
    for v in vline_dates:
        ax.axvline(x=v, linewidth=3, color="magenta", alpha=0.4, ls="--")
    ax.plot([], [], linewidth=3, color="magenta", label="New Season",
            alpha=0.4, ls="--")
    ax.set_xticks(api_xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
    ax.set_xlim(left=api_start_date, right=api_end_date)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("./figures/irrigation/irrigation_{:s}.png".format(p))
    plt.close()
# -----------------------------------------------------------------------------


# =============================================================================
# Populate directory "./figures/profile/profile_<probe_id>.png".
# =============================================================================
for i, p in enumerate(probe_ids):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Date")
    ax.set_ylabel("Profile Reading")
    ax.set_title("Profile Data for {:s}.".format(p))
    ax.grid(True)
    df = pd.read_excel("./data/processed_probe_data.xlsx",
                       sheet_name=p, header=0, index_col=0, parse_dates=True)
    ax.plot(df.index, df["profile"], label=p, lw=1.5, alpha=0.6)
    DESC = description_dict["data_blip_desc"]
    indices_data_blip = df["description"].str.contains(DESC, na=False)
    DESC = description_dict["large_profile_dip_desc"]
    indices_large_dips = df["description"].str.contains(DESC, na=False)
    ax.scatter(x=indices_data_blip.index[indices_data_blip],
               y=df.loc[indices_data_blip, "profile"], s=100, color="red",
               marker="*", label="Blips", edgecolors="black", alpha=0.5)
    ax.scatter(x=indices_large_dips.index[indices_large_dips],
               y=df.loc[indices_large_dips, "profile"], s=100, color="green",
               marker="X", label="Dips", edgecolors="black", alpha=0.5)
    for v in vline_dates:
        ax.axvline(x=v, linewidth=2, color="magenta", alpha=0.4, ls="--")
    ax.plot([], [], linewidth=2, color="magenta", label="New Season",
            alpha=0.4, ls="--")
    ax.set_xticks(api_xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
    ax.set_xlim(left=api_start_date, right=api_end_date)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("./figures/profile/profile_{:s}.png".format(p))
    plt.close()
# =============================================================================
