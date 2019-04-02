import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from cleaning_operations import BEGINNING_MONTH
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
zipped_meta = ((m, c) for m, c in zip(marker_list, color_list))  # here we used generator comprehension
# ======================================================================================================================


# Get a list of all the Probe-IDs involved for the cultivar
with open("./data/probe_ids.txt", "r") as f2:
    probe_ids = f2.readlines()
probe_ids = [x.strip() for x in probe_ids]

with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())

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


# ----------------------------------------------------------------------------------------------------------------------
# Define all the helper functions
# ----------------------------------------------------------------------------------------------------------------------
def get_dates_and_kcp(dataframe, probe_id):
    sub_df = dataframe.loc[(probe_id, ), ["kcp"]]
    return sub_df.index, sub_df["kcp"].values


def get_labels(begin, terminate):
    return [x for x in pd.date_range(start=begin, end=terminate, freq="MS")]
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot the Profile Readings (versus date) for each probe on a separate plot.
# For each plot, indicate the discrete points where the following occur:
#   1.  Data Blips --> DATA_BLIP_DESC
#   2.  Large Dips --> LARGE_PROFILE_DIP_DESC
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for p in probe_ids:
    df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(p), header=0, index_col=0,
                       parse_dates=True)
    indices_data_blip = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
    values = indices_data_blip.index[indices_data_blip].values
    print("\\"*80)
    print(p)
    print("-"*80)
    print(indices_data_blip.index[indices_data_blip])
    print("/"*80)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x=list(values), y=list(df.loc[indices_data_blip, "profile"].values), s=50,
               color="black", marker="*", label="Data blips", edgecolors="red")
    # major_xticks = get_labels(begin=df.index[0], terminate=df.index[-1])
    # ax.set_xticks(major_xticks)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%b'))
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
