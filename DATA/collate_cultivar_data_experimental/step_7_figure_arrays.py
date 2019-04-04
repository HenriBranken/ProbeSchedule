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

with open("./data/starting_year.txt", "r") as f:
    starting_year = int(f.readline().rstrip())
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Define all the helper functions
# ----------------------------------------------------------------------------------------------------------------------
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
beginning_datetime = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
end_datetime = datetime.datetime(year=starting_year + 1, month=BEGINNING_MONTH, day=1)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Plot array of cleaned kcp for all the probes
# ======================================================================================================================
fig, ax = plt.subplots(nrows=4, ncols=2, sharex="col", sharey="row", figsize=(7.07, 10))
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]
zipped_meta = ((m, c) for m, c in zip(marker_list, color_list))  # here we used generator comprehension

meta = next(zipped_meta)
ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax[0, 0].set_ylim(bottom=0.0, top=KCP_MAX)
ax[0, 0].grid(True)
dates, kcp = separate_dates_from_kcp(data_to_plot[0])
ax[0, 0].scatter(dates, kcp, marker=meta[0], color=meta[1], s=20, edgecolors="black", linewidth=1, alpha=0.5,
                 label=probe_ids[0])
ax[0, 0].legend()

meta = next(zipped_meta)
ax[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax[0, 1].set_ylim(bottom=0.0, top=KCP_MAX)
ax[0, 1].grid(True)
dates, kcp = separate_dates_from_kcp(data_to_plot[1])
ax[0, 1].scatter(dates, kcp, marker=meta[0], color=meta[1], s=20, edgecolors="black", linewidth=1, alpha=0.5,
                 label=probe_ids[1])
ax[0, 1].legend()

meta = next(zipped_meta)
ax[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax[1, 0].set_ylim(bottom=0.0, top=KCP_MAX)
ax[1, 0].grid(True)
dates, kcp = separate_dates_from_kcp(data_to_plot[2])
ax[1, 0].scatter(dates, kcp, marker=meta[0], color=meta[1], s=20, edgecolors="black", linewidth=1, alpha=0.5,
                 label=probe_ids[2])
ax[1, 0].legend()

meta = next(zipped_meta)
ax[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax[1, 1].set_ylim(bottom=0.0, top=KCP_MAX)
ax[1, 1].grid(True)
dates, kcp = separate_dates_from_kcp(data_to_plot[3])
ax[1, 1].scatter(dates, kcp, marker=meta[0], color=meta[1], s=20, edgecolors="black", linewidth=1, alpha=0.5,
                 label=probe_ids[3])
ax[1, 1].legend()

meta = next(zipped_meta)
ax[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax[2, 0].set_ylim(bottom=0.0, top=KCP_MAX)
ax[2, 0].grid(True)
dates, kcp = separate_dates_from_kcp(data_to_plot[4])
ax[2, 0].scatter(dates, kcp, marker=meta[0], color=meta[1], s=20, edgecolors="black", linewidth=1, alpha=0.5,
                 label=probe_ids[4])
ax[2, 0].legend()

meta = next(zipped_meta)
ax[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax[2, 1].set_ylim(bottom=0.0, top=KCP_MAX)
ax[2, 1].grid(True)
dates, kcp = separate_dates_from_kcp(data_to_plot[5])
ax[2, 1].scatter(dates, kcp, marker=meta[0], color=meta[1], s=20, edgecolors="black", linewidth=1, alpha=0.5,
                 label=probe_ids[5])
ax[2, 1].legend()

meta = next(zipped_meta)
ax[3, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax[3, 0].set_ylim(bottom=0.0, top=KCP_MAX)
ax[3, 0].grid(True)
dates, kcp = separate_dates_from_kcp(data_to_plot[6])
ax[3, 0].scatter(dates, kcp, marker=meta[0], color=meta[1], s=20, edgecolors="black", linewidth=1, alpha=0.5,
                 label=probe_ids[6])
ax[3, 0].legend()

fig.delaxes(ax[3, 1])
plt.subplots_adjust(wspace=0.05)
fig.suptitle("$k_{cp}$ versus Date")
fig.autofmt_xdate()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
major_xticks = get_labels(begin="2017-07-01", terminate="2018-06-01")
ax[2, 1].set_xticks(major_xticks)
ax[3, 0].set_xticks(major_xticks)
ax[2, 1].tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                     labelrotation=90, labelsize="small")
ax[3, 0].tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                     labelrotation=90, labelsize="small", axis="x")
plt.savefig("./figures/array_kcp.png")
plt.cla()
plt.clf()
plt.close()
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Make array of Total Irrigation Plots
# ----------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(nrows=4, ncols=2, sharex="col", sharey="row", figsize=(7.07, 10))
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]
zipped_meta = ((m, c) for m, c in zip(marker_list, color_list))  # here we used generator comprehension

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[0], header=0, index_col=0, parse_dates=True)
ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[0, 0].grid(True)
ax[0, 0].bar(df.index, df["total_irrig"].values, color="magenta", label="{}".format(probe_ids[0]))
indices_irr = df["description"].str.contains(description_dict["irr_desc"], na=False)
ax[0, 0].scatter(indices_irr.index[indices_irr], df.loc[indices_irr, ["total_irrig"]], color="black", marker="o", s=5,
                 alpha=0.7, edgecolors="black")
ax[0, 0].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[1], header=0, index_col=0, parse_dates=True)
ax[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[0, 1].grid(True)
ax[0, 1].bar(df.index, df["total_irrig"].values, color="magenta", label="{}".format(probe_ids[1]))
indices_irr = df["description"].str.contains(description_dict["irr_desc"], na=False)
ax[0, 1].scatter(indices_irr.index[indices_irr], df.loc[indices_irr, ["total_irrig"]], color="black", marker="o", s=5,
                 alpha=0.7, edgecolors="black")
ax[0, 1].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[2], header=0, index_col=0, parse_dates=True)
ax[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[1, 0].grid(True)
ax[1, 0].bar(df.index, df["total_irrig"].values, color="magenta", label="{}".format(probe_ids[2]))
indices_irr = df["description"].str.contains(description_dict["irr_desc"], na=False)
ax[1, 0].scatter(indices_irr.index[indices_irr], df.loc[indices_irr, ["total_irrig"]], color="black", marker="o", s=5,
                 alpha=0.7, edgecolors="black")
ax[1, 0].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[3], header=0, index_col=0, parse_dates=True)
ax[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[1, 1].grid(True)
ax[1, 1].bar(df.index, df["total_irrig"].values, color="magenta", label="{}".format(probe_ids[3]))
indices_irr = df["description"].str.contains(description_dict["irr_desc"], na=False)
ax[1, 1].scatter(indices_irr.index[indices_irr], df.loc[indices_irr, ["total_irrig"]], color="black", marker="o", s=5,
                 alpha=0.7, edgecolors="black")
ax[1, 1].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[4], header=0, index_col=0, parse_dates=True)
ax[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[2, 0].grid(True)
ax[2, 0].bar(df.index, df["total_irrig"].values, color="magenta", label="{}".format(probe_ids[4]))
indices_irr = df["description"].str.contains(description_dict["irr_desc"], na=False)
ax[2, 0].scatter(indices_irr.index[indices_irr], df.loc[indices_irr, ["total_irrig"]], color="black", marker="o", s=5,
                 alpha=0.7, edgecolors="black")
ax[2, 0].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[5], header=0, index_col=0, parse_dates=True)
ax[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[2, 1].grid(True)
ax[2, 1].bar(df.index, df["total_irrig"].values, color="magenta", label="{}".format(probe_ids[5]))
indices_irr = df["description"].str.contains(description_dict["irr_desc"], na=False)
ax[2, 1].scatter(indices_irr.index[indices_irr], df.loc[indices_irr, ["total_irrig"]], color="black", marker="o", s=5,
                 alpha=0.7, edgecolors="black")
ax[2, 1].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[6], header=0, index_col=0, parse_dates=True)
ax[3, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[3, 0].grid(True)
ax[3, 0].bar(df.index, df["total_irrig"].values, color="magenta", label="{}".format(probe_ids[6]))
indices_irr = df["description"].str.contains(description_dict["irr_desc"], na=False)
ax[3, 0].scatter(indices_irr.index[indices_irr], df.loc[indices_irr, ["total_irrig"]], color="black", marker="o", s=5,
                 alpha=0.7, edgecolors="black")
ax[3, 0].legend()

fig.delaxes(ax[3, 1])
plt.subplots_adjust(wspace=0.05)
fig.suptitle("Total Irrigation versus Date")
fig.autofmt_xdate()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
ax[2, 1].tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                     labelrotation=90, labelsize="small")
ax[3, 0].tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                     labelrotation=90, labelsize="small", axis="x")
plt.savefig("./figures/array_irrigation.png")
plt.cla()
plt.clf()
plt.close()
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Array of water profile plots
# ======================================================================================================================
fig, ax = plt.subplots(nrows=4, ncols=2, sharex="col", sharey="row", figsize=(7.07, 10))

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[0], header=0, index_col=0, parse_dates=True)
indices_data_blip = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
indices_large_dips = df["description"].str.contains(description_dict["large_profile_dip_desc"], na=False)
ax[0, 0].plot(df.index, df["profile"], label=probe_ids[0], lw=1, alpha=0.6)
ax[0, 0].scatter(x=indices_data_blip.index[indices_data_blip], y=df.loc[indices_data_blip, "profile"], s=50,
                 color="black", marker="*", label="Data blips", edgecolors="red")
ax[0, 0].scatter(x=indices_large_dips.index[indices_large_dips], y=df.loc[indices_large_dips, "profile"], s=50,
                 color="black", marker="X", label="'Large' Dips", edgecolors="green")
ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[0, 0].grid(True)
ax[0, 0].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[1], header=0, index_col=0, parse_dates=True)
indices_data_blip = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
indices_large_dips = df["description"].str.contains(description_dict["large_profile_dip_desc"], na=False)
ax[0, 1].plot(df.index, df["profile"], label=probe_ids[1], lw=1, alpha=0.6)
ax[0, 1].scatter(x=indices_data_blip.index[indices_data_blip], y=df.loc[indices_data_blip, "profile"], s=50,
                 color="black", marker="*", label="Data blips", edgecolors="red")
ax[0, 1].scatter(x=indices_large_dips.index[indices_large_dips], y=df.loc[indices_large_dips, "profile"], s=50,
                 color="black", marker="X", label="'Large' Dips", edgecolors="green")
ax[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[0, 1].grid(True)
ax[0, 1].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[2], header=0, index_col=0, parse_dates=True)
indices_data_blip = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
indices_large_dips = df["description"].str.contains(description_dict["large_profile_dip_desc"], na=False)
ax[1, 0].plot(df.index, df["profile"], label=probe_ids[2], lw=1, alpha=0.6)
ax[1, 0].scatter(x=indices_data_blip.index[indices_data_blip], y=df.loc[indices_data_blip, "profile"], s=50,
                 color="black", marker="*", label="Data blips", edgecolors="red")
ax[1, 0].scatter(x=indices_large_dips.index[indices_large_dips], y=df.loc[indices_large_dips, "profile"], s=50,
                 color="black", marker="X", label="'Large' Dips", edgecolors="green")
ax[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[1, 0].grid(True)
ax[1, 0].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[3], header=0, index_col=0, parse_dates=True)
indices_data_blip = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
indices_large_dips = df["description"].str.contains(description_dict["large_profile_dip_desc"], na=False)
ax[1, 1].plot(df.index, df["profile"], label=probe_ids[3], lw=1, alpha=0.6)
ax[1, 1].scatter(x=indices_data_blip.index[indices_data_blip], y=df.loc[indices_data_blip, "profile"], s=50,
                 color="black", marker="*", label="Data blips", edgecolors="red")
ax[1, 1].scatter(x=indices_large_dips.index[indices_large_dips], y=df.loc[indices_large_dips, "profile"], s=50,
                 color="black", marker="X", label="'Large' Dips", edgecolors="green")
ax[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[1, 1].grid(True)
ax[1, 1].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[4], header=0, index_col=0, parse_dates=True)
indices_data_blip = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
indices_large_dips = df["description"].str.contains(description_dict["large_profile_dip_desc"], na=False)
ax[2, 0].plot(df.index, df["profile"], label=probe_ids[4], lw=1, alpha=0.6)
ax[2, 0].scatter(x=indices_data_blip.index[indices_data_blip], y=df.loc[indices_data_blip, "profile"], s=50,
                 color="black", marker="*", label="Data blips", edgecolors="red")
ax[2, 0].scatter(x=indices_large_dips.index[indices_large_dips], y=df.loc[indices_large_dips, "profile"], s=50,
                 color="black", marker="X", label="'Large' Dips", edgecolors="green")
ax[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[2, 0].grid(True)
ax[2, 0].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[5], header=0, index_col=0, parse_dates=True)
indices_data_blip = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
indices_large_dips = df["description"].str.contains(description_dict["large_profile_dip_desc"], na=False)
ax[2, 1].plot(df.index, df["profile"], label=probe_ids[5], lw=1, alpha=0.6)
ax[2, 1].scatter(x=indices_data_blip.index[indices_data_blip], y=df.loc[indices_data_blip, "profile"], s=50,
                 color="black", marker="*", label="Data blips", edgecolors="red")
ax[2, 1].scatter(x=indices_large_dips.index[indices_large_dips], y=df.loc[indices_large_dips, "profile"], s=50,
                 color="black", marker="X", label="'Large' Dips", edgecolors="green")
ax[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[2, 1].grid(True)
ax[2, 1].legend()

df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=probe_ids[6], header=0, index_col=0, parse_dates=True)
indices_data_blip = df["description"].str.contains(description_dict["data_blip_desc"], na=False)
indices_large_dips = df["description"].str.contains(description_dict["large_profile_dip_desc"], na=False)
ax[3, 0].plot(df.index, df["profile"], label=probe_ids[6], lw=1, alpha=0.6)
ax[3, 0].scatter(x=indices_data_blip.index[indices_data_blip], y=df.loc[indices_data_blip, "profile"], s=50,
                 color="black", marker="*", label="Data blips", edgecolors="red")
ax[3, 0].scatter(x=indices_large_dips.index[indices_large_dips], y=df.loc[indices_large_dips, "profile"], s=50,
                 color="black", marker="X", label="'Large' Dips", edgecolors="green")
ax[3, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
ax[3, 0].grid(True)
ax[3, 0].legend()

fig.delaxes(ax[3, 1])
plt.subplots_adjust(wspace=0.05)
fig.suptitle("Profile versus Date")
fig.autofmt_xdate()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
ax[2, 1].tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                     labelrotation=90, labelsize="small")
ax[3, 0].tick_params(which="major", bottom=True, labelbottom=True, colors="black", labelcolor="black",
                     labelrotation=90, labelsize="small", axis="x")
plt.savefig("./figures/array_profile.png")
plt.cla()
plt.clf()
plt.close()
# ======================================================================================================================
