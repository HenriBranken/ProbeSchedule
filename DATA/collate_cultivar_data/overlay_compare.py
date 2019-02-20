import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from master_data import accepted_kcp_norm
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

with open("./data_to_plot", "rb") as f:
    data_to_plot = pickle.load(f)

with open("./probe_ids.txt", "r") as f2:
    probe_ids = f2.readlines()
probe_ids = [x.strip() for x in probe_ids]


def get_starting_year():
    date_values, _ = separate_dates_from_kcp(data_to_plot[0])
    return date_values[0].year


def separate_dates_from_kcp(tuple_object):
    date_values = tuple_object[0]
    kcp_values = tuple_object[1]
    return np.array(date_values), np.array(kcp_values)


def convert_master_data():
    month_values = accepted_kcp_norm.index
    global starting_year
    master_dates = []
    for m in month_values:
        if 7 <= m <= 12:
            master_dates.append(datetime.datetime(year=starting_year, month=m, day=15))
        else:
            master_dates.append(datetime.datetime(year=starting_year + 1, month=m, day=15))
    return np.array(master_dates), accepted_kcp_norm.values


starting_year = get_starting_year()
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]
zipped_meta = ((m, c) for m, c in zip(marker_list, color_list))  # a generator expression
norm_dates, norm_kcps = convert_master_data()

concatenated_data = np.empty(shape=(1, 2))
for i in range(len(data_to_plot)):
    date_arr = np.reshape(np.array(data_to_plot[i][0]), newshape=(-1, 1))
    kcp_arr = np.reshape(np.array(data_to_plot[i][1]), newshape=(-1, 1))
    arr = np.hstack((date_arr, kcp_arr))
    concatenated_data = np.vstack((concatenated_data, arr))
concatenated_data = np.delete(concatenated_data, 0, axis=0)
df = pd.DataFrame(data=concatenated_data[:, 1], index=concatenated_data[:, 0], columns=["kcp"])
df.index.name = "datetimestamp"
# offset = int(time.mktime(datetime.date(2017, 7, 1).timetuple()))
df["offset"] = df.index - datetime.datetime(year=2017, month=7, day=1)
df["days"] = df["offset"].dt.days
df.sort_values(by="days", axis=0, inplace=True)
print(df.head())
print(df.info())
df.to_excel("kcp_vs_dates.xlsx", sheet_name="sheet_1", header=True, index=True, index_label=True)
# z = np.polyfit(x=df["days"].values, y=df["kcp"].values, deg=2)
# print(z)

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlabel("Month")
ax.set_ylabel("$k_{cp}$")
ax.set_title("$k_{cp}$ versus time")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax.set_xlim(left=datetime.datetime(year=starting_year, month=7, day=1),
            right=datetime.datetime(year=starting_year+1, month=6, day=30))
ax.grid(True)
for i in range(len(data_to_plot)):
    dates, kcp = separate_dates_from_kcp(data_to_plot[i])  # extract the data from the zipped object
    meta = next(zipped_meta)
    marker, color = meta[0], meta[1]  # extract the marker and face-color
    ax.scatter(dates, kcp, color=color, marker=marker, s=60, edgecolors="black", linewidth=1, alpha=0.5,
               label=probe_ids[i])
ax.plot(norm_dates, norm_kcps, linewidth=2, label="Master Perennial Data")
# ax.scatter(concatenated_data[:, 0], concatenated_data[:, 1], color="black", marker="1", s=20, label="concat")
ax.legend()
fig.autofmt_xdate()
plt.savefig("overlay.png")
