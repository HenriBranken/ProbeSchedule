import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import calendar
from master_data import accepted_kcp_norm
from cleaning_operations import BEGINNING_MONTH
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the serialised data that were saved in `main.py`, and unpickle it
with open("./data_to_plot", "rb") as f:
    data_to_plot = pickle.load(f)

# Get a list of all the Probe-IDs involved for the cultivar
with open("./probe_ids.txt", "r") as f2:
    probe_ids = f2.readlines()
probe_ids = [x.strip() for x in probe_ids]
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


# "Wrap" the Master data points so that we only consider 1 whole Season
# We achieve this by manipulating the year values of the data points
def convert_master_data():
    month_values = accepted_kcp_norm.index  # imported from ./master_data.py
    global starting_year  # work with the global starting_year variable
    master_dates = []
    for m in month_values:
        if BEGINNING_MONTH <= m <= 12:  # if falling in the range of [July; December]
            master_dates.append(datetime.datetime(year=starting_year, month=m, day=15))
        else:  # if falling in the range of [January; June]
            master_dates.append(datetime.datetime(year=starting_year + 1, month=m, day=15))
    return np.array(master_dates), accepted_kcp_norm.values
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Plot all the data and save the plotted figure
# In addition to making the scatter plots, also plot the Master kcp data from the Perennial spreadsheet
# ======================================================================================================================
starting_year = get_starting_year()  # extract the starting year

# Create some meta data that will be used in the upcoming scatter plots
marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue", "darkorchid", "plum", "burlywood"]
zipped_meta = ((m, c) for m, c in zip(marker_list, color_list))  # here we used generator comprehension
wrapped_dates, wrapped_kcps = convert_master_data()

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Month of the Year")
ax.set_ylabel("$k_{cp}$")
ax.set_title("$k_{cp}$ versus Month of the Year")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax.set_xlim(left=datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1),
            right=datetime.datetime(year=starting_year+1, month=BEGINNING_MONTH - 1,
                                    day=calendar.monthrange(starting_year+1, BEGINNING_MONTH-1)[1]))
ax.grid(True)
for i in range(len(data_to_plot)):  # iterate over the datasets corresponding to the different probes
    dates, kcp = separate_dates_from_kcp(data_to_plot[i])  # extract the data from the zipped object, and sort by date
    meta = next(zipped_meta)
    ax.scatter(dates, kcp, color=meta[1], marker=meta[0], s=60, edgecolors="black", linewidth=1, alpha=0.5,
               label=probe_ids[i])
ax.plot(wrapped_dates, wrapped_kcps, linewidth=2, label="Master Perennial Data")  # Plot the Master data as a reference
ax.legend()
fig.autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig("overlay.png")
# ======================================================================================================================
