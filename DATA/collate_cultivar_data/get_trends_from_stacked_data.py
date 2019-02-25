import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
import datetime
import pandas as pd
import calendar
from cleaning_operations import BEGINNING_MONTH, KCP_MAX


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the serialised data that were saved in `main.py`, and unpickle it
with open("./data_to_plot", "rb") as f:
    data_to_plot = pickle.load(f)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Stack all the data from all the different probes
# Sort the data (ascending) according to the datetime column
# Save the sorted data in an Excel file
# Also store the numerical data columns `days` and `kcp` in a numpy array and save to file
# ----------------------------------------------------------------------------------------------------------------------
# Get the starting year
sample_dates = data_to_plot[0][0]
sample_dates.sort()
starting_year = sample_dates[0].year

# Stack all the data from all the different probes by using a for-loop
concatenated_data = np.empty(shape=(1, 2))  # initialise a placeholder to which data values will be stacked
for i in range(len(data_to_plot)):
    date_arr = np.reshape(np.array(data_to_plot[i][0]), newshape=(-1, 1))
    kcp_arr = np.reshape(np.array(data_to_plot[i][1]), newshape=(-1, 1))
    arr = np.hstack((date_arr, kcp_arr))
    concatenated_data = np.vstack((concatenated_data, arr))
concatenated_data = np.delete(concatenated_data, 0, axis=0)  # delete the first dummy row

# Initialise a pandas DataFrame called `df`
df = pd.DataFrame(index=concatenated_data[:, 0], data=concatenated_data[:, 1], columns=["kcp"])
df.index.name = "datetimeStamp"

# Get the starting month for the cultivar Season
df.sort_index(inplace=True)
starting_month = df.index[0].month

# Calculate the date offset for each row from the beginning of the cultivar Season
df["offset"] = df.index - datetime.datetime(year=starting_year, month=starting_month, day=1)

# Convert the time-delta objects of the `offset` column to integer type (therefore representing the number of days)
df["days"] = df["offset"].dt.days

# Sort the whole DataFrame by the `days` column in ascending order
df.sort_values(by="days", axis=0, inplace=True)
df.to_excel("kcp_vs_days.xlsx", sheet_name="sheet_1", header=True, index=True, index_label=True,
            columns=["days", "kcp"])

# Store `days` and `kcp` columns in numpy array (of type float)
# We want to use this array later for polynomial fitting
days = np.reshape(df.loc[:, "days"].values, newshape=(-1, 1)).astype(dtype=float)
kcp = np.reshape(df.loc[:, "kcp"].values, newshape=(-1, 1))
data_np_array = np.hstack((days, kcp))
np.save("kcp_vs_days_data", data_np_array)
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Perform polynomial fits to `data_np_array`
# Make plots of `kcp` versus `Days into the Season` of the Polynomial fits
# Here, the x-axis is simply an integer: the offset (in units of days) since the beginning of the season
# ======================================================================================================================
x = np.array(data_np_array[:, 0], dtype=float)
y = np.array(data_np_array[:, 1], dtype=float)

# Set the plotting meta-data, etc...
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("$n$ Days into the Season")
ax.set_ylabel("$k_{cp}$")
ax.set_title("$k_{cp}$ versus days")
loc = plticker.MultipleLocator(base=30)  # The spacing between the major-gridlines must be 30 (calendar days)
ax.xaxis.set_major_locator(loc)
ax.grid(True)
ax.set_xlim(left=0, right=365)
ax.set_ylim(bottom=0.05, top=KCP_MAX + 0.05)
ax.scatter(x, y, c="magenta", marker=".", edgecolors="black", alpha=0.5, label="Cleaned Probe Data")

fit_coeffs = []  # instantiate an empty list that will contain the coefficients of the np.polyfit executions
linspaced_x = np.arange(start=0, stop=365, step=1)  # linspaced_x = [0, 1, 2, ..., 364].  This represents number of days
# Notice that in the following we only consider quadratic and cubic polynomial fits to the data
# Performing a linear fit (y = mx + c) would be non-sensical
# From an educated guess, performing polynomial fits above order 3 would result in more oscillations,
# and we sould also have the risk of OVERFITTING the data
for highest_order in [2, 3]:
    z = np.polyfit(x=x, y=y, deg=highest_order)  # z contains the polynomial coefficients, with the highest power first
    fit_coeffs.append(list(z))
    p = np.poly1d(z)  # It is convenient to use poly1d objects for dealing with polynomials
    ax.plot(linspaced_x, p(linspaced_x), label="Order-{} Polynomial Fit".format(highest_order))
ax.legend()
plt.tight_layout()
plt.savefig("./fit_kcp_versus_days.png")
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create another plot where the x-axis is of type datetime.
# In this new plot we have kcp versus Month of the Year, i.e.: July/01, August/01, ..., June/01
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
starting_date = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
datetime_linspaced = []  # these dates will be used to plot the trend as inferred from the polynomial
for i in linspaced_x:
    datetime_linspaced.append(starting_date + datetime.timedelta(days=float(i)))

datetime_clouded = []  # These correspond to the scatter plot.  There are typically many overlapping dates
for i in x:
    datetime_clouded.append(starting_date + datetime.timedelta(days=float(i)))

trend = np.poly1d(fit_coeffs[0])  # Create a poly1d object with the coefficients of the quadratic polynomial
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Date (Month of the Year)")
ax.set_ylabel("$k_{cp}$")
ax.set_title("$k_{cp}$ versus Month of the Year")
ax.grid(True)
ax.set_xlim(left=datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1),
            right=datetime.datetime(year=starting_year+1, month=BEGINNING_MONTH - 1,
                                    day=calendar.monthrange(starting_year+1, BEGINNING_MONTH-1)[1]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))  # Month/01
ax.set_ylim(bottom=0.05, top=KCP_MAX + 0.05)
ax.scatter(datetime_clouded, y, c="magenta", marker=".", edgecolors="black", alpha=0.5, label="Cleaned Probe Data")
ax.plot(datetime_linspaced, trend(linspaced_x), label="Order-2 Polynomial Fit")
ax.legend()
fig.autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig("./fit_kcp_versus_month.png")

datetimestamp = np.reshape(datetime_linspaced, newshape=(-1, 1))
kcp = np.reshape(trend(linspaced_x), newshape=(-1, 1))
np.save("daily_trend_of_kcp_vs_datetime", np.hstack((datetimestamp, kcp)))
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
