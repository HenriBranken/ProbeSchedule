import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# Load the necessary data
# The kcp_vs_datetime array has 2 columns:
#   The first column is the datetime stamp.
#   The second column is the fitted (quadratic) kcp trend
# The cleaned_kcp_df is a pandas DataFrame containing the cleaned kcp data
# ----------------------------------------------------------------------------------------------------------------------
kcp_vs_datetime = np.load("daily_trend_of_kcp_vs_datetime.npy")

datetimestamp = kcp_vs_datetime[:, 0]
kcp_trend = kcp_vs_datetime[:, 1]

cleaned_kcp_df = pd.read_excel("kcp_vs_days.xlsx", sheet_name="sheet_1", names=["days", "kcp"], index_col=0,
                               parse_dates=True)
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Bin the data into 7-day buckets
# For each 7-day bucket, calculate the average kcp
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
counter = 0
one_week_data = []
binned_kcp = []
repeated_kcp = []
for kcp in kcp_trend:
    one_week_data.append(kcp)
    counter += 1
    if not (counter % 7):  # we have reached 7 elements in one_week_data
        computed_average = mean(one_week_data)  # get the average for the accrued 7 data points
        binned_kcp.append(computed_average)
        to_be_appended = [computed_average for i in range(7)]
        repeated_kcp += to_be_appended
        one_week_data = []  # flush the one_week_data container
        counter = 0  # reset the counter to 0

# Artificially append kcp data to reconcile shapes of repeated_kcp (y-axis) and datetimestamp (x-axis)
difference_in_lengths = np.abs(len(repeated_kcp) - len(datetimestamp))
for _ in range(difference_in_lengths):
    repeated_kcp.append(repeated_kcp[-1])

week_of_the_season = np.arange(start=1, stop=53, step=1, dtype=int)
week_of_the_season = week_of_the_season.reshape((-1, 1))
binned_kcp = np.array(binned_kcp).reshape((-1, 1))
weekly_kcp_data = np.hstack((week_of_the_season, binned_kcp))
np.save("weekly_kcp_trend", weekly_kcp_data, allow_pickle=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Date")
ax.set_ylabel("$k_{cp}$")
ax.set_xlim(left=datetimestamp[0], right=datetimestamp[-1])
ax.set_title("7-day averages of $k_{cp}$ vs date")
ax.plot(datetimestamp, repeated_kcp, label="7-day avg")
ax.scatter(cleaned_kcp_df.index, cleaned_kcp_df.loc[:, "kcp"],  c="magenta", marker=".", edgecolors="black",
           alpha=0.5, label="Cleaned Probe Data")
ax.legend()
ax.grid()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("7_day_kcp_averages_vs_date.png")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
