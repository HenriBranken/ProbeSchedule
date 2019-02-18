# import numpy as np
# import datetime
# from datetime import timedelta
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cleaning_operations


def load_probe_data(probe_name):
    dataframe = pd.read_excel("../Golden_Delicious_daily_data.xlsx", sheet_name=probe_name,
                              index_col=0, parse_dates=True)
    new_columns = []
    for c in dataframe.columns:
        if '0' in c:
            c = c.replace("0", "o")
        new_columns.append(c.lstrip())
    dataframe.columns = new_columns
    return dataframe


def initialize_flagging_columns(dataframe):
    dataframe["binary_value"] = 0.0
    dataframe["description"] = str()
    return dataframe


probe_ids = ["P-370", "P-371", "P-372", "P-384", "P-391", "P-392", "P-891"]

# TODO: encase all the code below inside a for-loop; we need to iterate over all the probes present in the list.

probe_id = "P-392"

# Loading data from excel sheet
df = load_probe_data(probe_name=probe_id)

# Initialise the flagging-related columns
df = initialize_flagging_columns(dataframe=df)

# Drop redundant columns
df = cleaning_operations.drop_redundant_columns(df=df)

# Detect stuck eto values, and flag stuck etc values:
bad_eto_days, df = cleaning_operations.flag_spurious_eto(df=df)

# Impute Koue Bokkeveld longterm data:
df = cleaning_operations.impute_kbv_data(df, bad_eto_days)

# Flag events in which rain > 2 mm.
df = cleaning_operations.flag_rain_events(df)

# flag events in which irrigation has taken place.
df = cleaning_operations.flag_irrigation_events(df)

# flag events in which data were simulated.
df = cleaning_operations.flag_simulated_events(df)

# flag spurious "profile" entries.
df = cleaning_operations.flag_suspicious_and_missing_profile_events(df)

# flag stuck heat_units events
df = cleaning_operations.flag_spurious_heat_units(df)

# flag unwanted etcp values
df = cleaning_operations.flag_unwanted_etcp(df)

# flag kcp values that deviate by more than 50% from the accepted norm
df = cleaning_operations.flag_unwanted_kcp(df)

# get the dates for which "binary_value" == 0.
starting_year, new_dates = cleaning_operations.get_final_dates(df)
kcp_values = df.loc[new_dates, "kcp"].values

fig, ax = plt.subplots()
fig.set_size_inches(8, 3)
ax.scatter(new_dates, kcp_values, color="gold", label="Remaining $k_{cp}$", marker="D", s=10, edgecolors="black",
           linewidth=1, alpha=0.6)
ax.set_xlabel("Month")
ax.set_ylabel("$k_{cp}$")
ax.set_title("$k_{cp}$ versus time")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d'))
ax.set_xlim(left=datetime.datetime(year=starting_year, month=8, day=1),
            right=datetime.datetime(year=starting_year+1, month=7, day=31))
ax.legend()
fig.autofmt_xdate()
plt.show()
