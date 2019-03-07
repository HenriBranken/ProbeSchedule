import pandas as pd
import pickle
import cleaning_operations
import os


# ----------------------------------------------------------------------------------------------------------------------
# Define some helper functions
# ----------------------------------------------------------------------------------------------------------------------
def load_probe_data(probe_name):
    dataframe = pd.read_excel("../Golden_Delicious_daily_data_new.xlsx", sheet_name=probe_name,
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
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Perform cleaning (flagging) operations on all the Probe Data
# ======================================================================================================================
# Create a list of all the Probe-IDs that are going to be used
probe_ids = ["P-370", "P-371", "P-372", "P-384", "P-391", "P-392", "P-891"]

# Initialise an empty list to which all our data will be appended in tuple form
data_to_plot = []

# Create a for loop iterating over all the different probe datasets
# In each iteration, perform all the necessary flagging
for probe_id in probe_ids:
    print("\n")
    print("*"*30)
    print("Busy with probe: {}.".format(probe_id))

    # Loading data from Excel sheet
    df = load_probe_data(probe_name=probe_id)

    # Initialise the flagging-related columns
    df = initialize_flagging_columns(dataframe=df)

    # Drop redundant columns
    df = cleaning_operations.drop_redundant_columns(df=df)

    # Detect stuck eto values, and flag stuck etc values:
    bad_eto_days, df = cleaning_operations.flag_spurious_eto(df=df)

    # Impute `Koue-Bokkeveld` longterm eto data:
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
    starting_year, new_dates, useful_dates = cleaning_operations.get_final_dates(df)
    kcp_values = df.loc[useful_dates, "kcp"].values

    data_to_plot.append((new_dates, kcp_values))
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Save all the cleaned data to file
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if not os.path.exists("./data"):
    os.makedirs("./data")

with open("data/data_to_plot", "wb") as f:
    pickle.dump(data_to_plot, f)

with open("data/probe_ids.txt", "w", encoding="utf-8") as f2:
    f2.write("\n".join(probe_ids))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
