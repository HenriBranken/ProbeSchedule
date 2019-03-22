import pandas as pd
import pickle
import cleaning_operations
import os
import datetime
from cleaning_operations import BEGINNING_MONTH


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define important input from the user.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create a list of all the Probe-IDs that are going to be used
probe_ids = [370, 371, 372, 384, 391, 392, 891]
probe_ids = ["P-{:.0f}".format(number) for number in probe_ids]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
# Perform cleaning (flagging) operations on all the Probe Data.
# ======================================================================================================================
# The order of the cleaning and data crunching operations are as follows:
#   1.      Load daily probe data from Excel Sheet.
#   2.      Create and Initialize 2 flagging-related columns in the DataFrame: "binary_value" and "description".
#   3.      Drop redundant columns; drop data columns that are of no use in the data crunching pipeline.
#   4.      Flag dates for which `eto` values are stuck (faulty weather-station data).
#   5.      Impute Koue-Bokkeveld (a.k.a. kbv) `eto` data onto stuck `eto` data described in the previous line.
#   6.      Flag raining events for which the rain exceeds 2 millimeters.
#   7.      Flag irrigation events.
#   8.      Flag events in which data were simulated.
#   9.      Flag dates in which data blips and data dips were observed for the `profile` measurements.
#   10.a    Flag events in which the heat units are equal to zero.
#   10.b    Interpolate missing heat_units; i.e. dates for which heat_units = np.nan = NaN.
#   10.c    Get the cumulative Growing Degree Days (GDD)
#   11.     Flag events in which `etcp` is known to be faulty/contaminated.
#   12.     Flag samples in which the calculated `kcp` values deviate by more than 50% from the accepted norm.
#   13.     Extract the dates and associated `kcp` values for which `binary_value == 0`.
#   14.     Append the cleaned dates and `kcp` values to the growing list `data_to_plot`.
#   15.     Save all the data and metadata of all the cleaning and crunching operations to an Excel sheet.
#           One Excel sheet represents ONE probe-id.
#           A set of Excel spreadsheets represents a set of probe-id's on a farm / farm-block / region / etc...
#   16.     Once all the sheets have been populated, save and close the .xlsx file.
# ======================================================================================================================
# Initialise an empty list to which all our data will be appended in tuple form
data_to_plot = []

writer = pd.ExcelWriter("./data/processed_probe_data.xlsx", engine="xlsxwriter")
# Create a for loop iterating over all the different probe datasets
# In each iteration, perform all the necessary flagging/cleaning/data crunching.
for probe_id in probe_ids:
    print("\n")
    print("*" * 80)
    print("Busy with probe: {:s}.".format(probe_id))

    # 1.
    # Load daily probe data from Excel Sheet.
    df = load_probe_data(probe_name=probe_id)

    # 2.
    # Create and Initialize 2 flagging-related columns in the DataFrame: "binary_value" and "description".
    df = initialize_flagging_columns(dataframe=df)

    # 3.
    # Drop redundant columns; drop data columns that are of no use in the data crunching pipeline.
    df = cleaning_operations.drop_redundant_columns(df=df)

    # 4.
    # Flag dates for which `eto` values are stuck (faulty weather-station data).
    bad_eto_days, df = cleaning_operations.flag_spurious_eto(df=df)

    # 5.
    # Impute Koue-Bokkeveld (a.k.a. kbv) `eto` data onto stuck `eto` data described in the previous line.
    df = cleaning_operations.impute_kbv_data(df, bad_eto_days)

    # 6.
    # Flag raining events for which the rain exceeds 2 millimeters.
    df = cleaning_operations.flag_rain_events(df)

    # 7.
    # Flag irrigation events.
    df = cleaning_operations.flag_irrigation_events(df)

    # 8.
    # Flag events in which data were simulated.
    df = cleaning_operations.flag_simulated_events(df)

    # 9.
    # Flag dates in which data blips and data dips were observed for the `profile` measurements.
    df = cleaning_operations.flag_suspicious_and_missing_profile_events(df)

    # 10.a
    # Flag events in which the heat units were stuck.
    df = cleaning_operations.flag_spurious_heat_units(df)

    # 10.b
    # Interpolate missing heat units
    df = cleaning_operations.interpolate_missing_heat_units(df, method="nearest")

    # 10.c
    # Get the cumulative Growing Degree Days
    df = cleaning_operations.cumulative_gdd(df)

    # 10.d
    # Flag events in which the heat units remain stuck after interpolation.
    df = cleaning_operations.flag_spurious_heat_units(df)

    # 11.
    # Flag events in which `etcp` is known to be faulty/contaminated.
    df = cleaning_operations.flag_unwanted_etcp(df)

    # 12.
    # Flag samples in which the calculated `kcp` values deviate by more than 50% from the accepted norm.
    df = cleaning_operations.flag_unwanted_kcp(df)

    # 13.
    # Extract the dates and associated `kcp` values for which `binary_value == 0`.
    starting_year, new_dates, useful_dates = cleaning_operations.get_final_dates(df)
    kcp_values = df.loc[useful_dates, "kcp"].values

    # 14.
    # Append the cleaned dates and `kcp` values to the growing list `data_to_plot`.
    data_to_plot.append((new_dates, kcp_values))

    # 15.
    # Save all the data and metadata of all the cleaning and crunching operations to an Excel sheet.
    df.to_excel(writer, sheet_name=probe_id)

# 16.
# Once all the sheets have been populated, save and close the .xlsx file.
writer.save()  # `./data/processed_probe_data.xlsx`
# ======================================================================================================================
# The columns in `./data/processed_probe_data.xlsx` are:
# ======================================================================================================================
# date, heat_units, rain, erain, total_irrig, tot_eff_irrig, etc, ety, eto, etcp, rzm_source, profile, cco,
# original_unit_system, binary_value, description, kcp
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Save all the cleaned data to file.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# The code below can be summarised as follows:
#   1.  Make sure the (relative) directory `./data` exists.
#   2.  Save the contents of the `data_to_plot` list.
#       This is saved at `./data/data_to_plot`
#   3.  Populate a .txt file containing all the names of the probeIDs:
#       This is saved at `./data/probe_ids.txt`
#   4.  Save dataframe containing reference crop coefficient data to an excel spreadsheet.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.
if not os.path.exists("./data"):
    os.makedirs("./data")

# 2.
# `data_to_plot` represents clean data samples associated with (unperturbed) normal water uptake.
# `data_to_plot` only consists of (datetime, kcp) samples for each probe.
with open("./data/data_to_plot", "wb") as f:
    pickle.dump(data_to_plot, f)

# 3.
# Populate a .txt file containing all the probe-ids (in the format of 'P-{some_number}').
# There is exactly one ProbeID per line in the .txt file.
with open("./data/probe_ids.txt", "w", encoding="utf-8") as f2:
    f2.write("\n".join(probe_ids))

# 4.
# Extract and save reference "cco" data
processed_df = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name="{}".format(probe_ids[0]), header=0,
                             index_col=0, squeeze=True, parse_dates=True)
starting_year = processed_df.index[0].year
cco_df = processed_df["cco"].to_frame()
cco_df["wrapped_date"] = datetime.datetime(year=starting_year, month=BEGINNING_MONTH, day=1)
for i, d in enumerate(cco_df.index):
    if BEGINNING_MONTH <= d.month <= 12:
        manipulated_date = datetime.datetime(year=starting_year, month=d.month, day=d.day)
        cco_df.loc[d, "wrapped_date"] = manipulated_date
    else:
        cco_df.loc[d, "wrapped_date"] = datetime.datetime(year=starting_year + 1, month=d.month, day=d.day)
cco_df.set_index(keys="wrapped_date", inplace=True)
cco_df = cco_df[~cco_df.index.duplicated(keep="first")]
cco_df.sort_index(ascending=True, inplace=True)
cco_df.to_excel("./data/reference_crop_coeff.xlsx", sheet_name="sheet_1", header=True, index=True)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
