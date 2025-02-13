import pandas as pd
import cleaning_operations
import os
import helper_meta_data as hm
from helper_functions import date_wrapper, safe_removal


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define important "constants".
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create a list, `probe_ids`, of all the probe-ids that are going to be used:
probe_ids = hm.probe_ids[:]

# Create the path "./figures" if it does not already exist:
if not os.path.exists("./figures"):
    os.makedirs("figures")

# Define the season start date.
season_start_date = hm.season_start_date
starting_year = hm.starting_year
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Remove old files generated by a previous execution of this script.
# =============================================================================
file_list = ["./data/processed_probe_data.xlsx",
             "./data/cleaned_data_for_overlay.xlsx",
             "./data/stacked_cleaned_data_for_overlay.xlsx",
             "./data/reference_crop_coeff.xlsx"]
safe_removal(file_list=file_list)
# =============================================================================


# -----------------------------------------------------------------------------
# Define some helper functions
# -----------------------------------------------------------------------------
def load_probe_data(probe_name):
    # Load a data stored in the Excel sheet_name indicated by `probe_name`, and
    # assign the data to a DataFrame called `dataframe`.
    dataframe = pd.read_excel("./data/cultivar_data_unique.xlsx",
                              sheet_name=probe_name, index_col=0,
                              parse_dates=True)
    # Create a new list of column names where any "0" characters are replaced
    # "o" characters.
    new_columns = []
    for c in dataframe.columns:
        if '0' in c:
            c = c.replace("0", "o")
        new_columns.append(c.lstrip())
    dataframe.columns = new_columns
    return dataframe


def initialize_flagging_columns(dataframe):
    # In the DataFrame passed to this function, instantiate two new series
    # columns.
    # 1. dataframe["binary_value"] stores the binary_value associated with a
    #    sample.
    # 2. dataframe["description"] stores the string descriptions describing the
    #    flagging iterations that were applied to the particular sample.
    dataframe["binary_value"] = 0
    dataframe["description"] = str()
    return dataframe
# -----------------------------------------------------------------------------


# =============================================================================
# Perform cleaning (flagging) operations on all the Probe Data.
# =============================================================================
# The order of the cleaning and data crunching operations are as follows:
# 1.   Load daily probe data from Excel Sheet.
# 2.   Create and Initialize 2 flagging-related columns in the DataFrame.
# 3.   Drop redundant columns.
# 4.   Flag dates for which `eto` values are stuck.
# 5.   Impute Koue-Bokkeveld (a.k.a. kbv) `eto` data onto stuck `eto` data.
# 6.   Flag raining events for which the rain exceeds 2 millimeters.
# 7.   Flag irrigation events.
# 8.   Flag events in which data were simulated.
# 9.   Flag dates in which data blips and data dips were observed.
# 10.a Flag events in which the heat units are equal to zero.
# 10.b Interpolate missing heat_units.
# 10.c Get the cumulative Growing Degree Days (GDD)
# 11.  Flag events in which `etcp` is known to be faulty/contaminated.
# 12.  Flag samples in which the calculated `kcp` values deviate by more than
#      50% from the accepted norm.
# 13.  Extract the dates and associated `kcp` values for which
#      `binary_value == 0`.
# 14.  Append the cleaned dates and `kcp` values to the growing list
#      `data_to_plot`.
# 15.  Save all the data and metadata of all the cleaning and crunching
#      operations to an Excel sheet.  One Excel sheet represents ONE probe-id.
#      A set of Excel spreadsheets represents a set of probe-id's.
# 16.  Once all the sheets have been populated, save and close the .xlsx file.
#      Also save .xlsx containing cleaned kcp data of all the probes.
# =============================================================================
writer = pd.ExcelWriter("./data/processed_probe_data.xlsx",
                        engine="xlsxwriter")
writer_2 = pd.ExcelWriter("./data/cleaned_data_for_overlay.xlsx",
                          engine="xlsxwriter")
# Create a for loop iterating over all the different probe datasets.
# In each iteration, perform all the necessary flagging.
for probe_id in probe_ids:
    print("\n")
    print("*" * 80)
    print("Busy with probe: {:s}.".format(probe_id))

    # 1.
    # Load daily probe data from Excel Sheet.
    df = load_probe_data(probe_name=probe_id)

    # 2.
    # Create and Initialize 2 flagging-related columns in the DataFrame.
    df = initialize_flagging_columns(dataframe=df)

    # 3.
    # Drop redundant columns.
    df = cleaning_operations.drop_redundant_columns(df=df)

    # 4.
    # Flag dates for which `eto` values are stuck.
    bad_eto_days, df = cleaning_operations.flag_spurious_et(df=df)

    # 5.
    # Impute Koue-Bokkeveld (a.k.a. kbv) `eto` data onto stuck `eto` data.
    # df = cleaning_operations.impute_kbv_data(df, bad_eto_days)

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
    # Flag dates in which data blips and data dips were observed.
    df = cleaning_operations.flag_suspicious_and_missing_profile_events(df)

    # 10.a
    # Flag events in which the heat units were stuck.
    df = cleaning_operations.flag_spurious_heat_units(df)

    # 10.b
    # Interpolate missing heat units
    df = cleaning_operations.interpolate_missing_heat_units(df,
                                                            method="nearest")

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
    # Flag samples in which the calculated `kcp` values deviate by more than
    # 50% from the accepted norm.
    df = cleaning_operations.flag_unwanted_kcp(df)

    # 13.
    # Extract the dates and associated `kcp` values for which
    # `binary_value == 0`.
    some_tuple = cleaning_operations.get_final_dates(df)
    starting_year, new_dates, useful_dates = some_tuple
    kcp_values = df.loc[useful_dates, "kcp"].values

    # 14.
    # Append the cleaned dates and `kcp` values to `data_to_plot`.
    df_2 = pd.DataFrame(data={"y_scatter": kcp_values}, index=new_dates,
                        columns=["y_scatter"], copy=True)
    df_2.to_excel(writer_2, sheet_name=probe_id, index_label="datetime_stamp",
                  float_format="%.7f")

    # 15.
    # Save all the data and metadata.
    df.to_excel(writer, sheet_name=probe_id)

# 16.
# Once all the sheets have been populated, save and close the .xlsx files.
writer.save()  # `./data/processed_probe_data.xlsx`
writer_2.save()  # `./data/cleaned_data_for_overlay.xlsx`


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Save all the cleaned data to file.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# The code below can be summarised as follows:
# 1. Stack the cleaned data, from `./data/cleaned_data_for_overlay.xlsx`,
#    together into one MultiIndex DataFrame.
# 2. Populate a .txt file containing all the names of the probeIDs:
#    This is saved at `./data/probe_ids.txt`
# 3. Save dataframe containing reference crop coefficient data to an excel
#    spreadsheet.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.
# Stack the cleaned data together into one MultiIndex DataFrame.
# Outer level index is "probe_id", and inner level index is "datetimeStamp".
cleaned_dict = pd.read_excel("./data/cleaned_data_for_overlay.xlsx",
                             sheet_name=None, header=0, parse_dates=True,
                             index_col=0)

dfs = []  # A list to be populated with the probe dataframes.
for probe_id in cleaned_dict.keys():
    temp_df = cleaned_dict[probe_id]
    temp_df["probe_id"] = probe_id
    dfs.append(temp_df)
stacked_df = pd.concat(dfs)
stacked_multi_df = stacked_df.set_index(["probe_id", stacked_df.index])
stacked_multi_df.sort_index(level=["probe_id", "datetime_stamp"], inplace=True)
stacked_multi_df.to_excel("./data/stacked_cleaned_data_for_overlay.xlsx",
                          float_format="%.7f", columns=["y_scatter"],
                          header=True, index=True)

# 3.
# Extract and save reference "cco" data
processed_df = pd.read_excel("./data/processed_probe_data.xlsx",
                             sheet_name="{}".format(probe_ids[0]), header=0,
                             index_col=0, squeeze=True, parse_dates=True)
cco_df = processed_df["cco"].to_frame()
dates = processed_df.index.to_series()
wrapped_dates = date_wrapper(date_iterable=dates, starting_year=starting_year)
cco_df["wrapped_date"] = wrapped_dates
# Make "wrapped_date" the new index of the cco_df DataFrame:
cco_df.set_index(keys="wrapped_date", inplace=True)
# Remove any duplicate entries:
cco_df = cco_df[~cco_df.index.duplicated(keep="first")]
# Sort the index in ascending order:
cco_df.sort_index(ascending=True, inplace=True)
# Calculate the season_day and convert it to type integer:
cco_df["season_day"] = cco_df.index - season_start_date
cco_df["season_day"] = cco_df["season_day"].dt.days
# Save the cco_df DataFrame to "./data/reference_crop_coeff.xlsx":
cco_df.to_excel("./data/reference_crop_coeff.xlsx", sheet_name="sheet_1",
                header=True, index=True)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
