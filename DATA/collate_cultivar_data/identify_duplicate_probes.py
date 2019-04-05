import pandas as pd


# ======================================================================================================================
# Declare "constants"
# ======================================================================================================================
probe_numbers = [370, 371, 372, 384, 391, 392, 891]  # Ouplaas, Kromfontein.
probe_ids = ["P-"+str(num) for num in probe_numbers]
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Import the necessary data
# ----------------------------------------------------------------------------------------------------------------------
# `../Golden_Delicious_daily_data.xlsx` is the raw data as extracted from an API call.
processed_dict = pd.read_excel("../Golden_Delicious_daily_data.xlsx", sheet_name=None)

dfs = []  # A list to be populated with the probe dataframes.  Each dataframe corresponds to an individual probe.
for probe_id in processed_dict.keys():
    temp_df = processed_dict[probe_id]
    temp_df["probe_id"] = probe_id
    dfs.append(temp_df)

# Create one massive DataFrame containing all the sheets' data.  The column "probe_id" specifies the probe_id
# associated with a particular sample.
df = pd.concat(dfs)

# Create a MultiIndex DataFrame where "probe_id" is the outermost index, and "date" is the innermost index.
# The outermost index is sorted according to "probe_id", whereas the innermost index is sorted according to "date".
multi_df = df.set_index(["probe_id", "date"])
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define some helper functions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Extract a sub_dataframe containing all the samples of a particular probe as indicated by the "label" parameter.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Extract a sub_dataframe containing all the samples of a particular probe as indicated by the "label" parameter.
def get_sub_df(multi_dataframe, label):
    return multi_dataframe.loc[(label, ), :]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Do the data manipulation to see if any sub-DataFrames are equal to one another
# ======================================================================================================================
# Populate list with probe DataFrames
probe_dfs = [get_sub_df(multi_dataframe=multi_df, label=p) for p in probe_ids]

master_list_of_duplicates = []
for i, p in enumerate(probe_ids):
    base_df = probe_dfs[i]
    base_probe_id = p
    remainder_list_of_probe_ids = [probe for probe in probe_ids if probe != base_probe_id]
    to_be_appended = [p]
    for pp in remainder_list_of_probe_ids:
        sub_df = get_sub_df(multi_dataframe=multi_df, label=pp)
        if base_df.equals(sub_df):
            to_be_appended.append(pp)
            to_be_appended.sort()
    if len(to_be_appended) > 1:
        master_list_of_duplicates.append(to_be_appended)

# Remove any duplicates sub-lists in `master_list_of_duplicates`
master_list_of_duplicates = [list(some_sub_tuple) for some_sub_tuple in set(map(tuple, master_list_of_duplicates))]
print("Sublists of probes that are identical:")
print(master_list_of_duplicates)
print("-" * 80)

probes_to_be_popped = []
for i in range(len(master_list_of_duplicates)):
    # Pop all the probes filling indices 1 and onward in the sublists.
    probes_to_be_popped.append(master_list_of_duplicates[i][1:])

# Flatten out `probes_to_be_popped` so that it is a simple 1-dimensional list.
probes_to_be_popped = [item for sublist in probes_to_be_popped for item in sublist]

print("\nProbes that are redundant, and thus need to be removed:")
print(probes_to_be_popped)
print("-" * 80)
# ======================================================================================================================
