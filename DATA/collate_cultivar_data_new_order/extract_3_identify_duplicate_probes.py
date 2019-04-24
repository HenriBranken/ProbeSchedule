import pandas as pd
from helper_functions import safe_removal


# =============================================================================
# Declare necessary "constants".
# =============================================================================
# Populate the list, `probe_ids`, containing a sequence of strings that
# represent the probe_ids.
with open("./data/probe_ids.txt", "r") as f:
    probe_ids = [p.rstrip() for p in list(f)]
# =============================================================================


# -----------------------------------------------------------------------------
# Import the necessary data
# -----------------------------------------------------------------------------
# `./data/cultivar_data.xlsx` is the raw data as extracted from an API call.
# We specify `sheet_name=None` to get all the sheets.
processed_dict = pd.read_excel("./data/cultivar_data.xlsx", sheet_name=None)

# A list to be populated with the probe dataframes.
# Each pandas dataframe corresponds to an individual probe.
dfs = []
for probe_id in processed_dict.keys():
    temp_df = processed_dict[probe_id]
    temp_df["probe_id"] = probe_id
    dfs.append(temp_df)

# Create one massive DataFrame containing all the sheets' (probes') data.
# The column "probe_id" is used to indicate the probe-id associated with any
# sample.
df = pd.concat(dfs)

# Create a MultiIndex DataFrame where "probe_id" is the outermost index, and
# "date" is the innermost index.
multi_df = df.set_index(["probe_id", "date"])
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define some helper functions
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Extract a sub_dataframe containing all the samples of a particular probe
#    as indicated by the `label` parameter.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.
def get_sub_df(multi_dataframe, label):
    return multi_dataframe.loc[(label, ), :]
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Remove old files generated from previous execution of this script.
# -----------------------------------------------------------------------------
file_list = ["./data/probes_to_be_discarded.txt", "./data/probe_ids.txt",
             "./data/cultivar_data_unique.xlsx"]
safe_removal(file_list=file_list)
# -----------------------------------------------------------------------------


# =============================================================================
# Do the data manipulation (checking) to see if any sub-DataFrames are
# duplicates.
# =============================================================================
# Populate list with all the probe DataFrames
probe_dfs = [get_sub_df(multi_dataframe=multi_df, label=p) for p in probe_ids]

master_list_of_duplicates = []
for i, p in enumerate(probe_ids):
    base_df = probe_dfs[i]
    base_probe_id = p
    remainder_list_of_probe_ids = [probe for probe in probe_ids
                                   if probe != base_probe_id]
    to_be_appended = [p]
    for pp in remainder_list_of_probe_ids:
        sub_df = get_sub_df(multi_dataframe=multi_df, label=pp)
        if base_df.equals(sub_df):  # here we perform the actual checking.
            to_be_appended.append(pp)
            to_be_appended.sort()
    if len(to_be_appended) > 1:
        master_list_of_duplicates.append(to_be_appended)

# Remove any duplicates sub-lists in `master_list_of_duplicates`
master_list_of_duplicates = [list(some_sub_tuple) for some_sub_tuple in
                             set(map(tuple, master_list_of_duplicates))]
print("Sublists of probes that are identical:")
print(master_list_of_duplicates)
print("-" * 80)

probes_to_be_popped = []
for i in range(len(master_list_of_duplicates)):
    # Pop all the probes filling indices 1 and onward in the sublists.
    probes_to_be_popped.append(master_list_of_duplicates[i][1:])

# Flatten out `probes_to_be_popped` so that it is a simple 1-dimensional list.
probes_to_be_popped = [item for sublist in probes_to_be_popped
                       for item in sublist]

print("\nProbes that are redundant, and thus need to be removed:")
print(probes_to_be_popped)
print("-" * 80)
# =============================================================================


# -----------------------------------------------------------------------------
# Write to two different `.txt` files the probe-id(s) that:
# 1. Need to be discarded from any future use whatsoever (they are redundant).
# 2. Need to be kept (i.e. any NON-redundant probe_id).
# -----------------------------------------------------------------------------
# The probe(s) to be discarded are written to
# `./data/probes_to_be_discarded.txt`.
with open("./data/probes_to_be_discarded.txt", "w") as f:
    f.write("\n".join(("{:s}".format(p)) for p in probes_to_be_popped))

# Generate the new list of probes that need to be kept.
# This list is written to `./data/probe_ids.txt`.
# Use list comprehension to get all the probes that do not belong to
# `probes_to_be_popped`.  (Note the use of `not in`).
new_probe_ids = [p for p in probe_ids if p not in probes_to_be_popped]
with open("./data/probe_ids.txt", "w") as f:
    f.write("\n".join(("{:s}".format(p)) for p in new_probe_ids))
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create a new Excel file that does not contain any duplicate probe datasets.
# Each sheet corresponds to a unique probe dataset.  Collectively, there are no
# duplicates in the probe-set.
# Store the Excel-file at `./data/cultivar_daily_data_unique.xlsx`.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Delete (probe) key from DataFrame Dictionary using the del keyword:
for p in probes_to_be_popped:
    del processed_dict[p]

# Instantiate a new Excel-File object at `./data/cultivar_data_unique.xlsx`.
writer = pd.ExcelWriter("./data/cultivar_data_unique.xlsx",
                        engine="xlsxwriter")

# Populate the Excel file with different sheets.  One sheet per (unique) probe.
# The sheet name is equal to the (unique) probe_id.
for p in new_probe_ids:
    sub_df = processed_dict[p]
    sub_df.drop(axis=1, columns=["probe_id"], inplace=True)
    sub_df.set_index(keys="date", drop=True, append=False, inplace=True)
    sub_df.to_excel(writer, sheet_name=p, header=True, index=True,
                    index_label=["date"])
writer.save()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
