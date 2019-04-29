import pandas as pd
from cleaning_operations import description_dict
from helper_functions import safe_removal

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Declare some constants
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# some string constants used when writing text to file later on in the report.
line = "+" + "-"*43 + "+" + "-"*15 + "+" + "-"*18 + "+"
border = "+" + "-"*78 + "+"
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Remove old files generated by a previous execution of this script.
# -----------------------------------------------------------------------------
file_list = ["./data/data_report.txt"]
safe_removal(file_list=file_list)
# -----------------------------------------------------------------------------


# =============================================================================
# Define some helper functions
# =============================================================================
# 1. reporter(file_to_write_to, dataframe, brief_desc, probe=None):
#    Gives us information on how many data samples are affected by a certain
#    event, such as, for example, `Rain perturbing etcp`, etc...
#    If probe is None, then we determine the amount of samples affected for the
#    entire set of probes.
#
# 2. conclusion(file_to_write_to, dataframe, probe=None):
#    Tells you how many samples are useful after flagging a probe dataset.
#    If probe is None, then we determine the amount of useful samples for the
#    entire set of probes on a farm/block.
# =============================================================================
def reporter(file_to_write_to, dataframe, brief_desc, probe=None):
    # Give us info on how many samples are affected by a certain flagging
    # event.  If probe=None, then we determine the amount of samples affected
    # in the entire set of probes.
    if probe:  # i.e. we are working with a subset DataFrame.
        nm = dataframe.loc[probe, "description"].str.contains(brief_desc).sum()
        n_tot_entries = len(list(dataframe.index.unique(level="date")))
        perc = nm / n_tot_entries * 100
        file_to_write_to.write("| {:<42s}".format(brief_desc) +
                               "|" + " "*8 + "{:>5.2f}%".format(perc) + " " +
                               "|" +
                               " "*8 +
                               "{:>4d}/{:<4d}".format(nm, n_tot_entries) +
                               " |\n")
    else:
        nm = dataframe["description"].str.contains(brief_desc).sum()
        n_tot_entries = len(df.index)
        perc = nm / n_tot_entries * 100
        file_to_write_to.write("| {:<42s}".format(brief_desc) +
                               "|" + " " * 8 + "{:>5.2f}%".format(perc) + " " +
                               "|" + " " * 8 +
                               "{:>4d}/{:<4d}".format(nm, n_tot_entries) +
                               " |\n")


def conclusion(file_to_write_to, dataframe, probe=None):
    # Report the number of samples that are useful for a probe (or a probe set
    # if probe=None) after all the flagging iterations were applied.
    if not probe:  # i.e. we are giving a conclusion for the entire probe SET.
        n_tot_entries = len(dataframe.index)
        n_affected = dataframe["binary_value"].sum()
        n_useful = n_tot_entries - n_affected
        calc = 100 - n_affected / n_tot_entries * 100
        if calc < 10.0:
            print("The quality of your data is not very good "
                  "(only {:.0f}% are useful).\n"
                  "Consider getting more probes.\n".format(calc))

        conc_string = "| Only {:.2f}% of data, that is {}/{} samples, " \
                      "are useful for the probe SET.".format(calc, n_useful,
                                                             n_tot_entries)
        n_empty_space = int(80 - len(conc_string) - 1)
        file_to_write_to.write(line + "\n")
        file_to_write_to.write(conc_string + " "*n_empty_space + "|\n")
    else:  # Only give a conclusion for a particular probe_id.
        n_tot_entries = len(dataframe.loc[probe])
        n_affected = dataframe.loc[(probe, ), "binary_value"].sum()
        n_useful = n_tot_entries - n_affected
        calc = 100 - n_affected / n_tot_entries * 100

        conc_string = "| Only {:.2f}% of data, that is {}/{} samples," \
                      " are useful for probe {}.".format(calc, n_useful,
                                                         n_tot_entries, probe)
        n_empty_space = int(80 - len(conc_string) - 1)
        file_to_write_to.write(line + "\n")
        file_to_write_to.write(conc_string + " " * n_empty_space + "|\n")
# =============================================================================


# -----------------------------------------------------------------------------
# Import the necessary data:
# -----------------------------------------------------------------------------
# `processed_dict` is of type OrderedDict.  `None` is passed to `sheet_name`,
# and so we read in simultaneously ALL the sheets.
processed_dict = pd.read_excel("./data/processed_probe_data.xlsx",
                               sheet_name=None)

dfs = []  # A list to be populated with the dataframes.
for probe_id in processed_dict.keys():
    temp_df = processed_dict[probe_id]
    temp_df["probe_id"] = probe_id
    dfs.append(temp_df)
# Create one massive DataFrame containing all the sheets' data.
# The column "probe_id" specifies the "probe_id" associated with a sample.
df = pd.concat(dfs)

# Create a MultiIndex DataFrame where "probe_id" is the outermost index,
# and "date" is the innermost index.
multi_df = df.set_index(["probe_id", "date"])
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Some DEMOS on how to extract information from a MultiIndex DataFrame.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# How to print a very specific sample from the MultiIndex DataFrame.
# print(df2.loc[("P-392", "2018-02-05"), ["heat_units"]])

# METHOD 1
# Collecting all the outer indices of the MultiIndex DataFrame; Elegant!
# In this example, the outer level corresponds to the probe_id.
# level_0 = list(df2.index.unique(level="probe_id"))
# print(level_0)

# METHOD 2
# Collecting all the inner indices of the MultiIndex DataFrame; Elegant!
# In this example, the inner level corresponds to dates.
# level_1 = list(df2.index.unique(level="date"))
# print(level_1[0], level_1[-1])

# METHOD 3
# Collecting all the outer/inner indices of the MultiIndex DataFrame; Elegant!
# outer_level = list(df2.index.get_level_values("probe_id").unique())
# inner_level = list(df2.index.get_level_values("date").unique())

# Extracting a sub-DataFrame that corresponds to a particular probe_id.
# p_370_df = df2.loc[("P-370", ), :]
# p_370_df = df2.loc["P-370"]
# p_370_df = df2.loc["P-370", :]
# print(p_370_df)

# Extracting all the entries for a particular date and column.
# print(df2.loc[(outer_level, "2018-02-02"), ["profile"]])

# idx = pd.IndexSlice
# print(df2.loc[idx[:, ("2018-02-02", "2019-02-03")], ["profile"]])
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Block of code to generate the report, which is saved at
# "./data/data_report.txt".
# =============================================================================
# Get a list of all the probe_ids.
probe_ids = list(multi_df.index.get_level_values("probe_id").unique())

# Get a list of all the description_dict keys.
dict_keys = list(description_dict.keys())

# Open a file in writing mode located at `./data/data_report.txt`.
f = open("./data/data_report.txt", mode="w")

for p in probe_ids:  # Loop over all the probes.
    f.write(border + "\n")
    f.write("|" + "{:^78}".format("Probe {:s} Report:".format(p)) + "|\n")
    f.write(line + "\n")
    for k in dict_keys:  # Loop over all the possible descriptions.
        reporter(file_to_write_to=f, dataframe=multi_df, probe=p,
                 brief_desc=description_dict[k])
    conclusion(file_to_write_to=f, dataframe=multi_df, probe=p)
    f.write(border + "\n\n\n")

f.write(border + "\n")
f.write("|" + "{:^78}".format("Report for the entire SET of probes:") + "|\n")
f.write(line + "\n")

for k in dict_keys:  # Loop over all the possible flagging descriptions.
    reporter(file_to_write_to=f, dataframe=multi_df,
             brief_desc=description_dict[k], probe=None)
conclusion(file_to_write_to=f, dataframe=multi_df, probe=None)
f.write(border + "\n")

# We may proceed to CLOSE the file.
f.close()
# =============================================================================
