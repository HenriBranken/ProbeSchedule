import numpy as np
import pandas as pd
import datetime
from cleaning_operations import description_dict


# ----------------------------------------------------------------------------------------------------------------------
# Import the necessary data:
# ----------------------------------------------------------------------------------------------------------------------
# `processed_dict` is of type OrderedDict.  `None` is passed to `sheet_name`, and so we read in simultaneously all the
# sheets
processed_dict = pd.read_excel("./data/processed_probe_data.xlsx", sheet_name=None)

dfs = []
for probe_id in processed_dict.keys():
    temp_df = processed_dict[probe_id]
    temp_df["probe_id"] = probe_id
    dfs.append(temp_df)
# Create one massive DataFrame containing all the sheets' data.  The column "probe_id" specifies the associated
# "probe_id"
df = pd.concat(dfs)

# Create a MultiIndex DataFrame where "probe_id" is the outermost index, and "date" is the innermost index.
df2 = df.set_index(["probe_id", "date"])
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Some examples/demos on how to extract information from MultiIndex DataFrames.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# How to print a very specific sample from the MultiIndex DataFrame.
# print(df2.loc[("P-392", "2018-02-05"), ["heat_units"]])

# METHOD 1
# Collecting all the outer indices of the MultiIndex DataFrame; more elegant.
# In this example, the outer level corresponds to the probe_id.
level_0 = list(df2.index.unique(level="probe_id"))
# print(level_0)

# METHOD 2
# Collecting all the inner indices of the MultiIndex DataFrame; more elegant.
# In this example, the inner level corresponds to dates.
level_1 = list(df2.index.unique(level="date"))
# print(level_1[0], level_1[-1])

# METHOD 3
# Collecting all the outer/inner indices of the MultiIndex DataFrame; elegant form.
outer_level = list(df2.index.get_level_values("probe_id").unique())
inner_level = list(df2.index.get_level_values("date").unique())

# Extracting a sub-DataFrame that corresponds to a particular probe_id.
p_370_df = df2.loc[("P-370", ), :]
# p_370_df = df2.loc["P-370"]
# p_370_df = df2.loc["P-370", :]
# print(p_370_df)

# Extracting all the entries for a particular date and column.
# print(df2.loc[(outer_level, "2018-02-02"), ["profile"]])

# pandas.IndexSlice --> Create an object to more easily perform multi-index slicing.
idx = pd.IndexSlice
# print(df2.loc[idx[:, ("2018-02-02", "2019-02-03")], ["profile"]])
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
