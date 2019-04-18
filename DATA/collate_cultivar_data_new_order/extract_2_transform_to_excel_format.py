import pandas as pd
import os


# =============================================================================
# Declare constants.
# The only constants here are:
# 1. path_to_probe_ids --> .txt file containing the 1 probe_id per line.
# 2. probe_ids --> A list of strings containing the probe ids.
# =============================================================================
path_to_probe_ids = "./data/probe_ids.txt"
with open(path_to_probe_ids, "r") as f:
    probe_ids = [x.rstrip() for x in f.readlines()]
# =============================================================================


# -----------------------------------------------------------------------------
# Remove the old `cultivar_*.xlsx` file(s).
# (If note removed, they will be overwritten at a later stage anyway.)
# -----------------------------------------------------------------------------
directory = "./data/"
files = os.listdir(directory)
for file in files:
    if file.startswith("cultivar_"):
        os.remove(os.path.join(directory, file))
        print("Removed the file named: {}.".format(file))
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Here we process the daily data (Heat Units (GDD), and the Waterbalance Data).
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# To use XlsxWriter with Pandas, you specify it as the Excel writer engine.
# To write to multiple sheets it is necessary to create an ExcelWriter object
# with a target file name, and specify a sheet in the file to write to.
writer = pd.ExcelWriter("./data/cultivar_data.xlsx", engine="xlsxwriter",
                        date_format="%Y-%m-%d")

# Loop over the different probe ids.
for p in probe_ids:
    print("Currently busy with probe {:s}.".format(p))
    sub_df = pd.read_csv("./data/" + p + "_daily_data.csv", sep=",", header=0,
                         parse_dates=True, index_col=0,
                         na_values=["nan", "None"])
    sub_df.to_excel(writer, header=True, index=True, sheet_name=p,
                    index_label="date", verbose=True)

writer.save()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
