import pandas as pd
import os


# =============================================================================
# Declare constants.
# The only constants here are:
# 1. path_to_probe_ids --> .txt File containing the 1 probe_id per line.
# 2. probe_ids --> A list of strings containing the probe ids.
# =============================================================================
path_to_probe_ids = "./probe_ids.txt"
with open(path_to_probe_ids, "r") as f:
    probe_ids = [x.rstrip() for x in f.readlines()]
# =============================================================================


# -----------------------------------------------------------------------------
# Remove the old *.xlsx file(s).
# -----------------------------------------------------------------------------
directory = "./"
files = os.listdir(directory)
for file in files:
    if file.endswith(".xlsx"):
        os.remove(os.path.join(directory, file))
        print("Removed the file named: {}.".format(file))
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Here we process the daily data (GDD, and the Waterbalance Data).
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
writer = pd.ExcelWriter("cultivar_data.xlsx", engine="xlsxwriter",
                        date_format="%Y-%m-%d")

for p in probe_ids:
    print("Currently busy with probe {:s}.".format(p))
    sub_df = pd.read_csv(p + "_daily_data.csv", sep=",", header=0,
                         parse_dates=True, index_col=0,
                         na_values=["nan", "None"])
    sub_df.to_excel(writer, header=True, index=True, sheet_name=p,
                    index_label="date", verbose=True)

writer.save()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
