import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# Declare constants.
# The only constants here are:
#   1.  path_to_probe_ids --> .txt File containing the 1 probe_id per line.
#   2.  probe_ids --> A list of strings containing the probe ids.
# ----------------------------------------------------------------------------------------------------------------------
path_to_probe_ids = "./collate_cultivar_data_new/data/probe_ids.txt"
with open(path_to_probe_ids, "r") as f:
    probe_ids = f.readlines()
    probe_ids = [x.rstrip("\n") for x in probe_ids]
print(probe_ids)
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Define some helper functions
# ======================================================================================================================
def dateparse(somestring):
    return pd.datetime.strptime(somestring, "%Y-%m-%d")


def datetimeparse(somestring):
    return pd.datetime.strptime(somestring, "%Y-%m-%d %H:%M:%S")
# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Firstly, let us work with the daily data (GDD, and the Waterbalance Data)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
p370_df = pd.read_csv("P-370_daily_data.csv", sep=",", header=0, parse_dates=True,
                      index_col=0, na_values="None", date_parser=dateparse)
p371_df = pd.read_csv("P-371_daily_data.csv", sep=",", header=0, parse_dates=True,
                      index_col=0, na_values="None", date_parser=dateparse)
p372_df = pd.read_csv("P-372_daily_data.csv", sep=",", header=0, parse_dates=True,
                      index_col=0, na_values="None", date_parser=dateparse)
p384_df = pd.read_csv("P-371_daily_data.csv", sep=",", header=0, parse_dates=True,
                      index_col=0, na_values="None", date_parser=dateparse)
p391_df = pd.read_csv("P-391_daily_data.csv", sep=",", header=0, parse_dates=True,
                      index_col=0, na_values="None", date_parser=dateparse)
p392_df = pd.read_csv("P-392_daily_data.csv", sep=",", header=0, parse_dates=True,
                      index_col=0, na_values="None", date_parser=dateparse)
p891_df = pd.read_csv("P-891_daily_data.csv", sep=",", header=0, parse_dates=True,
                      index_col=0, na_values="None", date_parser=dateparse)

with pd.ExcelWriter('Golden_Delicious_daily_data_new.xlsx', date_format="yyyy-mm-dd", datetime_format="yyyy-mm-dd",
                    engine="xlsxwriter") as writer:
    p370_df.to_excel(writer, header=True, index=True, sheet_name="P-370", index_label="date", verbose=True)
    p371_df.to_excel(writer, header=True, index=True, sheet_name="P-371", index_label="date", verbose=True)
    p372_df.to_excel(writer, header=True, index=True, sheet_name="P-372", index_label="date", verbose=True)
    p384_df.to_excel(writer, header=True, index=True, sheet_name="P-384", index_label="date", verbose=True)
    p391_df.to_excel(writer, header=True, index=True, sheet_name="P-391", index_label="date", verbose=True)
    p392_df.to_excel(writer, header=True, index=True, sheet_name="P-392", index_label="date", verbose=True)
    p891_df.to_excel(writer, header=True, index=True, sheet_name="P-891", index_label="date", verbose=True)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


"""
# ----------------------------------------------------------------------------------------------------------------------
# Now secondly, let us work with the Probe Measurements (azm, soil moisture, and soil temperature)
# ----------------------------------------------------------------------------------------------------------------------
p370_probe_df = pd.read_csv("P-370_probe_data.csv", sep=",", header=0, parse_dates=True,
                            index_col=0, na_values="None", date_parser=datetimeparse)
p371_probe_df = pd.read_csv("P-371_probe_data.csv", sep=",", header=0, parse_dates=True,
                            index_col=0, na_values="None", date_parser=datetimeparse)
p372_probe_df = pd.read_csv("P-371_probe_data.csv", sep=",", header=0, parse_dates=True,
                            index_col=0, na_values="None", date_parser=datetimeparse)
p384_probe_df = pd.read_csv("P-371_probe_data.csv", sep=",", header=0, parse_dates=True,
                            index_col=0, na_values="None", date_parser=datetimeparse)
p391_probe_df = pd.read_csv("P-371_probe_data.csv", sep=",", header=0, parse_dates=True,
                            index_col=0, na_values="None", date_parser=datetimeparse)
p392_probe_df = pd.read_csv("P-371_probe_data.csv", sep=",", header=0, parse_dates=True,
                            index_col=0, na_values="None", date_parser=datetimeparse)
p891_probe_df = pd.read_csv("P-371_probe_data.csv", sep=",", header=0, parse_dates=True,
                            index_col=0, na_values="None", date_parser=datetimeparse)

with pd.ExcelWriter('Golden_Delicious_probe_data.xlsx', date_format="yyyy-mm-dd", datetime_format="yyyy-mm-dd HH:MM:SS",
                    engine="xlsxwriter") as writer:
    p370_probe_df.to_excel(writer, header=True, index=True, sheet_name="P-370", index_label="datetime", verbose=True)
    p371_probe_df.to_excel(writer, header=True, index=True, sheet_name="P-371", index_label="datetime", verbose=True)
    p372_probe_df.to_excel(writer, header=True, index=True, sheet_name="P-372", index_label="datetime", verbose=True)
    p384_probe_df.to_excel(writer, header=True, index=True, sheet_name="P-384", index_label="datetime", verbose=True)
    p391_probe_df.to_excel(writer, header=True, index=True, sheet_name="P-391", index_label="datetime", verbose=True)
    p392_probe_df.to_excel(writer, header=True, index=True, sheet_name="P-392", index_label="datetime", verbose=True)
    p891_probe_df.to_excel(writer, header=True, index=True, sheet_name="P-891", index_label="datetime", verbose=True)
# ----------------------------------------------------------------------------------------------------------------------
"""
