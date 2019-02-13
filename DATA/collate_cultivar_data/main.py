import numpy as np
import datetime
from datetime import timedelta
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
# from cleaning_operations import *
import cleaning_operations


def load_probe_data(probe_name):
    dataframe = pd.read_excel("../Golden_Delicious_daily_data.xlsx", sheet_name=probe_name,
                              index_col=0, parse_dates=True)
    refined_columns = []
    for c in dataframe.columns:
        refined_columns.append(c.lstrip())
    dataframe.columns = refined_columns
    return dataframe


def initialize_flagging_columns(dataframe):
    dataframe["binary_value"] = 0.0
    dataframe["description"] = str()
    return dataframe


probe_ids = ["P-370", "P-371", "P-372", "P-384", "P-391", "P-392", "P-891"]

# TODO: encase all the code below inside a for-loop; we need to iterate over all the probes present in the list.

probe_id = "P-371"

df = load_probe_data(probe_name=probe_id)

df = initialize_flagging_columns(dataframe=df)

# Cleaning operation 1: drop redundant columns
cleaning_operations.drop_redundant_columns(df=df)
print(df.columns)
