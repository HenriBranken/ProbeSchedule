import numpy as np
import datetime
from datetime import timedelta
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from cleaning_operations import *


probe_ids = ["P-370", "P-371", "P-372", "P-384", "P-391", "P-392", "P-891"]

# TODO: place all the code below inside a for-loop; we need to iterate over all the probes.

probe_id = "P-371"

df = pd.read_excel("../Golden_Delicious_daily_data.xlsx", sheet_name=probe_id, index_col=0, parse_dates=True)
refined_columns = []
for c in df.columns:
    refined_columns.append(c.lstrip())
df.columns = refined_columns

# Add `binary_value` and `description` columns to the probe's DataFrame
# These columns will help to keep track of flagging operations
df["binary_value"] = 0.0
df["description"] = str()  # an empty string

# Cleaning operation 1: drop redundant columns
drop_redundant_columns(df)
