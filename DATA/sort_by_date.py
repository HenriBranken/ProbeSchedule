"""
In the command-line interface of your terminal, this script should be executed, for example, as follows:

python sort_by_date.py P-371_daily_data.csv

It will fetch the messy data from P-371_daily_data.csv, sort it chronologically, store it into a pandas dataframe,
and dump the contents into a .csv file.

The output file will have "_chron" appended to the end of the original file-name.  So in this example, your
output file will be:
P-371_daily_data_chron.csv
"""

import numpy as np
import datetime
import pandas as pd
import sys

f = sys.argv[1]

my_array = []

with open(f, 'r') as input_file:
    column_names = input_file.readline().rstrip().split(sep=",")
    for line in input_file:
        entries = line.rstrip().split(sep=",")
        my_array.append(entries)

length = len(sorted(my_array, key=lambda k: len(k), reverse=True)[0])
my_array = np.array([xi + ['None']*int((length - len(xi))) for xi in my_array])

my_array = np.asarray(sorted(my_array, key=lambda x: datetime.datetime.strptime(x[0], "%Y-%m-%d")))

df = pd.DataFrame(data=my_array[:, :],
                  columns=column_names)
df.set_index(keys="date", drop=True, inplace=True)
df.index = pd.to_datetime(df.index)

g = str(f).rstrip(".csv") + "_chron.csv"
df.to_csv(g, sep=",")
