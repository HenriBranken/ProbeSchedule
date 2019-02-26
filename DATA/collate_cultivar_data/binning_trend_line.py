import pandas as pd
import numpy as np
import datetime
from cleaning_operations import BEGINNING_MONTH
pd.set_option('display.max_columns', 6)

# ----------------------------------------------------------------------------------------------------------------------
# Load the necessary data
# The kcp_vs_datetime array has 2 columns:
#   The first column is the datetime stamp.
#   The second column is the fitted (quadratic) kcp trend
# The cleaned_kcp_df is a pandas DataFrame containing the cleaned kcp data
# ----------------------------------------------------------------------------------------------------------------------
# Data associated with the quadratic fit (2nd-order polynomial)
kcp_vs_datetime = np.load("data/daily_trend_of_kcp_vs_datetime.npy")
datetimestamp = kcp_vs_datetime[:, 0]
kcp_trend = kcp_vs_datetime[:, 1]

starting_year = datetimestamp[0].year
starting_week = datetimestamp[0].isocalendar()[1]
starting_date = datetimestamp[0]

# The scatter plot of cleaned kcp as a function of datetime
cleaned_kcp_df = pd.read_excel("data/kcp_vs_days.xlsx", sheet_name="sheet_1", names=["days", "kcp"], index_col=0,
                               parse_dates=True)
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Define helper functions
# ======================================================================================================================
def season_month_mapper(cal_m):
    if cal_m < 7:
        return int(cal_m + 13 - BEGINNING_MONTH)
    else:
        return int(cal_m + 1 - BEGINNING_MONTH)


def from_calendar_month_datetime_mapper(cal_m):
    if BEGINNING_MONTH <= cal_m <= 12:
        return datetime.datetime(year=starting_year, month=cal_m, day=15)
    else:
        return datetime.datetime(year=starting_year + 1, month=cal_m, day=15)


def calendar_week_mapper(season_w):
    delta = starting_week - 1
    if (season_w + delta) > 52:
        return int(season_w + delta - 52)
    else:
        return int(season_w + delta)


def from_season_week_datetime_mapper(season_w):
    return starting_date + datetime.timedelta(days=7*season_w - 4)


def calendar_day_mapper(season_d):
    specific_date = starting_date + datetime.timedelta(days=season_d - 1)
    return int(specific_date.timetuple()[-2])
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Daily for kcp.
# Save a pandas DataFrame, called kcp_vs_day_df.  kcp_vs_day_df contains the columns:
#   datetimestamp --> the index of the DataFrame
#   calendar_day --> self-explanatory
#   season_day --> self-explanatory
#   kcp_daily_basis --> self-explanatory
# ----------------------------------------------------------------------------------------------------------------------
season_day = 1
kcp_vs_season_day = []
for _, k in zip(datetimestamp, kcp_trend):
    kcp_vs_season_day.append([season_day, k])
    season_day += 1
kcp_vs_season_day = np.array(kcp_vs_season_day)

kcp_vs_day_df = pd.DataFrame(data={"season_day": kcp_vs_season_day[:, 0].astype(int),
                                   "day_averaged_kcp": kcp_vs_season_day[:, 1].astype(float)})
kcp_vs_day_df["calendar_day"] = kcp_vs_day_df["season_day"].map(calendar_day_mapper)
kcp_vs_day_df.sort_values(by="season_day", axis=0, ascending=True, inplace=True)
kcp_vs_day_df["datetimestamp"] = datetimestamp
kcp_vs_day_df.set_index(keys="datetimestamp", drop=True, inplace=True)
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create weekly averages for kcp.  I.e., one (averaged) kcp value for every 7 days.
# Save a pandas DataFrame, called kcp_vs_week_df.  kcp_vs_week_df contains the columns:
#   datetimestamp --> the index of the DataFrame
#   calendar_week --> self-explanatory
#   season_week --> self-explanatory
#   weekly_averaged_kcp --> self-explanatory
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
season_week = 1
kcp_vs_season_week = []
seven_kcp_values = []
counter = 0
for _, k in zip(datetimestamp, kcp_trend):
    seven_kcp_values.append(k)
    counter += 1
    if counter == 7:
        kcp_vs_season_week.append([season_week, np.average(seven_kcp_values)])
        season_week += 1
        counter = 0
        seven_kcp_values = []
    if season_week == 53:
        break

kcp_vs_season_week = np.array(kcp_vs_season_week)

kcp_vs_week_df = pd.DataFrame(data={"season_week": kcp_vs_season_week[:, 0].astype(int),
                                    "weekly_averaged_kcp": kcp_vs_season_week[:, 1]})
kcp_vs_week_df["calendar_week"] = kcp_vs_week_df["season_week"].map(calendar_week_mapper)
kcp_vs_week_df.sort_values(by="season_week", axis=0, ascending=True, inplace=True)
kcp_vs_week_df["datetimestamp"] = kcp_vs_week_df["season_week"].map(from_season_week_datetime_mapper)
kcp_vs_week_df.set_index(keys="datetimestamp", drop=True, inplace=True)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Create monthly averages for kcp.  I.e., one (averaged) kcp value for every month.
# Save a pandas DataFrame, called kcp_vs_month_df.  kcp_vs_month_df contains the columns:
#   datetimestamp --> the index of the DataFrame
#   calendar_month --> self-explanatory
#   season_month --> self-explanatory
#   monthly_averaged_kcp --> self-explanatory
# ======================================================================================================================
kcp_vs_calendar_month_dict = {"1": [], "2": [], "3": [],  "4": [],  "5": [],  "6": [],
                              "7": [], "8": [], "9": [], "10": [], "11": [], "12": []}

for d, k in zip(datetimestamp, kcp_trend):
    month_of_datetimestamp = d.month
    kcp_vs_calendar_month_dict[str(month_of_datetimestamp)].append(k)

kcp_vs_calendar_month = []
for m, kcp_list in kcp_vs_calendar_month_dict.items():
    kcp_vs_calendar_month_dict[m] = np.average(kcp_list)
    kcp_vs_calendar_month.append([int(m), np.average(kcp_list)])
kcp_vs_calendar_month = np.array(kcp_vs_calendar_month)

kcp_vs_month_df = pd.DataFrame(data={"calendar_month": kcp_vs_calendar_month[:, 0].astype(int),
                                     "monthly_averaged_kcp": kcp_vs_calendar_month[:, 1]})
kcp_vs_month_df["season_month"] = kcp_vs_month_df["calendar_month"].map(season_month_mapper)
kcp_vs_month_df["datetimestamp"] = \
    kcp_vs_month_df["calendar_month"].map(from_calendar_month_datetime_mapper)
kcp_vs_month_df.set_index(keys="datetimestamp", drop=True, inplace=True)
kcp_vs_month_df.sort_values(by="season_month", axis=0, ascending=True, inplace=True)


# ======================================================================================================================


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Finally, we save all the binned data.
#   1.  Create a pandas Excel Writer using XlsxWriter as the engine.
#   2.  Write each DataFrame to a different worksheet.
#   3.  Close the pandas Excel Writer and output the Excel file.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create a pandas Excel Writer using XlsxWriter as the engine.
writer = pd.ExcelWriter("data/binned_kcp_data.xlsx", engine="xlsxwriter")

# Write each DataFrame to a different worksheet.
kcp_vs_day_df.to_excel(writer, sheet_name="day_frequency",
                       columns=["season_day", "calendar_day", "day_averaged_kcp"], header=True, index=True,
                       index_label="datetimestamp")
kcp_vs_week_df.to_excel(writer, sheet_name="week_frequency",
                        columns=["season_week", "calendar_week", "weekly_averaged_kcp"], header=True, index=True,
                        index_label="datetimestamp")
kcp_vs_month_df.to_excel(writer, sheet_name="month_frequency",
                         columns=["season_month", "calendar_month", "monthly_averaged_kcp"], header=True, index=True,
                         index_label="datetimestamp")

# Close the pandas Excel Writer and output the Excel file
writer.save()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
