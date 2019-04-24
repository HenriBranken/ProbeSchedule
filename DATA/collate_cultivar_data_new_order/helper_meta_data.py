import numpy as np
import pandas as pd
import datetime
from cleaning_operations import BEGINNING_MONTH
import calendar
# -----------------------------------------------------------------------------


with open("./data/api_dates.txt", "r") as f:
    api_start_date = f.readline().rstrip()
    api_end_date = f.readline().rstrip()
api_start_date = datetime.datetime.strptime(api_start_date, "%Y-%m-%d")
api_end_date = datetime.datetime.strptime(api_end_date, "%Y-%m-%d")


pol_degree = 4


ALLOWED_TAIL_DEVIATION = 0.75


with open("./data/base_temperature.txt", "r") as f:
    temperature_base = float(f.readline())


with open("./data/probe_ids.txt", "r") as f:
    probe_ids = [x.rstrip() for x in list(f)]


with open("./data/starting_year.txt") as f:
    starting_year = int(f.readline().rstrip())


n_neighbours_list = list(np.arange(start=30, stop=1-1, step=-1))


delta_x = 1


x_limits = [0, 365]


marker_list = ["o", ">", "<", "s", "P", "*", "X", "D"]
color_list = ["red", "gold", "seagreen", "lightseagreen", "royalblue",
              "darkorchid", "plum", "burlywood"]
marker_color_meta = [(m, c) for m, c in zip(marker_list, color_list)]


color_ls_meta = [("goldenrod", "-"), ("green", "--"), ("blue", ":"),
                 ("silver", "-."), ("burlywood", "-"), ("lightsalmon", "--"),
                 ("chartreuse", ":")]


season_start_date = datetime.datetime(year=starting_year,
                                      month=BEGINNING_MONTH, day=1)
last_day = calendar.monthrange(year=starting_year+1,
                               month=BEGINNING_MONTH-1)[1]
season_end_date = datetime.datetime(year=starting_year+1,
                                    month=BEGINNING_MONTH-1, day=last_day)
datetime_stamp = list(pd.date_range(start=season_start_date,
                                    end=season_end_date, freq="D"))


CULTIVAR = "Golden Delicious Apples"


WEEKLY_BINNED_VERSION = True


mode = "WMA"
