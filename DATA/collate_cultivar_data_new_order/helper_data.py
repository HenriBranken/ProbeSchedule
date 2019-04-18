import pandas as pd
from cleaning_operations import BEGINNING_MONTH
import datetime
from helper_functions import get_labels
import helper_meta_data as hm


cleaned_multi_df = pd.read_excel("data/stacked_cleaned_data_for_overlay.xlsx",
                                 header=0, index_col=[0, 1], parse_dates=True)
outer_index = cleaned_multi_df.index.get_level_values("probe_id").unique()
outer_index = list(outer_index)
inner_index = cleaned_multi_df.index.get_level_values("datetimeStamp").unique()
inner_index = list(inner_index)


cco_df = pd.read_excel("./data/reference_crop_coeff.xlsx", sheet_name=0,
                       header=0, index_col=0, parse_dates=True)


processed_eg_df = pd.read_excel("./data/processed_probe_data.xlsx",
                                sheet_name=0, header=0, index_col=0,
                                squeeze=True, parse_dates=True)
vline_dates = []
for d in processed_eg_df.index:
    if (d.month == BEGINNING_MONTH) and (d.day == 1):
        new_season_date = datetime.datetime(year=d.year,
                                            month=d.month,
                                            day=d.day)
        vline_dates.append(new_season_date)
vline_date = vline_dates[0]


api_xticks = get_labels(begin=hm.api_start_date, terminate=hm.api_end_date,
                        freq="QS")
season_xticks = get_labels(begin=hm.season_start_date,
                           terminate=hm.season_end_date, freq="MS")
