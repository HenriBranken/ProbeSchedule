import numpy as np
import datetime
import pandas as pd
import master_data

# ----------------------------------------------------------------------------------------------------------------------
# Definitions of important constants
# ----------------------------------------------------------------------------------------------------------------------
ETO_MAX = 12
KCP_MAX = 0.8
BEGINNING_MONTH = 7
ETCP_MAX = ETO_MAX * KCP_MAX
DAY = datetime.timedelta(days=1)

# The following events are "irredeemable".  "irredeemable" is associated with a `binary_value` of 1.
RAIN_DESC = "Rain perturbing etcp"
SIMUL_DESC = "Software simulation"
IRR_DESC = "Irrigation perturbing etcp"
NULL_PROFILE_DESC = "Null profile value"
DATA_BLIP_DESC = "Profile data blip"
LARGE_PROFILE_DIP_DESC = "Large profile dip"
ETCP_POS_DESC = "Etcp is positive"
ETCP_OUTLIERS_DESC = "Etcp outliers"
LUX_DESC = "Luxurious water uptake"
BAD_KCP_DESC = "Unacceptable kcp"
UNREDEEMABLE = [RAIN_DESC, SIMUL_DESC, IRR_DESC, NULL_PROFILE_DESC, DATA_BLIP_DESC,
                LARGE_PROFILE_DIP_DESC, ETCP_POS_DESC, ETCP_OUTLIERS_DESC, LUX_DESC, BAD_KCP_DESC]

# The following events are "redeemable".  "redeemable" is associated with a `binary_value` of 0.
HU_ZERO_DESC = "Heat Units equal to zero"
ETO_STUCK_DESC = "Eto `stuck`"
ETC_STUCK_DESC = "Stuck etc due to stuck eto"
IMPUTED_ETO = "Imputed eto"
REDEEMABLE = [HU_ZERO_DESC, ETO_STUCK_DESC, ETC_STUCK_DESC, IMPUTED_ETO]

# With the definition of `description_dict`, we can use only 1 import statement instead of individually importing all
# the different _DESC constants.
description_dict = {"rain_desc": RAIN_DESC, "simul_desc": SIMUL_DESC, "irr_desc": IRR_DESC,
                    "null_profile_desc": NULL_PROFILE_DESC, "data_blip_desc": DATA_BLIP_DESC,
                    "large_profile_dip_desc": LARGE_PROFILE_DIP_DESC, "etcp_pos_desc": ETCP_POS_DESC,
                    "etcp_outliers_desc": ETCP_OUTLIERS_DESC, "lux_desc": LUX_DESC, "bad_kcp_desc": BAD_KCP_DESC,
                    "hu_zero_desc": HU_ZERO_DESC, "eto_stuck_desc": ETO_STUCK_DESC, "etc_stuck_desc": ETC_STUCK_DESC,
                    "imputed_eto": IMPUTED_ETO}
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define some helper functions.  These helper functions are:
#   1.  flagger(bad_dates, brief_desc, df, bin_value=0)
#   2.  reporter(df, brief_desc, remaining=False)
#   3.  calculate_kcp_deviation(df)
#   4.  kbv_imputer(flagged_dates, df, column_to_be_imputed)
#   5.  impute_kbv_data(df, bad_eto_days)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   1.
def flagger(bad_dates, brief_desc, df, bin_value=0):
    """
    Flag bad_dates with a binary value of 1 and append a brief description about why bad_dates were flagged.
    Descriptions about `redeemable` events are also added to the dataframe.

    Parameters:
    bad_dates (pandas.core.indexes.datetimes.DatetimeIndex):  Dates for which we cannot calculate kcp because our
    readings were perturbed/contaminated and rendered unuseful.
    brief_desc (str):  A very short description about why bad_dates were flagged.
    bin_value (int):  The binary value.  If Eto is imputed, Etc and heat_units are stuck, we can still get away with
    a new calculation of kcp; thus set binary_value=0 for such redeemable events.  However, if an `irredeemable` event
    is associated with a date, then bin_value=1.

    Returns:
    None.  It updates the DataFrame storing all the information related to flagging.  In this case the DataFrame is
    called `df`.
    """
    if df.loc[bad_dates, "description"].str.contains(brief_desc).all(axis=0):
        # The bad_dates have already been flagged for the reason given in brief_desc.
        # No use in duplicating brief_desc contents in the description column.
        # Therefore redundant information in the df DataFrame is avoided.
        print("You have already flagged these dates for the reason given in `brief_desc`; No flagging has taken place.")
        return
    else:
        for d in bad_dates:
            cond = (brief_desc in df.loc[d, "description"])
            if (df.loc[d, "binary_value"] == 0) & (bin_value == 0) & (cond is True):
                continue
            elif (df.loc[d, "binary_value"] == 0) & (bin_value == 0) & (cond is False):
                df.loc[d, "description"] += (" " + brief_desc + ".")
            elif (df.loc[d, "binary_value"] == 0) & (bin_value == 1) & (cond is True):
                df.loc[d, "binary_value"] = 1
            elif (df.loc[d, "binary_value"] == 0) & (bin_value == 1) & (cond is False):
                df.loc[d, "binary_value"] = 1
                df.loc[d, "description"] += (" " + brief_desc + ".")
            elif (df.loc[d, "binary_value"] == 1) & (bin_value == 0) & (cond is True):
                continue
            elif (df.loc[d, "binary_value"] == 1) & (bin_value == 0) & (cond is False):
                df.loc[d, "description"] += (" " + brief_desc + ".")
            elif (df.loc[d, "binary_value"] == 1) & (bin_value == 1) & (cond is True):
                continue
            else:  # (df.loc[d, "binary_value"] == 1) & (bin_value == 1) & (cond is False)
                df.loc[d, "description"] += (" " + brief_desc + ".")
        df.loc[bad_dates, "description"] = df.loc[:, "description"].apply(lambda s: s.lstrip().rstrip())
        return


#   2.
def reporter(df, brief_desc, remaining=False):
    tally = df["description"].str.contains(brief_desc).sum()
    n_tot_entries = len(df.index)
    perc = tally / n_tot_entries * 100
    print("{:.1f}% of data is affected due to [{}].".format(perc, brief_desc))

    if remaining:
        calc = 100 - df["binary_value"].sum() / len(df.index) * 100
        print("After all the flagging that has taken place for this probe data-set,"
              " only {:.0f}% of your data is useful.".format(calc))


#   3.
def calculate_kcp_deviation(df):
    df["kcp_perc_deviation"] = 0.0
    empirical_kcp = df["kcp"]
    associated_kcp_norm = df["cco"]
    df["kcp_perc_deviation"] = np.abs((empirical_kcp - associated_kcp_norm)/associated_kcp_norm) * 100.0
    return df["kcp_perc_deviation"]


#   4.
def kbv_imputer(flagged_dates, df, column_to_be_imputed):
    for d in flagged_dates:
        week_number = d.isocalendar()[1]
        try:
            df.loc[d, [column_to_be_imputed]] = master_data.df_kbv.loc[week_number, "kbv_eto"]
            for description in UNREDEEMABLE:
                if description in df.loc[d, "description"]:
                    break
            else:
                df.loc[d, "binary_value"] = 0  # we have 'salvaged' an entry.
                df.loc[d, "description"] = df.loc[d, "description"].replace(ETO_STUCK_DESC, IMPUTED_ETO)
        except KeyError:
            df.loc[d, column_to_be_imputed] = np.nan
    return df


#   5.
def impute_kbv_data(df, bad_eto_days):
    return kbv_imputer(flagged_dates=bad_eto_days, df=df, column_to_be_imputed="eto")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Define all the flagging functions.
# ======================================================================================================================
# The flagging functions are defined in the following order:
#   1.  drop_redundant_columns(df)
#   2.  flag_spurious_eto(df)
#   3.  flag_rain_events(df)
#   4.  flag_irrigation_events(df)
#   5.  flag_simulated_events(df)
#   6.  flag_suspicious_and_missing_profile_events(df)
#   7.a flag_spurious_heat_units(df)
#   7.b interpolate_missing_heat_units(df, method="nearest")
#   7.c cumulative_gdd(df)
#   8.  flag_unwanted_etcp(df)
#   9.  flag_unwanted_kcp(df)
# ======================================================================================================================
#   1.
def drop_redundant_columns(df):
    labels = ["rzone", "available", "days_left", "deficit_current",
              "rzm", "fcap", "deficit_want", "refill", "eto_forecast_yr"]
    df.drop(labels=labels, axis=1, inplace=True)
    return df


#   2.
def flag_spurious_eto(df):
    interim_df = pd.DataFrame(data={"eto_diff1": df["eto"].diff(periods=1), "eto_diff2": df["eto"].diff(periods=2)},
                              index=df.index)
    condition = (interim_df["eto_diff1"] == 0.0) | (interim_df["eto_diff2"] == 0)
    bad_eto_days = df[condition].index
    flagger(bad_dates=bad_eto_days, brief_desc=ETO_STUCK_DESC, df=df, bin_value=1)
    df.loc[bad_eto_days, ["eto"]] = np.nan

    df.loc[bad_eto_days, ["etc"]] = np.nan
    flagger(bad_dates=bad_eto_days, brief_desc=ETC_STUCK_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=ETC_STUCK_DESC)
    return bad_eto_days, df


#   3.
def flag_rain_events(df):
    condition = (df["rain"] > 2)
    bad_rain_dates = df[condition].index
    flagger(bad_dates=bad_rain_dates, brief_desc=RAIN_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=RAIN_DESC)
    return df


#   4.
def flag_irrigation_events(df):
    conditions = (df["total_irrig"] > 0) & (df["etcp"] > 0.5*df["etc"]) & (df["rain"] == 0)
    flag_irrigation_dates = df[conditions].index
    flagger(bad_dates=flag_irrigation_dates, brief_desc=IRR_DESC, df=df)
    reporter(df=df, brief_desc=IRR_DESC)
    return df


#   5.
def flag_simulated_events(df):
    condition = df["rzm_source"].str.contains("software")
    flag_software_dates = df[condition].index
    flagger(bad_dates=flag_software_dates, brief_desc=SIMUL_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=SIMUL_DESC)
    return df


#   6.
def flag_suspicious_and_missing_profile_events(df):
    df["profile"].replace(0.0, np.nan, inplace=True)
    condition = df["profile"].isnull()
    bad_profile_days = df[condition].index
    flagger(bad_dates=bad_profile_days, brief_desc=NULL_PROFILE_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=NULL_PROFILE_DESC)

    interim_df = pd.DataFrame(data={"profile_difference": df["profile"].diff(), "profile": df["profile"]},
                              index=df.index)
    data_blip_days = []
    for d in interim_df.index:
        try:
            if (interim_df.loc[d, "profile_difference"] < 0) and pd.isnull(interim_df.loc[d + DAY, "profile"]):
                data_blip_days.append(d)
        except KeyError:
            pass
    data_blip_days = pd.to_datetime(data_blip_days)
    flagger(bad_dates=data_blip_days, brief_desc=DATA_BLIP_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=DATA_BLIP_DESC)

    interim_df.loc[data_blip_days, ["profile_difference"]] = np.nan
    negative_differences = interim_df[interim_df["profile_difference"] < 0]["profile_difference"].values
    percentile_value = np.quantile(negative_differences, q=[0.01, 0.02, 0.03, 0.04, 0.05,
                                                            0.06, 0.07, 0.08, 0.09, 0.10])[4]
    large_dip_days = []
    for d in interim_df.index:
        try:
            if (interim_df.loc[d, "profile_difference"] < percentile_value) \
                    and (interim_df.loc[d + DAY, "profile_difference"] > 0):
                large_dip_days.append(d)
        except KeyError:
            pass
    large_dip_days = pd.to_datetime(large_dip_days)
    flagger(bad_dates=large_dip_days, brief_desc=LARGE_PROFILE_DIP_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=LARGE_PROFILE_DIP_DESC)
    return df


#   7.a.
def flag_spurious_heat_units(df):
    # df["hu_diff1"] = df["heat_units"].diff(periods=1)
    # df["hu_diff2"] = df["heat_units"].diff(periods=2)
    # condition = (df["hu_diff1"] == 0.0) | (df["hu_diff2"] == 0)
    condition = df["heat_units"] == 0
    bad_hu_days = df[condition].index
    flagger(bad_dates=bad_hu_days, brief_desc=HU_ZERO_DESC, df=df, bin_value=0)
    reporter(df=df, brief_desc=HU_ZERO_DESC)
    df.loc[bad_hu_days, ["heat_units"]] = np.nan
    return df


#   7.b
def interpolate_missing_heat_units(df, method="nearest"):
    """
    Interpolate all NaN entries in the "heat_units" column using some interpolation scheme.
    The default method is "nearest".
    """
    df["interpolated_hu"] = df["heat_units"].copy(deep=True)
    df.loc[:, "interpolated_hu"].interpolate(method=method, axis=0, inplace=True)
    df.loc[:, "interpolated_hu"].fillna(value=0.0, inplace=True)
    return df


#   7.c
def cumulative_gdd(df):
    season_candidates = []
    df["cumulative_gdd"] = 0
    for datestamp in df.index:
        if (datestamp.month == BEGINNING_MONTH) and (datestamp.day == 1):
            season_candidates.append(datestamp)
            df.loc[datestamp, ["cumulative_gdd"]] = 0.0
        elif (datestamp.month == BEGINNING_MONTH) and (datestamp.day == 2):
            df.loc[datestamp, ["cumulative_gdd"]] = 0.0 + df.loc[datestamp, "interpolated_hu"]
        else:
            try:
                df.loc[datestamp, ["cumulative_gdd"]] = df.loc[(datestamp - DAY), "cumulative_gdd"] + \
                                                        df.loc[datestamp, "interpolated_hu"]
            except KeyError:  # we cannot go 1 day back into the past; we are at start_date.  Use h.u. of start_date
                df.loc[datestamp, ["cumulative_gdd"]] = df.loc[datestamp, "interpolated_hu"]

    first_season_date = min(season_candidates)
    condition = df.index < first_season_date
    pre_first_season = df[condition].index
    df.loc[pre_first_season, "cumulative_gdd"] = np.nan
    return df


#   8.
def flag_unwanted_etcp(df):
    condition = df["etcp"] >= 0.0
    positive_etcp_days = df[condition].index
    flagger(bad_dates=positive_etcp_days, brief_desc=ETCP_POS_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=ETCP_POS_DESC)

    condition = df["binary_value"] == 1
    junk_data_dates = df[condition].index
    df.loc[junk_data_dates, ["etcp"]] = np.nan

    # to simplify programming, multiply `etcp` column with -1
    df["etcp"] = df["etcp"].multiply(-1, fill_value=np.nan)

    condition = df["etcp"] > ETCP_MAX
    etcp_outlier_dates = df[condition].index
    df.loc[etcp_outlier_dates, ["etcp"]] = np.nan
    flagger(bad_dates=etcp_outlier_dates, brief_desc=ETCP_OUTLIERS_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=ETCP_OUTLIERS_DESC)

    condition = df["etcp"] > df["eto"].mul(KCP_MAX, fill_value=np.nan)
    luxurious_dates = df[condition].index
    df.loc[luxurious_dates, ["etcp"]] = np.nan
    flagger(bad_dates=luxurious_dates, brief_desc=LUX_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=LUX_DESC)
    return df


#   9.
def flag_unwanted_kcp(df):
    df["kcp"] = df["etcp"].div(df["eto"])
    perc_series = calculate_kcp_deviation(df)
    condition = (perc_series.isnull()) | (perc_series > 50) | df["kcp"].isnull()
    bad_calc_kcp_dates = df[condition].index
    df.loc[bad_calc_kcp_dates, "kcp"] = np.nan
    flagger(bad_dates=bad_calc_kcp_dates, brief_desc=BAD_KCP_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=BAD_KCP_DESC, remaining=True)
    return df
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Define a function that wraps all the dates into 1 Season interval.
# These wrapped dates are stored in `new_dates`.
# `useful_dates` represent the original probe dates for which `binary_value == 0`.
# ----------------------------------------------------------------------------------------------------------------------
def get_final_dates(df):
    condition = df["binary_value"] == 0
    useful_dates = df[condition].index
    starting_year = useful_dates[0].year
    new_dates = []
    for d in useful_dates:
        extracted_month = d.month
        if BEGINNING_MONTH <= extracted_month <= 12:  # Normalise to `starting_year`
            new_dates.append(datetime.datetime(year=starting_year, month=d.month, day=d.day))
        else:  # 1 <= extracted_month < BEGINNING_MONTH.  Normalise to `starting_year + 1`
            new_dates.append(datetime.datetime(year=starting_year + 1, month=d.month, day=d.day))
    return starting_year, new_dates, useful_dates
# ----------------------------------------------------------------------------------------------------------------------
