import numpy as np
import datetime
import pandas as pd
import master_data
import sys

# -----------------------------------------------------------------------------
# Definitions of important constants
# -----------------------------------------------------------------------------
RAIN_THRESHOLD = 2  # The value (in mm) above which flagging needs to occur.
ETO_MAX = 12  # Maximum allowed eto for the cultivar.
KCP_MAX = 0.8  # Maximum allowed kcp for the cultivar.
BEGINNING_MONTH = 7  # i.e. beginning of July.
ETCP_PERC_DEVIATION = 0.30  # The smaller the value, the more strict flagging.
KCP_PERC_DEVIATION = 0.50  # Maximum allowed percentage deviation
ETCP_MAX = ETO_MAX * KCP_MAX  # Maximum allowed etcp for the cultivar.
DAY = datetime.timedelta(days=1)  # Define a time-delta object of 1-day long.
SUSP_LOW_VAL = 1 - ETCP_PERC_DEVIATION  # Derived
SUSP_HIGH_VAL = 1 + ETCP_PERC_DEVIATION  # Derived

# The following events are "unredeemable".
# An "unredeemable" event is associated with a `binary_value` of 1.
RAIN_DESC = "Rain perturbing etcp"
SIMUL_DESC = "Software simulation"
IRR_DESC = "`Possible` irrigation"
NULL_PROFILE_DESC = "Null profile value"
DATA_BLIP_DESC = "Profile data blip"
LARGE_PROFILE_DIP_DESC = "Large profile dip"
ETCP_POS_DESC = "Etcp is positive"
ETCP_SUSP_LOW_DESC = "Etcp suspiciously low"
ETCP_SUSP_HIGH_DESC = "Etcp suspiciously high"
ETCP_OUTLIERS_DESC = "Etcp larger than allowed maximum"
EXCEEDING_KCP_MAX_DESC = "Ratio of etcp over eto exceeds KCP_MAX"
BAD_KCP_DESC = "kcp deviation too big"
KCP_IS_NAN_DESC = "kcp is NaN"
ETO_BAD_DESC = "Stuck or faulty eto"
ETC_BAD_DESC = "Stuck or faulty etc"
UNREDEEMABLE = [RAIN_DESC, SIMUL_DESC, IRR_DESC, NULL_PROFILE_DESC,
                DATA_BLIP_DESC, LARGE_PROFILE_DIP_DESC, ETCP_POS_DESC,
                ETCP_SUSP_LOW_DESC, ETCP_SUSP_HIGH_DESC, ETCP_OUTLIERS_DESC,
                EXCEEDING_KCP_MAX_DESC, BAD_KCP_DESC, ETO_BAD_DESC,
                ETC_BAD_DESC, KCP_IS_NAN_DESC]

# The following events are "redeemable".
# "redeemable" is associated with a `binary_value` of 0.
HU_BAD_DESC = "Faulty or Missing Heat Units"
IMPUTED_ETO = "Imputed eto"
REDEEMABLE = [HU_BAD_DESC, IMPUTED_ETO]

# With the definition of `description_dict`, we can use only 1 import
# statement in any follow-up Python scripts, instead of importing each
# individual `*_DESC` description.
description_dict = {"rain_desc": RAIN_DESC,
                    "simul_desc": SIMUL_DESC,
                    "irr_desc": IRR_DESC,
                    "null_profile_desc": NULL_PROFILE_DESC,
                    "data_blip_desc": DATA_BLIP_DESC,
                    "large_profile_dip_desc": LARGE_PROFILE_DIP_DESC,
                    "etcp_pos_desc": ETCP_POS_DESC,
                    "etcp_outliers_desc": ETCP_OUTLIERS_DESC,
                    "etcp_susp_low_desc": ETCP_SUSP_LOW_DESC,
                    "etcp_susp_high": ETCP_SUSP_HIGH_DESC,
                    "exceeding_kcp_max_desc": EXCEEDING_KCP_MAX_DESC,
                    "bad_kcp_desc": BAD_KCP_DESC,
                    "kcp_is_nan_desc": KCP_IS_NAN_DESC,
                    "hu_bad_desc": HU_BAD_DESC,
                    "imputed_eto": IMPUTED_ETO,
                    "eto_bad_desc": ETO_BAD_DESC,
                    "etc_bad_desc": ETC_BAD_DESC
                    }
# -----------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define some helper functions.  These helper functions are:
# 1.  flagger(bad_dates, brief_desc, df, bin_value=0)
# 2.  reporter(df, brief_desc, remaining=False)
# 3.  calculate_kcp_deviation(df)
# 4.  kbv_imputer(flagged_dates, df, column_to_be_imputed)
# 5.  impute_kbv_data(df, bad_eto_days)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1.
def flagger(bad_dates, brief_desc, df, bin_value=0, affected_cols=None,
            set_to_nan=False):
    """
    Flag bad_dates with a binary value of 1 and append a brief description
    about why bad_dates were flagged.
    Descriptions about `redeemable` events are also added to the dataframe.

    Parameters:
    bad_dates (pandas.core.indexes.datetimes.DatetimeIndex):  Dates for which
    we cannot calculate kcp because our readings were perturbed/contaminated
    and rendered not useful.
    brief_desc (str):  A  short description about why bad_dates were flagged.
    bin_value (int):  The binary value.  If Eto is imputed, Etc and heat_units
    are stuck, we can still get away with a new calculation of kcp; thus set
    binary_value=0 for such redeemable events.  However, if an `unredeemable`
    event is associated with a date, then bin_value=1 always.

    Returns:
    None.  It updates the DataFrame storing all the information related to
    flagging.  In this case the DataFrame is called `df`.
    """
    if df.loc[bad_dates, "description"].str.contains(brief_desc).all(axis=0):
        # if True:  Then the `bad_dates` have already been flagged.
        # No use in duplicating brief_desc contents in the description column.
        # Therefore redundant information in the df DataFrame is avoided.
        print("You have already flagged these dates for the reason given in "
              "`brief_desc`; No flagging has taken place.")
    else:
        for d in bad_dates:
            cond = (brief_desc in df.loc[d, "description"])
            if (df.loc[d, "binary_value"] == 0) & (bin_value == 0) & \
                    (cond is True):
                continue
            elif (df.loc[d, "binary_value"] == 0) & (bin_value == 0) & \
                    (cond is False):
                df.loc[d, "description"] += (" " + brief_desc + ".")
            elif (df.loc[d, "binary_value"] == 0) & (bin_value == 1) & \
                    (cond is True):
                df.loc[d, "binary_value"] = 1
            elif (df.loc[d, "binary_value"] == 0) & (bin_value == 1) & \
                    (cond is False):
                df.loc[d, "binary_value"] = 1
                df.loc[d, "description"] += (" " + brief_desc + ".")
            elif (df.loc[d, "binary_value"] == 1) & (bin_value == 0) & \
                    (cond is True):
                continue
            elif (df.loc[d, "binary_value"] == 1) & (bin_value == 0) & \
                    (cond is False):
                df.loc[d, "description"] += (" " + brief_desc + ".")
            elif (df.loc[d, "binary_value"] == 1) & (bin_value == 1) & \
                    (cond is True):
                continue
            else:
                df.loc[d, "description"] += (" " + brief_desc + ".")
        df.loc[bad_dates, "description"] = \
            df.loc[:, "description"].apply(lambda s: s.lstrip().rstrip())
    if set_to_nan:
        if not affected_cols:
            print("You must specify a list of column(s) whose bad entries you "
                  "want to set to NaN via the `affected_cols` argument."
                  "  E.g. affected_cols=['profile', 'etcp']")
            sys.exit(1)
        elif not isinstance(affected_cols, (list,)):
            print("Argument `affected_cols` must be specified in LIST form, "
                  "e.g.: ['profile', 'etcp']")
            sys.exit(1)
        else:
            df.loc[bad_dates, affected_cols] = np.nan
    return


# 2.
def reporter(df, brief_desc, remaining=False):
    """
    Print out a statement to STDOUT, about how many data samples have been
    affected by a flagging event/flagging iteration.
    :param df: The DataFrame on which we would like to report some flagging
    statistics.
    :param brief_desc: A short string explaining the flagging event.  Read all
    the *_DESC constants defined above.
    :param remaining: If True, then also print out a statement to STDOUT about
    how many data samples are left with a binary value of 0, and thus, at this
    stage, still useful.
    :return: None
    """
    # Calculate the percentage of data-samples affected by a flagging event.
    tally = df["description"].str.contains(brief_desc).sum()
    n_tot_entries = len(df.index)
    perc = tally / n_tot_entries * 100
    print("{:.2f}% of data is affected due to [{}].".format(perc, brief_desc))

    # Print a message to STDOUT about the percentage of data left with a
    # binary_value of 0.
    if remaining:
        calc = 100 - df["binary_value"].sum() / len(df.index) * 100
        print("After all the flagging that has taken place for this probe "
              "data-set, only {:.2f}% of your data is useful.".format(calc))


# 3.
def calculate_kcp_deviation(df):
    """
    Calculate the percentage deviation of the calculated kcp = etcp/eto from
    the reference crop coefficients cco.
    :param df: The DataFrame on which we we like to calculate the percentage
    deviation.
    :return: Return the pandas Series `df["kcp_perc_deviation"]`.
    """
    df["kcp_perc_deviation"] = 0.0
    empirical_kcp = df["kcp"]
    associated_kcp_norm = df["cco"]
    difference = empirical_kcp - associated_kcp_norm
    # Formula used to calculate the percentage deviation:
    df["kcp_perc_deviation"] = np.abs(difference/associated_kcp_norm) * 100.0
    return df["kcp_perc_deviation"]


# 4.
def kbv_imputer(flagged_dates, df, column_to_be_imputed):
    """
    This function was mostly used in the prototype/testing phase, and will not
    be invoked when working with new data.
    :param flagged_dates: The dates for which eto or etc has been flagged.
    :param df: The DataFrame.
    :param column_to_be_imputed:  Column on which data imputation needs to be
    performed.
    :return: Return the updated DataFrame containing the imputed data.
    """
    for d in flagged_dates:
        week_number = d.isocalendar()[1]
        try:
            df.loc[d, [column_to_be_imputed]] = \
                master_data.df_kbv.loc[week_number, "kbv_eto"]
            df.loc[d, "description"] = \
                df.loc[d, "description"].replace(ETO_BAD_DESC, IMPUTED_ETO)
            for description in UNREDEEMABLE:
                if description in df.loc[d, "description"]:
                    break
            else:
                df.loc[d, "binary_value"] = 0  # we have 'salvaged' an entry.
        except KeyError:
            df.loc[d, column_to_be_imputed] = np.nan
    reporter(df=df, brief_desc=IMPUTED_ETO, remaining=False)
    reporter(df=df, brief_desc=ETO_BAD_DESC, remaining=False)
    return df


# 5.
def impute_kbv_data(df, bad_eto_days):
    """
    Function used to invoke `kbv_imputer`.
    """
    return kbv_imputer(flagged_dates=bad_eto_days, df=df,
                       column_to_be_imputed="eto")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Define all the flagging functions.
# =============================================================================
# The flagging functions are defined in the following order:
# 1.  drop_redundant_columns(df)
# 2.  flag_spurious_eto(df)
# 3.  flag_rain_events(df)
# 4.  flag_irrigation_events(df)
# 5.  flag_simulated_events(df)
# 6.  flag_suspicious_and_missing_profile_events(df)
# 7.a flag_spurious_heat_units(df)
# 7.b interpolate_missing_heat_units(df, method="nearest")
# 7.c cumulative_gdd(df)
# 8.  flag_unwanted_etcp(df)
# 9.  flag_unwanted_kcp(df)
# =============================================================================
# 1.
def drop_redundant_columns(df):
    """
    A function used to drop column that are of no interest in our data-analysis
    pipeline.
    :param df: The DataFrame in which we need to drop the redundant columns.
    :return: Return a pruned DataFrame containing less columns.  The reamining
    columns are the columns of interest.
    """
    labels = ["rzone", "available", "days_left", "deficit_current", "erain",
              "tot_eff_irrig", "ety", "rzm", "fcap", "deficit_want", "refill",
              "eto_forecast_yr"]
    df.drop(labels=labels, axis=1, inplace=True)
    return df


# 2.
def flag_spurious_et(df):
    """
    Flag the dates for which `eto` and `etc` are stuck.  I.e. they repeat.
    :param df: The DataFrame in which we need to perform the flagging.
    :return: Return updated DataFrame in which stuck et events are flagged.
    """
    interim_df = pd.DataFrame(data={"eto_diff1": df["eto"].diff(periods=1),
                                    "eto": df["eto"]},
                              index=df.index, copy=True)
    condition = (interim_df["eto_diff1"] == 0.0) | (interim_df["eto"] == 0)
    bad_eto_days = df[condition].index
    flagger(bad_dates=bad_eto_days, brief_desc=ETO_BAD_DESC, df=df,
            bin_value=1, affected_cols=["eto", "etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=ETO_BAD_DESC)

    # Do the same for etc -----------------------------------------------------
    interim_df = pd.DataFrame(data={"etc_diff1": df["etc"].diff(periods=1),
                                    "etc": df["etc"]},
                              index=df.index, copy=True)
    condition = (interim_df["etc_diff1"] == 0.0) | (interim_df["etc"] == 0)
    bad_etc_days = df[condition].index
    flagger(bad_dates=bad_etc_days, brief_desc=ETC_BAD_DESC, df=df,
            bin_value=1, affected_cols=["etc", "etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=ETC_BAD_DESC)

    return bad_eto_days, df


# 3.
def flag_rain_events(df):
    """
    Flag events in which the rain, if present, exceeded 2 millimeters.
    :param df: The DataFrame in which we need to perform the flagging.
    :return: Return updated DataFrame in which the rain events were flagged.
    """
    condition = (df["rain"] > RAIN_THRESHOLD)
    bad_rain_dates = df[condition].index
    flagger(bad_dates=bad_rain_dates, brief_desc=RAIN_DESC, df=df,
            bin_value=1, affected_cols=["etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=RAIN_DESC)
    return df


# 4.
def flag_irrigation_events(df):
    """
    Flag events in which irrigation has taken place.  Simultaneously, it must
    also hold True that `etcp` is positive and `rain` is 0.
    :param df: The DataFrame in which we need to perform the flagging.
    :return: Return updated DataFrame in which the irrigation events were
    flagged.
    """
    conditions = (df["total_irrig"] > 0) & (df["etcp"] >= 0) & \
                 (df["rain"] == 0)
    flag_irrigation_dates = df[conditions].index
    flagger(bad_dates=flag_irrigation_dates, brief_desc=IRR_DESC, df=df,
            bin_value=1, affected_cols=["etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=IRR_DESC)
    return df


# 5.
def flag_simulated_events(df):
    """
    Flag readings in the probe dataset that are simulated instead of being
    derived from the electronic probe telemetry.
    :param df: The DataFrame in which we need to perform the flagging.
    :return: Return the updated DataFrame in which simulated events were
    flagged.
    """
    # case=False means case insensitive.
    condition = df["rzm_source"].str.contains("software", case=False)
    flag_software_dates = df[condition].index
    flagger(bad_dates=flag_software_dates, brief_desc=SIMUL_DESC, df=df,
            bin_value=1, affected_cols=["etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=SIMUL_DESC)

    return df


# 6.
def flag_suspicious_and_missing_profile_events(df):
    """
    Flag events in which the profile readings are equal to 0.
    Flag events in which data blips occur; a data blip is defined as an event
    in which the profile reading is NaN, and the profile reading of the
    previous day dips low.
    Flag events in which unrealistically LARGE profile dips occur.
    :param df: The DataFrame for which we need to perform flagging.
    :return: Return updated DataFrame in which flagging has taken place.
    """
    # Flag events in which the profile readings are equal to 0.
    df["profile"].replace(0.0, np.nan, inplace=True)
    condition = df["profile"].isnull()
    bad_profile_days = df[condition].index
    flagger(bad_dates=bad_profile_days, brief_desc=NULL_PROFILE_DESC, df=df,
            bin_value=1, affected_cols=["profile", "etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=NULL_PROFILE_DESC)

    # Create interim_df containing the profile entries, as well as the
    # difference between consecutive profile entries.
    interim_df = pd.DataFrame(data={"profile_difference": df["profile"].diff(),
                                    "profile": df["profile"]},
                              index=df.index)

    # Flag events in which data blips occur; a data blip is defined as an event
    # in which the profile reading is NaN, and the profile reading of the
    # previous day dips very low.
    data_blip_days = []
    for d in interim_df.index:
        try:
            cond_1 = interim_df.loc[d, "profile_difference"] < 0
            cond_2 = pd.isnull(interim_df.loc[d + DAY, "profile"])
            if cond_1 & cond_2:
                data_blip_days.append(d)
        except KeyError:  # i.e. .loc[d + DAY] does not exist in the DataFrame.
            pass
    data_blip_days = pd.to_datetime(data_blip_days)
    flagger(bad_dates=data_blip_days, brief_desc=DATA_BLIP_DESC, df=df,
            bin_value=1, affected_cols=["etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=DATA_BLIP_DESC)

    # Set the profile dips preceding the data blips to NaN values:
    interim_df.loc[data_blip_days, ["profile_difference"]] = np.nan
    # Find all the dates in which the profile reading dips from its previous
    # value on the preceding day:
    bool_series = interim_df["profile_difference"] < 0
    # Collect all these negative_differences into an array-like object:
    negative_differences = interim_df[bool_series]["profile_difference"].values
    # Identify the "profile_difference" value corresponding to 5th percentile.
    # Note that low percentiles, say e.g. the 2nd percentile, correspond with
    # 'very negative' "profile_difference" readings, whereas large percentile
    # values, say e.g. the 98th percentile, correspond to 'slightly negative'
    # "profile_difference" values.
    percentile_value = np.quantile(negative_differences, q=[0.01, 0.02, 0.03,
                                                            0.04, 0.05, 0.06,
                                                            0.07, 0.08, 0.09,
                                                            0.10])[4]

    # Flag the dates for which "profile_difference" is less than the 5th
    # percentile, and where simultaneously the following day's profile reading
    # spikes up from the previous day's profile dip.
    large_dip_days = []
    for d in interim_df.index:
        try:
            cond_1 = interim_df.loc[d, "profile_difference"] < percentile_value
            cond_2 = interim_df.loc[d + DAY, "profile_difference"] > 0
            if cond_1 & cond_2:
                large_dip_days.append(d)
        except KeyError:  # i.e. .loc[d + DAY] does not exist.
            pass
    large_dip_days = pd.to_datetime(large_dip_days)
    flagger(bad_dates=large_dip_days, brief_desc=LARGE_PROFILE_DIP_DESC, df=df,
            bin_value=1, affected_cols=["etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=LARGE_PROFILE_DIP_DESC)
    return df


# 7.a.
def flag_spurious_heat_units(df):
    """
    Flag events in which the "heat_units" are NaN.
    :param df: The DataFrame in which flagging needs to take place.
    :return: The updated DataFrame in which flagging has taken place.
    """
    condition = ~np.isfinite(df["heat_units"])
    bad_hu_days = df[condition].index
    flagger(bad_dates=bad_hu_days, brief_desc=HU_BAD_DESC, df=df, bin_value=0,
            affected_cols=["heat_units"], set_to_nan=True)
    reporter(df=df, brief_desc=HU_BAD_DESC)
    return df


# 7.b
def interpolate_missing_heat_units(df, method="nearest"):
    """
    Interpolate all NaN entries in the "heat_units" column using some
    interpolation scheme.  The default method is "nearest".
    """
    df["interpolated_hu"] = df["heat_units"].copy(deep=True)
    df.loc[:, "interpolated_hu"].interpolate(method=method, axis=0,
                                             inplace=True)
    return df


# 7.c
def cumulative_gdd(df):
    """
    Create a new series, df["cumulative_gdd"], in the DataFrame df which gives
    the cumulative growing degree days as a function of date.  The cumulative
    GDD is simply calculated by accumulating the "heat_units" values from the
    start of any season.
    :param df: The DataFrame in which we want to instantiate the new series.
    :return: Return the updated DataFrame containing the df["cumulative_gdd"]
    series.
    """
    season_candidates = []
    df["cumulative_gdd"] = 0
    for datestamp in df.index:
        if (datestamp.month == BEGINNING_MONTH) and (datestamp.day == 1):
            season_candidates.append(datestamp)
            df.loc[datestamp, "cumulative_gdd"] = 0.0
        elif (datestamp.month == BEGINNING_MONTH) and (datestamp.day == 2):
            df.loc[datestamp, "cumulative_gdd"] = \
                df.loc[datestamp, "interpolated_hu"]
        else:
            try:
                df.loc[datestamp, "cumulative_gdd"] = \
                    df.loc[(datestamp - DAY), "cumulative_gdd"] + \
                    df.loc[datestamp, "interpolated_hu"]
            except KeyError:  # we cannot go 1 day back into the past.
                df.loc[datestamp, "cumulative_gdd"] = \
                    df.loc[datestamp, "interpolated_hu"]

    # In the date range [api_start_date; api_end_date], identify the first
    # season starting date.  All dates preceding first_season_date will have a
    # df["cumulative_gdd"] value of np.nan.  It is ambiguous to calculate the
    # GDD for these dates since they cannot be anchored to a start of a season
    # before `first_season_date`.
    first_season_date = min(season_candidates)
    condition = df.index < first_season_date
    pre_first_season = df[condition].index
    df.loc[pre_first_season, "cumulative_gdd"] = np.nan
    return df


# 8.
def flag_unwanted_etcp(df):
    """
    Flag events in which df["etcp"] is positive.
    Flag water uptake events that are suspiciously low and suspiciously high.
    Flag events in which etcp exceeds the maximum allowed ETCP_MAX.
    Flag events in which etcp is greater than eto*KCP_MAX.  Stated differently,
    flag the events in which the ratio etcp/eto exceeds KCP_MAX.
    :param df: The DataFrame for which flagging needs to take place.
    :return: Return updated DataFrame that has been flagged.
    """
    # Flag events in which df["etcp"] is positive.
    condition = (df["etcp"] >= 0.0)
    positive_etcp_days = df[condition].index
    flagger(bad_dates=positive_etcp_days, brief_desc=ETCP_POS_DESC, df=df,
            bin_value=1, affected_cols=["etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=ETCP_POS_DESC)

    # Flag water uptake events that are suspiciously low.
    c_1 = df["etc"].multiply(-1*SUSP_LOW_VAL, fill_value=np.nan) < df["etcp"]
    c_2 = df["etcp"] < 0
    condition = c_1 & c_2
    susp_low_etcp_dates = df[condition].index
    flagger(bad_dates=susp_low_etcp_dates, brief_desc=ETCP_SUSP_LOW_DESC,
            df=df, bin_value=1, affected_cols=["etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=ETCP_SUSP_LOW_DESC)

    # Flag water uptake events that are suspiciously high.
    c_1 = df["etcp"] < df["etc"].multiply(-1*SUSP_HIGH_VAL, fill_value=np.nan)
    c_2 = df["etcp"] < 0
    condition = c_1 & c_2
    susp_high_etcp_dates = df[condition].index
    flagger(bad_dates=susp_high_etcp_dates, brief_desc=ETCP_SUSP_HIGH_DESC,
            df=df, bin_value=1, affected_cols=["etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=ETCP_SUSP_HIGH_DESC)

    # To simplify the remaining programming, multiply the `etcp` column with -1
    # so that we henceforth only work with positive values.
    df["etcp"] = df["etcp"].multiply(-1, fill_value=np.nan)

    # Flag water uptake events exceeding ETCP_MAX.
    condition = df["etcp"] > ETCP_MAX
    etcp_outlier_dates = df[condition].index
    flagger(bad_dates=etcp_outlier_dates, brief_desc=ETCP_OUTLIERS_DESC, df=df,
            bin_value=1, affected_cols=["etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=ETCP_OUTLIERS_DESC)

    # Flag events in which the ratio etcp/eto exceeds KCP_MAX.
    condition = df["etcp"] > df["eto"].mul(KCP_MAX, fill_value=np.nan)
    exceeding_normal_uptake_dates = df[condition].index
    flagger(bad_dates=exceeding_normal_uptake_dates,
            brief_desc=EXCEEDING_KCP_MAX_DESC, df=df, bin_value=1,
            affected_cols=["etcp"], set_to_nan=True)
    reporter(df=df, brief_desc=EXCEEDING_KCP_MAX_DESC)
    return df


# 9.
def flag_unwanted_kcp(df):
    """
    Flag events where kcp deviates too much from the reference crop
    coefficients, cco.  Also flag events where kcp is NaN.
    :param df: The DataFrame that needs to be flagged.
    :return: Return the updated DataFrame in which flagging has taken place.
    """
    # Flag events where kcp deviates too much from the reference crop
    # coefficient, cco.
    df["kcp"] = df["etcp"].div(df["eto"])
    perc_series = calculate_kcp_deviation(df)
    condition = (perc_series > KCP_PERC_DEVIATION*100)
    bad_calc_kcp_dates = df[condition].index
    flagger(bad_dates=bad_calc_kcp_dates, brief_desc=BAD_KCP_DESC, df=df,
            bin_value=1, affected_cols=["kcp"], set_to_nan=True)
    reporter(df=df, brief_desc=BAD_KCP_DESC)
    # -------------------------------------------------------------------------
    # Flag events where kcp is NaN.
    condition = ~np.isfinite(df['kcp'])
    missing_kcp_dates = df[condition].index
    flagger(bad_dates=missing_kcp_dates, brief_desc=KCP_IS_NAN_DESC, df=df,
            bin_value=1, affected_cols=["kcp"], set_to_nan=True)
    reporter(df, brief_desc=KCP_IS_NAN_DESC, remaining=True)
    return df
# =============================================================================


# -----------------------------------------------------------------------------
# Define a function that wraps `useful_dates` into 1 Season interval (spanning
# an entire year).
# These 'wrapped' dates are stored in `new_dates`.
# `useful_dates` represent the original dates for which `binary_value = 0`.
# -----------------------------------------------------------------------------
def get_final_dates(df):
    condition = df["binary_value"] == 0
    useful_dates = df[condition].index
    starting_year = useful_dates[0].year
    new_dates = []
    for d in useful_dates:
        extracted_month = d.month
        if BEGINNING_MONTH <= extracted_month <= 12:
            new_dates.append(datetime.datetime(year=starting_year,
                                               month=d.month, day=d.day))
        else:
            new_dates.append(datetime.datetime(year=starting_year + 1,
                                               month=d.month, day=d.day))
    return starting_year, new_dates, useful_dates
# -----------------------------------------------------------------------------
