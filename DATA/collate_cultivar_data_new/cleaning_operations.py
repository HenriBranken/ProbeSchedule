import numpy as np
import datetime
import pandas as pd
import master_data

# ----------------------------------------------------------------------------------------------------------------------
# Definitions of certain constants
# ----------------------------------------------------------------------------------------------------------------------
DAY = datetime.timedelta(days=1)

# The following events are "irredeemable"
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

# The following events are "redeemable"
HU_STUCK_DESC = "Heat Units `stuck`"
ETO_STUCK_DESC = "Eto `stuck`"
ETC_STUCK_DESC = "Stuck etc due to stuck eto"
IMPUTED_ETO = "Imputed eto"
REDEEMABLE = [HU_STUCK_DESC, ETO_STUCK_DESC, ETC_STUCK_DESC, IMPUTED_ETO]

ETO_MAX = 12
KCP_MAX = 0.8
ETCP_MAX = ETO_MAX * KCP_MAX
BEGINNING_MONTH = 7
# ----------------------------------------------------------------------------------------------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define some helper functions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def flagger(bad_dates, brief_desc, df, bin_value=0):
    """
        Flag bad_dates with a binary value of 1 and append a brief description about why bad_dates were flagged.

        Parameters:
        bad_dates (pandas.core.indexes.datetimes.DatetimeIndex):  Dates for which we cannot calculate k_cp because our
        readings were perturbed and rendered unuseful.
        brief_desc (str):  A very short description about why bad_dates were flagged.
        bin_value (int):  The binary value.  If Eto is imputed, Etc and heat_units are stuck, we can still get away with
        a new calculation of kcp; thus set binary_value=0 for such redeemable events.

        Returns:
        None.  It updates the DataFrame storing all the information related to flagging.  In this case the DataFrame is
        called `df`
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


def reporter(df, brief_desc, remaining=False):
    tally = df["description"].str.contains(brief_desc).sum()
    n_tot_entries = len(df.index)
    perc = tally / n_tot_entries * 100
    print("{:.1f}% of data is affected due to [{}].".format(perc, brief_desc))

    if remaining:
        calc = 100 - df["binary_value"].sum() / len(df.index) * 100
        print("After all the flagging that has taken place for this probe data-set,"
              " only {:.0f}% of your data is useful.".format(calc))


def drop_redundant_columns(df):
    labels = ["rzone", "available", "days_left", "deficit_current",
              "rzm", "fcap", "deficit_want", "refill", "eto_forecast_yr"]
    df.drop(labels=labels, axis=1, inplace=True)
    return df


def calculate_kcp_deviation(df):
    df["kcp_perc_deviation"] = 0.0
    empirical_kcp = df["kcp"]
    associated_kcp_norm = df["cco"]
    df["kcp_perc_deviation"] = np.abs((empirical_kcp - associated_kcp_norm)/associated_kcp_norm) * 100.0
    return df["kcp_perc_deviation"]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ======================================================================================================================
# Define all the flagging functions
# ======================================================================================================================
def flag_spurious_eto(df):
    df["eto_diff1"] = df["eto"].diff(periods=1)
    df["eto_diff2"] = df["eto"].diff(periods=2)
    condition = (df["eto_diff1"] == 0.0) | (df["eto_diff2"] == 0)
    bad_eto_days = df[condition].index
    flagger(bad_dates=bad_eto_days, brief_desc=ETO_STUCK_DESC, df=df, bin_value=1)
    df.loc[bad_eto_days, ["eto"]] = np.nan

    df.loc[bad_eto_days, ["etc"]] = np.nan
    flagger(bad_dates=bad_eto_days, brief_desc=ETC_STUCK_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=ETC_STUCK_DESC)
    return bad_eto_days, df


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


def impute_kbv_data(df, bad_eto_days):
    return kbv_imputer(flagged_dates=bad_eto_days, df=df, column_to_be_imputed="eto")


def flag_rain_events(df):
    condition = (df["rain"] > 2)
    bad_rain_dates = df[condition].index
    flagger(bad_dates=bad_rain_dates, brief_desc=RAIN_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=RAIN_DESC)
    return df


def flag_irrigation_events(df):
    conditions = (df["total_irrig"] > 0) & (df["etcp"] > 0.5*df["etc"]) & (df["rain"] == 0)
    flag_irrigation_dates = df[conditions].index
    flagger(bad_dates=flag_irrigation_dates, brief_desc=IRR_DESC, df=df)
    reporter(df=df, brief_desc=IRR_DESC)
    return df


def flag_simulated_events(df):
    condition = df["rzm_source"].str.contains("software")
    flag_software_dates = df[condition].index
    flagger(bad_dates=flag_software_dates, brief_desc=SIMUL_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=SIMUL_DESC)
    return df


def flag_suspicious_and_missing_profile_events(df):
    df["profile"].replace(0.0, np.nan, inplace=True)
    condition = df["profile"].isnull()
    bad_profile_days = df[condition].index
    flagger(bad_dates=bad_profile_days, brief_desc=NULL_PROFILE_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=NULL_PROFILE_DESC)

    df["profile_difference"] = df["profile"].diff()
    data_blip_days = []
    for d in df.index:
        try:
            if (df.loc[d, "profile_difference"] < 0) and pd.isnull(df.loc[d + DAY, "profile"]):
                data_blip_days.append(d)
        except KeyError:
            pass
    data_blip_days = pd.to_datetime(data_blip_days)
    flagger(bad_dates=data_blip_days, brief_desc=DATA_BLIP_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=DATA_BLIP_DESC)

    df.loc[data_blip_days, ["profile_difference"]] = np.nan
    negative_differences = df[df["profile_difference"] < 0]["profile_difference"].values
    percentile_value = np.quantile(negative_differences, q=[0.01, 0.02, 0.03, 0.04, 0.05,
                                                            0.06, 0.07, 0.08, 0.09, 0.10])[4]
    large_dip_days = []
    for d in df.index:
        try:
            if (df.loc[d, "profile_difference"] < percentile_value) and (df.loc[d + DAY, "profile_difference"] > 0):
                large_dip_days.append(d)
        except KeyError:
            pass
    large_dip_days = pd.to_datetime(large_dip_days)
    flagger(bad_dates=large_dip_days, brief_desc=LARGE_PROFILE_DIP_DESC, df=df, bin_value=1)
    reporter(df=df, brief_desc=LARGE_PROFILE_DIP_DESC)
    return df


def flag_spurious_heat_units(df):
    df["hu_diff1"] = df["heat_units"].diff(periods=1)
    df["hu_diff2"] = df["heat_units"].diff(periods=2)
    condition = (df["hu_diff1"] == 0.0) | (df["hu_diff2"] == 0)
    bad_hu_days = df[condition].index
    flagger(bad_dates=bad_hu_days, brief_desc=HU_STUCK_DESC, df=df, bin_value=0)
    reporter(df=df, brief_desc=HU_STUCK_DESC)
    df.loc[bad_hu_days, ["heat_units"]] = 0.0
    df.loc[bad_hu_days, ["heat_units"]] = 0.0
    return df


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
# Define a function that wraps all the dates into 1 Season interval
# ----------------------------------------------------------------------------------------------------------------------
def get_final_dates(df):
    condition = df["binary_value"] == 0
    useful_dates = df[condition].index
    starting_year = useful_dates[0].year
    new_dates = []
    for d in useful_dates:
        extracted_month = d.month
        if BEGINNING_MONTH <= extracted_month <= 12:
            new_dates.append(datetime.datetime(year=starting_year, month=d.month, day=d.day))
        else:
            new_dates.append(datetime.datetime(year=starting_year + 1, month=d.month, day=d.day))
    return starting_year, new_dates, useful_dates
# ----------------------------------------------------------------------------------------------------------------------
