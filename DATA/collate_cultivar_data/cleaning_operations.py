import numpy as np
import datetime
import pandas as pd

# Definitions of certain constants
DAY = datetime.timedelta(days=1)

ERAIN_DESC = "erain perturbing etcp"
SIMUL_DESC = "Software simulation"
IRR_DESC = "Irrigation perturbing etcp"
NULL_PROFILE_DESC = "Null profile value"
HUGE_PROFILE_DIP_DESC = "Huge profile dip"
LARGE_PROFILE_DIP_DESC = "Large profile dip"
HU_STUCK_DESC = "Heat Units `stuck`"
ET0_STUCK_DESC = "et0 `stuck`"
ETCP_POS_DESC = "etcp is positive"
ETCP_OUTLIERS_DESC = "etcp outliers"
LUX_DESC = "Luxurious water uptake"

ET0_MAX = 12
KCP_MAX = 0.8
ETCP_MAX = ET0_MAX * KCP_MAX


def flagger(bad_dates, brief_desc, df):
    """
    Flag bad_dates with a binary value of 1 and append a brief description about why bad_dates were flagged.

    Parameters:
    bad_dates (pandas.core.indexes.datetimes.DatetimeIndex):  Dates for which we cannot calculate k_cp because our
    readings were perturbed and rendered unuseful.
    brief_desc (str):  A very short description about why bad_dates were flagged.

    Returns:
    None.  It updates the DataFrame storing all the information related to flagging.  In this case the DataFrame is
    called `df_flag`
    """
    if df.loc[bad_dates, "description"].str.contains(brief_desc).all(axis=0):
        # The bad_dates have already been flagged for the reason given in brief_desc.
        # No use in duplicating brief_desc contents in the description column.
        # Therefore redundant information in the df_flag DataFrame is avoided.
        print("You have already flagged these dates for the reason given in `brief_desc`; No flagging has taken place.")
        return
    else:
        df.loc[bad_dates, "binary_value"] = 1
        df.loc[bad_dates, "description"] += (" " + brief_desc + ".")
        df.loc[:, "description"] = df.loc[:, "description"].apply(lambda s: s.lstrip().rstrip())


def drop_redundant_columns(df):
    labels = ["rzone", "available", "days_left", "deficit_current",
              "rzm", "fcap", "deficit_want", "refill", "et0_forecast_yr"]
    df.drop(labels=labels, axis=1, inplace=True)


def flag_erain_events(df):
    df_rain = df.filter(["rain", "erain"], axis=1)

    condition = (df_rain["erain"] > 0) & (df["total_irrig"] == 0) & (df["etcp"] > 0)
    erain_dates = df_rain[condition].index

    flagger(bad_dates=erain_dates, brief_desc=ERAIN_DESC, df=df)


def flag_simulated_events(df):
    condition = df["rzm_source"].str.contains("software")
    flag_software_dates = df[condition].index

    flagger(bad_dates=flag_software_dates, brief_desc=SIMUL_DESC, df=df)


def flag_irrigation_events(df):
    df_irr = df.filter(["total_irrig"], axis=1)

    conditions = (df_irr["total_irrig"] > 0) & (df["etcp"] > 0) & (df["rain"] == 0)
    flag_irrigation_dates = df[conditions].index

    flagger(bad_dates=flag_irrigation_dates, brief_desc=IRR_DESC, df=df)


def flag_suspicious_and_missing_profile_events(df):
    df_profile = df.filter(["profile"], axis=1)
    df_profile["difference"] = df_profile["profile"].diff()

    df_profile["profile"].replace(0.0, np.nan, inplace=True)
    condition = df_profile["profile"].isnull()
    bad_profile_days = df_profile[condition].index
    flagger(bad_dates=bad_profile_days, brief_desc=NULL_PROFILE_DESC, df=df)

    huge_dip_days = []
    for d in df_profile.index:
        try:
            if (df_profile.loc[d, "difference"] < 0) and pd.isnull(df_profile.loc[d + DAY, "profile"]):
                huge_dip_days.append(d)
        except KeyError:
            pass
    huge_dip_days = pd.to_datetime(huge_dip_days)
    flagger(bad_dates=huge_dip_days, brief_desc=HUGE_PROFILE_DIP_DESC, df=df)
    df_profile.loc[huge_dip_days, ["profile"]] = np.nan
    df.loc[huge_dip_days, ["profile"]] = np.nan

    df_profile.loc[huge_dip_days, ["difference"]] = np.nan
    negative_differences = df_profile[df_profile["difference"] < 0]["difference"].values
    percentile_value = np.quantile(negative_differences, q=[0.01, 0.015, 0.02])[2]
    large_dip_days = []
    for d in df_profile.index:
        try:
            if (df_profile.loc[d, "difference"] < percentile_value) and (df_profile.loc[d + DAY, "difference"] > 0):
                large_dip_days.append(d)
        except KeyError:
            pass
    large_dip_days = pd.to_datetime(large_dip_days)
    flagger(bad_dates=large_dip_days, brief_desc=LARGE_PROFILE_DIP_DESC, df=df)
    df_profile.loc[large_dip_days, ["profile"]] = np.nan
    df.loc[large_dip_days, ["profile"]] = np.nan


def flag_spurious_heat_units(df):
    df_gdd = df.filter(["heat_units"], axis=1)
    df_gdd["hu_diff1"] = df_gdd["heat_units"].diff(periods=1)
    df_gdd["hu_diff2"] = df_gdd["heat_units"].diff(periods=2)
    condition = (df_gdd["hu_diff1"] == 0.0) | (df_gdd["hu_diff2"] == 0)
    bad_hu_days = df_gdd[condition].index
    flagger(bad_dates=bad_hu_days, brief_desc=HU_STUCK_DESC, df=df)
    df_gdd.loc[bad_hu_days, ["heat_units"]] = 0.0
    df.loc[bad_hu_days, ["heat_units"]] = 0.0


def flag_spurious_et0(df):
    df_et0 = df.filter(["et0"], axis=1)
    df_et0["et0_diff1"] = df_et0["et0"].diff(periods=1)
    df_et0["et0_diff2"] = df_et0["et0"].diff(periods=2)
    condition = (df_et0["et0_diff1"] == 0.0) | (df_et0["et0_diff2"] == 0)
    bad_et0_days = df_et0[condition].index
    flagger(bad_dates=bad_et0_days, brief_desc=ET0_STUCK_DESC, df=df)
    df_et0.loc[bad_et0_days, ["et0"]] = np.nan
    df.loc[bad_et0_days, ["et0"]] = np.nan


def flag_unwanted_etcp(df):
    df_etcp = df.filter(["etcp"], axis=1)

    condition = df_etcp["etcp"] >= 0.0
    positive_etcp_days = df_etcp[condition].index
    flagger(bad_dates=positive_etcp_days, brief_desc=ETCP_POS_DESC, df=df)
    df_etcp.loc[positive_etcp_days, ["etcp"]] = np.nan
    df.loc[positive_etcp_days, ["etcp"]] = np.nan

    condition = df["binary_value"] == 1
    junk_data_dates = df[condition].index
    df_etcp.loc[junk_data_dates, ["etcp"]] = np.nan
    df.loc[junk_data_dates, ["etcp"]] = np.nan

    # to simply programming, multiply `etcp` column with -1
    df_etcp["etcp"] = df_etcp["etcp"].multiply(-1, fill_value=np.nan)
    df["etcp"] = df["etcp"].multiply(-1, fill_value=np.nan)

    condition = df_etcp["etcp"] > ETCP_MAX
    etcp_outlier_dates = df_etcp[condition].index
    df_etcp.loc[etcp_outlier_dates, ["etcp"]] = np.nan
    df.loc[etcp_outlier_dates, ["etcp"]] = np.nan
    flagger(bad_dates=etcp_outlier_dates, brief_desc=ETCP_OUTLIERS_DESC, df=df)

    condition = df_etcp["etcp"] > df["et0"].mul(KCP_MAX, fill_value=np.nan)
    luxurious_dates = df_etcp[condition].index
    df_etcp.loc[luxurious_dates, ["etcp"]] = np.nan
    df.loc[luxurious_dates, ["etcp"]] = np.nan
    flagger(bad_dates=luxurious_dates, brief_desc=LUX_DESC, df=df)
