import numpy as np
import math
from scipy.signal import argrelextrema
import pandas as pd


class NoProperWMATrend(Exception):
    pass


pol_degree = 4


def rectify_trend(fitted_trend_values):
    if all(j > 0 for j in fitted_trend_values):
        return fitted_trend_values
    else:
        negative_indices = np.where(fitted_trend_values <= 0)[0]
        # identify all the indices where the fit is negative
        diff_arr = np.ediff1d(negative_indices, to_begin=1)
        if all(diff_arr == 1):
            if negative_indices[0] == 0:
                fitted_trend_values[negative_indices] = \
                    fitted_trend_values[negative_indices[-1] + 1]
            else:
                fitted_trend_values[negative_indices] = \
                    fitted_trend_values[negative_indices[0] - 1]
        else:
            special_index = np.where(diff_arr != 1)[0][0]
            index_where_left_neg_portion_ends = \
                negative_indices[special_index - 1]
            fitted_trend_values[0: index_where_left_neg_portion_ends + 1] = \
                fitted_trend_values[index_where_left_neg_portion_ends + 1]
            index_where_right_neg_portion_starts = \
                negative_indices[special_index]
            fitted_trend_values[index_where_right_neg_portion_starts:] = \
                fitted_trend_values[index_where_right_neg_portion_starts - 1]
        return fitted_trend_values


def find_nearest_index(model_array, raw_value):
    model_array = np.asarray(model_array)
    nearest_index = (np.abs(model_array - raw_value)).argmin()
    return nearest_index


def gaussian(x, amp=1, mean=0, sigma=10):
    return amp*np.exp(-(x - mean)**2 / (2*sigma**2))


def weighted_moving_average(x, y, step_size=1.0, width=10, x_lims=None):
    if x_lims:
        x_min, x_max = x_lims[0], x_lims[1]
    else:
        x_min, x_max = math.floor(min(x)), math.ceil(max(x))
    num = int((x_max - x_min) // step_size + 1)
    bin_coords = np.linspace(start=x_min, stop=x_max, num=num, endpoint=True)
    bin_avgs = np.zeros(len(bin_coords))

    for index in range(len(bin_coords)):
        weights = gaussian(x=x, mean=bin_coords[index], sigma=width)
        bin_avgs[index] = np.average(y, weights=weights)
    y_better = rectify_trend(fitted_trend_values=bin_avgs)
    return bin_coords, y_better


def get_r_squared(x_raw, y_raw, x_fit, y_fit):
    indices = []
    y_proxies = []
    # value of y_fit, whose x-coord is closest to the x-coord of y_raw.
    for x in x_raw:
        indices.append(find_nearest_index(x_fit, x))
    for j in indices:
        y_proxies.append(y_fit[j])
    ssres = 0
    for k in range(len(x_raw)):
        ssres += (y_proxies[k] - y_raw[k])**2
    return np.sqrt(ssres/len(y_raw))


def get_n_local_extrema(y_fit):
    args_loc_minima = argrelextrema(y_fit, np.less)
    num_loc_minima = len(args_loc_minima[0])
    args_loc_maxima = argrelextrema(y_fit, np.greater)
    num_loc_maxima = len(args_loc_maxima[0])
    num_loc_extrema = num_loc_minima + num_loc_maxima
    return num_loc_extrema


def get_prized_index(n_bumps_list):
    some_list = np.where(np.asarray(n_bumps_list) == 1)[0]
    if len(some_list) > 0:
        return some_list[-1]
    else:
        raise NoProperWMATrend("No index at which the number of extrema is 1.")


def create_xy_df(x_vals, y_vals, iteration, status):
    if status == "scatter":
        dataframe = pd.DataFrame(data={"x_scatter": x_vals,
                                       "y_scatter": y_vals,
                                       "iteration": iteration},
                                 copy=True)
        dataframe.index.name = "i"
    else:
        dataframe = pd.DataFrame(data={"x_smoothed": x_vals,
                                       "y_smoothed": y_vals,
                                       "iteration": iteration},
                                 copy=True)
        dataframe.index.name = "i"
    return dataframe


def get_xs_and_ys(dataframe, iteration, status="scatter"):
    x_var, y_var = "x_" + status, "y_" + status
    sub_df = dataframe.loc[(iteration, ), [x_var, y_var]]
    return sub_df[x_var].values, sub_df[y_var].values


def collapse_dataframe(multi_index_df, tbr_probe_list, start_date):
    df = multi_index_df.copy(deep=True)
    for pr in tbr_probe_list:
        df.drop(index=pr, level=0, inplace=True)
    df.index = df.index.droplevel(0)
    df.sort_index(axis=0, level="datetimeStamp", ascending=True, inplace=True)
    df["days"] = df.index - start_date
    df["days"] = df["days"].dt.days
    return df["days"].values, df["kcp"].values


def extract_probe_df(multi_index_df, probe, start_date):
    df = multi_index_df.loc[(probe, ), ["kcp"]]
    df["days"] = df.index - start_date
    df["days"] = df["days"].dt.days
    df.sort_values("days", ascending=True, inplace=True)
    return df[["days", "kcp"]]


def get_final_polynomial_fit(x_raw, y_raw, step_size=1.0, degree=pol_degree,
                             x_lims=None):
    if x_lims:
        x_min, x_max = x_lims[0], x_lims[1]
    else:
        x_min, x_max = math.floor(min(x_raw)), math.ceil(max(x_raw))
    num = int((x_max - x_min) // step_size + 1)
    bin_coords = np.linspace(start=x_min, stop=x_max, num=num, endpoint=True)
    coeffs = np.polyfit(x_raw, y_raw, degree)
    pol = np.poly1d(coeffs)
    bin_avgs = pol(bin_coords)
    y_better = rectify_trend(fitted_trend_values=bin_avgs)
    y_best = simplify_trend(fitted_trend_values=y_better)
    return bin_coords, y_best


def simplify_trend(fitted_trend_values):
    loc_maxima_index = argrelextrema(fitted_trend_values, np.greater)[0]
    loc_minima_indices = argrelextrema(fitted_trend_values, np.less)[0]
    if len(loc_minima_indices) >= 1:
        number_of_minima = len(loc_minima_indices)
        for j in range(number_of_minima):
            minima_index = loc_minima_indices[j]
            if minima_index < loc_maxima_index[0]:
                fitted_trend_values[0: minima_index] = \
                    fitted_trend_values[minima_index]
            else:
                fitted_trend_values[minima_index:] = \
                    fitted_trend_values[minima_index]
        return fitted_trend_values
    else:
        return fitted_trend_values


def get_offsets_from_trendline(x_raw, y_raw, x_fit, y_fit):
    offsets = []
    indices = []
    for x in x_raw:
        indices.append(find_nearest_index(x_fit, x))
    y_proxies = []
    for idx in indices:
        y_proxies.append(y_fit[idx])
    for j in range(len(y_raw)):
        offsets.append(np.abs(y_raw[j] - y_proxies[j]))
    return offsets


def create_interim_df(x_raw, y_raw, dev):
    dataframe = pd.DataFrame({"x": x_raw, "y": y_raw, "dev": dev}, copy=True)
    dataframe.index.name = "i"
    dataframe.sort_values(by="dev", axis=0, ascending=False, inplace=True)
    return dataframe


def create_new_generation_xy_scatter(dataframe, threshold):
    dataframe["to_be_removed"] = dataframe["dev"] >= threshold
    dataframe = dataframe[~dataframe["to_be_removed"]]
    dataframe.sort_values(by="x", axis=0, ascending=True, inplace=True)
    return dataframe["x"].values, dataframe["y"].values
