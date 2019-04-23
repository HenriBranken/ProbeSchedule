import pandas as pd
import numpy as np
import os
from itertools import cycle
import helper_functions as hf
import helper_meta_data as hm
import helper_data as hd
pd.options.mode.chained_assignment = None  # default='warn'


# =============================================================================
# Define some other important constants
# =============================================================================
n_neighbours_list = hm.n_neighbours_list
delta_x = hm.delta_x
x_limits = hm.x_limits[:]
marker_color_meta = hm.marker_color_meta[:]
marker_color_meta = cycle(marker_color_meta)
ALLOWED_TAIL_DEVIATION = hm.ALLOWED_TAIL_DEVIATION
top_border_1 = "+" + "-"*78 + "+\n"
line_1 = "+" + "-"*7 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + \
       "-"*19 + "+" + "-"*11 + "+\n"
top_border_2 = "+" + "-"*78 + "+\n"
line_2 = "+" + "-"*11 + "+" + "-"*16 + "+" + "-"*14 + "+" + "-"*17 + "+" + \
         "-"*16 + "+\n"
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import all the necessary data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Get the starting year: the year of the most "historic/past" sample.
starting_year = hm.starting_year
season_start_date = hm.season_start_date

# Load the cleaned (scatterplot) data of kcp versus datetimeStamp from
# "./data/stacked_cleaned_data_for_overlay.xlsx" that was saved in
# `step_1_perform_cleaning.py`.  We are interested in the `datetimeStamp`
# index, and the newly created `days` column.
cleaned_multi_df = hd.cleaned_multi_df
outer_index = hd.outer_index[:]
inner_index = hd.inner_index[:]

scatter_df = cleaned_multi_df.copy(deep=True)
scatter_df.index = scatter_df.index.droplevel(0)
scatter_df.sort_index(axis=0, level="datetimeStamp", ascending=True,
                      inplace=True)
scatter_df["days"] = scatter_df.index - season_start_date
scatter_df["days"] = scatter_df["days"].dt.days
x_scatter = scatter_df["days"].values
y_scatter = scatter_df["kcp"].values


# Import the (first/beginning) kcp vs datetimestamp smoothed trend from
# "binned_kcp_data.xlsx".  We need to import the sheet `day_frequency` from
# "binned_kcp_data.xlsx".  We are interested in the `season_day` column, and
# the `day_averaged_kcp` column.
smoothed_kcp_vs_date_df = pd.read_excel("./data/binned_kcp_data.xlsx",
                                        sheet_name="day_frequency", header=0,
                                        index_col=0, parse_dates=True,
                                        squeeze=False)
x_smoothed = smoothed_kcp_vs_date_df["season_day"].values
y_smoothed = smoothed_kcp_vs_date_df["day_averaged_kcp"].values


# Get the DataFrame holding the reference crop coefficients.
cco_df = hd.cco_df.copy(deep=True)
cco_x_scatter = cco_df["season_day"].values
cco_y_scatter = cco_df["cco"].values

# Get all the probe_ids.
probe_ids = hm.probe_ids[:]
n_iterations = int(len(probe_ids) - 1)


# Determine the final mode that was used in `step_3_smoothed_version.py`.
# mode is either "Polynomial-fit" or "WMA".  Preferably it should be "WMA".
with open("./data/mode.txt", "r") as f:
    mode = f.readline().rstrip()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Define some helper functions
# -----------------------------------------------------------------------------
def check_tails(y_trendline_1, leading_ref=cco_df["cco"].values[0],
                trailing_ref=cco_df["cco"].values[-1],
                allowed_dev=ALLOWED_TAIL_DEVIATION, suppress_check=False):
    leading_val, trailing_val = y_trendline_1[0], y_trendline_1[-1]
    leading_dev = np.absolute(leading_val - leading_ref)/leading_ref
    trailing_dev = np.absolute(trailing_val - trailing_ref)/trailing_ref
    if suppress_check:
        return True
    if (leading_dev <= allowed_dev) and (trailing_dev <= allowed_dev):
        return True
    else:
        return False


def check_improvement(latest_stat, stat_list, suppress_check=False):
    if suppress_check:
        return True
    if latest_stat <= stat_list[-1]:
        return True
    else:
        return False


def info_filler_1(iter_, pnts, cco_rs, sc_rs, delta, removed,
                  just_strings=False):
    if just_strings:
        a = "| {:^5s} ".format(iter_)
        b = "| {:^10s} ".format(pnts)
        c = "| {:^10s} ".format(cco_rs)
        d = "| {:^10s} ".format(sc_rs)
        g = "| {:^17s} ".format(delta)
        h = "| {:^9s} |\n".format(removed)
    else:
        a = "| {:>5.0f} ".format(iter_)
        b = "| {:>10.0f} ".format(pnts)
        c = "| {:>10.7f} ".format(cco_rs)
        d = "| {:>10.7f} ".format(sc_rs)
        if isinstance(delta, float):
            g = "| {:>17.7f} ".format(delta)
        else:
            g = "| {:>17s} ".format(delta)
        h = "| {:>9s} |\n".format(removed)
    return a + b + c + d + g + h


def info_filler_2(iter_, n_healthy_p, num_imp, num_det, bool_val,
                  just_strings=False):
    if just_strings:
        a = "| {:^9s} ".format(iter_)
        b = "| {:^14s} ".format(n_healthy_p)
        c = "| {:^12s} ".format(num_imp)
        d = "| {:^15s} ".format(num_det)
        g = "| {:^14s} |\n".format(bool_val)
    else:
        a = "| {:>9s} ".format("[{}, {}]".format(iter_, iter_ + 1))
        b = "| {:>14.0f} ".format(n_healthy_p)
        c = "| {:>12.0f} ".format(num_imp)
        d = "| {:>15.0f} ".format(num_det)
        g = "| {:>14s} |\n".format(str(bool_val))
    return a + b + c + d + g


def generate_output(the_string, out_file):
    print(the_string)
    out_file.write(the_string)
    return


def compare_old_vs_new(healthy_probes, x_fit, old_trend, new_trend):
    old_r_squares = []
    new_r_squares = []
    for healthy_probe in healthy_probes:
        sub_df = hf.extract_probe_df(multi_index_df=cleaned_multi_df,
                                     probe=healthy_probe,
                                     starting_date=season_start_date)
        old_r_squared = hf.get_r_squared(x_raw=sub_df["days"].values,
                                         y_raw=sub_df["kcp"].values,
                                         x_fit=x_fit, y_fit=old_trend)
        new_r_squared = hf.get_r_squared(x_raw=sub_df["days"].values,
                                         y_raw=sub_df["kcp"].values,
                                         x_fit=x_fit, y_fit=new_trend)
        old_r_squares.append(old_r_squared)
        new_r_squares.append(new_r_squared)
    boolean_list = [new_r_squares[ii] <= old_r_squares[ii] for ii in
                    range(len(healthy_probes))]
    number_improvements = sum(boolean_list)
    number_deterioriations = len(healthy_probes) - number_improvements
    comparison = number_improvements >= number_deterioriations
    return number_improvements, number_deterioriations, comparison
# -----------------------------------------------------------------------------


# =============================================================================
# Initialise the "starting points" corresponding to iteration i = 0.
# =============================================================================
r_squared_stat = hf.get_r_squared(x_raw=x_scatter, y_raw=y_scatter,
                                  x_fit=x_smoothed, y_fit=y_smoothed)
r_sqr_cco_stat = hf.get_r_squared(x_raw=cco_x_scatter, y_raw=cco_y_scatter,
                                  x_fit=x_smoothed, y_fit=y_smoothed)
xy_scatter_df = hf.create_xy_df(x_vals=x_scatter, y_vals=y_scatter,
                                iteration=int(0), status="scatter")
xy_smoothed_df = hf.create_xy_df(x_vals=x_smoothed, y_vals=y_smoothed,
                                 iteration=int(0), status="smoothed")
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data Munging.  In the for-loop we iteratively remove "bad" probes.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Populate lists with our "starting points" that we created in the above block:
r_squared_stat_list = [r_squared_stat]
r_sqr_cco_stat_list = [r_sqr_cco_stat]
xy_scatter_dfs = [xy_scatter_df]
xy_smoothed_dfs = [xy_smoothed_df]
removed_probes = []
healthy_old_vs_new = []

# Assert that the number of flagging iterations is less than the num_probes:
assert len(probe_ids) > n_iterations, "Number of iterations is greater than " \
                                      "the number of probes.\n" \
                                      "Decrease n_iterations such that " \
                                      "n_iterations < len(probe_ids)."

fn = open("./probe_screening/execution_output.txt", "w")
imps_and_dets = open("./probe_screening/probe_information.txt", "w")
imps_and_dets.write(top_border_2)
imps_and_dets.write("| {:^76s} |\n".format("Probe Information"))
imps_and_dets.write(line_2)
imps_and_dets.write(info_filler_2(iter_="", n_healthy_p="Number of",
                                  num_imp="Number of", num_det="Number of",
                                  bool_val="", just_strings=True))
imps_and_dets.write(info_filler_2(iter_="Iteration",
                                  n_healthy_p="healthy probes",
                                  num_imp="Improvements",
                                  num_det="Deterioriations",
                                  bool_val="n_imp >= n_det",
                                  just_strings=True))
imps_and_dets.write(line_2)

q = 0
# Cycle through the screening logic `n_iterations` times.
for i in range(n_iterations):
    y_smoothed_old = np.copy(y_smoothed)
    try:
        # Populate probes_r_squared list with r-squares between probe scatter
        # data and the current smoothed trend line_1:
        probes_r_squared = []
        # Loop over all the probes:
        for p in probe_ids:
            # Extract an individual probe:
            probe_df = hf.extract_probe_df(multi_index_df=cleaned_multi_df,
                                           probe=p,
                                           starting_date=season_start_date)
            # Compute r-squared between probe scatter and current trend line_1:
            val = hf.get_r_squared(x_raw=probe_df["days"].values,
                                   y_raw=probe_df["kcp"].values,
                                   x_fit=x_smoothed, y_fit=y_smoothed)
            # Append r-squared `val` to the growing probes_r_squared list:
            probes_r_squared.append(val)
        # Extract the probe index whose probe scatter data is the most poorly
        # fit by the smoothed trendline_1.  Call this probe the "worst probe":
        max_arg_index = np.where(probes_r_squared == max(probes_r_squared))[0]
        max_arg_index = max_arg_index[0]
        # Append the "worst probe" to `removed_probes`.
        removed_probes.append(probe_ids.pop(max_arg_index))
        probes_r_squared.pop(max_arg_index)
        # Generate a "pruned" (xy) scatter dataset, and assign to `x_scatter`
        # and `y_scatter`:
        some_tuple = hf.collapse_dataframe(multi_index_df=cleaned_multi_df,
                                           tbr_probe_list=removed_probes,
                                           starting_date=season_start_date)
        # We have finally garnered our `x_scatter` and `y_scatter` data.
        x_scatter, y_scatter = some_tuple
        # Determine the mode; whether it be "WMA" or "Polynomial-fit".
        if mode == "WMA":
            # This will store all the various trend line_1s associated with
            # different values of `n_neighbours`:
            saved_trend_line_1s = []
            num_bumps = []
            tracker = 0
            for n_neighbours in n_neighbours_list:
                try:
                    # Generate new smoothed trendline_1 based on `n_neighbours`
                    # hyperparameter:
                    x_smoothed, y_smoothed = \
                        hf.weighted_moving_average(x=x_scatter, y=y_scatter,
                                                   step_size=delta_x,
                                                   width=n_neighbours,
                                                   x_lims=x_limits,
                                                   append_=True)
                    # Append smoothed trend line_1 to `saved_trend_line_1s`.
                    saved_trend_line_1s.append(zip(x_smoothed, y_smoothed))
                    # Append n_bumps to `num_bumps` list.
                    num_bumps.append(hf.get_n_local_extrema(y_smoothed))
                    tracker += 1
                # I.e. out scatter is too sparse to generate a WMA trendline_1:
                except ZeroDivisionError:
                    # Truncate `n_neighbours_list` to contain only the
                    # `n_neighbours` that we have looped over in the for-loop.
                    n_neighbours_list = n_neighbours_list[:tracker]
                    # Exit the for-loop; `n_neighbours` has become too small.
                    break
            try:
                # Get index at which the smoothed trend line_1 has only 1 bump.
                prized_index = hf.get_prized_index(num_bumps)
                # Extract the associated smoothed trend_line_1 having 1 bump.
                trend_line_1 = saved_trend_line_1s[prized_index]
                # Extract `x_smoothed` and `y_smoothed` from `trend_line_1`.
                some_tuple = [list(t) for t in zip(*trend_line_1)]
                x_smoothed, y_smoothed = some_tuple[0], some_tuple[1]
            # I.e. there is no trendline_1 just having 1 bump as desired.
            # Therefore we have to switch over to "Polynomial-fit" mode.
            except hf.NoProperWMATrend as e_1:
                generate_output(the_string="Cannot perform Gaussian-WMA.\n"
                                           "Proceeding with Polynomial Fit.\n",
                                out_file=fn)
                mode = "Polynomial-fit"  # Switch over to "Polynomial-fit" mode
                # Perform 4th-order pol-fit to `x_scatter` and `y_scatter`.
                x_smoothed, y_smoothed = \
                    hf.get_final_polynomial_fit(x_raw=x_scatter,
                                                y_raw=y_scatter,
                                                step_size=delta_x,
                                                degree=hf.pol_degree,
                                                x_lims=x_limits)
        # I.e. we know for sure that mode is "Polynomial-fit" and we do not
        # have to bother with "WMA" trendline_1s.
        else:
            # Perform a 4th-order polynomial fit to the scatter data.
            x_smoothed, y_smoothed = \
                hf.get_final_polynomial_fit(x_raw=x_scatter,
                                            y_raw=y_scatter,
                                            step_size=delta_x,
                                            degree=hf.pol_degree,
                                            x_lims=x_limits)
        # At this point we have finally garnered our new `x_smoothed` and
        # `y_smoothed` dataset.
        r_squared_stat = hf.get_r_squared(x_raw=x_scatter,
                                          y_raw=y_scatter,
                                          x_fit=x_smoothed,
                                          y_fit=y_smoothed)
        r_sqr_cco_stat = hf.get_r_squared(x_raw=cco_x_scatter,
                                          y_raw=cco_y_scatter,
                                          x_fit=x_smoothed,
                                          y_fit=y_smoothed)
        perf_status = check_improvement(latest_stat=r_squared_stat,
                                        stat_list=r_squared_stat_list,
                                        suppress_check=False)

        if perf_status:
            pass
        else:
            probe_ids.append(removed_probes.pop())
            generate_output(the_string="There is no improvement in the "
                                       "r-squared statistic.\n"
                                       "Exiting the for-loop prematurely.\n",
                            out_file=fn)
            break
        tail_status = check_tails(y_trendline_1=y_smoothed,
                                  suppress_check=False)

        # I.e., if `tail_status` is True:
        if tail_status:
            pass
        else:
            probe_ids.append(removed_probes.pop())
            generate_output(the_string="The Tail standards are NOT "
                                       "satisfied.\n"
                                       "Have to exit the for-loop "
                                       "prematurely.\n", out_file=fn)
            break

        n_imp, n_det, majority_vote = \
            compare_old_vs_new(healthy_probes=probe_ids,
                               x_fit=x_smoothed, old_trend=y_smoothed_old,
                               new_trend=y_smoothed)
        imps_and_dets.write(info_filler_2(iter_=i,
                                          n_healthy_p=len(probe_ids),
                                          num_imp=n_imp,
                                          num_det=n_det,
                                          bool_val=majority_vote))
        healthy_old_vs_new.append(majority_vote)
        if majority_vote:
            pass
        # I.e. majority_vote is FALSE.
        else:
            probe_ids.append(removed_probes.pop())
            generate_output(the_string="The trendline fit for the "
                                       "MAJORITY of the probes did "
                                       "not improve.\n"
                                       "Exiting the for-loop "
                                       "prematurely.",
                            out_file=fn)
            break
        xy_smoothed_dfs.append(hf.create_xy_df(x_vals=x_smoothed,
                                               y_vals=y_smoothed,
                                               iteration=int(i + 1),
                                               status="smoothed"))
        xy_scatter_dfs.append(hf.create_xy_df(x_vals=x_scatter,
                                              y_vals=y_scatter,
                                              iteration=int(i + 1),
                                              status="scatter"))
        r_squared_stat_list.append(r_squared_stat)
        r_sqr_cco_stat_list.append(r_sqr_cco_stat)
        q += 1
    # I.e. we cannot cycle through the screening logic `n_iterations` times:
    except IndexError as e_2:
        generate_output(the_string="Cannot generate a new trendline_1.\n"
                                   "Have to exit the for-loop prematurely.\n",
                        out_file=fn)
        break  # exit the for-loop
n_iterations = q  # we re-assign n_iterations
deltas_r_squared = list(np.ediff1d(r_squared_stat_list))
deltas_r_squared.insert(0, "-")

imps_and_dets.write(line_2)
imps_and_dets.close()
fn.close()


# -----------------------------------------------------------------------------
# Stack all the dataframes together into a MultiIndex DataFrame
# -----------------------------------------------------------------------------
xy_scatter_df = pd.concat(xy_scatter_dfs, ignore_index=False, sort=False,
                          copy=True)
xy_scatter_df = xy_scatter_df.set_index(["iteration", xy_scatter_df.index])

xy_smoothed_df = pd.concat(xy_smoothed_dfs, ignore_index=False, sort=False,
                           copy=True)
xy_smoothed_df = xy_smoothed_df.set_index(["iteration", xy_smoothed_df.index])
# -----------------------------------------------------------------------------


# =============================================================================
# Print out some basic results of the probe-screening algorithm:
# =============================================================================
print("\n" + "."*80)
print("Number of successful probe removals = {:.0f}.".format(q))
print("len(xy_scatter_dfs) = {:.0f}.".format(len(xy_scatter_dfs)))
print("len(xy_smoothed_dfs) = {:.0f}.".format(len(xy_smoothed_dfs)))
print("len(r_squared_stat_list) = {:.0f}.".format(len(r_squared_stat_list)))
print("len(r_sqr_cco_stat_list) = {:.0f}.".format(len(r_sqr_cco_stat_list)))
print("len(removed_probes) = {:.0f}.".format(len(removed_probes)))
print("len(probe_ids) = {:.0f}.".format(len(probe_ids)))
print("num_probes = len(probe_ids) + len(removed_probes) "
      "= {:.0f}.".format(len(probe_ids) + len(removed_probes)))
print("removed_probes = {}.".format(removed_probes))
print("healthy_old_vs_new = {}.".format(healthy_old_vs_new))
print("."*80 + "\n")
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Save the results to disk.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. "./probe_screening/xy_scatter_dfs.xlsx"
# 2. "./probe_screening/xy_smoothed_dfs.xlsx"
#    "./probe_screening/pruned_kcp_vs_days.xlsx"
# 3. "./probe_screening/r_squared_stat_list.txt"
# 4. "./probe_screening/r_sqr_cco_stat_list.txt"
# 5. "./probe_screening/removed_probes.txt"
# 6. "./probe_screening/healthy_probes.txt"
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
removed_probes.insert(0, "-")

if not os.path.exists("./probe_screening/"):
    os.makedirs("probe_screening")

directory = "./probe_screening/"
files = os.listdir(directory)
for file in files:
    if file.endswith((".xlsx", "_stat_list.txt", "_.idx.txt")):
        os.remove(os.path.join(directory, file))
        print("Removed the file named: {}.".format(file))
    if os.path.exists("./probe_screening/removed_probes.txt"):
        os.remove("./probe_screening/removed_probes.txt")
        print("Removed the file named: removed_probes.txt.")
    if os.path.exists("./probe_screening/screening_report.txt"):
        os.remove("./probe_screening/screening_report.txt")
        print("Removed the file named: screening_report.txt.")
    if os.path.exists("./probe_screening/healthy_probes.txt"):
        os.remove("./probe_screening/healthy_probes.txt")
        print("Removed the file named: healthy_probes.txt.")

# 1:
writer_sc = pd.ExcelWriter("./probe_screening/xy_scatter_dfs.xlsx",
                           engine="xlsxwriter")
# 2:
writer_sm = pd.ExcelWriter("./probe_screening/xy_smoothed_dfs.xlsx",
                           engine="xlsxwriter")
for i in range(q + 1):
    xy_scatter_dfs[i].to_excel(writer_sc, sheet_name="iter_{:.0f}".format(i),
                               columns=["x_scatter", "y_scatter"],
                               header=True, index=True, index_label="j",
                               float_format="%.7f")
    xy_smoothed_dfs[i].to_excel(writer_sm, sheet_name="iter_{:.0f}".format(i),
                                columns=["x_smoothed", "y_smoothed"],
                                header=True, index=True, index_label="j",
                                float_format="%.7f")
writer_sc.save()
writer_sm.save()

xy_scatter_dfs[-1].to_excel("./probe_screening/screened_kcp_vs_days.xlsx",
                            columns=["x_scatter", "y_scatter"], header=True,
                            index=True, index_label="j", float_format="%.7f")

# The second part of number 2:
xy_smoothed_dfs[-1].to_excel("./probe_screening/pruned_kcp_vs_days.xlsx",
                             columns=["x_smoothed", "y_smoothed"],
                             header=True, index=True, index_label="j",
                             float_format="%.7f")

# 3:
with open("./probe_screening/r_squared_stat_list.txt", "w") as f:
    for val in r_squared_stat_list:
        f.write(str(val) + "\n")
# 4:
with open("./probe_screening/r_sqr_cco_stat_list.txt", "w") as f:
    for val in r_sqr_cco_stat_list:
        f.write(str(val) + "\n")
# 5:
with open("./probe_screening/removed_probes.txt", "w") as f:
    for p in removed_probes:
        f.write(str(p) + "\n")
# 6:
with open("./probe_screening/healthy_probes.txt", "w") as f:
    for p in probe_ids:
        f.write(str(p) + "\n")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============================================================================
# Write a `Screening Report`, and save to disk at
# "./probe_screening/screening_report.txt"
# =============================================================================
# Open a file in writing mode located at
# "./data/probe_screening/screening_report.txt".
f = open("./probe_screening/screening_report.txt", mode="w")

f.write(top_border_1)
f.write("|" + "{:^78}".format("Screening Report") + "|\n")
f.write(line_1)
content = info_filler_1(iter_="", pnts="n scatter", cco_rs="cco",
                        sc_rs="scatter", delta="delta", removed="probe_id",
                        just_strings=True)
f.write(content)
content = info_filler_1(iter_="iter", pnts="points", cco_rs="r-squared",
                        sc_rs="r-squared", delta="scatter r-sqr",
                        removed="removed", just_strings=True)
f.write(content)
f.write(line_1)
for i in range(q + 1):
    content = info_filler_1(iter_=i, pnts=len(xy_scatter_dfs[i].index),
                            cco_rs=r_sqr_cco_stat_list[i],
                            sc_rs=r_squared_stat_list[i],
                            delta=deltas_r_squared[i],
                            removed=removed_probes[i])
    f.write(content)
f.write(line_1)
f.close()


# =============================================================================
