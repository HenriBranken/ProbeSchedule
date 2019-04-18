import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleaning_operations import KCP_MAX
from matplotlib.offsetbox import AnchoredText
import helper_meta_data as hm
import helper_functions as hf
import helper_data as hd

# -----------------------------------------------------------------------------
# Import the necessary data
# -----------------------------------------------------------------------------
# 1. "./probe_screening/xy_scatter_dfs.xlsx"
# 2. "./probe_screening/xy_smoothed_dfs.xlsx"
# 3. helper_data.cco_df
# 4. "./probe_screening/r_squared_stat_list.txt"
# 5. helper_data.cleaned_multi_df
# 6. "./probe_screening/healthy_probes.txt"
# -----------------------------------------------------------------------------
# 1.
xy_scatter_dict = pd.read_excel("./probe_screening/xy_scatter_dfs.xlsx",
                                sheet_name=None, header=0,
                                names=["x_scatter", "y_scatter"], index_col=0)
# 2.
xy_smoothed_dict = pd.read_excel("./probe_screening/xy_smoothed_dfs.xlsx",
                                 sheet_name=None, header=0,
                                 names=["x_smoothed", "y_smoothed"],
                                 index_col=0)
# 3.
cco_df = hd.cco_df.copy(deep=True)
# 4.
with open("./probe_screening/r_squared_stat_list.txt", "r") as f:
    r_squared_stat_list = []
    for val in f.readlines():
        r_squared_stat_list.append(float(val.rstrip()))
# 5.
cleaned_multi_df = hd.cleaned_multi_df.copy(deep=True)
# 6.
with open("./probe_screening/healthy_probes.txt", "r") as f:
    healthy_probes = []
    for p in f.readlines():
        healthy_probes.append(p.rstrip())
# -----------------------------------------------------------------------------


# =============================================================================
# Declare some necessary "constants"
# =============================================================================
x_limits = hm.x_limits[:]
dfs_keys = list(xy_scatter_dict.keys())
n_iterations = len(dfs_keys)
season_xticks = list(np.arange(start=x_limits[0], stop=x_limits[1], step=30))
season_start_date = hm.season_start_date
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot the progression of the scatter plots and trendlines.
# Files are saved at "./probe_screening/progression_iter_{}.png"
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for i in range(n_iterations):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.set_ylim(bottom=0, top=KCP_MAX)
    ax.set_xlabel("$n$ Days into the Season")
    ax.set_ylabel("$k_{cp}$")
    ax.set_title("$k_{cp}$ versus Days")
    ax.set_xticks(season_xticks)
    ax.grid(True)
    sc_df = xy_scatter_dict[dfs_keys[i]]
    sm_df = xy_smoothed_dict[dfs_keys[i]]
    ax.scatter(sc_df["x_scatter"].values, sc_df["y_scatter"].values,
               marker="o", color="lightseagreen", edgecolors="k",
               label="Probe Data", alpha=0.7)
    ax.scatter(cco_df["season_day"].values, cco_df["cco"].values,
               marker=".", color="yellow", label="cco", alpha=0.5)
    ax.plot(sm_df["x_smoothed"].values, sm_df["y_smoothed"].values,
            color="tomato", linewidth=2, label="Smoothed Trend", alpha=0.7)
    at = AnchoredText("Iteration: {}.\n"
                      "$r^2$ = {:.4f}.".format(i, r_squared_stat_list[i]),
                      prop=dict(size=12), frameon=True, loc="upper left",
                      pad=0.3, borderpad=0.5)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    ax.legend(loc="upper right", prop=dict(size=12), fancybox=True,
              framealpha=1.0, edgecolor="inherit")
    plt.tight_layout()
    plt.savefig("./probe_screening/progression_{:s}.png".format(dfs_keys[i]))
    plt.close()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Compare the oldest and latest trendlines with remaining healthy probes.
# Files are saved at "./probe_screening/healthy_iter_{}.png"
# -----------------------------------------------------------------------------
x_old = xy_smoothed_dict[dfs_keys[0]]["x_smoothed"].values
y_old = xy_smoothed_dict[dfs_keys[0]]["y_smoothed"].values
x_new = xy_smoothed_dict[dfs_keys[-1]]["x_smoothed"].values
y_new = xy_smoothed_dict[dfs_keys[-1]]["y_smoothed"].values
for i, p in enumerate(healthy_probes):
    probe_df = cleaned_multi_df.loc[(p, ), ["kcp"]]
    probe_df["days"] = probe_df.index - season_start_date
    probe_df["days"] = probe_df["days"].dt.days
    probe_df.sort_values(by="days", inplace=True)
    r_sqrd_old = hf.get_r_squared(x_raw=probe_df["days"].values,
                                  y_raw=probe_df["kcp"].values,
                                  x_fit=x_old, y_fit=y_old)
    r_sqrd_new = hf.get_r_squared(x_raw=probe_df["days"].values,
                                  y_raw=probe_df["kcp"].values,
                                  x_fit=x_new, y_fit=y_new)
    if r_sqrd_new <= r_sqrd_old:
        improvement = "True"
    else:
        improvement = "False"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.set_ylim(bottom=0, top=KCP_MAX)
    ax.set_xlabel("$n$ Days into the Season")
    ax.set_ylabel("$k_{cp}$")
    ax.set_title("$k_{cp}$ versus Days")
    ax.set_xticks(season_xticks)
    ax.grid(True)
    ax.scatter(cco_df["season_day"].values, cco_df["cco"].values,
               marker=".", color="yellow", label="cco", alpha=0.5)
    ax.scatter(probe_df["days"].values, probe_df["kcp"].values, marker="P",
               edgecolors="black", color="mediumslateblue", alpha=0.6,
               label="Probe Data")
    ax.plot(x_old, y_old, ls="-.", lw=2, alpha=0.6, color="sienna",
            label="Old Trend")
    ax.plot(x_new, y_new, lw=2, alpha=0.6, color="orchid",
            label="Latest Trend")
    plt.legend(loc="upper right", prop=dict(size=12), fancybox=True,
               framealpha=1.0, edgecolor="inherit")
    at = AnchoredText("ID: {}.\n"
                      "Old $r^2$ = {:.4f}.\n"
                      "New $r^2$ = {:.4f}.\n"
                      "Improvement: {}.".format(p, r_sqrd_old, r_sqrd_new,
                                                improvement),
                      prop=dict(size=12), frameon=True, loc="upper left",
                      pad=0.3, borderpad=0.5)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    plt.tight_layout()
    plt.savefig("./probe_screening/healthy_iter_{}.png".format(i))
    plt.close()
# -----------------------------------------------------------------------------


# =============================================================================
# Save the index range information.
# =============================================================================
# 1. "./probe_screening/progression_idx.txt"
# 2. "./probe_screening/healthy_idx.txt"
# =============================================================================
with open("./probe_screening/progression_idx.txt", "w") as f:
    f.write("{}\n".format(n_iterations - 1))
with open("./probe_screening/healthy_idx.txt", "w") as f:
    f.write("{}\n".format(len(healthy_probes) - 1))
# =============================================================================
