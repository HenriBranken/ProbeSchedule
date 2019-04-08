#!/bin/bash

# Execute the Python scripts related to "collate_cultivar_data" in the correct order.
# Remove (delete) any old files beforehand that are no longer of interest.

response=99
while [[ "${response}" != y && "${response}" != n ]]
do
    read -p "Are you sure you want to continue? (y/n)  Executing this script will delete current files stored in ./data/ and ./figures/  " response
done
if [[ "${response}" = y ]]
then
    rm -rfv data/*
    rm -rfv figures/*
    python3 step_1a_perform_cleaning.py
    python3 step_1b_write_report.py
    python3 step_2_overlay_compare.py
    python3 step_3_smoothed_version.py
    python3 step_4_binning_trend_line.py
    python3 step_5_plotting_binned_kcp_trends.py
    python3 step_6_et_al.py
    python3 step_7_figure_arrays.py
    python3 step_8_kcp_versus_gdd.py
    python3 step_9_fit_kcp_vs_gdd.py
else
    echo "User decided against executing run_pipeline.sh"
    sleep 3s
fi

