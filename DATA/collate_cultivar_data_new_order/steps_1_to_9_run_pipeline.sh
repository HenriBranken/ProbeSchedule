#!/bin/bash

# Keep the .csv and .xlsx files generated in the extraction pipeline.
# Remove all other old files.
# Perform data crunching and reproduce the results.

response=99
while [[ "${response}" != y && "${response}" != n ]]
do
    read -p "Are you sure you want to continue? (y/n)  Executing this script will delete files stored in ./data/ and ./figures/  " response
done
if [[ "${response}" = y ]]
then
    rm -v ./data/binned_kcp_data.xlsx
    rm -v ./data/cleaned_data_for_overlay.xlsx
    rm -v ./data/data_report.txt
    rm -v ./data/fit_of_kcp_vs_cumulative_gdd.xlsx
    rm -v ./data/kcp_vs_days.xlsx
    rm -v ./data/kcp_vs_smoothed_cumul_gdd.xlsx
    rm -v ./data/mode.txt
    rm -v ./data/prized_index.txt
    rm -v ./data/prized_n_neighbours.txt
    rm -v ./data/processed_probe_data.xlsx
    rm -v ./data/projected_weekly_data.xlsx
    rm -v ./data/reference_crop_coeff.xlsx
    rm -v ./data/smoothed_cumul_gdd_vs_season_day.xlsx
    rm -v ./data/smoothed_kcp_trend_vs_datetime.xlsx
    rm -v ./data/stacked_cleaned_data_for_overlay.xlsx
    rm -v ./data/statistics_wma_trend_lines.txt
    rm -rfv ./figures/*
    python3 step_1a_perform_cleaning.py
    python3 step_1b_write_report.py
    python3 step_2_overlay_compare.py
    python3 step_3_smoothed_version.py
    python3 step_4_binning_trend_line.py
    python3 step_5_plotting_binned_kcp_trends.py
    python3 step_6_et_al.py
    python3 step_7_figure_arrays.py
    python3 step_8a_tailor_for_weekly_bins.py
    python3 step_8b_kcp_versus_gdd.py
    python3 step_8c_fit_kcp_vs_gdd.py
#    python3 screen_outliers_henri.py
#    python3 screen_outliers_jacobus.py
else
    echo "User decided against executing run_pipeline.sh..."
    sleep 3s
fi
