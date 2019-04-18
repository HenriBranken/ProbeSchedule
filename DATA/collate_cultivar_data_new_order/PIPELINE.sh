#!/bin/bash

# Delete all the files stored under `./data/*` and `./figures/*`.
# Perform all the extraction steps, encapsulated in "extracts_1_to_3_pipeline.sh".
# Perform all the data crunching on the extracted data, encapsulated in "steps_1_to_9_run_pipeline.sh".

response=99
while [[ "${response}" != y && "${response}" != n ]]
do
    read -p "Are you sure you want to continue? (y/n)  Executing this script will delete files stored in ./data/ and ./figures/  " response
done
if [[ "${response}" = y ]]
then
    rm -rfv ./data/*
    rm -rfv ./figures/*
    python3 extract_1_get_daily_data_csv_format.py
    python3 extract_2_transform_to_excel_format.py
    python3 extract_3_identify_duplicate_probes.py
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
