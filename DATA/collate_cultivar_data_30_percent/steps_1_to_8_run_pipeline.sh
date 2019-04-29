#!/bin/bash

# Keep the .csv and .xlsx files generated in the extraction pipeline.
# Remove all other old files.
# Perform data crunching and reproduce the results.

response=99
while [[ "${response}" != y && "${response}" != n ]]
do
    read -p "Are you sure you want to continue? (y/n)  Executing this script will delete all old files.  " response
done
if [[ "${response}" = y ]]
then
    rm -v ./data/binned_kcp_data.xlsx
    rm -v ./data/cleaned_and_sorted_kcp.xlsx
    rm -v ./data/cleaned_data_for_overlay.xlsx
    rm -v ./data/data_report.txt
    rm -v ./data/first_smoothed_kcp_trend.xlsx
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
    rm -v ./data/stacked_cleaned_data_for_overlay.xlsx
    rm -v ./data/statistics_polynomial_fit.txt
    rm -v ./data/statistics_wma_trend_lines.txt
    rm -rfv ./figures/*
    rm -rfv ./probe_screening/*.txt
    rm -rfv ./probe_screening/*.gif
    rm -rfv ./probe_screening/*.png
    rm -rfv ./probe_screening/*.xlsx

    python3 step_1a_perform_cleaning.py

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_1b_write_report.py
    else
        echo "The Python script step_1a_perform_cleaning.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_2_overlay_compare.py
    else
        echo "The Python script step_1b_write_report.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_3_smoothed_version.py
    else
        echo "The Python script step_2_overlay_compare.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_4a_probe_screening.py
    else
        echo "The Python script step_3_smoothed_version.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_4b_show_progression.py
    else
        echo "The Python script step_4a_probe_screening.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        cd ./probe_screening/
        bash create_gifs.sh
        cd ../
    else
        echo "The Python script step_4b_show_progression.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_5_tailor_for_weekly_bins.py
    else
        echo "Something went wrong while executing create_gifs.sh."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_5_tailor_for_weekly_bins.py
    else
        echo "The Python script step_4b_show_progression.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_6a_kcp_versus_gdd.py
    else
        echo "The Python script step_5_tailor_for_weekly_bins.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_6b_fit_kcp_vs_gdd.py
    else
        echo "The Python script step_6a_kcp_versus_gdd.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_7_et_al.py
    else
        echo "The Python script step_6b_fit_kcp_vs_gdd.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 step_8_figure_arrays.py
    else
        echo "The Python script step_7_et_al.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        echo "All Python scripts were successfully executed."
    else
        echo "The Python script step_8_figure_arrays.py failed."
        exit 1
    fi
else
    echo "User decided against executing steps_1_to_8_run_pipeline.sh..."
    sleep 3s
fi
