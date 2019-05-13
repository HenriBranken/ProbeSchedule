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
    cd ./data/
    GLOBIGNORE=*.csv:probe_ids.txt:api_dates.txt:base_temperature.txt:starting_year.txt:cultivar_data.xlsx:probes_to_be_discarded.txt:cultivar_data_unique.xlsx
    rm -v *
    unset GLOBIGNORE
    cd ../

    rm -rfv ./figures/*

    cd ./probe_screening/
    GLOBIGNORE=create_gifs.sh
    rm -v *
    unset GLOBIGNORE
    cd ../

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
