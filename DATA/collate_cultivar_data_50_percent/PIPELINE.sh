#!/bin/bash

# Delete all the files stored under "./data/*", "./figures/*" and "./probe_screening/*".
# Perform all the extraction steps, encapsulated in "extracts_1_to_3_pipeline.sh".
# Perform all the data crunching on the extracted data, encapsulated in "steps_1_to_8_run_pipeline.sh".

shopt -s extglob
response=99
while [[ "${response}" != y && "${response}" != n ]]
do
    read -p "Are you sure you want to continue? (y/n)  Executing this script will delete all old files.  " response
done
if [[ "${response}" = y ]]
then
    ls ./data/* | grep -v ./data/probe_numbers.txt | xargs rm
    rm -rfv ./figures/*
    ls ./probe_screening/* | grep -v ./probe_screening/create_gifs.sh | xargs rm

    python3 extract_1_get_daily_data_csv_format.py

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 extract_2_transform_to_excel_format.py
    else
        echo "The Python script extract_1_get_daily_data_csv_format.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        python3 extract_3_identify_duplicate_probes.py
    else
        echo "The Python script extract_2_transform_to_excel_format.py failed."
        exit 1
    fi

    RESULT=$?
    if [[ ${RESULT} -eq 0 ]]; then
        echo "All the extraction Python scripts succeeded."
    else
        echo "The Python script extract_3_identify_duplicate_probes.py failed."
        exit 1
    fi

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
        echo "Something went wrong while executing the create_gifs.sh script."
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
    echo "User decided against executing run_pipeline.sh..."
    sleep 3s
fi
