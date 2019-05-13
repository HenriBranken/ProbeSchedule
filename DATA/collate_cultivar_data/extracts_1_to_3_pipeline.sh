#!/bin/bash

# Extract probe data from the API.
# 1. Generate a .csv file for each probe ID.
# 2. Combine the .csv files into a multi-sheet Excel spreadsheet.
# 3. Check for duplicate data sets, and remove copies if present.

response=99
while [[ "${response}" != y && "${response}" != n ]]
do
    read -p "Are you sure you want to continue? (y/n)  Executing this script will delete old files.  " response
done
if [[ "${response}" = y ]]
then
    rm -v ./data/api_dates.txt
    rm -v ./data/probe_ids.txt
    rm -v ./data/starting_year.txt
    rm -rfv ./data/*_daily_data.csv

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
        echo "All the Python scripts succeeded."
    else
        echo "The Python script extract_3_identify_duplicate_probes.py failed."
        exit 1
    fi
else
    echo "User decided against executing extracts_1_to_3_pipeline.sh..."
    sleep 3s
fi
