#!/bin/bash

response=99
while [[ "${response}" != y && "${response}" != n ]]
do
    read -p "Are you sure you want to continue? (y/n)  Executing this script will delete files stored in ./data/  " response
done
if [[ "${response}" = y ]]
then
    python3 extract_1_get_daily_data_csv_format.py
    python3 extract_2_transform_to_excel_format.py
    python3 extract_3_identify_duplicate_probes.py
else
    echo "User decided against executing pipeline.sh..."
    sleep 3s
fi