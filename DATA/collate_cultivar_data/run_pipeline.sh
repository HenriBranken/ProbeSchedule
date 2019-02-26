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
    rm -fv data/*
    rm -fv figures/*
    python3 main.py
    python3 overlay_compare.py
    python3 get_trends_from_stacked_data.py
    python3 binning_trend_line.py
    python3 plotting_binned_kcp_trends.py
else
    echo "User decided against executing run_pipeline.sh"
    sleep 3s
fi

