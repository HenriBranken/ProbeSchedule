#!/bin/bash
files_of_interest=($(ls P-*_daily_data.csv))

for f in ${files_of_interest[@]}
do
    python sort_by_date.py $f
    rm -v $f
done
