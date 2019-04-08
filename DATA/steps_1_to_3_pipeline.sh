#!/bin/bash

python3 step_1_get_daily_data_csv_format.py
python3 step_2_transform_to_excel_format.py
python3 step_3_identify_duplicate_probes.py

