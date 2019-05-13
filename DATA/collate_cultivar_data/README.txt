---Concerning the "constants.txt" file---
RAIN_THRESHOLD:  2 mm is a reasonable default value.
ETO_MAX:  Is some educated guess.  (e.g. 12 mm for Golden Delicious Apples)
KCP_MAX:  Is some educated guess.  (e.g. 0.8 for Golden Delicious Apples)
BEGINNING_MONTH:  The calendar month when cultivar season starts.  Must be an
integer in the [1; 12] range.
ETCP_PERC_DEVIATION:  A float in the range [0.30; 0.50].  The smaller the value,
the more stringent the flagging becomes.
KCP_PERC_DEVIATION:  0.50 is a reasonable default value.
start_date:  The beginning date you feed in the API call.
T_base:  The base temperature of the cultivar.  (e.g. 10 Celsius for Golden
Delicious Apples).
pol_degree:  4 is the best choice, and should remain as such.
ALLOWED_TAIL_DEVIATION:  A float in the range [0.50; 0.75].  0.75 is a
reasonable default value.
delta_x:  1 is a reasonable value.
x_limits_left:  This should be 0.
x_limits_right:  This should be 365.
CULTIVAR:  The name of the cultivar you are investigating.  (e.g. Golden
Delicious Apples).
WEEKLY_BINNED_VERSION:  This should be True.
mode:  This should be WMA.

Below is a copy-paste for investigating Golden Delicious Apples data (in the
Southern Hemisphere):
RAIN_THRESHOLD=2
ETO_MAX=12
KCP_MAX=0.8
BEGINNING_MONTH=7
ETCP_PERC_DEVIATION=0.30
KCP_PERC_DEVIATION=0.50
start_date=2017-01-01
T_base=10
pol_degree=4
ALLOWED_TAIL_DEVIATION=0.75
delta_x=1
x_limits_left=0
x_limits_right=365
CULTIVAR=Golden Delicious Apples
WEEKLY_BINNED_VERSION=True
mode=WMA

---Concerning the extracts_1_to_3_pipeline.sh script---
This bash script fetches and formats cultivar data from the API.
To execute this script, cd to the folder containing this script, and then from
the terminal execute:
bash extracts_1_to_3_pipeline.sh

---Concerning the steps_1_to_8_run_pipeline.sh script---
This performs all the data manipulation to produce the results and figures.
You need to execute extracts_1_to_3_pipeline.sh before you can execute
steps_1_to_8_run_pipeline.sh.
To execute this script, cd to the folder containing this script, and then from
the terminal execute:
bash steps_1_to_8_run_pipeline.sh

---Concerning the PIPELINE.sh script---
This script combines all the steps in extracts_1_to_3_pipeline.sh and
steps_1_to_8_run_pipeline.sh in one script.
To execute this script, cd to the folder containing this script, and then from
the terminal execute:
bash PIPELINE.sh
