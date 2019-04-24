# -----------------------------------------------------------------------------
# Notice that in this script we extract data from:
#   https://app.probeschedule.com/data_api/v3/custom/heat_units/<ProbeID>/
#   <start_date>/<end_date>
#   https://app.probeschedule.com/data_api/v3/custom/wb/<ProbeID>/
#   <start_date>/<end_date>
# -----------------------------------------------------------------------------
import requests
import json
import operator
import numpy as np
import datetime
from datetime import timedelta
import os
from helper_functions import safe_removal


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define important parameters used in the extraction process.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Parameters involved are:
# 1. The date range for which we want probe data.  This is represented by
#    [start_date; end_date].
# 2. The probe NUMBERS for which we need to extract data.  This is stored in
#    `devices`.  Later in the code, we loop over `devices` to extract the
#    individual probe data.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
start_date = '2017-01-01'
yesterday = datetime.datetime.today() - timedelta(days=1)
end_date = yesterday.strftime("%Y-%m-%d")  # get yesterday's date.
T_base = 10.0  # Base Temperature in Degrees Celsius.

# Contains the raw probe numbers, e.g.: 370, 392, 891, etc...
with open("./data/probe_numbers.txt") as f:
    devices = [int(x.rstrip()) for x in list(f) if x != '\n']
devices.sort()

# The year in which we start collecting probe data.
starting_year = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").year)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Remove the old files as we don't need them anymore.
# (If they will not be removed they will in any case be overwritten later on).
# -----------------------------------------------------------------------------
if not os.path.exists("./data"):
    os.makedirs("data")
directory = "./data/"
files = os.listdir(directory)
for file in files:
    if file.endswith("_daily_data.csv"):
        os.remove(os.path.join(directory, file))
        print("Removed the file named: {}.".format(file))
file_list = ["./data/probe_ids.txt", "./data/api_dates.txt",
             "./data/base_temperature.txt", "./data/starting_year.txt"]
safe_removal(file_list=file_list)
# -----------------------------------------------------------------------------


# =============================================================================
# Create a simple .txt file containing the probe_ids.  One line per probe_id.
# The format is P-<some_probe_number_here>, e.g.: P-370
# This .txt file lives at `./data/probe_ids.txt`.
# Create `./data/api_dates.txt` containing the API start and ending dates.
# Create `./data/base_temperature.txt` containing the base temperature of the
# cultivar.
# =============================================================================
# Text file containing all the probe IDs.
with open("./data/probe_ids.txt", "w") as f:
    f.write("\n".join(("P-{:s}".format(str(device_number))) for device_number
                      in devices))

# Text file containing the API start date and API end date.
with open("./data/api_dates.txt", "w") as f:
    f.write(start_date + "\n")
    f.write(end_date + "\n")

# Text file containing the base temperature in degrees Celsius.
with open("./data/base_temperature.txt", "w") as f:
    f.write(str(T_base))

# Text file containing the starting year in which we collect probe data.
with open("./data/starting_year.txt", "w") as f:
    f.write(str(starting_year))
# =============================================================================


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Extract probe data from the API.  In particular, collect:
# 1. heat_units data
# 2. water balance data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# We first need to get an authorisation token.
headers = {'content-type': 'application/json'}
url = 'http://api.probeschedule.com/data_api/v3/token'
data = {"type": "login",
        "data": {"username": "info@ai.matogen.com",
                 "password": "oolee9ai"}}
params = {}  # empty dictionary
response = requests.post(url, params=params, data=json.dumps(data),
                         headers=headers)
token = json.loads(response.text)
token = token['data']['token']
data = {}
headers = {'authorization': 'Bearer ' + token}


# Loop over all the different probes.
for device in devices:
    print("Currently busy with probe {}...\n".format(device))
    # In the following, extract heat_unit data.
    url = "https://app.probeschedule.com/data_api/v3/custom/heat_units/P-" + \
          str(device)+'/'+start_date+'/'+end_date

    response = requests.get(url, params=params, data=json.dumps(data),
                            headers=headers)
    response = json.loads(response.text)

    daily_data = {}

    for reading_date, nested_dict in response['data'].items():
        temps_list = nested_dict["temps"]
        temps_list = [np.float(t) for t in temps_list]
        T_min = np.min(temps_list)
        T_max = np.max(temps_list)
        T_24hour_avg = round(np.average(temps_list), 4)
        heat_units_raw = round((T_min + T_max)/2.0 - T_base, 4)
        heat_units = heat_units_raw if heat_units_raw >= 0 else float(0)
        daily_data[reading_date] = {"heat_units": heat_units,
                                    "T_min": T_min,
                                    "T_max": T_max,
                                    "T_24hour_avg": T_24hour_avg}

    # In the following, extract water_balance data.
    url = "https://app.probeschedule.com/data_api/v3/custom/wb/P-" + \
          str(device)+'/'+start_date+'/'+end_date

    response = requests.get(url, params=params, data=json.dumps(data),
                            headers=headers)
    response = json.loads(response.text)

    for reading_date, values in response['data'].items():
        if reading_date not in daily_data:
            daily_data[reading_date] = {"heat_units": np.nan,
                                        "T_min": np.nan,
                                        "T_max": np.nan,
                                        "T_24hour_avg": np.nan}
            for k, v in values.items():
                daily_data[reading_date][k] = v
        else:
            for k, v in values.items():
                daily_data[reading_date][k] = v

    # I want to find a DATE that contains data for all the other variables so
    # that I can extract ALL the COLUMN HEADINGS.
    n_keys_dict = {}
    for key, value in daily_data.items():
        n_keys = 0
        for k, v in value.items():
            n_keys += 1
        n_keys_dict[key] = n_keys
    sorted_n_keys = sorted(n_keys_dict.items(), key=operator.itemgetter(1),
                           reverse=True)
    magic_date = sorted_n_keys[0][0]

    # Now that we finally have all the data stored in the daily_data
    # dictionary, we can write it to an output file:
    f = open("./data/P-"+str(device)+"_daily_data.csv", "w")
    f.write("date")
    # Firstly write all the column headings in one line.
    for k, v in daily_data[magic_date].items():
        f.write(", "+k)
    f.write("\n")

    # Secondly, write all the column values/readings.
    for date, values in daily_data.items():
        f.write(date)
        for _, v in values.items():
            f.write(', '+str(v))  # write the value
        f.write("\n")
    f.close()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
