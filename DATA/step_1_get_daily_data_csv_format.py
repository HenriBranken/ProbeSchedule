# ----------------------------------------------------------------------------------------------------------------------
# Notice that in this script we extract data from:
#   https://app.probeschedule.com/data_api/v3/custom/heat_units/<ProbeID>/<start_date>/<end_date>
#   https://app.probeschedule.com/data_api/v3/custom/wb/<ProbeID>/<start_date>/<end_date>
# ----------------------------------------------------------------------------------------------------------------------

import requests
import json
import operator
import numpy as np
import datetime
from datetime import timedelta
import os
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Important parameters used in the extraction process.
# Parameters involved are:
#   1.  The date range for which we want probe data.  This is represented by [start_date; end_date].
#   2.  The probe NUMBERS for which we need to extract data.  This needs to be specified in `devices`.
#       Later in the code, we loop over `devices` to extract the individual probe data.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
start_date = '2017-01-01'
yesterday = datetime.datetime.today() - timedelta(days=1)
end_date = yesterday.strftime("%Y-%m-%d")  # get yesterday's date
devices = [370, 371, 372, 384, 391, 392, 891]
T_base = 10.0  # Base Temperature in Degrees Celcius
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------------------
# Remove the old *_daily_data.csv files
# ----------------------------------------------------------------------------------------------------------------------
directory = "./"
files = os.listdir(directory)
for file in files:
    if file.endswith("_daily_data.csv"):
        os.remove(os.path.join(directory, file))
        print("Removed the file named: {}.".format(file))
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Create a simple .txt file containing the probe_ids.  One line per probe_id.
# This .txt file lives at `./probe_ids.txt`
# ======================================================================================================================
with open("./probe_ids.txt", "w") as f:
    f.write("\n".join(("P-{:s}".format(str(device_number))) for device_number in devices))
# ======================================================================================================================


headers = {'content-type': 'application/json'}
url = 'http://api.probeschedule.com/data_api/v3/token'

data = {"type": "login", "data": {"username": "info@ai.matogen.com", "password": "oolee9ai"}}
params = {}  # empty dictionary

response = requests.post(url, params=params, data=json.dumps(data), headers=headers)
token = json.loads(response.text)
token = token['data']['token']

data = {}
headers = {'authorization': 'Bearer ' + token}


for device in devices:
    print("Currently busy with probe {}...\n".format(device))
    url = "https://app.probeschedule.com/data_api/v3/custom/heat_units/P-"+str(device)+'/'+start_date+'/'+end_date

    response = requests.get(url, params=params, data=json.dumps(data), headers=headers)
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

    url = "https://app.probeschedule.com/data_api/v3/custom/wb/P-"+str(device)+'/'+start_date+'/'+end_date

    response = requests.get(url, params=params, data=json.dumps(data), headers=headers)
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

    # I want to find a date that contains data for all the other variables so that I can extract ALL the column
    # headings
    n_keys_dict = {}
    for key, value in daily_data.items():
        n_keys = 0
        for k, v in value.items():
            n_keys += 1
        n_keys_dict[key] = n_keys
    sorted_n_keys = sorted(n_keys_dict.items(), key=operator.itemgetter(1), reverse=True)
    magic_date = sorted_n_keys[0][0]

    # Now that we finally have all the data stored in the daily_data dictionary, we can write it to an output file:
    f = open("P-"+str(device)+"_daily_data.csv", "w")
    f.write("date")
    for k, v in daily_data[magic_date].items():
        f.write(", "+k)
    f.write("\n")  # after writing all the column headings

    for date, values in daily_data.items():
        f.write(date)
        for _, v in values.items():
            f.write(', '+str(v))  # write the value
        f.write("\n")
    f.close()
