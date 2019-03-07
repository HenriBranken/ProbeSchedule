# ----------------------------------------------------------------------------------------------------------------------
# Notice that in this script we extract data from:
#   https://app.probeschedule.com/data_api/v3/custom/heat_units/<ProbeId>/<start_date>/<end_date>
#   https://app.probeschedule.com/data_api/v3/custom/wb/<ProbeId>/<start_date>/<end_date>
# ----------------------------------------------------------------------------------------------------------------------

import requests
import json
import operator

headers = {'content-type': 'application/json'}
url = 'http://api.probeschedule.com/data_api/v3/token'

data = {"type": "login", "data": {"username": "info@ai.matogen.com", "password": "oolee9ai"}}
params = {}  # empty dictionary

response = requests.post(url, params=params, data=json.dumps(data), headers=headers)
token = json.loads(response.text)
token = token['data']['token']

data = {}
headers = {'authorization': 'Bearer ' + token}

# Insert here your beginning date and end date
start_date = '2017-08-01'
start_datetime = start_date + ' 00:00:00'
end_date = '2019-02-01'
end_datetime = end_date + ' 00:00:00'
devices = [370, 371, 372, 384, 391, 392, 891]

for device in devices:
    print("Currently busy with probe {}...".format(device))
    url = "https://app.probeschedule.com/data_api/v3/custom/heat_units/P-"+str(device)+'/'+start_date+'/'+end_date

    response = requests.get(url, params=params, data=json.dumps(data), headers=headers)
    response = json.loads(response.text)

    daily_data = {}

    for reading_date, value in response['data'].items():
        daily_data[reading_date] = {"heat_units": value}

    url = "https://app.probeschedule.com/data_api/v3/custom/wb/P-"+str(device)+'/'+start_date+'/'+end_date

    response = requests.get(url, params=params, data=json.dumps(data), headers=headers)
    response = json.loads(response.text)

    for reading_date, values in response['data'].items():
        if reading_date not in daily_data:
            daily_data[reading_date] = {"heat_units": 0}  # basically padding empty heat unit entries to dictionary
            for k, v in values.items():
                daily_data[reading_date][k] = v
        else:
            for k, v in values.items():
                daily_data[reading_date][k] = v

    # I want to find a date that contains data for all the other 21 variables so that I can extract ALL the column
    # headings
    n_keys_dict = {}
    for key, value in daily_data.items():
        n_keys = 0
        for k, v in value.items():
            n_keys += 1
        n_keys_dict[key] = n_keys
    sorted_n_keys = sorted(n_keys_dict.items(), key=operator.itemgetter(1), reverse=True)
    magic_date = sorted_n_keys[0][0]

    # Now that we finally have all the data stored in the daily_data dictionary, we can write it output file:
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
