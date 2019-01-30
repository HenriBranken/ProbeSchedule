import requests
import json
import datetime
import collections

headers = {'content-type': 'application/json'}
url = 'http://api.probeschedule.com/data_api/v3/token'

data = {"type": "login", "data": {"username": "info@ai.matogen.com", "password": "oolee9ai"}}
params = {}

response = requests.post(url, params=params, data=json.dumps(data), headers=headers)
token = json.loads(response.text)
token = token['data']['token']

data = {}
headers = {'authorization': 'Bearer ' + token}

start_date = '2018-08-01'
start_datetime = start_date + ' 00:00:00'
end_date = '2018-08-01'
end_datetime = end_date + ' 00:00:00'
devices = [370]

for device in devices:

    url = 'https://app.probeschedule.com/data_api/v3/devices/P-'+str(device)+'/data/'+start_datetime+'/'+end_datetime

    response = requests.get(url, params=params, data=json.dumps(data), headers=headers)
    response = json.loads(response.text)

    f = open("P-"+str(device)+"_azm_data.csv", "w")
    for reading in response['data']['sensors']['azm']['readings']:
        reading_datetime = datetime.datetime.fromtimestamp(reading[0])
        f.write(reading_datetime.strftime("%Y-%m-%d %H-%M-%S")+','+str(reading[1])+"\n")

    f.close()

    url = "https://app.probeschedule.com/data_api/v3/custom/heat_units/P-"+str(device)+'/'+start_date+'/'+end_date

    response = requests.get(url, params=params, data=json.dumps(data), headers=headers)
    response = json.loads(response.text)

    daily_data = {}

    for reading_date, value in response['data'].iteritems():
        daily_data[reading_date] = {"heat_units": value}

    url = "https://app.probeschedule.com/data_api/v3/custom/wb/P-"+str(device)+'/'+start_date+'/'+end_date

    response = requests.get(url, params=params, data=json.dumps(data), headers=headers)
    response = json.loads(response.text)

    for reading_date, values in response['data'].iteritems():
        if reading_date not in daily_data:
            daily_data[reading_date] = {"heat_units": 0}

        for k, v in values.iteritems():
            daily_data[reading_date][k] = v

    ordered_daily_data = collections.OrderedDict(sorted(daily_data.items()))

    f = open("P-"+str(device)+"_daily_data.csv", "w")
    for date, values in ordered_daily_data.iteritems():
        f.write("date")
        for k, v in values.iteritems():
            f.write(','+k)

        f.write("\n")
        break

    for date, values in ordered_daily_data.iteritems():
        f.write(date)
        for k, v in values.iteritems():
            f.write(','+str(v))

        f.write("\n")

    f.close()
