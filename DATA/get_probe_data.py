import requests
import json
import datetime

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
end_date = '2018-08-01'
end_datetime = end_date + ' 00:00:00'

# Insert here all the Probe-IDs of interest
devices = [370, 371, 372, 384, 391, 392, 891, 1426]

for device in devices:
    print("Currently working with Probe: {}...".format(device))
    url = 'https://app.probeschedule.com/data_api/v3/devices/P-'+str(device)+'/data/'+start_datetime+'/'+end_datetime
    
    response = requests.get(url, params=params, data=json.dumps(data), headers=headers)
    response = json.loads(response.text)
    my_data = dict()

    variables = []
    for i in response["data"]["sensors"]:
        variables.append(i)
    
    variables_and_names = []
    for j in variables:
        variables_and_names.append(j + " [" + str(response["data"]["sensors"][j]["name"]) + "]")

    for d in response["data"]["sensors"]["azm"]["readings"]:
        my_data[d[0]] = dict()

    for i in my_data.items():
        for j in variables_and_names:
            my_data[i[0]][j] = dict()

    for i, j in zip(variables, variables_and_names):
        for k1, k2 in response["data"]["sensors"][i]["readings"]:
            # i->variable, j->variable+name, k1->timestamp, k2->value
            my_data[k1][j] = k2

    f = open("P-"+str(device)+"_probe_data.csv", "w")
    f.write("datatime")
    for i in variables_and_names:
        f.write(", "+str(i))
    f.write("\n")

    for timestamp in my_data:
        f.write(datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"))
        for i in variables_and_names:
            f.write(", " + str(my_data[timestamp][i]))
        f.write("\n")
    f.close()
