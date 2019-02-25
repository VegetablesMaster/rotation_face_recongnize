import requests
import json

url = 'http://192.168.1.102:5000/'
r = requests.get(url)

print("Status code:",r.status_code)

request_dict = r.json()

angle = request_dict['camera1']['length']

print(angle)
